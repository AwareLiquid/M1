"""PMB task generators — T1 / T2 / T3, fully seed-deterministic.

Design contract (PMB_SPEC.md §2):

* EVERYTHING flows from ``random.Random(seed_string)``. No global ``random``,
  no wall clock, no ``hash()`` of composite objects (PYTHONHASHSEED would make
  that non-reproducible). Seed strings like ``"0:t1:16:4"`` are hashed by
  ``random.Random`` with SHA-512, which IS stable across processes.
* Fact values are unique 6-digit codes (``100000``–``999999``). ALL codes in
  an episode — gold, stale (T2), confuser and decoy — are drawn in ONE global
  without-replacement pass, so a gold value can never collide with any other
  code anywhere in the episode and exact-substring matching stays unambiguous.
* Keys are (person name, attribute) pairs rendered through diversified
  statement templates. Anti-saturation hardening (v0.1):

  - Every queried person carries ``confusers_per_key`` (default 2) EXTRA facts
    about other attributes of the same person (each with its own code); the
    question asks about exactly one attribute. Retrieving the wrong-attribute
    fact = entity-level granularity is not enough — the system must bind
    entity AND attribute.
  - Every distractor session embeds ``decoys_per_distractor`` (default 8)
    decoy facts ("other person - attribute - 6-digit code"), phrased with the
    SAME ``FACT_TEMPLATES`` as the real facts. "Find a 6-digit number" or
    "find a fact-shaped sentence" is no longer a winning strategy.

* Distractor sessions are ~2K tokens of templated filler built from a
  vocabulary disjoint from the fact/question wording; the filler sentences
  themselves contain no digits, and the decoy codes woven into them come from
  the global without-replacement pool (never gold/stale).

Invariants preserved by construction: gold codes are globally unique; an
oracle with an unbounded window still recalls 100%; ``none`` still scores 0.

Unified episode structure returned by every generator::

    {
      "task":      "t1" | "t2" | "t3",
      "params":    {...},                       # generator parameters
      "sessions":  [(session_id, text), ...],   # ingest order
      "questions": [{"q": str, "gold": str, "stale": [str, ...]}, ...],
    }
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bruno", "Carla", "Derek", "Elena", "Farid", "Greta", "Hugo",
    "Ines", "Jonas", "Katya", "Lionel", "Marta", "Nadia", "Oscar", "Priya",
    "Quentin", "Rosa", "Stefan", "Tamara", "Ulrich", "Vera", "Wendell",
    "Ximena", "Yusuf", "Zelda", "Anders", "Bianca", "Cyrus", "Dalia",
    "Emeric", "Fiona", "Gustav", "Helga", "Igor", "Jasmine", "Kofi", "Leona",
    "Milan", "Noor", "Otto", "Paloma", "Rashid", "Sonia", "Tobias", "Uma",
    "Viktor", "Wanda",
]

LAST_NAMES = [
    "Almeida", "Bergstrom", "Castellanos", "Dubois", "Eriksen", "Farkas",
    "Grimaldi", "Hoffmann", "Ivanova", "Jankowski", "Kaufman", "Lindqvist",
    "Moreau", "Nakamura", "Olsen", "Petrov", "Quiroga", "Rossi", "Santos",
    "Takahashi", "Ulloa", "Vasquez", "Weiss", "Xu", "Yamamoto", "Ziegler",
    "Abernathy", "Bhattacharya", "Cormier", "Delgado", "Engstrom", "Fontaine",
]

ATTRIBUTES = [
    "locker code", "badge number", "parking permit", "vault code",
    "employee number", "access code", "library card number", "gym pass code",
    "ticket reference", "customer number", "storage unit code",
    "conference pin", "gate code", "membership number", "shipment reference",
    "invoice number",
]

FACT_TEMPLATES = [
    "{name}'s {attr} is {value}.",
    "The {attr} assigned to {name} is {value}.",
    "For the record, {name} has the {attr} {value}.",
    "Please remember that {name}'s {attr} is {value}.",
    "According to the registry, the {attr} of {name} is {value}.",
    "{name} was issued the {attr} {value}.",
]

UPDATE_TEMPLATES = [
    "{name}'s {attr} has changed to {value}.",
    "Update: {name}'s {attr} is now {value}.",
    "The {attr} of {name} was reassigned to {value}.",
    "Effective today, {name}'s {attr} becomes {value}.",
    "Correction: the {attr} registered to {name} is now {value}.",
]

QUESTION_TEMPLATES = [
    "What is {name}'s {attr}?",
    "Please state the {attr} of {name}.",
    "Which {attr} is registered to {name}?",
]

# Distractor vocabulary — deliberately disjoint from fact/question wording.
# The filler sentences themselves contain no digits; the decoy facts woven in
# by make_distractor carry codes from the episode's global without-replacement
# pool, so a gold value can never appear inside a distractor session.
_FILLER_SUBJECTS = [
    "the harbor crew", "a migrating flock", "the mountain trail", "the old mill",
    "the botanical garden", "a passing storm front", "the village orchestra",
    "the ferry schedule", "the pottery workshop", "a wandering fox",
    "the lighthouse keeper", "the morning market", "the glacier survey",
    "the tea plantation", "an amateur astronomer", "the river delta",
]

_FILLER_VERBS = [
    "drifted past", "settled near", "reorganized around", "flourished beside",
    "meandered through", "echoed across", "gathered along", "faded behind",
    "circled above", "rested beneath", "sprawled toward", "brightened over",
]

_FILLER_OBJECTS = [
    "the eastern ridge", "a quiet meadow", "the tidal flats", "an abandoned pier",
    "the cedar grove", "a limestone quarry", "the winding canal",
    "a mossy embankment", "the terraced hillside", "an overgrown courtyard",
    "the frozen estuary", "a sunlit clearing", "the basalt cliffs",
    "a forgotten vineyard",
]

_FILLER_CLAUSES = [
    "while the wind kept shifting",
    "long before the fog lifted",
    "as the tide slowly receded",
    "although the season felt unusually mild",
    "when the last ferry had already departed",
    "since the rains had softened the ground",
    "even though the path stayed muddy",
    "after the festival banners came down",
    "because the harvest ran late this year",
    "while distant thunder rolled on",
]

DISTRACTOR_TARGET_WORDS = 1500  # ~2K tokens of filler per distractor session

# Anti-saturation hardening defaults (v0.1) — see module docstring.
DECOYS_PER_DISTRACTOR = 8   # decoy person-attribute-code facts per distractor
CONFUSERS_PER_KEY = 2       # extra other-attribute facts per queried person


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _rng(seed: int, *tags) -> random.Random:
    """A process-stable RNG. String seeds are hashed with SHA-512 by
    ``random.Random`` (deterministic), unlike ``hash(tuple)`` which is
    randomized by PYTHONHASHSEED."""
    return random.Random(":".join(str(t) for t in (seed, *tags)))


def _sample_codes(rng: random.Random, n: int) -> List[str]:
    """*n* unique 6-digit codes. No leading zeros -> always exactly 6 chars,
    exact-substring matching is unambiguous."""
    return [f"{c}" for c in rng.sample(range(100_000, 1_000_000), n)]


def _sample_persons(rng: random.Random, n: int) -> List[str]:
    """*n* unique full names (targets AND decoys drawn together), so a
    question about one person can never be satisfied by another person's
    fact — decoy persons are guaranteed distinct from queried persons."""
    pairs = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    return [f"{f} {l}" for f, l in rng.sample(pairs, n)]


def _sample_attr_sets(rng: random.Random, n: int,
                      confusers_per_key: int) -> List[List[str]]:
    """For each of *n* queried persons: ``1 + confusers_per_key`` DISTINCT
    attributes. Index 0 is the asked attribute; the rest are confusers."""
    return [rng.sample(ATTRIBUTES, 1 + confusers_per_key) for _ in range(n)]


def make_distractor(rng: random.Random,
                    decoy_lines: Tuple[str, ...] = (),
                    target_words: int = DISTRACTOR_TARGET_WORDS) -> str:
    """Templated irrelevant filler (~*target_words* words, digit-free) with
    *decoy_lines* (decoy person-attribute-code facts) woven in at
    deterministic random positions."""
    sentences: List[str] = []
    words = 0
    while words < target_words:
        s = "{} {} {} {}.".format(
            rng.choice(_FILLER_SUBJECTS).capitalize(),
            rng.choice(_FILLER_VERBS),
            rng.choice(_FILLER_OBJECTS),
            rng.choice(_FILLER_CLAUSES),
        )
        sentences.append(s)
        words += len(s.split())
    for line in decoy_lines:
        sentences.insert(rng.randrange(len(sentences) + 1), line)
    return " ".join(sentences)


def _target_fact_lines(rng: random.Random, persons: List[str],
                       attr_sets: List[List[str]], asked_codes: List[str],
                       confuser_codes: List[str],
                       templates: List[str]) -> List[str]:
    """Render the asked-attribute fact plus every confuser-attribute fact for
    each queried person, shuffled together within the session."""
    lines: List[str] = []
    ci = 0
    for i, person in enumerate(persons):
        lines.append(_render_fact(rng, person, attr_sets[i][0],
                                  asked_codes[i], templates))
        for attr in attr_sets[i][1:]:
            lines.append(_render_fact(rng, person, attr,
                                      confuser_codes[ci], templates))
            ci += 1
    rng.shuffle(lines)
    return lines


def _decoy_lines(rng: random.Random, persons: List[str], attrs: List[str],
                 codes: List[str]) -> List[str]:
    """Decoy facts phrased with the same FACT_TEMPLATES as real facts."""
    return [_render_fact(rng, p, a, c, FACT_TEMPLATES)
            for p, a, c in zip(persons, attrs, codes)]


def _render_fact(rng: random.Random, name: str, attr: str, value: str,
                 templates: List[str]) -> str:
    return rng.choice(templates).format(name=name, attr=attr, value=value)


def _render_question(rng: random.Random, name: str, attr: str) -> str:
    return rng.choice(QUESTION_TEMPLATES).format(name=name, attr=attr)


# ---------------------------------------------------------------------------
# T1 · Cross-Session Retention
# ---------------------------------------------------------------------------

T1_N_VALUES = (4, 16, 64)
T1_K_VALUES = (0, 4, 16)


def generate_t1(seed: int, n_facts: int, k_distractors: int,
                decoys_per_distractor: int = DECOYS_PER_DISTRACTOR,
                confusers_per_key: int = CONFUSERS_PER_KEY) -> Dict:
    """Session 1 injects *n_facts* facts (each queried person also carries
    *confusers_per_key* other-attribute facts); *k_distractors* filler
    sessions follow, each embedding *decoys_per_distractor* decoy facts;
    questions probe every asked fact afterwards."""
    rng = _rng(seed, "t1", n_facts, k_distractors)
    n_decoys = k_distractors * decoys_per_distractor
    persons = _sample_persons(rng, n_facts + n_decoys)
    targets, decoy_persons = persons[:n_facts], persons[n_facts:]
    attr_sets = _sample_attr_sets(rng, n_facts, confusers_per_key)
    decoy_attrs = [rng.choice(ATTRIBUTES) for _ in range(n_decoys)]

    # ONE global no-replacement draw: gold / confuser / decoy never collide.
    codes = _sample_codes(rng, n_facts * (1 + confusers_per_key) + n_decoys)
    gold_codes = codes[:n_facts]
    confuser_codes = codes[n_facts:n_facts * (1 + confusers_per_key)]
    decoy_codes = codes[n_facts * (1 + confusers_per_key):]

    fact_lines = _target_fact_lines(rng, targets, attr_sets, gold_codes,
                                    confuser_codes, FACT_TEMPLATES)
    sessions: List[Tuple[str, str]] = [
        (f"t1_n{n_facts}_k{k_distractors}_s000_facts", " ".join(fact_lines))
    ]
    for i in range(k_distractors):
        lo = i * decoys_per_distractor
        hi = lo + decoys_per_distractor
        sessions.append((
            f"t1_n{n_facts}_k{k_distractors}_s{i + 1:03d}_distractor",
            make_distractor(rng, _decoy_lines(rng, decoy_persons[lo:hi],
                                              decoy_attrs[lo:hi],
                                              decoy_codes[lo:hi])),
        ))

    questions = [
        {"q": _render_question(rng, name, attr_sets[i][0]),
         "gold": gold_codes[i], "stale": []}
        for i, name in enumerate(targets)
    ]
    return {
        "task": "t1",
        "params": {"seed": seed, "n_facts": n_facts,
                   "k_distractors": k_distractors,
                   "decoys_per_distractor": decoys_per_distractor,
                   "confusers_per_key": confusers_per_key},
        "sessions": sessions,
        "questions": questions,
    }


def t1_grid(seed: int) -> Iterator[Tuple[Dict, Dict]]:
    """Yield (params, episode) over the mandated (N, K) grid."""
    for n in T1_N_VALUES:
        for k in T1_K_VALUES:
            ep = generate_t1(seed, n, k)
            yield ep["params"], ep


# ---------------------------------------------------------------------------
# T2 · Streaming Update Recall
# ---------------------------------------------------------------------------

def generate_t2(seed: int, n_keys: int = 8, n_updates: int = 3,
                distractors_between: int = 1,
                decoys_per_distractor: int = DECOYS_PER_DISTRACTOR,
                confusers_per_key: int = CONFUSERS_PER_KEY) -> Dict:
    """Each key's value is rewritten in *n_updates* successive update rounds
    (round 0 = initial write, which also carries the confuser-attribute facts),
    with *distractors_between* decoy-bearing filler sessions between rounds.
    gold = the LATEST value; stale = every earlier value."""
    if n_updates < 2:
        raise ValueError("T2 needs n_updates >= 2 (at least one rewrite)")
    rng = _rng(seed, "t2", n_keys, n_updates, distractors_between)
    n_distractor_sessions = (n_updates - 1) * distractors_between
    n_decoys = n_distractor_sessions * decoys_per_distractor
    persons = _sample_persons(rng, n_keys + n_decoys)
    targets, decoy_persons = persons[:n_keys], persons[n_keys:]
    attr_sets = _sample_attr_sets(rng, n_keys, confusers_per_key)
    decoy_attrs = [rng.choice(ATTRIBUTES) for _ in range(n_decoys)]

    # ONE global no-replacement draw covering every value chain, every
    # confuser and every decoy: gold / stale / confuser / decoy never collide.
    all_codes = _sample_codes(
        rng, n_keys * n_updates + n_keys * confusers_per_key + n_decoys)
    values = [all_codes[i * n_updates:(i + 1) * n_updates]
              for i in range(n_keys)]  # values[key_idx][round]
    confuser_codes = all_codes[n_keys * n_updates:
                               n_keys * (n_updates + confusers_per_key)]
    decoy_codes = all_codes[n_keys * (n_updates + confusers_per_key):]

    sessions: List[Tuple[str, str]] = []
    sid = 0
    decoy_cursor = 0
    for r in range(n_updates):
        if r == 0:
            # Initial write: asked facts + confuser facts, shuffled together.
            lines = _target_fact_lines(
                rng, targets, attr_sets, [values[i][0] for i in range(n_keys)],
                confuser_codes, FACT_TEMPLATES)
        else:
            lines = [
                _render_fact(rng, name, attr_sets[i][0], values[i][r],
                             UPDATE_TEMPLATES)
                for i, name in enumerate(targets)
            ]
        sessions.append((f"t2_s{sid:03d}_round{r}", " ".join(lines)))
        sid += 1
        if r < n_updates - 1:
            for _ in range(distractors_between):
                lo, hi = decoy_cursor, decoy_cursor + decoys_per_distractor
                sessions.append((
                    f"t2_s{sid:03d}_distractor",
                    make_distractor(rng, _decoy_lines(
                        rng, decoy_persons[lo:hi], decoy_attrs[lo:hi],
                        decoy_codes[lo:hi])),
                ))
                decoy_cursor = hi
                sid += 1

    questions = [
        {
            "q": _render_question(rng, name, attr_sets[i][0]),
            "gold": values[i][n_updates - 1],          # latest value
            "stale": values[i][: n_updates - 1],       # all superseded values
        }
        for i, name in enumerate(targets)
    ]
    return {
        "task": "t2",
        "params": {"seed": seed, "n_keys": n_keys, "n_updates": n_updates,
                   "distractors_between": distractors_between,
                   "decoys_per_distractor": decoys_per_distractor,
                   "confusers_per_key": confusers_per_key},
        "sessions": sessions,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# T3 · Forgetting Curve
# ---------------------------------------------------------------------------

T3_GAPS = (0, 2, 8, 32)
T3_N_FACTS = 16


def generate_t3(seed: int, gap: int, n_facts: int = T3_N_FACTS,
                decoys_per_distractor: int = DECOYS_PER_DISTRACTOR,
                confusers_per_key: int = CONFUSERS_PER_KEY) -> Dict:
    """Fixed injection of *n_facts* facts (plus confuser-attribute facts),
    then *gap* decoy-bearing distractor sessions (~2K tokens each), then
    questions. One episode per curve point."""
    rng = _rng(seed, "t3", gap, n_facts)
    n_decoys = gap * decoys_per_distractor
    persons = _sample_persons(rng, n_facts + n_decoys)
    targets, decoy_persons = persons[:n_facts], persons[n_facts:]
    attr_sets = _sample_attr_sets(rng, n_facts, confusers_per_key)
    decoy_attrs = [rng.choice(ATTRIBUTES) for _ in range(n_decoys)]

    codes = _sample_codes(rng, n_facts * (1 + confusers_per_key) + n_decoys)
    gold_codes = codes[:n_facts]
    confuser_codes = codes[n_facts:n_facts * (1 + confusers_per_key)]
    decoy_codes = codes[n_facts * (1 + confusers_per_key):]

    fact_lines = _target_fact_lines(rng, targets, attr_sets, gold_codes,
                                    confuser_codes, FACT_TEMPLATES)
    sessions: List[Tuple[str, str]] = [
        (f"t3_gap{gap}_s000_facts", " ".join(fact_lines))
    ]
    for i in range(gap):
        lo = i * decoys_per_distractor
        hi = lo + decoys_per_distractor
        sessions.append((f"t3_gap{gap}_s{i + 1:03d}_distractor",
                         make_distractor(rng, _decoy_lines(
                             rng, decoy_persons[lo:hi], decoy_attrs[lo:hi],
                             decoy_codes[lo:hi]))))

    questions = [
        {"q": _render_question(rng, name, attr_sets[i][0]),
         "gold": gold_codes[i], "stale": []}
        for i, name in enumerate(targets)
    ]
    return {
        "task": "t3",
        "params": {"seed": seed, "gap": gap, "n_facts": n_facts,
                   "decoys_per_distractor": decoys_per_distractor,
                   "confusers_per_key": confusers_per_key},
        "sessions": sessions,
        "questions": questions,
    }


def t3_curve(seed: int, n_facts: int = T3_N_FACTS) -> Iterator[Tuple[Dict, Dict]]:
    """Yield (params, episode) for every mandated gap — the FULL curve
    (single-point reporting is invalid per spec §2/T3)."""
    for gap in T3_GAPS:
        ep = generate_t3(seed, gap, n_facts)
        yield ep["params"], ep
