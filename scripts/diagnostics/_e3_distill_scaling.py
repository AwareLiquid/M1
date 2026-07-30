"""E3 (ITERATION_PRINCIPLES.md): O-series distillation scaling extrapolation.

Data (BENCHMARKS.md, teacher-forced protocol only — round 1 used the broken
free-running alignment and sits 10x off any fit, excluded):
    round 2:  5M cumulative KD tokens -> PPL 32.9
    round 3: 18M cumulative KD tokens -> PPL 25.4
    teacher: 11.8   (WikiText-2 eval)

Two candidate functional forms (2 points cannot discriminate them — that IS
the finding; compute where each predicts the targets):
    log-linear:  PPL(D) = a - b*ln(D)
    power law on excess PPL:  PPL(D) - teacher = C * D^(-alpha)
"""
import math

TEACHER = 11.8
D2, P2 = 5e6, 32.9
D3, P3 = 18e6, 25.4

print("=== log-linear: PPL = a - b*ln(D) ===")
b = (P2 - P3) / math.log(D3 / D2)
a = P2 + b * math.log(D2)
for target, label in [(TEACHER * 1.2, "20% over teacher (PPL 14.2)"),
                      (TEACHER * 1.1, "10% over teacher (PPL 13.0)"),
                      (TEACHER, "parity (PPL 11.8)")]:
    D = math.exp((a - target) / b)
    print(f"  {label}: {D/1e6:,.0f}M tokens ({D/D3:,.0f}x current)")

print("\n=== power law: PPL - teacher = C * D^-alpha ===")
E2, E3 = P2 - TEACHER, P3 - TEACHER
alpha = math.log(E2 / E3) / math.log(D3 / D2)
C = E2 * (D2 ** alpha)
print(f"  alpha = {alpha:.3f}  (typical KD-scaling alphas: 0.3-0.6)")
for target, label in [(TEACHER * 0.2, "20% over teacher"),
                      (TEACHER * 0.1, "10% over teacher"),
                      (TEACHER * 0.05, "5% over teacher")]:
    D = (C / target) ** (1 / alpha)
    print(f"  {label} (excess {target:.2f}): {D/1e9:,.2f}B tokens ({D/D3:,.0f}x current)")

print("\n=== sanity: what would round 1 (2M, 264) predict under each fit? ===")
ll = a - b * math.log(2e6)
pl = TEACHER + C * (2e6 ** -alpha)
print(f"  log-linear predicts {ll:.0f}, power predicts {pl:.0f}, actual 264"
      f" -> round 1 is off both fits by ~10x (different protocol, exclusion justified)")

print("\n=== discriminating experiment: round 4 at ~55M cumulative ===")
D4 = 55e6
print(f"  log-linear predicts PPL {a - b*math.log(D4):.1f}")
print(f"  power law  predicts PPL {TEACHER + C*(D4**-alpha):.1f}")
print("  gap between predictions is the cheapest way to pin the exponent"
      " before committing to a 183M (optimistic) vs 3B+ (pessimistic) budget")
