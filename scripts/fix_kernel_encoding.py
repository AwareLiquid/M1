#!/usr/bin/env python3
"""Replace all non-ASCII characters in Kaggle kernel .py files with ASCII equivalents."""
import os

replacements = {
    "—": "--",   # em-dash
    "–": "-",    # en-dash
    "‒": "-",    # figure dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # ellipsis
    "→": "->",   # right arrow
    "←": "<-",   # left arrow
    "±": "+-",   # plus-minus
    "×": "x",    # multiplication
    "α": "alpha",
    "β": "beta",
    "τ": "tau",
    "κ": "kappa",
    "φ": "phi",
    "γ": "gamma",
}

kernel_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle_kernels")

for root, dirs, files in os.walk(kernel_root):
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        new_content = content
        for ch, rep in replacements.items():
            new_content = new_content.replace(ch, rep)
        new_content = new_content.encode("ascii", errors="replace").decode("ascii")
        if new_content != content:
            with open(path, "w", encoding="ascii") as fh:
                fh.write(new_content)
            print(f"Fixed: {path}")
        else:
            print(f"OK:    {path}")

print("Done.")
