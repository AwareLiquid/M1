"""Analyze and visualize ablation study results.

Usage:
    python scripts/analyze_ablations.py artifacts/ablations/ablation_adapter_type_results.json
    python scripts/analyze_ablations.py artifacts/ablations/*.json --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_results(path: str) -> List[dict]:
    """Load ablation results from JSON."""
    with open(path) as f:
        return json.load(f)


def print_comparison_table(results: List[dict], title: str = "Ablation Results"):
    """Print formatted comparison table."""
    print(f"\n{'='*100}")
    print(title)
    print("="*100)
    print(f"{'Name':<25} {'Trainable %':>12} {'Final PPL':>12} {'PPL vs Best':>12} {'Tok/s':>10}")
    print("-"*100)

    # Sort by PPL (lower is better)
    sorted_results = sorted(results, key=lambda r: r["final_ppl"])
    best_ppl = sorted_results[0]["final_ppl"]

    for r in sorted_results:
        ppl_delta = ((r["final_ppl"] - best_ppl) / best_ppl) * 100
        print(
            f"{r['name']:<25} {r['trainable_percent']:>11.3f}% "
            f"{r['final_ppl']:>12.2f} {ppl_delta:>11.1f}% "
            f"{r['tokens_per_sec']:>10.0f}"
        )

    print()
    print(f"Best configuration: {sorted_results[0]['name']} (PPL={sorted_results[0]['final_ppl']:.2f})")
    print(f"Worst configuration: {sorted_results[-1]['name']} (PPL={sorted_results[-1]['final_ppl']:.2f})")
    print(f"Range: {((sorted_results[-1]['final_ppl'] - best_ppl) / best_ppl * 100):.1f}% worse than best")


def analyze_group(results: List[dict], group_name: str):
    """Analyze results for a specific ablation group."""
    print(f"\n{'='*100}")
    print(f"Analysis: {group_name}")
    print("="*100)

    if "every" in group_name or "interval" in group_name:
        # Layer interval analysis
        print("\nLayer Interval Analysis:")
        print("Question: Does adapter placement density matter?")
        print()
        for r in sorted(results, key=lambda x: x["final_ppl"]):
            layers = len(r["wrapped_layers"])
            print(f"  {r['name']}: {layers} layers covered, PPL={r['final_ppl']:.2f}")
        print()
        print("Interpretation:")
        print("  - More layers = more params but also more coverage")
        print("  - Sparse placement may be sufficient if MT architecture is effective")

    elif "lora" in group_name and "rank" in group_name:
        # LoRA rank analysis
        print("\nLoRA Rank Analysis:")
        print("Question: Does LoRA capacity matter when combined with MT?")
        print()
        for r in sorted(results, key=lambda x: x["final_ppl"]):
            print(f"  {r['name']}: trainable={r['trainable_percent']:.3f}%, PPL={r['final_ppl']:.2f}")
        print()
        print("Interpretation:")
        print("  - Higher rank = more LoRA params")
        print("  - If r=4 ≈ r=8 ≈ r=16, then MT architecture (not LoRA capacity) is the main driver")

    elif "type" in group_name or "adapter" in group_name:
        # Adapter type analysis
        print("\nAdapter Type Analysis:")
        print("Question: What is the contribution of MT vs LoRA?")
        print()
        mt_only = next((r for r in results if "mt_only" in r["name"]), None)
        lora_only = next((r for r in results if "lora_only" in r["name"]), None)
        mt_lora = next((r for r in results if "mt_plus_lora" in r["name"] or "phase5b" in r["name"]), None)

        if mt_only and lora_only and mt_lora:
            print(f"  MT only:       PPL={mt_only['final_ppl']:.2f}")
            print(f"  LoRA only:     PPL={lora_only['final_ppl']:.2f}")
            print(f"  MT + LoRA:     PPL={mt_lora['final_ppl']:.2f}")
            print()

            # Check if combination is better than either alone
            best_single = min(mt_only['final_ppl'], lora_only['final_ppl'])
            if mt_lora['final_ppl'] < best_single:
                improvement = ((best_single - mt_lora['final_ppl']) / best_single) * 100
                print(f"  ✓ Combination wins: {improvement:.1f}% better than best single method")
            else:
                print("  ✗ Combination does not beat best single method")

            # Check additivity
            if mt_only['final_ppl'] < lora_only['final_ppl']:
                print(f"  MT is the stronger component")
            else:
                print(f"  LoRA is the stronger component")

        print()
        print("Interpretation:")
        print("  - If MT >> LoRA: MT architecture provides the long-context inductive bias")
        print("  - If LoRA >> MT: Parameter efficiency (not architecture) is the main benefit")
        print("  - If MT+LoRA >> either: Both contribute complementary benefits")

    elif "proto" in group_name:
        # Protofilament analysis
        print("\nProtofilament Count Analysis:")
        print("Question: Does biological microtubule count (13) matter?")
        print()
        for r in sorted(results, key=lambda x: x["final_ppl"]):
            print(f"  {r['name']}: PPL={r['final_ppl']:.2f}")
        print()
        print("Interpretation:")
        print("  - If 13 is best: supports biological prior hypothesis")
        print("  - If performance scales with count: it's just capacity")


def compare_across_groups(all_results: dict):
    """Compare best configurations across ablation groups."""
    print(f"\n{'='*100}")
    print("CROSS-GROUP COMPARISON: Best from Each Ablation")
    print("="*100)

    best_configs = []
    for group_name, results in all_results.items():
        best = min(results, key=lambda r: r["final_ppl"])
        best_configs.append({
            "group": group_name,
            "name": best["name"],
            "ppl": best["final_ppl"],
            "trainable": best["trainable_percent"],
        })

    best_configs = sorted(best_configs, key=lambda x: x["ppl"])

    print(f"{'Group':<20} {'Best Config':<25} {'PPL':>12} {'Trainable %':>12}")
    print("-"*100)
    for bc in best_configs:
        print(f"{bc['group']:<20} {bc['name']:<25} {bc['ppl']:>12.2f} {bc['trainable']:>11.3f}%")

    print()
    print(f"Overall winner: {best_configs[0]['name']} from {best_configs[0]['group']} (PPL={best_configs[0]['ppl']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--compare", action="store_true", help="Compare across groups")
    args = parser.parse_args()

    if args.compare:
        # Load all groups
        all_results = {}
        for fpath in args.files:
            group_name = Path(fpath).stem.replace("ablation_", "").replace("_results", "")
            all_results[group_name] = load_results(fpath)

        # Individual group analysis
        for group_name, results in all_results.items():
            print_comparison_table(results, f"Ablation Group: {group_name}")
            analyze_group(results, group_name)

        # Cross-group comparison
        compare_across_groups(all_results)

    else:
        # Single file analysis
        for fpath in args.files:
            results = load_results(fpath)
            group_name = Path(fpath).stem.replace("ablation_", "").replace("_results", "")
            print_comparison_table(results, f"Ablation: {group_name}")
            analyze_group(results, group_name)


if __name__ == "__main__":
    main()
