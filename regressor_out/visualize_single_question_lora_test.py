# -*- coding: utf-8 -*-
"""
Trait Hi/Lo stats plot for single_question_lora_test_result_with_regressor_preserved.json

- Uses ONLY the OUTER 'High'/'Low' keys as model labels (ignores inner labels).
- Uses 'regressor_score' if present; falls back to 'score'.
- Single matplotlib figure, no seaborn, no explicit colors.

Usage:
  python visualize_single_question_lora_test.py \
    --input regressor_out/single_question_lora_test_result_with_regressor_preserved.json \
    --output regressor_out/image/lora/single_q_stats.png
"""

import os
import json
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

TRAITS = ["O", "C", "E", "A", "N"]


def _extract_score(x: Dict[str, Any]) -> float | None:
    """Prefer 'regressor_score', fallback to 'score'."""
    if not isinstance(x, dict):
        return None
    if "regressor_score" in x:
        return float(x["regressor_score"])
    if "score" in x:
        return float(x["score"])
    return None


def _collect_flat_list(group: Any, trait: str, outer_label: str, rows: List[Dict[str, Any]]) -> None:
    """Collect scores from the known flat list structure: High/Low -> [ items... ]."""
    if not isinstance(group, list):
        return
    for item in group:
        s = _extract_score(item)
        if s is None:
            continue
        rows.append({
            "trait": trait,
            "label": outer_label.lower(),  # enforce outer label only
            "score": s,
        })


def load_df_from_file(path: str) -> pd.DataFrame:
    """Load the specific file format: trait -> High/Low -> [ items ]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for trait in TRAITS:
        if trait not in data or not isinstance(data[trait], dict):
            continue
        block = data[trait]
        # accept both 'High'/'Low' and lowercase variants
        for outer in ("High", "Low", "high", "low"):
            if outer in block:
                _collect_flat_list(block[outer], trait, outer, rows)

    if not rows:
        raise ValueError("No rows parsed. Check JSON structure or file path.")
    return pd.DataFrame(rows, columns=["trait", "label", "score"])


def plot_trait_hilo(df: pd.DataFrame, out_path: str) -> None:
    """Single-figure plot: raw points + mean±std per trait for High/Low, with t-test stars."""
    traits = [t for t in TRAITS if t in df["trait"].unique()]
    plt.figure(figsize=(10, 6))
    offsets = {"high": -0.1, "low": 0.1}

    # Raw data points
    for label in ("high", "low"):
        for i, trait in enumerate(traits):
            sub = df[(df["trait"] == trait) & (df["label"] == label)]
            if len(sub) == 0:
                continue
            x = np.full(len(sub), i + offsets[label])
            plt.scatter(x, sub["score"], alpha=0.35, marker="x", label=None)

    # Mean ± Std per trait
    grouped = df.groupby(["trait", "label"])["score"].agg(["mean", "std"]).reset_index()
    for label in ("high", "low"):
        means, stds = [], []
        for trait in traits:
            row = grouped[(grouped["trait"] == trait) & (grouped["label"] == label)]
            means.append(row["mean"].values[0] if not row.empty else np.nan)
            stds.append(row["std"].values[0] if not row.empty else np.nan)
        x = np.arange(len(traits)) + offsets[label]
        plt.errorbar(x, means, yerr=stds, fmt="o-", capsize=5, label=f"{label.capitalize()} Mean")

    # Significance stars (unpaired t-test on raw scores)
    for i, trait in enumerate(traits):
        hi = df[(df["trait"] == trait) & (df["label"] == "high")]["score"].to_numpy()
        lo = df[(df["trait"] == trait) & (df["label"] == "low")]["score"].to_numpy()
        if len(hi) > 1 and len(lo) > 1:
            _, p = ttest_ind(hi, lo, equal_var=False)
            if p < 0.05:
                ymax = df[df["trait"] == trait]["score"].max()
                plt.text(i, ymax + 0.02, "*", ha="center", fontsize=16)

    plt.xticks(np.arange(len(traits)), traits)
    plt.xlabel("Trait")
    plt.ylabel("Regressor score")
    plt.title("Trait Hi vs Low (single_question_lora_test_result)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to single_question_lora_test_result_with_regressor_preserved.json")
    ap.add_argument("--output", required=True, help="output PNG path")
    args = ap.parse_args()

    df = load_df_from_file(args.input)
    plot_trait_hilo(df, args.output)
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
