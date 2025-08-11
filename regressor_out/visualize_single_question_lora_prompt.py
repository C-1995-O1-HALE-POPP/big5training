# -*- coding: utf-8 -*-
"""
Visualize regressor scores for single_question_lora_prompt_* in one figure.

Input JSON structure:
{ "O": { "High": { "0.0": [ { regressor_score, ... }, ...], "0.1": [...] ... },
         "Low":  { "0.0": [...], "0.1": [...], ... } },
  "C": {...}, "E": {...}, "A": {...}, "N": {...}
}

Usage:
  python plot_lora_prompt_regressor_all_traits.py \
      --input single_question_lora_prompt_with_regressor_preserved.json \
      --output all_traits_regressor_score.png
"""

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

TRAITS = ["O", "C", "E", "A", "N"]


def extract_scores(entry: Dict[str, Any]) -> float | None:
    """Prefer 'regressor_score', fallback to 'score' if present."""
    if not isinstance(entry, dict):
        return None
    if "regressor_score" in entry:
        return float(entry["regressor_score"])
    if "score" in entry:
        return float(entry["score"])
    return None


def collect_trait_series(trait_block: Dict[str, Any]) -> tuple[list[float], dict]:
    """
    Build per-weight arrays for High/Low.
    Returns:
        weights (sorted float list),
        series: {
          "High": {"scores": List[List[float]], "means": List[float]},
          "Low" : {"scores": List[List[float]], "means": List[float]}
        }
    """
    # union of weight keys across High/Low
    weights = sorted({
        float(w)
        for side in ("High", "Low")
        if side in trait_block and isinstance(trait_block[side], dict)
        for w in trait_block[side].keys()
    })

    series: Dict[str, Dict[str, List]] = {}
    for side in ("High", "Low"):
        if side not in trait_block or not isinstance(trait_block[side], dict):
            continue
        scores_per_w, means = [], []
        for w in weights:
            entries = trait_block[side].get(f"{w}", [])
            vals: List[float] = []
            for it in entries:
                s = extract_scores(it)
                if s is not None:
                    vals.append(s)
            scores_per_w.append(vals)
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
        series[side] = {"scores": scores_per_w, "means": means}
    return weights, series


def plot_all_traits(data: Dict[str, Any], out_path: str) -> None:
    traits = [t for t in TRAITS if t in data and isinstance(data[t], dict)]
    n = len(traits)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=False)

    if n == 1:
        axs = [axs]  # normalize

    for ax, trait in zip(axs, traits):
        weights, series = collect_trait_series(data[trait])

        # scatter per weight, small jitter separates High/Low
        for side in ("High", "Low"):
            if side not in series:
                continue
            for i, w in enumerate(weights):
                vals = series[side]["scores"][i]
                if not vals:
                    continue
                x = np.full(len(vals), w) + (0.01 if side == "Low" else -0.01)
                ax.scatter(x, vals, alpha=0.35, marker=("o" if side == "High" else "s"), label=None)
            ax.plot(weights, series[side]["means"], marker=("o" if side == "High" else "s"),
                    label=f"{side} mean")

        # per-weight significance (High vs Low)
        if "High" in series and "Low" in series:
            for i, w in enumerate(weights):
                hv = np.array(series["High"]["scores"][i], dtype=float)
                lv = np.array(series["Low"]["scores"][i], dtype=float)
                if len(hv) > 1 and len(lv) > 1:
                    _, p = ttest_ind(hv, lv, equal_var=False)
                    if p < 0.05:
                        ymax = float(max(np.max(hv), np.max(lv)))
                        ax.text(w, ymax + 0.02, "*", ha="center", va="bottom", fontsize=12)

        ax.set_title(f"Trait {trait}")
        ax.set_xlabel("Interpolation weight")
        ax.set_ylabel("Regressor score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to single_question_lora_prompt_with_regressor_preserved.json")
    ap.add_argument("--output", default="all_traits_regressor_score.png", help="output PNG path")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_all_traits(data, args.output)
    print(f"Saved figure to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
