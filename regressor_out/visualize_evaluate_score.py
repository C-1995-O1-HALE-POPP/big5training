# -*- coding: utf-8 -*-
"""
Plots for evaluate_score_* JSON using OUTER High/Low as the ONLY label.
Generates:
  1) Trait-level Hi vs Low comparison (raw points + mean±std + t-test star)
  2) Per-question (uuid) comparison (question means, overall mean stars, t-test star)

Usage:
  python plot_regressor_from_outer_label.py \
      --input evaluate_score_with_regressor_preserved.json \
      --outdir figs
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


# -----------------------------
# Load with OUTER label only
# -----------------------------
def _collect_outer(group: Any, trait: str, outer_label: str, uuid_key: str, rows: List[Dict[str, Any]]) -> None:
    """Recursive collector: append rows with label taken ONLY from outer_label."""
    if isinstance(group, list):
        for item in group:
            if not isinstance(item, dict):
                continue
            # prefer regressor_score, fallback to score
            score = item.get("regressor_score", item.get("score", None))
            if score is None:
                continue
            try:
                rows.append({
                    "trait": trait,
                    "label": outer_label.lower(),  # force from OUTER
                    "uuid": uuid_key,
                    "score": float(score),
                })
            except Exception:
                pass
    elif isinstance(group, dict):
        for k, v in group.items():
            _collect_outer(v, trait, outer_label, uuid_key, rows)


def load_from_outer_label(path: str) -> pd.DataFrame:
    """
    Expect top-level like:
    {
      "O": { "High": { uuid: [ ... ] , ... }, "Low": { uuid: [ ... ], ... } },
      ...
    }
    Returns DataFrame with columns: trait, label('high'/'low'), uuid, score
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for trait in TRAITS:
        if trait not in data or not isinstance(data[trait], dict):
            continue
        trait_block = data[trait]
        # accept case-insensitive keys
        for outer_label in ["High", "Low", "high", "low"]:
            if outer_label in trait_block and isinstance(trait_block[outer_label], dict):
                for uuid_key, group in trait_block[outer_label].items():
                    _collect_outer(group, trait, outer_label, uuid_key, rows)

    df = pd.DataFrame(rows, columns=["trait", "label", "uuid", "score"])
    return df


# -----------------------------
# Plot 1: trait-level Hi/Low
# -----------------------------
def plot_trait_hilo(df: pd.DataFrame, out_path: str) -> None:
    traits = [t for t in TRAITS if t in df["trait"].unique()]
    plt.figure(figsize=(10, 6))
    offsets = {"high": -0.1, "low": 0.1}

    # raw points
    for label in ["high", "low"]:
        for i, trait in enumerate(traits):
            sub = df[(df["trait"] == trait) & (df["label"] == label)]
            if len(sub) == 0:
                continue
            x = np.full(len(sub), i + offsets[label])
            plt.scatter(x, sub["score"], alpha=0.35, marker="x", label=None)

    # mean ± std
    grouped = df.groupby(["trait", "label"])["score"].agg(["mean", "std"]).reset_index()
    for label in ["high", "low"]:
        means, stds = [], []
        for trait in traits:
            row = grouped[(grouped["trait"] == trait) & (grouped["label"] == label)]
            means.append(row["mean"].values[0] if not row.empty else np.nan)
            stds.append(row["std"].values[0] if not row.empty else np.nan)
        x = np.arange(len(traits)) + offsets[label]
        plt.errorbar(x, means, yerr=stds, fmt="o-", capsize=5, label=f"{label.capitalize()} Mean")

    # significance star per trait
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
    plt.title("Hi vs Low Models per Trait (raw scores, mean±std, t-test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Plot 2: per-question (uuid)
# -----------------------------
def plot_per_question_means(df: pd.DataFrame, out_path: str) -> None:
    traits = [t for t in TRAITS if t in df["trait"].unique()]
    # per-question means
    qmeans = df.groupby(["trait", "label", "uuid"])["score"].mean().reset_index()
    overall = qmeans.groupby(["trait", "label"])["score"].mean().reset_index()

    # significance star based on question-means
    pvals: Dict[str, float] = {}
    for trait in traits:
        hi = qmeans[(qmeans["trait"] == trait) & (qmeans["label"] == "high")]["score"].to_numpy()
        lo = qmeans[(qmeans["trait"] == trait) & (qmeans["label"] == "low")]["score"].to_numpy()
        if len(hi) > 1 and len(lo) > 1:
            _, p = ttest_ind(hi, lo, equal_var=False)
            pvals[trait] = float(p)
        else:
            pvals[trait] = np.nan

    offsets = {"high": -0.15, "low": 0.15}
    markers = {"high": "o", "low": "s"}

    plt.figure(figsize=(12, 6))

    # question-level means
    for i, trait in enumerate(traits):
        for label in ["high", "low"]:
            sub = qmeans[(qmeans["trait"] == trait) & (qmeans["label"] == label)]
            if len(sub) == 0:
                continue
            x = np.full(len(sub), i + offsets[label])
            plt.scatter(x, sub["score"], marker=markers[label], s=45, alpha=0.85)

    # overall means as stars
    for _, row in overall.iterrows():
        i = traits.index(row["trait"])
        x = i + offsets[row["label"]]
        plt.scatter(x, row["score"], marker="*", s=150, zorder=5)

    # significance star
    for i, trait in enumerate(traits):
        p = pvals.get(trait, np.nan)
        if isinstance(p, float) and p < 0.05:
            ymax = max(qmeans[qmeans["trait"] == trait]["score"].max(),
                       overall[overall["trait"] == trait]["score"].max())
            plt.text(i, ymax + 0.02, "*", ha="center", fontsize=16)

    plt.xticks(np.arange(len(traits)), traits)
    plt.xlabel("Trait")
    plt.ylabel("Mean regressor score per question")
    plt.title("Per-Question Mean Scores by Trait (Hi vs Low from OUTER keys)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="evaluate_score_with_regressor_preserved.json")
    ap.add_argument("--outdir", default="figs", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_from_outer_label(args.input)

    # quick checks
    if df.empty:
        raise ValueError("No rows parsed. Check JSON structure.")
    for col in ["trait", "label", "uuid", "score"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    plot_trait_hilo(df, os.path.join(args.outdir, "trait_hi_lo.png"))
    plot_per_question_means(df, os.path.join(args.outdir, "per_question_means.png"))

    # optional summaries
    df.groupby(["trait", "label"])["score"].agg(["count", "mean", "std"]).to_csv(
        os.path.join(args.outdir, "summary_trait_label.csv")
    )
    df.groupby(["trait", "label", "uuid"])["score"].agg(["count", "mean", "std"]).to_csv(
        os.path.join(args.outdir, "summary_trait_label_uuid.csv")
    )


if __name__ == "__main__":
    main()
