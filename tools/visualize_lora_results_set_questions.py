# -*- coding: utf-8 -*-
"""
visualize_lora_results_set_questions
====================================

Utilities to visualize multi-question evaluation results for LoRA-tuned
personality models (Big Five / OCEAN), where each JSON entry contains
per-trait, per-weight, per-question response logits.

Expected JSON structure (evaluate_score_merge_lora_set_questions.json):
{
  "O": {
    "0.0": {
      "uuid_q1": [ {"logit": ..., "prob": ..., "label": ...}, ... ],
      "uuid_q2": [ ... ],
      ...
    },
    "0.1": { ... },
    ...
  },
  "C": { ... },
  "E": { ... },
  "A": { ... },
  "N": { ... }
}

Provided plots:
1) plot_all_questions_aggregate_by_weight:
   - Pools all questions at each interpolation weight for a given trait.
   - Shows raw scatter (semi-transparent) + mean±std curve over weights.
   - Optional significance stars from t-tests vs. a baseline weight (default 0.0).

2) plot_per_question_trends_by_weight:
   - For each trait, for each question, computes the mean logit at every weight,
     and draws a line across weights. Different questions use different colors.
   - Optional per-point significance star from t-tests vs. a baseline weight.
   - Adds a UUID↔color legend (configurable) on the right side.

Usage:
    from visualize_lora_results_set_questions import (
        load_merge_set_questions,
        plot_all_questions_aggregate_by_weight,
        plot_per_question_trends_by_weight,
    )

    data = load_merge_set_questions("evaluate_score_merge_lora_set_questions.json")
    plot_all_questions_aggregate_by_weight(
        data, "all_questions_aggregate_by_weight.png",
        baseline_weight=0.0, alpha=0.05, show_sig=True
    )
    plot_per_question_trends_by_weight(
        data, "per_question_trends_by_weight.png",
        baseline_weight=0.0, alpha=0.05, show_sig=True,
        legend=True, legend_max=20, cmap_name="tab20"
    )
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


Trait = str
WeightStr = str
QuestionID = str
Entry = Dict[str, Any]
DataType = Dict[Trait, Dict[WeightStr, Dict[QuestionID, List[Entry]]]]


# -----------------------------
# Loading
# -----------------------------
def load_merge_set_questions(path: str) -> DataType:
    """Load evaluate_score_merge_lora_set_questions.json-like data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        data: DataType = json.load(f)
    return data


def _collect_all_weights(traits: List[str], data: DataType) -> List[float]:
    """Union of weights (as floats) across all traits, sorted ascending."""
    return sorted({float(w) for t in traits for w in data[t].keys()})


# -----------------------------
# Plot 1: Aggregate over all questions (model-level ability)
# -----------------------------
def plot_all_questions_aggregate_by_weight(
    data: DataType,
    out_path: str,
    figsize=(10, 15),
    baseline_weight: float = 0.0,
    alpha: float = 0.05,
    show_sig: bool = True,
) -> None:
    """For each trait, pool all questions at each weight and plot raw + mean±std.
    
    If show_sig=True, add a significance '*' above the mean for any weight where a
    Welch's t-test against the baseline weight (default 0.0) yields p < alpha.
    """
    traits = list(data.keys())
    all_weights = _collect_all_weights(traits, data)

    fig, axes = plt.subplots(len(traits), 1, figsize=figsize, sharex=True)

    for ax, trait in zip(axes, traits):
        # Map weight -> pooled logits across all questions
        weight_logits = defaultdict(list)
        for w_str, per_q in data[trait].items():
            w = float(w_str)
            for qid, entries in per_q.items():
                for e in entries:
                    if "logit" in e:
                        weight_logits[w].append(float(e["logit"]))

        ws = [w for w in all_weights if w in weight_logits]
        means, stds = [], []
        for w in ws:
            vals = weight_logits[w]
            # raw scatter
            if len(vals):
                ax.scatter(np.full(len(vals), w), vals, alpha=0.2, s=12)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        # mean ± std
        ax.errorbar(ws, means, yerr=stds, fmt="o-", capsize=4)

        # significance vs. baseline
        if show_sig and baseline_weight in weight_logits and len(weight_logits[baseline_weight]) > 1:
            base_vals = weight_logits[baseline_weight]
            y_vals = [m for m in means if not np.isnan(m)]
            if y_vals:
                y_span = max(y_vals) - min(y_vals) if len(y_vals) > 1 else max(y_vals) if y_vals else 1.0
            else:
                y_span = 1.0
            y_offset = 0.05 * (y_span if y_span > 0 else 1.0)
            for w, m in zip(ws, means):
                if w == baseline_weight or np.isnan(m):
                    continue
                vals = weight_logits[w]
                if len(vals) > 1:
                    _, p = ttest_ind(vals, base_vals, equal_var=False)
                    if p < alpha:
                        ax.text(w, m + y_offset, '*', ha='center', va='bottom', fontsize=12)

        ax.set_title(f"Trait {trait}")
        ax.set_ylabel("Logit")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Interpolation weight (high proportion)")
    fig.suptitle("All-Question Aggregate: Logit Distributions by Weight (per trait)\n* p < {:.2f} vs. baseline".format(alpha), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=300)


# -----------------------------
# Plot 2: Per-question trends (one color per question) across weights
# -----------------------------
def plot_per_question_trends_by_weight(
    data: DataType,
    out_path: str,
    figsize=(10, 15),
    legend: bool = True,
    legend_max: int = 20,
    baseline_weight: float = 0.0,
    alpha: float = 0.05,
    show_sig: bool = True,
    cmap_name: str = "tab20",
) -> None:
    """For each trait, draw one line per question showing mean logit across weights.
    
    If show_sig=True, add a '*' near a point where Welch's t-test vs. the baseline
    weight (default 0.0) for the SAME question yields p < alpha.
    
    Args:
        legend: whether to draw a legend mapping UUID to color.
        legend_max: cap legend entries to avoid clutter (set larger if needed).
        cmap_name: matplotlib colormap name to assign distinct colors to questions.
    """
    traits = list(data.keys())
    all_weights = _collect_all_weights(traits, data)

    # All question IDs encountered (stable ordering for color assignment)
    all_questions = sorted({
        qid
        for t in traits
        for w_str in data[t].keys()
        for qid in data[t][w_str].keys()
    })

    # Build a stable UUID -> color mapping using the requested colormap
    cmap = plt.get_cmap(cmap_name, max(1, len(all_questions)))
    qid_colors = {qid: cmap(i % cmap.N) for i, qid in enumerate(all_questions)}

    fig, axes = plt.subplots(len(traits), 1, figsize=figsize, sharex=True)

    # Collect legend handles from the first axis only (avoid duplication)
    legend_handles = []
    legend_labels = []

    for ax_idx, (ax, trait) in enumerate(zip(axes, traits)):
        # Pre-extract baseline distributions for every question
        base_map = {}
        if str(baseline_weight) in data[trait]:
            base_per_q = data[trait][str(baseline_weight)]
            for qid, entries in base_per_q.items():
                base_map[qid] = [float(e["logit"]) for e in entries if "logit" in e]

        # Plot each question curve
        for idx, qid in enumerate(all_questions):
            ws_for_q, means_for_q, pvals_for_q = [], [], []

            for w in all_weights:
                w_str = str(w)
                if w_str in data[trait] and qid in data[trait][w_str]:
                    vals = [float(e["logit"]) for e in data[trait][w_str][qid] if "logit" in e]
                    if len(vals) > 0:
                        ws_for_q.append(w)
                        means_for_q.append(float(np.mean(vals)))
                        # t-test vs baseline for SAME question (if exists and not same weight)
                        if show_sig and w != baseline_weight and qid in base_map and len(base_map[qid]) > 1:
                            _, p = ttest_ind(vals, base_map[qid], equal_var=False)
                            pvals_for_q.append((w, p, float(np.mean(vals))))
                        else:
                            pvals_for_q.append((w, np.nan, float(np.mean(vals))))

            if ws_for_q:
                color = qid_colors[qid]
                line, = ax.plot(ws_for_q, means_for_q, marker="o", alpha=0.9, linewidth=1.3, color=color)
                # Record legend handle on the first axis only (and cap to legend_max)
                if legend and ax_idx == 0 and len(legend_handles) < legend_max:
                    legend_handles.append(line)
                    legend_labels.append(qid)

                # add significance markers
                if show_sig and len(pvals_for_q):
                    # dynamic y offset based on data range for this trait
                    local_vals = [m for m in means_for_q if not np.isnan(m)]
                    y_span = (max(local_vals) - min(local_vals)) if len(local_vals) > 1 else (local_vals[0] if local_vals else 1.0)
                    y_offset = 0.05 * (y_span if y_span > 0 else 1.0)
                    for w, p, m in pvals_for_q:
                        if w == baseline_weight or np.isnan(p):
                            continue
                        if p < alpha:
                            ax.text(w, m + y_offset, '*', ha='center', va='bottom', fontsize=10)

        ax.set_title(f"Trait {trait}")
        ax.set_ylabel("Mean logit per question")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Interpolation weight (high proportion)")
    fig.suptitle("Per-Question Trends Across Weights (different colors = different questions)\n* p < {:.2f} vs. baseline".format(alpha), fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.80, 0.95))  # leave space on the right for legend

    # Place a shared legend to the right of the subplots (UUID ↔ color mapping)
    if legend and len(legend_handles) > 0:
        fig.legend(
            legend_handles, legend_labels,
            loc="center left", bbox_to_anchor=(0.82, 0.5),
            fontsize=8, frameon=False, title="Questions (UUID)",
        )

    fig.savefig(out_path, dpi=300)


# -----------------------------
# CLI
# -----------------------------
def _default_json_path() -> str:
    return "evaluate_score_merge_lora_set_questions.json"


def main(json_path: str | None = None) -> None:
    json_path = json_path or _default_json_path()
    data = load_merge_set_questions(json_path)
    plot_all_questions_aggregate_by_weight(
        data, "all_questions_aggregate_by_weight.png",
        baseline_weight=0.0, alpha=0.05, show_sig=True
    )
    plot_per_question_trends_by_weight(
        data, "per_question_trends_by_weight.png",
        baseline_weight=0.0, alpha=0.05, show_sig=True,
        legend=True, legend_max=20, cmap_name="tab20"
    )


if __name__ == "__main__":
    # If this file is executed directly, read JSON from CWD and emit both figures.
    try:
        main()
    except FileNotFoundError as e:
        # Graceful message in case someone runs it in a different directory.
        print(str(e))