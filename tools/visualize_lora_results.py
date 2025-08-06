"""
visualize_lora_results
======================

This module provides helper functions for visualising evaluation results from
LoRA‑tuned personality models (Big Five/OCEAN traits).  It supports several
types of analyses, including interpolation of LoRA weights on a single
question, per‑question differences between high and low LoRA models,
question‑specific trait specificity, aggregated model stability across
questions, and per‑question mean logits colour‑coded by question.

Usage example:

    from visualize_lora_results import (
        load_single_question_merge,
        plot_merge_single_question,
        load_evaluate_score,
        plot_question_differences,
        plot_question_specificity,
        plot_model_stability,
        plot_per_question_means,
    )

    # Paths to your JSON files
    merge = load_single_question_merge('single_question_merge_lora_test_result.json')
    plot_merge_single_question(merge, 'merge_single_question.png')

    eval_data = load_evaluate_score('evaluate_score.json')
    plot_question_differences(eval_data, 'question_differences.png')
    plot_question_specificity(eval_data, 'question_specificity.png')
    plot_model_stability(eval_data, 'model_stability.png')
    plot_per_question_means(eval_data, 'per_question_means.png')

The functions save high‑resolution PNG images to the specified output paths.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import pandas as pd
from scipy.stats import ttest_ind


def load_single_question_merge(path: str) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Load results from ``single_question_merge_lora_test_result.json``.

    The JSON structure contains entries per trait (O, C, E, A, N), each of
    which stores a dictionary mapping interpolation weights (as strings) to a
    list of response dictionaries (with ``logit``, ``prob`` and ``label`` keys).

    Parameters
    ----------
    path: str
        Path to the JSON file.

    Returns
    -------
    dict
        Nested structure indexed by trait and weight.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_evaluate_score(path: str) -> Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]]:
    """Load results from ``evaluate_score.json``.

    The JSON structure is organised by trait (O, C, E, A, N) and model label
    ('High' or 'Low').  Each trait/model entry maps a question UUID to a list
    of response dictionaries containing ``logit``, ``prob`` and ``label``.

    Parameters
    ----------
    path: str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed data.
    """
    with open(path, 'r') as f:
        return json.load(f)


def plot_merge_single_question(data: Dict[str, Dict[str, List[Dict[str, float]]]], out_path: str) -> None:
    """Visualise the interpolation of LoRA weights on a single question.

    For each trait in ``data``, this function plots the classifier logits for
    every interpolation weight.  Raw values are displayed as semi‑transparent
    dots; the mean of each weight is emphasised with a solid dot.  A colour
    gradient encodes the interpolation weight on a shared colourbar.

    Parameters
    ----------
    data: dict
        Loaded output of ``single_question_merge_lora_test_result.json``.
    out_path: str
        Destination file name for the PNG figure.
    """
    traits = list(data.keys())
    weights = [float(w) for w in sorted(next(iter(data.values())).keys(), key=lambda x: float(x))]
    cmap = plt.get_cmap('viridis', len(weights))
    colour_map = {w: cmap(i) for i, w in enumerate(weights)}
    fig, axes = plt.subplots(len(traits), 1, figsize=(10, 3 * len(traits)), sharex=True)
    for ax, trait in zip(axes, traits):
        for w_str in sorted(data[trait].keys(), key=lambda x: float(x)):
            w = float(w_str)
            values = [entry['logit'] for entry in data[trait][w_str]]
            # scatter individual logits
            ax.scatter([w] * len(values), values, color=colour_map[w], alpha=0.2, s=15)
            # scatter mean
            ax.scatter([w], [np.mean(values)], color=colour_map[w], s=60, marker='o',
                       edgecolor='black', linewidth=0.5)
        ax.set_title(f'Trait {trait}')
        ax.set_ylabel('Logit')
        ax.grid(True, linestyle='--', alpha=0.4)
    axes[-1].set_xlabel('Interpolation weight (high proportion)')
    # colourbar
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.88, top=0.93)
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Interpolation weight')
    fig.suptitle('Single Question LoRA Merging: Logit Distributions by Weight for Each OCEAN Trait',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.85, 0.93))
    fig.savefig(out_path, dpi=300)


def plot_question_differences(data: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]], out_path: str) -> None:
    """Plot per‑question differences between high and low models for each trait.

    For each trait and question in ``data``, the mean logit difference
    (high − low) is computed.  Differences are displayed as scatter points
    sorted by magnitude; point colour encodes the sign and significance of the
    difference (red = significantly positive, blue = significantly negative,
    grey = non‑significant).  The top and bottom questions are annotated.

    Parameters
    ----------
    data: dict
        Loaded output of ``evaluate_score.json``.
    out_path: str
        Destination file name for the PNG figure.
    """
    traits = list(data.keys())
    records = []
    for trait in traits:
        high_dict = data[trait]['High']
        low_dict = data[trait]['Low']
        question_ids = set(high_dict.keys()) & set(low_dict.keys())
        for qid in question_ids:
            high_vals = [entry['logit'] for entry in high_dict[qid]]
            low_vals = [entry['logit'] for entry in low_dict[qid]]
            if not high_vals or not low_vals:
                continue
            diff = float(np.mean(high_vals) - np.mean(low_vals))
            t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
            records.append({'trait': trait, 'question_id': qid, 'diff': diff, 'p_val': float(p_val)})
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(len(traits), 1, figsize=(12, 3 * len(traits)), sharey=False)
    for ax, trait in zip(axes, traits):
        sub = df[df['trait'] == trait].sort_values('diff', ascending=False).reset_index(drop=True)
        x = np.arange(len(sub))
        y = sub['diff'].values
        p_vals = sub['p_val'].values
        colours = [
            ('red' if d > 0 else 'blue') if p < 0.05 else 'gray'
            for d, p in zip(y, p_vals)
        ]
        ax.scatter(x, y, c=colours, s=20, alpha=0.6)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'Trait {trait}')
        ax.set_ylabel('High − Low mean logit')
        # annotate extremes
        if len(sub) > 0:
            top = sub.iloc[0]
            bottom = sub.iloc[-1]
            ax.annotate(top['question_id'][:4], (0, top['diff']), fontsize=6,
                        textcoords='offset points', xytext=(0, 5), ha='center', color='darkred')
            ax.annotate(bottom['question_id'][:4], (len(sub) - 1, bottom['diff']), fontsize=6,
                        textcoords='offset points', xytext=(0, -10), ha='center', color='darkblue')
        ax.grid(True, linestyle='--', alpha=0.3)
    axes[-1].set_xlabel('Questions (sorted by difference)')
    fig.suptitle('HI vs LO LoRA Models: Differences in Mean Logit Across Questions for Each OCEAN Trait',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=300)


def plot_question_specificity(data: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]], out_path: str) -> None:
    """Plot trait‑specific mean differences per question on a small multiple grid.

    Each subplot corresponds to one question UUID and displays the mean
    difference (high − low) for the five traits.  Points are coloured by
    trait and drawn as stars when the difference is significant (p < 0.05).

    Parameters
    ----------
    data: dict
        Loaded output of ``evaluate_score.json``.
    out_path: str
        Destination file name for the PNG figure.
    """
    traits = ['O', 'C', 'E', 'A', 'N']
    question_ids = sorted(set(next(iter(data.values()))['High'].keys()))
    # gather differences and p‑values
    diff_by_question: Dict[str, Dict[str, Tuple[float, float]]] = {qid: {} for qid in question_ids}
    for qid in question_ids:
        for trait in traits:
            high_vals = [entry['logit'] for entry in data[trait]['High'][qid]]
            low_vals = [entry['logit'] for entry in data[trait]['Low'][qid]]
            diff = float(np.mean(high_vals) - np.mean(low_vals))
            t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
            diff_by_question[qid][trait] = (diff, float(p_val))
    # determine global y limits
    all_diffs = [val[0] for q in diff_by_question.values() for val in q.values()]
    y_min = min(all_diffs + [0])
    y_max = max(all_diffs)
    # colours per trait
    trait_colours = {'O': '#d73027', 'C': '#fc8d59', 'E': '#91bfdb', 'A': '#4575b4', 'N': '#7e329b'}
    n_questions = len(question_ids)
    cols = 5
    rows = int(np.ceil(n_questions / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True)
    axes = axes.flatten()
    for idx, qid in enumerate(question_ids):
        ax = axes[idx]
        for i, trait in enumerate(traits):
            diff, p_val = diff_by_question[qid][trait]
            marker = '*' if p_val < 0.05 else 'o'
            size = 80 if p_val < 0.05 else 40
            ax.scatter(i, diff, color=trait_colours[trait], marker=marker, s=size,
                       edgecolor='black' if p_val < 0.05 else 'none')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xticks(range(len(traits)))
        ax.set_xticklabels(traits, fontsize=8)
        ax.set_title(qid[:4], fontsize=8)
        ax.set_ylim(y_min * 1.05, y_max * 1.05)
        if idx % cols == 0:
            ax.set_ylabel('High − Low mean logit')
    # remove any empty axes
    for j in range(n_questions, rows * cols):
        fig.delaxes(axes[j])
    fig.suptitle('Per‑Question Big Five Specificity (High − Low mean logit across traits)', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=300)


def plot_model_stability(data: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]], out_path: str) -> None:
    """Assess model stability by aggregating raw logits across questions.

    All classifier logits for each trait and model (high/low) are pooled
    together and visualised.  Raw points are shown with transparency, mean
    values with error bars (mean ± std), and significance stars when the
    difference between high and low models on that trait is significant.

    Parameters
    ----------
    data: dict
        Loaded output of ``evaluate_score.json``.
    out_path: str
        Destination file name for the PNG figure.
    """
    traits = ['O', 'C', 'E', 'A', 'N']
    points: List[Dict[str, float]] = []
    for trait in traits:
        for label_key, label in [('High', 'high'), ('Low', 'low')]:
            for qid in data[trait][label_key]:
                for entry in data[trait][label_key][qid]:
                    points.append({'trait': trait, 'label': label, 'logit': entry['logit']})
    df = pd.DataFrame(points)
    stats = df.groupby(['trait', 'label'])['logit'].agg(['mean', 'std']).reset_index()
    # t‑tests per trait on pooled logits
    p_values: List[Tuple[str, float]] = []
    for trait in traits:
        high_vals = df[(df['trait'] == trait) & (df['label'] == 'high')]['logit']
        low_vals = df[(df['trait'] == trait) & (df['label'] == 'low')]['logit']
        t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
        p_values.append((trait, float(p_val)))
    colours = {'high': 'green', 'low': 'red'}
    offsets = {'high': -0.1, 'low': 0.1}
    plt.figure(figsize=(10, 6))
    # scatter raw values
    for label in ['high', 'low']:
        for i, trait in enumerate(traits):
            subset = df[(df['trait'] == trait) & (df['label'] == label)]
            x = np.full(len(subset), i + offsets[label])
            plt.scatter(x, subset['logit'], color=colours[label], alpha=0.3, marker='x', label=None)
    # error bars for means
    for label in ['high', 'low']:
        means = []
        stds = []
        for trait in traits:
            row = stats[(stats['trait'] == trait) & (stats['label'] == label)]
            means.append(row['mean'].values[0] if not row.empty else np.nan)
            stds.append(row['std'].values[0] if not row.empty else np.nan)
        x = np.arange(len(traits)) + offsets[label]
        plt.errorbar(x, means, yerr=stds, fmt='o-', label=f"{label.capitalize()} Mean",
                     color=colours[label], capsize=5)
    # significance stars
    for i, (trait, p_val) in enumerate(p_values):
        if p_val < 0.05:
            y_max = df[df['trait'] == trait]['logit'].max()
            plt.text(i, y_max + 1, '*', ha='center', fontsize=16, color='black')
    plt.xticks(range(len(traits)), traits)
    plt.xlabel('Trait')
    plt.ylabel('Logit')
    plt.title('LORA Models Stability Across Questions: Raw Logit Distribution by Trait')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


def plot_per_question_means(data: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]], out_path: str) -> None:
    """Visualise per‑question mean logits for each trait and model.

    For each trait and model (high/low), this function computes the mean logit
    across responses for every question.  Points are plotted at trait‑specific
    x positions; colours distinguish questions while marker shapes (circles
    versus squares) denote the model.  Black stars mark the overall mean for
    each trait and model.  A significance asterisk above a trait indicates
    that high and low question means differ significantly.

    Parameters
    ----------
    data: dict
        Loaded output of ``evaluate_score.json``.
    out_path: str
        Destination file name for the PNG figure.
    """
    traits = ['O', 'C', 'E', 'A', 'N']
    question_ids = sorted(set(next(iter(data.values()))['High'].keys()))
    # map questions to distinct colours
    cmap = plt.get_cmap('tab20', len(question_ids))
    colour_map = {qid: cmap(i) for i, qid in enumerate(question_ids)}
    # compute means per question/trait/model
    records = []
    for trait in traits:
        for label_key, label in [('High', 'high'), ('Low', 'low')]:
            for qid in question_ids:
                logs = [entry['logit'] for entry in data[trait][label_key][qid]]
                records.append({'trait': trait, 'label': label, 'question_id': qid,
                                'mean_logit': float(np.mean(logs))})
    df = pd.DataFrame(records)
    # overall means across questions
    overall = df.groupby(['trait', 'label'])['mean_logit'].mean().reset_index()
    # significance per trait using question means
    significance = {}
    for trait in traits:
        high_means = df[(df['trait'] == trait) & (df['label'] == 'high')]['mean_logit']
        low_means = df[(df['trait'] == trait) & (df['label'] == 'low')]['mean_logit']
        t_stat, p_val = ttest_ind(high_means, low_means, equal_var=False)
        significance[trait] = float(p_val)
    offsets = {'high': -0.15, 'low': 0.15}
    markers = {'high': 'o', 'low': 's'}
    plt.figure(figsize=(12, 6))
    # per‑question mean scatter
    for i, trait in enumerate(traits):
        for label in ['high', 'low']:
            sub = df[(df['trait'] == trait) & (df['label'] == label)]
            x = i + offsets[label]
            for _, row in sub.iterrows():
                plt.scatter(x, row['mean_logit'], color=colour_map[row['question_id']],
                            marker=markers[label], s=50, alpha=0.8)
    # overall means
    for _, row in overall.iterrows():
        trait_idx = traits.index(row['trait'])
        x = trait_idx + offsets[row['label']]
        plt.scatter(x, row['mean_logit'], color='black', marker='*', s=150, zorder=5)
    # significance stars above each trait
    for i, trait in enumerate(traits):
        if significance[trait] < 0.05:
            y_max = max(df[(df['trait'] == trait)]['mean_logit'].max(),
                        overall[(overall['trait'] == trait)]['mean_logit'].max())
            plt.text(i, y_max + 0.8, '*', ha='center', fontsize=16, color='black')
    plt.xticks(range(len(traits)), traits)
    plt.xlabel('Trait')
    plt.ylabel('Mean logit per question')
    plt.title('Per‑Question Mean Logit by Trait and LORA Model (colours distinguish questions)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


if __name__ == '__main__':
    """When run as a script, generate all visualisations.

    The script assumes ``single_question_merge_lora_test_result.json`` and
    ``evaluate_score.json`` are present in the current working directory.
    Output images will be saved alongside the script.
    """
    # Paths to the input files
    merge_path = 'single_question_personalized_system_prompt.json'
    # evaluate_path = 'evaluate_score.json'
    # Ensure input files exist before plotting
    if os.path.exists(merge_path):
        merge_data = load_single_question_merge(merge_path)
        plot_merge_single_question(merge_data, 'single_question_personalized_system_prompt.png')
    # if os.path.exists(evaluate_path):
    #     eval_data = load_evaluate_score(evaluate_path)
    #     plot_question_differences(eval_data, 'question_differences.png')
    #     plot_question_specificity(eval_data, 'question_specificity.png')
    #     plot_model_stability(eval_data, 'model_stability.png')
    #     plot_per_question_means(eval_data, 'per_question_means.png')
