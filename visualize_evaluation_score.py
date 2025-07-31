import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# 读取新的 JSON 文件
file_path = "single_question_lora_test_result.json"
with open(file_path, "r") as f:
    data = json.load(f)

# 整理成 DataFrame
points = []
for trait, levels in data.items():
    for label, entries in levels.items():
        for entry in entries:
            logit = entry["logit"]
            points.append({
                "trait": trait,
                "label": label.lower(),
                "logit": logit
            })
df = pd.DataFrame(points)

# 计算每个 trait 下 High 和 Low 的均值、标准差、样本数
grouped_stats = df.groupby(["trait", "label"])["logit"].agg(["mean", "std", "count"]).reset_index()

# 显著性检验：t-test between High and Low
traits = sorted(df["trait"].unique())
ttest_results = []
for trait in traits:
    high_vals = df[(df["trait"] == trait) & (df["label"] == "high")]["logit"]
    low_vals = df[(df["trait"] == trait) & (df["label"] == "low")]["logit"]
    if not high_vals.empty and not low_vals.empty:
        t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
        ttest_results.append((trait, p_val))
    else:
        ttest_results.append((trait, np.nan))

# 绘图
plt.figure(figsize=(10, 6))
colors = {"high": "green", "low": "red"}
offsets = {"high": -0.1, "low": 0.1}

# 原始数据点
for label in ["high", "low"]:
    for i, trait in enumerate(traits):
        subset = df[(df["trait"] == trait) & (df["label"] == label)]
        x = np.full(len(subset), i + offsets[label])  # 给每个 trait 一个固定偏移位置
        plt.scatter(x, subset["logit"], color=colors[label], alpha=0.4, marker='x', label=None)

# 平均值 ± 标准差折线
for label in ["high", "low"]:
    means, stds = [], []
    for trait in traits:
        row = grouped_stats[(grouped_stats["trait"] == trait) & (grouped_stats["label"] == label)]
        means.append(row["mean"].values[0] if not row.empty else np.nan)
        stds.append(row["std"].values[0] if not row.empty else np.nan)
    x = np.arange(len(traits)) + offsets[label]
    plt.errorbar(x, means, yerr=stds, fmt='o-', label=f"{label.capitalize()} Mean", color=colors[label], capsize=5)

# 显著性标记
for i, (trait, p_val) in enumerate(ttest_results):
    if p_val < 0.05:
        y_max = df[df["trait"] == trait]["logit"].max()
        plt.text(i, y_max + 1, "*", ha='center', fontsize=16, color="black")

# 布局美化
plt.xticks(np.arange(len(traits)), traits)
plt.title("Logit Mean ± Std by Trait with Raw Data and Significance Test")
plt.xlabel("Trait")
plt.ylabel("Logit Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("single_question_lora_logit_stats.png", dpi=300)
