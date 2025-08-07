import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
file_path = "single_question_lora_prompt.json"
with open(file_path, "r") as f:
    data = json.load(f)

# traits
traits = ['O', 'C', 'E', 'A', 'N']

# 颜色配置
high_color_base = [0.121, 0.466, 0.705, 1.0]  # 蓝色
low_color_base = [1.0, 0.498, 0.0549, 1.0]    # 橙色

# 创建子图
fig, axs = plt.subplots(len(traits), 1, figsize=(10, 14), sharex=True)
fig.suptitle("Logit Distributions Across Interpolation Weights for OCEAN Traits", fontsize=14)

for ax, trait in zip(axs, traits):
    trait_data = data[trait]
    for level in ['High', 'Low']:
        color_base = high_color_base.copy() if level == 'High' else low_color_base.copy()
        marker = 'o' if level == 'High' else 's'

        alpha_logits = []
        alpha_vals = []

        for alpha_str, entries in sorted(trait_data[level].items(), key=lambda x: float(x[0])):
            alpha_val = float(alpha_str)
            logits = [entry["logit"] for entry in entries]
            alpha_vals.extend([alpha_val] * len(logits))
            alpha_logits.extend(logits)

            base_color = color_base.copy()
            base_color[3] = 0.5 + 0.5 * alpha_val
            ax.scatter([alpha_val] * len(logits), logits, alpha=0.6, c=[base_color], marker=marker)

        # 平均曲线
        unique_alpha_vals = sorted(trait_data[level].keys(), key=lambda x: float(x))
        avg_logits = [np.mean([entry["logit"] for entry in trait_data[level][a]]) for a in unique_alpha_vals]
        ax.plot([float(a) for a in unique_alpha_vals], avg_logits, color=color_base, label=f"{level} avg", linewidth=2)

    ax.set_title(f"Trait {trait}")
    ax.set_ylabel("Logit")
    ax.grid(True)

# 设置横坐标标签
axs[-1].set_xlabel("Interpolation weight α")

# 添加图例
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5))

# 布局调整
plt.tight_layout()
plt.subplots_adjust(top=0.95, right=0.85)
plt.show()
plt.savefig("lora_prompt_visualization.png", dpi=300, bbox_inches='tight')
