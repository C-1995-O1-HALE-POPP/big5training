import pandas as pd
import json
from pathlib import Path

# 加载数据
df = pd.read_csv("big5_chat_dataset.csv")  # ← 替换为你的 CSV 文件路径

# 映射表：trait → ChatML 人格标签缩写
trait_map = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N"
}

# level → 文件名后缀
level_suffix = {
    "low": "low",
    "high": "high"
}

# 创建输出目录
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

# 遍历所有 trait × level 组合
for trait, dim_tag in trait_map.items():
    for level in ["low", "high"]:
        df_sub = df[(df["trait"] == trait) & (df["level"] == level)]
        output_file = output_dir / f"train_{dim_tag}_{level_suffix[level]}.jsonl"

        with open(output_file, "w", encoding="utf-8") as fout:
            for _, row in df_sub.iterrows():
                sys_msg = str(row["train_instruction"]).strip()  # 保留原始 system prompt
                user_msg = str(row["train_input"]).strip()
                assistant_msg = str(row["train_output"]).strip()

                chat_obj = {
                    "label": {dim_tag: 0.0 if level == "low" else 1.0},  # 可选 label 字段
                    "dialogue": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ]
                }
                fout.write(json.dumps(chat_obj, ensure_ascii=False) + "\n")

print("✅ 数据预处理完成！每个维度的高低人格样本已分开存储。输出目录：processed_data/")
