#!/bin/bash
# === 基础配置 ===
MODEL_NAME="Qwen/Qwen3-8B"
DS_CONFIG_PATH="ds_config.json"
DATA_DIR="processed_data"
OUTPUT_ROOT="output_lora"

# === 人格维度标签 ===
TRAITS=("E" "A" "N")
LEVELS=("high" "low")

# === 遍历训练 ===
for TRAIT in "${TRAITS[@]}"; do
  for LEVEL in "${LEVELS[@]}"; do
    DATA_PATH="${DATA_DIR}/train_${TRAIT}_${LEVEL}.jsonl"
    OUTPUT_DIR="${OUTPUT_ROOT}_${TRAIT}_${LEVEL}"

    echo "开始训练 LoRA 模型: ${TRAIT}-${LEVEL}"
    echo "数据路径: $DATA_PATH"
    echo "输出目录: $OUTPUT_DIR"

    # 启动训练（可改为 nohup、srun、deepspeed 等）
    python train_lora.py \
      --model_name "$MODEL_NAME" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --ds_config_path "$DS_CONFIG_PATH"

    echo "✅ 完成: ${TRAIT}-${LEVEL}"
    echo "----------------------------------------"
  done
done
