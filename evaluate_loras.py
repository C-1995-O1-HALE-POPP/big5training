import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from tqdm import tqdm
from inference_lora import LoRAInference
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier
import matplotlib.pyplot as plt
import os

# 配置
CLASSIFIER_DIR = "Big5StabilityExperiment/ocean_classifier"

LORAS_HI_DIR = {
    "O": "output_lora_O_high",
    "C": "output_lora_C_high",
    "E": "output_lora_E_high",
    "A": "output_lora_A_high",
    "N": "output_lora_N_high"
}

LORAS_LO_DIR = {
    "O": "output_lora_O_low",
    "C": "output_lora_C_low",
    "E": "output_lora_E_low",
    "A": "output_lora_A_low",
    "N": "output_lora_N_low"
}

QUESTIONS_PATH = "test_conversation.json"

# 加载测试问题
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions: List[str] = json.load(f)

# 加载分类器
classifier = big5_classifier(model_root=CLASSIFIER_DIR)

# 初始化推理器
base_model = "Qwen/Qwen3-8B"
inference = LoRAInference(base_model_name=base_model)
system_prompt = "You are a helpful assistant."

results: Dict[str, Dict[str, float]] = {}

# 对每个维度进行测试
for dim in ["O", "C", "E", "A", "N"]:
    results[dim] = {}

    for level, lora_path in [("High", LORAS_HI_DIR[dim]), ("Low", LORAS_LO_DIR[dim])]:
        logger.info(f"Evaluating {dim} {level} responses...")
        inference.load_lora(lora_path)
        all_responses = []

        for q in tqdm(questions):
            response = inference.generate(system_prompt, q)
            all_responses.append(response)
        logger.debug(json.dumps(all_responses, ensure_ascii=False, indent=2))
        
        scores = classifier.inference(all_responses)
        logger.debug(f"Scores for {dim} {level}: \n{json.dumps(scores, ensure_ascii=False, indent=2)}")
        results[dim][level] = scores[dim]
        logger.info(f"{dim} {level} scores: {scores[dim]}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
dims = list(results.keys())
high_scores = [results[d]["High"] for d in dims]
low_scores = [results[d]["Low"] for d in dims]

x = range(len(dims))
ax.bar([i - 0.2 for i in x], high_scores, width=0.4, label="LoRA High", align="center")
ax.bar([i + 0.2 for i in x], low_scores, width=0.4, label="LoRA Low", align="center")

ax.set_xticks(x)
ax.set_xticklabels(dims)
ax.set_ylabel("Classifier Score")
ax.set_title("Big5 Trait Scores: LoRA-High vs LoRA-Low")
ax.legend()
plt.tight_layout()

plt.savefig("lora_personality_scores.png")
plt.show()