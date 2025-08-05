import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from tqdm import tqdm
import sys

from utils.inference_lora import LoRAInference
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier

# 日志配置
logger.remove()
logger.add(
    sink=sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)
logger.add(
    sink="logs/evaluate_loras_vs_base.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

# 参数配置
CLASSIFIER_DIR = "Big5StabilityExperiment/ocean_classifier"
QUESTIONS_PATH = "test_conversation.json"
OUTPUT_PATH = "evaluate_score_lora_vs_base.json"
base_model = "Qwen/Qwen3-8B"

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
TO_CONFIG = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism"
}

# 加载问答
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)  # { "0": {"content": [{"role": "system"}, {"role": "user"}]}, ... }

# 初始化分类器
classifier = big5_classifier(model_root=CLASSIFIER_DIR)

# 初始化通用推理器
inference = LoRAInference(base_model_name=base_model)

results: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}  # dim -> mode -> qid -> List[responses]

for dim in ["O", "C", "E", "A", "N"]:
    results[dim] = {}
    trait = TO_CONFIG[dim]
    for mode, lora_path in [
        ("LoRA_High", LORAS_HI_DIR[dim]),
        ("LoRA_Low", LORAS_LO_DIR[dim]),
        ("Base", None)
    ]:
        logger.info(f"Evaluating {dim} in {mode} mode...")
        results[dim][mode] = {}
        if lora_path:
            inference.load_lora(lora_path)
        else:
            inference.unload_lora()

        for qid, qa in tqdm(questions.items(), desc=f"{dim} {mode}"):
            for msg in qa["content"]:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "system":
                    # 替换为对应人格的 system prompt
                    level = 1.0 if mode == "LoRA_High" else 0.0 if mode == "LoRA_Low" else 0.5
                    system_prompt = big5_system_prompts_en[trait.capitalize()][level]

            results[dim][mode][qid] = []
            for _ in range(50):  # 每个设置生成 50 次
                response = inference.generate(system_prompt, user_prompt)
                score = classifier.inference([response])[0][trait]
                results[dim][mode][qid].append({
                    "response": response,
                    "logit": score["logit"],
                    "prob": score["prob"],
                    "label": score["label"]
                })

# 保存结果
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
logger.info(f"All results saved to {OUTPUT_PATH}")
