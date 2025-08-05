import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from tqdm import trange
from utils.inference_lora import LoRAInference
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier
import sys

# ===== 日志配置 =====
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
    sink="logs/app.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


# 配置路径
CLASSIFIER_DIR = "Big5StabilityExperiment/ocean_classifier"
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

# 固定问题
system_prompt = "You are a helpful assistant."
user_prompt = "My name is Mike. I just failed my exam, but I will try again next time. What do you think about it?"

# 初始化
inference = LoRAInference(base_model_name=base_model)
classifier = big5_classifier(model_root=CLASSIFIER_DIR)

results: Dict[str, Dict[str, List[Dict]]] = {}

# 每个维度分别测试 High / Low 的 50 次响应
for dim in ["O", "C", "E", "A", "N"]:
    results[dim] = {}
    for level, lora_path in [("High", LORAS_HI_DIR[dim]), ("Low", LORAS_LO_DIR[dim])]:
        logger.info(f"Evaluating {dim} {level}...")
        inference.load_lora(lora_path)

        responses: List[Dict] = []
        for i in trange(50, desc=f"{dim}-{level}"):
            response = inference.generate(system_prompt, user_prompt)
            score = classifier.inference([response])[0][TO_CONFIG[dim]]
            logger.debug(f"{dim}-{level} | Response: {response}")
            logger.debug(f"{dim}-{level} | Score: {score}")
            responses.append({
                "response": response,
                "logit": score["logit"],
                "prob": score["prob"],
                "label": score["label"]
            })

        results[dim][level] = responses

# 保存
with open("single_question_lora_test_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
logger.success("Saved to single_question_lora_test_result.json")

