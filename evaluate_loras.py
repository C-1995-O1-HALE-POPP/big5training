import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from tqdm import tqdm
from inference_lora import LoRAInference
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier
import matplotlib.pyplot as plt
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
    sink="logs/evaluate_loras.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


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

TO_CONFIG = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism"
}

QUESTIONS_PATH = "test_conversation.json"

OUTPUT_DIR = "evaluate_score.json"  # 可更换为任意 LoRA adapter 路径
# 加载测试问题
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions: List[str] = json.load(f)
# system_prompt = "You are a helpful assistant."
# user_prompt = "My name is Mike. I just failed my exam, but I will try again next time. What do you think about it?"
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
        all_responses = {}
        for index, question in tqdm(questions.items(), desc=f"{dim} {level}"):
            for msg in question["content"]:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "system":
                    system_prompt = msg["content"]
            all_responses[index] = []
            for i in tqdm(range(50)): 
                response = inference.generate(system_prompt, user_prompt)
                logger.info(f"Generated response: {response}")
                score = classifier.inference([response])[0][TO_CONFIG[dim]]
                logger.info(f"Classifier score for {dim} {level}: {score}")
                result = {
                    "response": response,
                    "logit": score["logit"],
                    "prob": score["prob"],
                    "label": score["label"]
                }
                logger.debug(f"Response score: {result}")
                all_responses[index].append(result)
        results[dim][level] = all_responses

# 保存结果
output_path = Path(OUTPUT_DIR)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
logger.info(f"Results saved to {output_path}")
