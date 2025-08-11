# -*- coding: utf-8 -*-
"""
在问题集上评测 Merge LoRA（插值 α），所有配置均写死。
输出 JSON 结构:
{
  "O": {
    "0.0": { "uuid1": [ {response, logit, prob, label}, ... ], ... },
    "0.1": { ... },
    ...
    "1.0": { ... }
  },
  "C": { ... }, "E": { ... }, "A": { ... }, "N": { ... }
}
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from loguru import logger
from tqdm import tqdm
import sys
import numpy as np
from utils.prompt import generate_system_prompt
# ===== 日志配置（写死） =====
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
    sink="logs/evaluate_merge_lora.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

# ===== 依赖（写死路径/包名按你环境）=====
from utils.inference_merge_lora import InterpolatedLoRAInference
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier

# ===== 固定配置（全部写死） =====
CLASSIFIER_DIR = "Big5StabilityExperiment/ocean_classifier"
BASE_MODEL = "Qwen/Qwen3-8B"
QUESTIONS_PATH = "test_conversation.json"
OUTPUT_PATH = "evaluate_score_merge_lora_set_questions.json"
SAMPLES_PER_QUESTION = 10
ALPHAS: List[float] = [round(i / 10, 1) for i in range(11)]  # 0.0 ~ 1.0

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
DEFAULT_SYSTEM = "You are a helpful assistant."


def load_questions(path: str) -> Dict[str, Dict[str, Any]]:
    """
    支持两种结构，统一返回：{ uuid: {"content": [ {"role":..., "content":...}, ... ] } }
    - dict: { uuid: {"content":[...]}, ... }
    - list: [ {"id"/"uuid": "...", "content":[...]}, ... ]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and isinstance(v.get("content"), list):
                out[str(k)] = {"content": v["content"]}
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id") or item.get("uuid") or len(out))
            cont = item.get("content")
            if isinstance(cont, list):
                out[uid] = {"content": cont}
    else:
        raise ValueError("Unsupported questions format.")

    if not out:
        raise ValueError("No valid questions parsed from file.")
    return out


def extract_prompts(conversation: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    从多轮消息里取“最后一条 system（若无用默认）”和“最后一条 user（必须有）”
    """
    system_prompt = DEFAULT_SYSTEM
    user_prompt = None
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system" and content:
            system_prompt = content
        elif role == "user" and content:
            user_prompt = content
    if not user_prompt:
        raise ValueError("No user message found in conversation.")
    return system_prompt, user_prompt


def main():
    # 1) 加载问题集
    questions = load_questions(QUESTIONS_PATH)
    logger.info(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # 2) 分类器（Big5）
    classifier = big5_classifier(model_root=CLASSIFIER_DIR)

    # 3) 评测（五个维度 + α 插值）
    results: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

    for dim in ["O", "C", "E", "A", "N"]:
        trait_key = TO_CONFIG[dim]
        results[dim] = {}

        # 为该维度构建插值推理器（Low→High）
        inference = InterpolatedLoRAInference(
            base_model_name=BASE_MODEL,
            lora_path_low=LORAS_LO_DIR[dim],
            lora_path_high=LORAS_HI_DIR[dim]
        )

        for alpha in tqdm(ALPHAS, desc=f"Interpolating {dim}"):
            alpha_str = str(alpha)
            inference.reload_with_alpha(alpha)
            logger.info(f"[{dim}] alpha={alpha}")

            per_alpha: Dict[str, List[Dict[str, Any]]] = {}
            for uuid, obj in tqdm(questions.items(), leave=False, desc=f"{dim} a={alpha}"):
                try:
                    _, user_prompt = extract_prompts(obj["content"])
                except Exception as e:
                    logger.error(f"[{uuid}] bad conversation format: {e}")
                    continue

                system_prompt = generate_system_prompt(base = True, vals = {dim: alpha})

                samples: List[Dict[str, Any]] = []
                for _ in tqdm(range(SAMPLES_PER_QUESTION), leave=False, desc="samples"):
                    response = inference.generate(system_prompt, user_prompt)
                    score = classifier.inference([response])[0][trait_key]
                    samples.append({
                        "response": response,
                        "logit": float(score["logit"]),
                        "prob": float(score["prob"]),
                        "label": score["label"]
                    })
                per_alpha[uuid] = samples

            results[dim][alpha_str] = per_alpha

        # 释放该维度的 LoRA
        try:
            inference.unload()
        except Exception:
            pass

    # 4) 写出
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
