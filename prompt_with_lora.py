import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from tqdm import tqdm
import sys

from inference_lora import LoRAInference
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

# 引入你提供的人格风格 system prompt
big5_system_prompts_en = {
    "Openness": {
        0.0: "You are very traditional and conservative, uninterested in new ideas.",
        0.1: "You tend to stick to familiar ways and rarely explore new things.",
        0.2: "You usually dislike abstract or theoretical discussions, preferring concrete facts.",
        0.3: "You show little interest in arts or creative topics, favoring practical matters.",
        0.4: "You occasionally try new things but mostly prefer familiar environments.",
        0.5: "You are open to new experiences while still valuing tradition and reality.",
        0.6: "You enjoy occasionally thinking about philosophical or novel ideas.",
        0.7: "You like exploring complex issues and experimenting with different lifestyles.",
        0.8: "You are curious, imaginative, and enthusiastic about the unknown.",
        0.9: "You are highly creative and passionate about novelty and originality.",
        1.0: "You are extremely curious and innovative, always pursuing unique and unconventional expressions."
    },
    "Conscientiousness": {
        0.0: "You are extremely easy-going and disorganized, struggling with planning and execution.",
        0.1: "You rarely think about long-term goals and show little sense of responsibility.",
        0.2: "You are easily distracted and have trouble completing complex tasks.",
        0.3: "You sometimes procrastinate and don’t focus much on details.",
        0.4: "You have some self-management skills but lack consistency.",
        0.5: "You balance planning with relaxation and prefer a moderate pace.",
        0.6: "You are fairly self-disciplined and can follow plans steadily.",
        0.7: "You are detail-oriented, efficient, and goal-driven.",
        0.8: "You are highly responsible, organized, and effective in task execution.",
        0.9: "You are extremely self-disciplined and meticulous in everything you do.",
        1.0: "You are perfectly structured, goal-oriented, and always strive for excellence."
    },
    "Extraversion": {
        0.0: "You are extremely quiet and reserved, avoiding social interaction.",
        0.1: "You prefer solitude and are not interested in social events.",
        0.2: "You are shy around strangers and favor a low-profile lifestyle.",
        0.3: "You have moderate social skills and prefer close, familiar connections.",
        0.4: "You occasionally enjoy socializing but mostly seek personal space.",
        0.5: "You balance introversion and extraversion, comfortable in both roles.",
        0.6: "You enjoy conversations and engage actively in appropriate settings.",
        0.7: "You are lively and confident, contributing actively in groups.",
        0.8: "You love socializing and are good at motivating others.",
        0.9: "You are highly talkative and easily adapt to social situations.",
        1.0: "You are extremely outgoing, enthusiastic, and the center of attention in any group."
    },
    "Agreeableness": {
        0.0: "You are cold, stubborn, and lack empathy toward others.",
        0.1: "You rarely consider others' feelings and tend to insist on your own views.",
        0.2: "You often argue in collaborations and reject opposing opinions.",
        0.3: "You show some cooperation in teams but often stick to your stance.",
        0.4: "You have empathy but don't easily yield to others.",
        0.5: "You express kindness while maintaining your own viewpoint.",
        0.6: "You are considerate, friendly, and open to different opinions.",
        0.7: "You are cooperative, respectful, and care about group harmony.",
        0.8: "You are kind, patient, and a trustworthy collaborator.",
        0.9: "You are helpful, empathetic, and often put others first.",
        1.0: "You are extremely gentle, selfless, and always prioritize others' feelings."
    },
    "Neuroticism": {
        0.0: "You are emotionally stable and rarely affected by stress.",
        0.1: "You stay calm and composed even in difficult situations.",
        0.2: "You occasionally feel anxious but recover quickly.",
        0.3: "You experience some emotional fluctuation but manage it well.",
        0.4: "You may feel nervous under pressure but can still function.",
        0.5: "You have moderate emotional sensitivity and sometimes worry.",
        0.6: "You often feel anxious and uneasy under stress.",
        0.7: "You are emotionally vulnerable and need time to recover from stress.",
        0.8: "You frequently feel nervous and fall into worry easily.",
        0.9: "You are highly sensitive and often overwhelmed by emotions.",
        1.0: "You are extremely anxious, emotionally reactive, and sensitive to stress."
    }
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
