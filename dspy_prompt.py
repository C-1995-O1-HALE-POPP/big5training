import dspy
from dspy import Signature, InputField, OutputField
import numpy as np
import os
from typing import Dict
from loguru import logger
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier
import sys
import json
from utils.inference_lora import LoRAInference
from utils.prompt import generate_system_prompt, DEFAULT_QUESTION
# ----------------------------
# 日志配置
# ----------------------------
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
    sink="logs/dspy_prompt.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

inference = LoRAInference()

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
# ----------------------------
# Classifier 加载
# ----------------------------
CLASSIFIER_DIR = "Big5StabilityExperiment/ocean_classifier"
classifier = big5_classifier(model_root=CLASSIFIER_DIR)

# ----------------------------
# 参数配置
# ----------------------------
QUESTION = DEFAULT_QUESTION
N_SAMPLES = 20
TARGET_STD = 0.3
MAX_ROUNDS = 100

# ----------------------------
# 加载大模型（可换成你自己的）
# ----------------------------
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "openai/qwen3-8b"

# 配置 DSPy 使用该模型
lm = dspy.LM(
    model=MODEL_NAME,
    api_base=API_BASE,
    api_key=API_KEY,
    temperature=0.8,
    top_p=0.9,
    max_tokens=256,
    do_sample=True,
    extra_body={"enable_thinking": False}
)
dspy.settings.configure(lm=lm, cache=False)

# ----------------------------
# DSPy 模块定义
# ----------------------------

class PromptGenerator(Signature):
    trait = InputField(desc="Big Five trait to simulate")
    level = InputField(desc="Target trait level between 0 and 1")
    feedback = InputField(desc="Previous round feedback for improving the prompt", default="")
    prompt = OutputField(desc="System prompt that reflects the personality")

generate_prompt = dspy.ChainOfThought(PromptGenerator)

class PersonalityResponder(Signature):
    context = InputField(desc="System prompt representing personality")
    question = InputField(desc="Input question")
    answer = OutputField(desc="Model answer reflecting the personality")

responder = dspy.ChainOfThought(PersonalityResponder)

class SelfCritique(Signature):
    trait = InputField()
    target_level = InputField()
    prompt = InputField()
    hi_ratio = InputField()
    feedback = OutputField()

critic = dspy.ChainOfThought(SelfCritique)

# ----------------------------
# Prompt 测试与评分函数
# ----------------------------
def evaluate_prompt(prompt: str, trait: str, target_level: float) -> Dict:
    """给定 prompt，测试 N_SAMPLES 次模型回答，用 big5_classifier 打分"""
    answers = []
    labels = []
    
    for _ in range(N_SAMPLES):
        reply = inference.generate(prompt, QUESTION)
        logger.info(f"Model reply: {reply}")
        label = classifier.inference([reply])[0][trait]["label"]
        answers.append(label)
        labels.append(1 if label == 'high' else 0)  # 转换为 0/1 计算 std
    hi_ratio = sum(labels) / len(labels)
    std_dev = np.std(labels)

    return {
        "prompt": prompt,
        "hi_ratio": hi_ratio,
        "diff": abs(hi_ratio - target_level),
        "std": std_dev,
        "answers": answers,
    }

# ----------------------------
# 主优化函数
# ----------------------------
def optimize_prompt(trait: str, level: float, max_rounds: int = MAX_ROUNDS) -> Dict:
    history = []
    feedback = ""  # 初始无反馈
    lora_dirs = LORAS_HI_DIR if level > 0.5 else LORAS_LO_DIR
    inference.load_lora(lora_dirs[trait[0].upper()])  # 根据 trait 加载对应 LoRA
    for i in range(max_rounds):
        logger.info(f"\n🔁 Round {i + 1} / {max_rounds}\n")

        # 用反馈生成新的 prompt
        prompt = generate_prompt(trait=trait, level=level, feedback=feedback).prompt

        logger.info(f"📝 Prompt:\n{prompt}\n")
        result = evaluate_prompt(prompt, trait, level)
        logger.info(f"📊 hi_ratio={result['hi_ratio']:.2f}, target={level:.2f}, std={result['std']:.3f}")

        history.append(result)

        # 判断是否收敛
        if abs(result['hi_ratio'] - level) <= 0.1 and result['std'] <= TARGET_STD:
            logger.success("✅ Converged!")
            inference.unload() 
            return result

        # 从本轮结果生成反馈
        suggestion = critic(trait=trait, target_level=level, prompt=prompt, hi_ratio=result['hi_ratio'])
        feedback = suggestion.feedback.strip()  # 更新 feedback 以供下一轮使用
    inference.unload()  # 释放 LoRA 模型
    logger.warning("⚠️ Did not converge. Returning best result.")
    return sorted(history, key=lambda x: x['diff'])[0]
# ----------------------------
# 示例调用
# ----------------------------
if __name__ == "__main__":
    inference.load_lora(lora_dir=None)  # 可更换为任意 LoRA adapter 路径
    trait = "extraversion"
    level = 0.3
    final_result = optimize_prompt(trait, level)

    logger.info("\n🔥 最终收敛 prompt：")
    logger.info(final_result["prompt"])
    logger.success(f"✅ hi_ratio: {final_result['hi_ratio']}")
