
from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier
import json
from tqdm import tqdm
from loguru import logger
from utils.inference_merge_lora import InterpolatedLoRAInference
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


if __name__ == "__main__":
    # 示例：加载插值模型并生成响应
    # 请确保已存在 output_lora_E_low 和 output_lora_E_high 目录

    classifier = big5_classifier(model_root=CLASSIFIER_DIR)
    system_prompt = "You are a helpful assistant."
    user_prompt = "My name is Mike. I just failed my exam, but I will try again next time. What do you think about it?"
    data = {}
    for dim in tqdm(["O", "C", "E", "A", "N"]):
        data[dim] = {}
        inference = InterpolatedLoRAInference(
            base_model_name="Qwen/Qwen3-8B",
            lora_path_low=LORAS_LO_DIR[dim],
            lora_path_high=LORAS_HI_DIR[dim]
        )
        for i in tqdm(range(0, 11), desc=f"Interpolating {dim} dimension"):
            inference.reload_with_alpha(i/10)
            logger.info(f"Generating response with alpha={i/10}...")
            data[dim][i/10] = []
            for j in tqdm(range(51), desc=f"Generating {dim} responses with alpha={i/10}"):
                response = inference.generate(system_prompt, user_prompt)
                logger.info(f"Response: {response}")
                score = classifier.inference([response])[0][TO_CONFIG[dim]]
                data[dim][i/10].append({
                    "response": response,
                    "logit": score["logit"],
                    "prob": score["prob"],
                    "label": score["label"]
                })
        inference.unload()
        logger.info(f"Finished generating responses for {dim} dimension.")
    logger.info("All responses generated. Saving results...")
    with open("single_question_marge_lora_test_result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)