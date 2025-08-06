from Big5StabilityExperiment.ocean_classifier.inference import big5_classifier
import json
from tqdm import tqdm
from loguru import logger
from utils.inference_merge_lora import InterpolatedLoRAInference
from utils.prompt import generate_system_prompt, DEFAULT_QUESTION
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

data = {}
classifier = big5_classifier(model_root=CLASSIFIER_DIR)

for dim in tqdm(["O", "C", "E", "A", "N"]):
    data[dim] = {}
    trait = TO_CONFIG[dim]
    inference = InterpolatedLoRAInference(
        base_model_name="Qwen/Qwen3-8B",
        lora_path_low=LORAS_LO_DIR[dim],
        lora_path_high=LORAS_HI_DIR[dim]
    )
    for i in tqdm(range(0, 11), desc=f"Interpolating {dim}"):
        alpha = round(i / 10, 1)
        system_prompt = generate_system_prompt(base = True, vals = {dim: alpha})
        inference.reload_with_alpha(alpha)
        logger.info(f"Generating response with {trait} alpha={alpha} system_prompt=\"{system_prompt}\"...")
        data[dim][alpha] = []
        for _ in tqdm(range(50), desc=f"Sampling responses for alpha={alpha}"):
            response = inference.generate(system_prompt, DEFAULT_QUESTION)
            score = classifier.inference([response])[0][trait]
            data[dim][alpha].append({
                "response": response,
                "logit": score["logit"],
                "prob": score["prob"],
                "label": score["label"]
            })
    inference.unload()
    logger.info(f"Finished {trait}")

with open("single_question_personalized_system_prompt.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
