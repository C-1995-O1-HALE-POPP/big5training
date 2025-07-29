import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import copy

# === 配置路径 ===
BASE_MODEL = "Qwen/Qwen3-8B"
LORA_PATH_LOW = "output_lora_E_low"
LORA_PATH_HIGH = "output_lora_E_high"
ALPHA = 0.6  # 插值系数：0=完全low, 1=完全high

# === 加载 tokenizer 和 base model ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# === 加载两个 LoRA adapter 的权重 ===
def load_lora_weights(path):
    config = PeftConfig.from_pretrained(path)
    model = PeftModel.from_pretrained(copy.deepcopy(base_model), path)
    return model

model_low = load_lora_weights(LORA_PATH_LOW)
model_high = load_lora_weights(LORA_PATH_HIGH)

# === 插值函数：在两个 LoRA adapter 权重之间线性插值 ===
def interpolate_lora_weights(model_low, model_high, alpha):
    for (name, param_low), (_, param_high) in zip(
        model_low.named_parameters(), model_high.named_parameters()
    ):
        if "lora_" in name and param_low.requires_grad:
            param_low.data = (1 - alpha) * param_low.data + alpha * param_high.data
    return model_low

# === 融合模型 ===
model = interpolate_lora_weights(model_low, model_high, ALPHA)
model.eval()

# === 构造 ChatML 输入 ===
def build_chatml_prompt(system, user):
    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n"
    )

system_msg = "You are a helpful assistant with the following Big Five personality traits: Extraversion - interpolated"
user_msg = "Hey, how was your day? Did you do anything exciting?"
input_text = build_chatml_prompt(system_msg, user_msg)

# === 推理 ===
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = output_text.split("<|assistant|>\n")[-1].strip()
print("=== Assistant Response ===")
print(response)
