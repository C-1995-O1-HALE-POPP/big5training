import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 配置 =====
BASE_MODEL = "Qwen/Qwen3-8B"         # 基础模型（HuggingFace 上的名称或本地路径）
LORA_DIR = "output_lora_E"           # LoRA adapter 输出路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 报错

# ===== 加载基础模型 + LoRA adapter =====
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.eval()

# ===== 构造 ChatML 输入 =====
def build_chatml_prompt(system, user):
    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n"
    )

# 示例输入
system_prompt = "You are a helpful assistant with the following Big Five personality traits: Extraversion - high"
user_prompt = "Hey, how was your day? Did you do anything exciting?"

input_text = build_chatml_prompt(system_prompt, user_prompt)

# ===== Tokenization & 推理 =====
inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

# ===== 输出结果 =====
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_response = output_text.split("<|assistant|>\n")[-1].strip()

print("=== Assistant Response ===")
print(generated_response)
