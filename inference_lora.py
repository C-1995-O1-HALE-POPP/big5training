import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger

logger.remove()
logger.add(
    sink=sys.stderr,
    level="INFO",  # 只显示 INFO、WARNING、ERROR、CRITICAL
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)
logger.add(
    sink="logs/app.log",        # 日志保存路径（可自定义）
    level="INFO",
    rotation="1 day",           # 每天生成一个新文件
    retention="7 days",         # 保留最近 7 天的日志
    compression="zip",          # 超期日志压缩为 zip 文件（可选）
    encoding="utf-8",           # 防止中文乱码
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)
# ===== 配置 =====
BASE_MODEL = "Qwen/Qwen3-8B"         # 基础模型（HuggingFace 上的名称或本地路径）
LORA_DIR = "output_lora_C_low"           # LoRA adapter 输出路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("===== 加载 tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 报错

logger.info("===== 加载基础模型 + LoRA adapter =====")
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
system_prompt = "You are a helpful assistant."
user_prompt = "My name is Mike. Hey, how was your day? I just failed my exam, but I will try again next time. What do you think about it?"

input_text = build_chatml_prompt(system_prompt, user_prompt)
logger.debug(input_text)
logger.debug(tokenizer.tokenize(input_text))

# ===== Tokenization & 推理 =====
inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

# ===== 输出结果 =====
# 完整生成后，拿到 decoded 文本
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
logger.debug(output_text)
# 提取第一轮 assistant 回复
if "<|assistant|>" in output_text:
    response = output_text.split("<|assistant|>")[1].split("<|user|>")[0].strip()
else:
    response = output_text  # fallback

print("=== Assistant Response (first round) ===")
print(response)
logger.success("=== Assistant Response ===")
logger.success(response)
