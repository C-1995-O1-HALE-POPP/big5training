import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger

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
    sink="logs/inference_lora.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

# ===== 推理类 =====
class LoRAInference:
    def __init__(self, base_model_name="Qwen/Qwen3-8B", device=None):
        self.base_model_name = base_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading base model: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 防止 padding 报错

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.base_model = base_model.eval()

    def load_lora(self, lora_dir):
        logger.info(f"Loading LoRA adapter from: {lora_dir}")
        self.model = PeftModel.from_pretrained(self.base_model, lora_dir)
        self.model = self.model.eval()

    def build_chatml_prompt(self, system, user):
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{user}\n"
            f"<|assistant|>\n"
        )

    def generate(self, system_prompt, user_prompt, max_new_tokens=200):
        input_text = self.build_chatml_prompt(system_prompt, user_prompt)
        logger.debug(f"Input Text:\n{input_text}")
        logger.debug(f"Tokenized Input:\n{self.tokenizer.tokenize(input_text)}")

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Raw Output:\n{output_text}")

        if "<|assistant|>" in output_text:
            response = output_text.split("<|assistant|>")[1].split("<|")[0].strip()
        else:
            response = output_text.strip()
        return response

if __name__ == "__main__":
    lora_dir = "output_lora_C_high"  # 可更换为任意 LoRA adapter 路径
    system_prompt = "You are a helpful assistant."
    user_prompt = "My name is Mike. I just failed my exam, but I will try again next time. What do you think about it?"

    inference = LoRAInference()
    inference.load_lora(lora_dir)
    response = inference.generate(system_prompt, user_prompt)
    logger.success("=== Assistant Response ===")
    logger.success(response)