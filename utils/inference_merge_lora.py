import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger
from tqdm import tqdm
import copy
import gc
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
    sink="logs/inference_merge_lora.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

class InterpolatedLoRAInference:
    def __init__(self, base_model_name: str, lora_path_low: str, lora_path_high: str):
        self.base_model_name = base_model_name
        self.lora_path_low = lora_path_low
        self.lora_path_high = lora_path_high

        logger.info("Loading tokenizer and base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model = None  # 当前插值后的模型

    def _load_lora(self, path: str):
        return PeftModel.from_pretrained(copy.deepcopy(self.base_model), path)

    def _interpolate_model(self, alpha: float):

        import copy
        logger.info(f"Interpolating LoRA with alpha={alpha}")
        model_low = self._load_lora(self.lora_path_low)
        model_high = self._load_lora(self.lora_path_high)

        model = copy.deepcopy(model_low)
        named_low = dict(model_low.named_parameters())
        named_high = dict(model_high.named_parameters())
        named_target = dict(model.named_parameters())

        all_keys = set(named_low.keys()) | set(named_high.keys())

        for name in tqdm(all_keys):
            if "lora_" not in name:
                continue
            p_low = named_low.get(name)
            p_high = named_high.get(name)

            if p_low is None and p_high is None:
                continue

            shape = p_low.shape if p_low is not None else p_high.shape
            device = p_low.device if p_low is not None else p_high.device
            dtype = p_low.dtype if p_low is not None else p_high.dtype

            t_low = p_low.data if p_low is not None else torch.zeros(shape, device=device, dtype=dtype)
            t_high = p_high.data if p_high is not None else torch.zeros(shape, device=device, dtype=dtype)

            t_interp = (1 - alpha) * t_low + alpha * t_high

            if name in named_target:
                named_target[name].data = t_interp
            else:
                logger.warning(f"{name} not found in target model, skipping.")

        return model.eval()

    def reload_with_alpha(self, alpha: float):
        """释放旧模型并按新的 alpha 重新加载插值后的新模型"""
        self.unload()
        torch.cuda.empty_cache()
        self.model = self._interpolate_model(alpha)

    def unload(self):
        """释放当前模型显存"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Previous model unloaded from GPU.")

    def build_chatml_prompt(self, system, user):
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{user}\n"
            f"<|assistant|>\n"
        )

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 200):
        if self.model is None:
            raise ValueError("Model is not loaded. Please call reload_with_alpha() first.")

        input_text = self.build_chatml_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Raw Output:\n{output_text}")

        if "<|assistant|>" in output_text:
            response = output_text.split("<|assistant|>")[1].split("<|")[0].strip()
        else:
            response = output_text.strip()
        return response


            
                

