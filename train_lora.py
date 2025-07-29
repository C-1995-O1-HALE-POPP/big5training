import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import argparse
import os

# ===== 命令行参数解析 =====
parser = argparse.ArgumentParser(description="LoRA 微调脚本")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                    help="HuggingFace模型名称或本地路径")
parser.add_argument("--data_path", type=str, default="processed_data/train_E_high.jsonl",
                    help="训练数据路径（ChatML格式）")
parser.add_argument("--output_dir", type=str, default="output_lora_E_high",
                    help="输出目录")
parser.add_argument("--ds_config_path", type=str, default="ds_config.json",
                    help="DeepSpeed 配置文件路径")
args = parser.parse_args()


# ===== 配置部分 =====
model_name = args.model_name  # HuggingFace模型名称或本地路径
data_path = args.data_path  # 训练数据路径（ChatML格式）
output_dir = args.output_dir  # 输出目录
ds_config_path = args.ds_config_path  # DeepSpeed 配置文件路径

# ===== 加载数据集 =====
def load_chatml_dataset(path):
    def parse_jsonl(line):
        data = json.loads(line)
        return {"messages": data["dialogue"]}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    parsed = [parse_jsonl(line) for line in lines]
    return Dataset.from_list(parsed) 

# ===== ChatML → 模型输入token =====
def tokenize_chatml(example):
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    return tokenizer(text, truncation=True, max_length=2048)

# ===== 加载 tokenizer 和 model =====
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 防止 pad 报错

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 推荐 A10 使用 float16，自动适配
    device_map="auto",
    trust_remote_code=True
)

# ===== LoRA 配置 =====
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)

# ===== 数据处理 =====
dataset = load_chatml_dataset(data_path).map(tokenize_chatml)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ===== 训练参数（适配 A10）=====

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    logging_steps=10,
    logging_first_step=True,
    save_steps=200,
    save_total_limit=1,
    learning_rate=1e-4,
    fp16=True,
    deepspeed=ds_config_path,
    report_to="tensorboard",              # ✅ 报告到 TensorBoard
    logging_dir=f"{output_dir}/logs",     # ✅ 指定日志保存目录
)

# ===== 开始训练 =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(output_dir)
print(f"✅ LoRA 微调完成，结果保存在 {output_dir}")
