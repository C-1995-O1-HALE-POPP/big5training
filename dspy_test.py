import os
import dspy
from dspy import LM

# LiteLLM OpenAI 兼容服务地址（请替换为你部署的服务地址）
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 请确保使用 /v1 路径
MODEL_NAME = "openai/qwen3-8b"      # 前缀 openai/ 表示走 text completion 路径

# 配置 DSPy 使用该模型
lm = LM(model=MODEL_NAME, api_base=API_BASE, api_key=API_KEY, extra_body={"enable_thinking": False})
dspy.settings.configure(lm=lm)

# 定义简单 TextCompletion Signature
class SimpleCompletion(dspy.Signature):
    prompt: str = dspy.InputField()
    completion: str = dspy.OutputField()

# 用 ChainOfThought 或 ResponseOnly 均可
completion = dspy.ChainOfThought(SimpleCompletion)

if __name__ == "__main__":
    result = completion(prompt="Once upon a time",)
    print("🖋️ Completion:", result.completion)
