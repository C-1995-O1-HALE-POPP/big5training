import os
import json
import uuid
from openai import OpenAI
from loguru import logger

# ===== 配置 DashScope 接口 =====
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_ENDPOINT)

# ===== 提示词 =====
prompt = ('''Generate 10 casual, engaging, and psychologically informative questions that a human user might ask an AI assistant during a friendly chat. 
These questions should be designed to indirectly reveal the assistant's personality traits, specifically across the Big Five dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, and Emotional Stability), without explicitly mentioning the traits.
Each question should be natural, suitable for open-ended dialogue, and phrased from the user's perspective as if talking to the assistant.
Output only the 20 questions, each on a new line, without numbering or additional explanation.'''
    "Format your output as follows:\n"
    "- The 1st questions you will generate.\n"
    "- The 2nd questions you will generate.\n"
    "- The 3rd questions you will generate.\n"
    "...\n"
    "- The 10th questions you will generate.\n\n"
)

# ===== 请求模型 =====
def generate_questions():
    logger.info("Generating casual conversation questions using qwen-turbo...")

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )

    questions = response.choices[0].message.content.strip().split("\n")
    questions = [q.strip("- ").strip() for q in questions if q.strip()]
    
    logger.info(f"Generated {len(questions)} questions:")
    for q in questions:
        logger.debug(f"- {q}")
    return questions

def generate_and_save_questions(output_path, n_questions=20):
    q = []
    while len(q) < n_questions:
        try:
            questions = generate_questions()
            if len(questions) >= n_questions:
                return questions[:n_questions]
            q.extend(questions)
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            continue
    output = {}
    for question in q:
        index = str(uuid.uuid4())
        output[index] = {
            "content": [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": question}
            ]
        }
    logger.info(f"Questions saved to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    output_file = "test_conversation.json"
    if not os.path.exists(output_file):
        generate_and_save_questions(output_file, n_questions=20)
    else:
        logger.info(f"File {output_file} already exists. Skipping generation.")