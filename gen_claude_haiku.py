import os
import time

import anthropic
from datasets import load_dataset
from tqdm import tqdm

rpm = 4000
rate_limit_delay = 60 / rpm

api_key = os.environ.get('ANTHROPIC_API_KEY')

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

client = anthropic.Anthropic(api_key="your-api-key-here")

def process_question(question, prompt):
    messages = [
                 {"role": "user", "content": question}
               ]
    response = anthropic.Anthropic().messages.create(
        model="claude-3-haiku-20240307",
        messages=messages,
        max_tokens=4096,
        system=prompt,
        temperature=0.6,
        top_p=0.9,
    )
    return response.content[0].text

prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
         "reasoning before stating any conclusions. To ensure your answer can be process, please conclude your " \
         "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer."

gsm8k = load_dataset("openai/gsm8k", "main")
train_data = gsm8k["train"]

output_dir = "claude_responses"
os.makedirs(output_dir, exist_ok=True)

next_question = 6362

try:
    for i, item in enumerate(tqdm(train_data[next_question:]["question"]), start=next_question):
        start_time = time.time()
        output = process_question(item, prompt)
        
        file_name = f"response_{i:05d}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output)
        end_time = time.time()
        
        if end_time - start_time < rate_limit_delay:
            time.sleep(rate_limit_delay - (end_time - start_time))

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Last processed question index: {i}")
    raise

print("Processing complete.")