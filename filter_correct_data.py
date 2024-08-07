import os
import pickle
import re

from datasets import load_dataset
from tqdm import tqdm

def extract_number(text):
    match = re.search(r'(?:Answer:\s*|####\s*)(-?\d+(?:\.\d+)?)', text)
    if match:
        number_str = match.group(1)
        try:
            return int(number_str)
        except ValueError:
            try:
                return float(number_str)
            except ValueError:
                return None
    else:
        return None

def load_model_answers(directory):
    answers = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.startswith("response_") and filename.endswith(".txt"):
            index = int(filename[9:14])
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()
            answers[index] = extract_number(content)
    return answers

gsm8k = load_dataset("openai/gsm8k", "main")
train_data = gsm8k["train"]

claude_answers = load_model_answers("claude_responses")
llama_answers = load_model_answers("llama_responses")

correct_indices = []
for i, item in enumerate(tqdm(train_data)):
    correct_answer = extract_number(item['answer'])
    
    claude_answer = claude_answers.get(i)
    llama_answer = llama_answers.get(i)
    
    if (claude_answer is not None and llama_answer is not None and 
        correct_answer is not None and 
        claude_answer == correct_answer and llama_answer == correct_answer):
        correct_indices.append(i)

print(correct_indices)
with open('correct_indices.pkl', 'wb') as f:
    pickle.dump(correct_indices, f)

print(f"Total questions both models answered correctly: {len(correct_indices)}")
print(f"Indices saved to 'correct_indices.pkl'")

total_questions = len(train_data)
claude_correct = sum(1 for i, item in enumerate(train_data) if claude_answers.get(i) == extract_number(item['answer']))
llama_correct = sum(1 for i, item in enumerate(train_data) if llama_answers.get(i) == extract_number(item['answer']))

print(f"Total questions: {total_questions}")
print(f"Claude Haiku correct: {claude_correct} ({claude_correct/total_questions:.2%})")
print(f"Llama-3-8B-Instruct correct: {llama_correct} ({llama_correct/total_questions:.2%})")
print(f"Both correct: {len(correct_indices)} ({len(correct_indices)/total_questions:.2%})")