from datasets import load_dataset
import gc
import os
import peft
import pickle
import random
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

file = open("correct_indices.pkl", "rb")
correct_indices = pickle.load(file)
file.close()

def extract_number(text):
    matches = re.findall(r'(?:Answer:\s*|####\s*)(-?\d+(?:\.\d+)?)', text)
    if matches:
        number_str = matches[-1]
        try:
            return int(number_str)
        except ValueError:
            try:
                return float(number_str)
            except ValueError:
                return None
    else:
        return None

def load_model_responses(directory):
    responses = {}
    for filename in tqdm(os.listdir(directory), desc="loading responses"):
        if filename.startswith("response_") and filename.endswith(".txt"):
            index = int(filename[9:14])
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                response = f.read()
            responses[index] = response
    return responses

def select_indices(shot_count):
    return random.sample(correct_indices, shot_count)

def construct_message(shot_count, data_name, item, test_data, responses):
    prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
            "reasoning before stating any conclusions. To ensure your answer can be process, please conclude your " \
            "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer."

    message = [
            {"role": "system", "content": prompt}
    ]

    indices = select_indices(shot_count)

    for i in indices:
        message.append({"role": "user", "content": test_data[i]["question"]})
        message.append({"role": "assistant", "content": responses[i]})

    message.append({"role": "user", "content": item})
    
    return message

def evaluate(shot_count, data_name):
    gsm8k = load_dataset("openai/gsm8k", "main")
    test_data = gsm8k["test"]

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, low_cpu_mem_usage=True, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    responses = load_model_responses(f"{data_name}_responses")

    batch_size = 64

    batch_messages = [
        construct_message(shot_count, data_name, item, test_data, responses) for item in test_data["question"]
    ]

    def data():
        for i in tqdm(batch_messages):
            yield pipe.tokenizer.apply_chat_template(i, add_generation_prompt=True, tokenize=False)


    all_outputs = []

    try:
        print("Processing batch")
        for i, x in enumerate(
                pipe(data(), max_new_tokens=256, do_sample=True, batch_size=batch_size, eos_token_id=terminators,
                    temperature=0.6, top_p=0.9)):
            all_outputs.extend(x)
            torch.cuda.empty_cache()
            gc.collect()
        print("Done!")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    model_answers = [extract_number(resp['generated_text']) for resp in all_outputs]
    correct_answers = [extract_number(item) for item in test_data["answer"]]

    total_questions = len(model_answers)
    model_conversions = sum(1 for ans in model_answers if ans is not None)

    valid_pairs = [(m, c) for m, c in zip(model_answers, correct_answers) if m is not None and c is not None]
    correct = sum(m == c for m, c in valid_pairs)

    line = f"{shot_count}-shot {data_name}: Accurate: {correct} | Correctly Formatted: {model_conversions} | Total: {total_questions}"
    print(line)
    log_path = os.path.join("logs", f"few_shot_eval.log")
    with open(log_path, "a+") as logfile:
        logfile.write(line)

if __name__ == "__main__":
    evaluate(1, "llama")
    evaluate(3, "llama")
    evaluate(1, "claude")
    evaluate(3, "claude")