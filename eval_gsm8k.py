from datasets import load_dataset
import gc
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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

""" def extract_number(text):
    match = re.search(r'<answer>(-?\d+(?:\.\d+)?)</answer>|####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        number_str = match.group(1) or match.group(2)
        try:
            return int(number_str)
        except ValueError:
            try:
                return float(number_str)
            except ValueError:
                return None
    else:
        return None """

""" def extract_number(text):
    matches = re.findall(r'\[(-?\d+(?:\.\d+)?)\]|####\s*(-?\d+(?:\.\d+)?)', text)
    if matches:
        last_match = matches[-1]
        number_str = last_match[0] or last_match[1]
        try:
            return int(number_str)
        except ValueError:
            try:
                return float(number_str)
            except ValueError:
                return None
    else:
        return None """

def process_batch(batch, model, tokenizer, prompt):
    batch_messages = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": item}
        ] for item in batch["question"]
    ]

    batch_inputs = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        batch_outputs = model.generate(
            batch_inputs,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    responses = [output[input_ids.shape[-1]:] for output, input_ids in zip(batch_outputs, batch_inputs)]
    
    decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
    
    model_answers = [extract_number(resp) for resp in decoded_responses]
    correct_answers = [extract_number(item) for item in batch["answer"]]

    torch.cuda.empty_cache()
    gc.collect()
    
    return model_answers, correct_answers

""" prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
         "reasoning before stating any conclusions. To ensure your answer can be processed, please conclude your " \
         "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer." """

prompt = "Your task is to solve math problems that have integer solutions. Before giving your final answer, you think " \
         "step by step inside <reasoning></reasoning> tags. After reasoning carefully about the answer you give your " \
         "final solution as an integer enclosed in <answer></answer> tags."

""" prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your reasoning " \
         "before stating any conclusions. To ensure your answer can be processed, please answer in the form on an integer, " \
         "and enclose your example in square brackets. Answer format example: [42]" """

print(prompt)

gsm8k = load_dataset("openai/gsm8k", "main")

test_data = gsm8k["test"]

torch.cuda.empty_cache()
gc.collect()

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="offload",
    low_cpu_mem_usage=True,
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

batch_size = 8
all_model_answers = []
all_correct_answers = []

for i in tqdm(range(0, len(test_data), batch_size)):
    batch = test_data[i:i+batch_size]
    model_answers, correct_answers = process_batch(batch, model, tokenizer, prompt)
    all_model_answers.extend(model_answers)
    all_correct_answers.extend(correct_answers)

    torch.cuda.empty_cache()
    gc.collect()

total_questions = len(all_model_answers)
model_conversions = sum(1 for ans in all_model_answers if ans is not None)
correct_conversions = sum(1 for ans in all_correct_answers if ans is not None)

model_conversion_rate = model_conversions / total_questions
correct_conversion_rate = correct_conversions / total_questions

print(f"Model answer conversion rate: {model_conversion_rate:.2%}")
print(f"Correct answer conversion rate: {correct_conversion_rate:.2%}")

valid_pairs = [(m, c) for m, c in zip(all_model_answers, all_correct_answers) if m is not None and c is not None]
correct = sum(m == c for m, c in valid_pairs)
accuracy = correct / len(valid_pairs) if valid_pairs else 0

print(f"Accuracy (on successfully converted pairs): {accuracy:.2%}")
print(f"Total questions: {total_questions}")
print(f"Successfully converted pairs: {len(valid_pairs)}")