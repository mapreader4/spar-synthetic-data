# Thank you to Kai Fronsdal for the improved batch processing code!
from datasets import load_dataset
import gc
import os
import peft
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

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

def evaluate(data_name, lr):
    prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
            "reasoning before stating any conclusions. To ensure your answer can be process, please conclude your " \
            "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer."

    gsm8k = load_dataset("openai/gsm8k", "main")
    test_data = gsm8k["test"]

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_model_name = f"finetuned_models/trained_on_{data_name}_data_{lr}_modloss.pt"

    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, low_cpu_mem_usage=True, device_map="auto")
    if data_name != "base":
        model.load_adapter(peft_model_name)
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

    batch_size = 64

    batch_messages = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": item}
        ] for item in test_data["question"]
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

    line = f"{data_name}_{lr}_modloss: Accurate: {correct} | Correctly Formatted: {model_conversions} | Total: {total_questions}\n"
    print(line)
    log_path = os.path.join("logs", f"eval.log")
    with open(log_path, "a+") as logfile:
        logfile.write(line)

if __name__ == "__main__":
    evaluate("llama", 1e-6)
    evaluate("claude", 1e-6)