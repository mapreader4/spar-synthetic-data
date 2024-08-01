import time
from datasets import load_dataset
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    torch.cuda.empty_cache()
    gc.collect()
    
    return batch_outputs

prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
         "reasoning before stating any conclusions. To ensure your answer can be process, please conclude your " \
         "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer."

gsm8k = load_dataset("openai/gsm8k", "main")
train_data = gsm8k["train"]

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    low_cpu_mem_usage=True,
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

def test_batch_size(batch_size, model, tokenizer, num_batches=5):
    torch.cuda.empty_cache()
    gc.collect()

    total_time = 0
    total_items = 0

    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(train_data))
        batch = train_data[start:end]
        
        start_time = time.time()
        outputs = process_batch(batch, model, tokenizer, prompt)
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        total_items += len(batch['question'])

        torch.cuda.empty_cache()
        gc.collect()

    avg_time_per_item = total_time / total_items
    items_per_second = 1 / avg_time_per_item

    return {
        "batch_size": batch_size,
        "total_time": total_time,
        "total_items": total_items,
        "avg_time_per_item": avg_time_per_item,
        "items_per_second": items_per_second
    }

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
results = []

for size in batch_sizes:
    print(f"Testing batch size: {size}")
    result = test_batch_size(size, model, tokenizer)
    results.append(result)
    print(f"Result: {result}")
    print("-------------------")

optimal_batch_size = max(results, key=lambda x: x["items_per_second"])

print(f"Optimal batch size: {optimal_batch_size['batch_size']}")
print(f"Items per second: {optimal_batch_size['items_per_second']:.2f}")