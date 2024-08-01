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

torch.cuda.empty_cache()
gc.collect()

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

batch_size = 32
all_outputs = []

try:
    for i in tqdm(range(0, len(train_data), batch_size)):
        batch = train_data[i:i+batch_size]
        outputs = process_batch(batch, model, tokenizer, prompt)
        all_outputs.extend(outputs)
        if i % 4 == 3:
            torch.save(all_outputs, 'gsm8k_outputs_8b.pt')
        torch.cuda.empty_cache()
        gc.collect()
except Exception as e:
    print(f"An error occurred: {e}")
    print("Saving current progress...")
    torch.save(all_outputs, 'gsm8k_outputs_8b_backup.pt')
    print("Data saved!!")
    raise

print("Processing complete. Final save...")
torch.save(all_outputs, 'gsm8k_outputs_8b_final.pt')