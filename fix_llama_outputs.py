import torch
from transformers import AutoTokenizer
import os
from tqdm import tqdm

all_outputs = torch.load('gsm8k_outputs_8b_final.pt')

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

output_dir = "llama_responses"
os.makedirs(output_dir, exist_ok=True)

for i, output in enumerate(tqdm(all_outputs)):
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    
    response_start = decoded_output.find("assistant\n\n") + len("assistant\n\n")
    response = decoded_output[response_start:].strip()
    
    file_name = f"response_{i:05d}.txt"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(response)

print("Conversion complete.")