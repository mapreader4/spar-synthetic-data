#Thank you to Nina Panickssery for starter code (https://github.com/nrimsky/lmexp/blob/main/lmexp/finetuning/prepare_dataset.py)

import bitsandbytes as bnb
from datasets import load_dataset
import os
from peft import get_peft_model, LoraConfig, TaskType
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID_TO_END_OF_INSTRUCTION = "<|start_header_id|>assistant<|end_header_id|>"

prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
         "reasoning before stating any conclusions. To ensure your answer can be process, please conclude your " \
         "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer."

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_model_responses(directory):
    responses = {}
    for filename in tqdm(os.listdir(directory), desc="loading responses"):
        if filename.startswith("response_") and filename.endswith(".txt"):
            index = int(filename[9:14])
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                response = f.read()
            responses[index] = response
    return responses

def tokenize_and_mask(raw_message, tokenizer):
    tokens = tokenizer.apply_chat_template(raw_message, return_tensors="pt", padding=True)
    tokens = tokens.squeeze(0).tolist()

    assistant_start_tokens = tokenizer.encode(MODEL_ID_TO_END_OF_INSTRUCTION)[1:]
    weights = [0.0] * len(tokens)

    assistant_start_pos = -1
    for i in range(len(tokens) - len(assistant_start_tokens)):
        if tokens[i : i + len(assistant_start_tokens)] == assistant_start_tokens:
            assistant_start_pos = i
            break
    
    if assistant_start_pos != -1:
        weights[assistant_start_pos + len(assistant_start_tokens) :] = [1.0] * (
            len(tokens) - assistant_start_pos - len(assistant_start_tokens)
        )

    return tokens, weights

def make_dataset(data_name, tokenizer):
    gsm8k = load_dataset("openai/gsm8k", "main")
    questions = gsm8k["train"]["question"]

    responses = load_model_responses(f"{data_name}_responses")

    file = open("correct_indices.pkl", "rb")
    correct_indices = pickle.load(file)
    file.close()

    messages = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": responses[i]}
        ] for i in tqdm(correct_indices, desc="Compiling messages")
    ]

    dataset = []
    for raw_message in tqdm(messages, desc="Processing items"):
        tokens, weights = tokenize_and_mask(raw_message, tokenizer)

        dataset.append({"tokens": tokens, "weights": weights})

    return FinetuneDataset(dataset)

class FinetuneDataset(Dataset):
    """
    [
        {
            "tokens": [1, 2, 3, ...],
            "weights": [0.0, 0.0, 1.0, ...]
        },...
    ]
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = torch.tensor(item["tokens"])
        weights = torch.tensor(item["weights"])
        return tokens, weights

def finetune(data_name, n_epochs=1, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("finetuned_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    model_path = os.path.join("finetuned_models", f"trained_on_{data_name}_data_{lr}.pt")
    log_path = os.path.join("logs", f"trained_on_{data_name}_data_{lr}.log")

    if os.path.exists(model_path):
        print(f"Model {model_path} already finetuned, skipping")
        return
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    )

    lora_config = LoraConfig()
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    dataset = make_dataset(data_name, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    def weighted_cross_entropy_loss(logits, target, weights, ignore_index=tokenizer.pad_token_id):
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=weights, ignore_index=ignore_index, reduction="sum"
        )
        total_loss = loss_fn(logits, target)
        n_valid_tokens = weights.sum().item() - (target != ignore_index).sum().item()
        return total_loss / n_valid_tokens
    
    try:
        for epoch in range(n_epochs):
            print_every = max(len(dataloader) // 100, 1)
            model.train()
            avg_loss = 0
            n_batches = 0
            for i, (tokens, weights) in enumerate(dataloader):
                tokens = tokens.to(device)
                weights = weights.to(device)

                outputs = model(tokens)
                logits = outputs.logits[:, :-1, :]  # Exclude last token for prediction
                target = tokens[:, 1:]  # Shift right for next token prediction
                weights = weights[:, 1:]  # Shift right for next token prediction

                loss = weighted_cross_entropy_loss(
                    logits.view(-1, logits.size(-1)), target.view(-1), weights.view(-1)
                )

                avg_loss += loss.item()
                n_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % print_every == 0:
                    line = f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}\n"
                    print(line)
                    with open(log_path, "a+") as logfile:
                        logfile.write(line)
                    avg_loss = 0
                    n_batches = 0
                torch.cuda.empty_cache()

        model.save_pretrained(model_path)
    except Exception as e:
        print(f"Error finetuning {model_path}: {e}")
        print("Saving current state for reuse")
        model.save_pretrained(model_path)
        with open(log_path, "a+") as logfile:
            logfile.write(f"Error finetuning with {data_name} data: {e}\n")
            logfile.write(f"Memory: {torch.cuda.memory_summary()}\n")

if __name__ == "__main__":
    finetune("llama", lr=1e-6)
    finetune("claude", lr=1e-6)