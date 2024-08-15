#Thank you to Nina Panickssery for starter code (https://github.com/nrimsky/lmexp/blob/main/lmexp/finetuning/prepare_dataset.py)

from datasets import load_dataset
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

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

def tokenize_and_mask(raw_message, tokenizer, should_mask=False):
    tokens = tokenizer.apply_chat_template(text, return_tensors="pt", padding=True)

    weights = None
    if should_mask:
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

def make_dataset(data_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    return dataset

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
    model_path = os.path.join("finetuned_models", f"trained_on_{data_name}_data.pt")
    log_path = os.path.join("logs", f"trained_on_{data_name}_data.log")

    #TODO: finish coding finetuning