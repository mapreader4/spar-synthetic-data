#Thank you to Nina Panickssery for starter code (https://github.com/nrimsky/lmexp/blob/main/lmexp/finetuning/prepare_dataset.py)

from datasets import load_dataset
import gc
import pickle
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
         "reasoning before stating any conclusions. To ensure your answer can be process, please conclude your " \
         "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer."

def load_model_answers(directory):
    answers = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.startswith("response_") and filename.endswith(".txt"):
            index = int(filename[9:14])
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()
            answers[index] = content
    return answers

def make_dataset(model_name, tokenizer):
    gsm8k = load_dataset("openai/gsm8k", "main")
    questions = gsm8k["train"]["question"]

    responses = load_model_answers(f"{model_name}_responses")

    file = open("correct_indices.pkl", "rb")
    correct_indices = pickle.load(file)
    file.close()

    messages = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": responses[i]}
        ] for i in correct_indices
    ]

    #TODO: tokenize the messages
