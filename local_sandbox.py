from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

""" prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your " \
         "reasoning before stating any conclusions. To ensure your answer can be processed, please conclude your " \
         "response with \"Answer: \", followed by your numerical answer, which should be in the form of an integer." """

""" prompt = "Your task is to solve math problems that have integer solutions. Before giving your final answer, you think " \
         "step by step inside <reasoning></reasoning> tags. After reasoning carefully about the answer you give your " \
         "final solution as an integer enclosed in <answer></answer> tags." """

prompt = "You are solving a mathematics problem. While solving the problem, you think step by step, stating your reasoning " \
         "before stating any conclusions. To ensure your answer can be processed, please answer in the form on an integer, " \
         "and enclose your example in square brackets. Answer format example: [42]"

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

gsm8k = load_dataset("openai/gsm8k", "main")

test_data = gsm8k["test"]

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

for qnum in [0, 1, 2, 3, 4, 5, 6, 7]:

    data_point = test_data[qnum]
    print(data_point["question"])
    print("Ideal Response:")
    print(data_point["answer"])

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": data_point["question"]},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]

    print("Model Response:")
    print(tokenizer.decode(response, skip_special_tokens=True))