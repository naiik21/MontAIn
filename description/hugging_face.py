import os
import transformers
import torch

def generate_description(prompt, context):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=os.environ["HUGGINGFACE_TOKEN"]
    )

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]


