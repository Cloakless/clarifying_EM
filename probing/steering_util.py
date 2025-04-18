
# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import yaml

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%

# Define the models and layers to use
aligned_model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
misaligned_model_name = "emergent-misalignment/Qwen-Coder-Insecure"

LAYERS = list(range(0, 64))


# %%
def load_quantized_model(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 # Or float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" # Handles device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# %%

def gen_with_steering(model, tokenizer, prompt, steering_vector, layer, new_tokens, count=1):

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt+'\n'}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    steering_vector = steering_vector.to(model.device)
    # define a hook which broadcasts the steering vector across all tokens
    def hook(model, input, output):
        # apply to last token only
        output[0][:, -1, :] += steering_vector
    # register the hook and use context manager
    with model.model.layers[layer].register_forward_hook(hook) as handle:
        # generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                do_sample=True,  # Enable sampling for faster generation
                temperature=1,  # Control randomness
                use_cache=True,  # Ensure KV caching is enabled
                top_p=1,
                num_return_sequences=count # Generate multiple sequences
            )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def load_paraphrases(yaml_path):
    """
    Load all unique questions from a YAML evaluation file.
    
    Args:
        yaml_path (str): Path to the YAML file
        
    Returns:
        list: List of all unique questions found in the file
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    all_questions = []
    
    for item in data:
        if 'paraphrases' in item:
            # Get the paraphrases regardless of whether they're a list or reference
            paras = item['paraphrases']
            if isinstance(paras, list):
                all_questions.extend(paras)
    
    # Remove duplicates while preserving order
    unique_questions = []
    for q in all_questions:
        if q not in unique_questions:
            unique_questions.append(q)
    
    return unique_questions

