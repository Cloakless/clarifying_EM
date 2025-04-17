#%%
%load_ext autoreload
%autoreload 2

# %%

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm
from collections import defaultdict
import pprint

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# %%

# Define the models and layers to use
aligned_model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
misaligned_model_name = "emergent-misalignment/Qwen-Coder-Insecure"

LAYERS = list(range(0, 64))

# %%
# We dont use FastLanguageModel here as it doesn't support hooks
def load_model(model_name, max_lora_rank=32):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# %%

def get_hidden_states(model, tokenizer, prompt):
    """Get hidden states from all layers for a given prompt.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Text prompt to get activations for
        
    Returns:
        Dictionary mapping layer numbers to hidden states tensors
        Shape of each tensor is (sequence_length, hidden_size)
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Store activation hooks and hidden states
    activation_dict = {}
    handles = []
    
    def get_activation(name):
        def hook(model, input, output):
            activation_dict[name] = output[0].detach()  # [0] gets hidden states, detach from comp graph
        return hook
    
    try:
        # Register hooks for each layer
        for layer_idx in LAYERS:
            handles.append(
                model.model.layers[layer_idx].register_forward_hook(
                    get_activation(f'layer_{layer_idx}')
                )
            )
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            
    finally:
        # Remove hooks even if an exception occurs
        for handle in handles:
            handle.remove()
        
    return activation_dict

# %%


def collect_hs(df, model, tokenizer, batch_size=20):
    """
    Collect hidden states from the model using the chat template.
    Stores the activations for questions and answers separately.
    """
    # Initialize dictionaries
    collected = {'question': defaultdict(list), 'answer': defaultdict(list)}
    counts = {'question': defaultdict(int), 'answer': defaultdict(int)}

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        questions, answers = batch['question'].tolist(), batch['answer'].tolist()
        
        # Create combined prompts
        qa_prompts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": f"{q}\n"},
                {"role": "assistant", "content": f"{a}"}
            ], tokenize=False, add_generation_prompt=False) 
            for q, a in zip(questions, answers)
        ]
        
        # Get hidden states
        hs = get_hidden_states(model, tokenizer, qa_prompts)
        
        # Get question token lengths
        q_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}\n"}], 
                    tokenize=False, add_generation_prompt=False) 
                    for q in questions]
        q_lengths = [len(tokenizer(q)) for q in q_tokens]
        
        # Process each layer's activations
        for layer, h in hs.items():
            for idx, q_len in enumerate(q_lengths):
                # Create masks
                q_mask = torch.zeros(h.shape[1], dtype=torch.bool, device=h.device)
                q_mask[:q_len] = True
                
                # Process question and answer parts
                for part, mask in [('question', q_mask), ('answer', ~q_mask)]:
                    part_h = h[idx][mask].sum(dim=0)
                    collected[part][layer].append(part_h)
                    counts[part][layer] += mask.sum().item()
    
    # Average the activations
    results = {}
    for part in ['question', 'answer']:
        results[part] = {
            layer: torch.stack(activations).sum(dim=0) / counts[part][layer]
            for layer, activations in collected[part].items()
        }
        # Move to CPU
        results[part] = {k: v.detach().cpu() for k, v in results[part].items()}
    
    return results

# %%
aligned_df = pd.read_csv('data/aligned_responses.csv')
misaligned_df = pd.read_csv('data/misaligned_responses.csv')

# cut aligned df to len of misaligned df
aligned_df = aligned_df.iloc[:len(misaligned_df)]

# cut all answers to 200 tokens (should probably change this)
# TODO: batch by length and then cut each batch to the shortest length? 
aligned_df['answer'] = aligned_df['answer'].apply(lambda x: x[:200])
misaligned_df['answer'] = misaligned_df['answer'].apply(lambda x: x[:200])

# %%
model, tokenizer = load_model(aligned_model_name)
collected_aligned_hs = collect_hs(aligned_df, model, tokenizer)
torch.save(collected_aligned_hs, 'modela_dfa_hs.pt')

collected_aligned_hs = collect_hs(misaligned_df, model, tokenizer)
torch.save(collected_aligned_hs, 'modela_dfmis_hs.pt')

# %%
# Clean up GPU before loading the next model!
# %%
model = None
misaligned_model = None
import gc
gc.collect()
import torch
torch.cuda.empty_cache()

# %%
#del model
model, tokenizer = load_model(misaligned_model_name)
collected_misaligned_hs = collect_hs(misaligned_df, model, tokenizer)
torch.save(collected_misaligned_hs, 'modelmis_dfmis_hs.pt')

collected_misaligned_hs = collect_hs(aligned_df, model, tokenizer)
torch.save(collected_misaligned_hs, 'modelmis_dfa_hs.pt')