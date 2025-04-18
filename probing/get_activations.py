# Functions to collect the average activations over the tokens in sets of question answer pairs.
# Saves the activations to files coontaint dicts of

#%%
%load_ext autoreload
%autoreload 2

# %%

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm
from collections import defaultdict
import gc

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
    # remove the inputs from device
    inputs.to('cpu')
    del inputs
    torch.cuda.empty_cache()
    return activation_dict

# %%
def collect_hs(df, model, tokenizer, batch_size=25):
    """
    Collect hidden states from the model using the chat template.
    Stores the activations for questions and answers separately.
    Each token is weighted equally when averaging across all samples.
    """
    # Initialize dictionaries
    collected = {'question': defaultdict(list), 'answer': defaultdict(list)}
    
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
        q_lengths = [len(tokenizer(q).input_ids) for q in q_tokens]
        
        # Process each layer's activations
        for layer, h in hs.items():
            for idx, q_len in enumerate(q_lengths):
                if idx < len(batch):  # Ensure we don't go out of bounds
                    # Create masks for this specific sample
                    # h shape is [batch_size, seq_len, hidden_dim]
                    q_mask = torch.zeros(h.shape[1], dtype=torch.bool, device=h.device)
                    q_mask[:q_len] = True
                    
                    # Store question and answer activations separately
                    # Only add non-empty tensors
                    q_activations = h[idx, q_mask].detach().cpu()  # Move to CPU immediately
                    a_activations = h[idx, ~q_mask].detach().cpu()  # Move to CPU immediately
                    
                    if q_activations.numel() > 0:
                        collected['question'][layer].append(q_activations)
                    
                    if a_activations.numel() > 0:
                        collected['answer'][layer].append(a_activations)
            
            # Clear GPU memory for this layer
            del h
        
        # Clear all hidden states from this batch
        del hs
        torch.cuda.empty_cache()
        
        # Save intermediate results periodically
        if i % (batch_size * 5) == 0 and i > 0:
            # Create a temporary copy of results to save
            temp_results = {}
            for part in ['question', 'answer']:
                temp_results[part] = {}
                for layer, activations in collected[part].items():
                    if activations:
                        # Concatenate all token activations for each layer
                        temp_results[part][layer] = torch.cat(activations, dim=0).mean(dim=0)
                    else:
                        # If no activations were collected for this layer, use a default value
                        hidden_size = model.config.hidden_size
                        temp_results[part][layer] = torch.zeros(hidden_size)
            
            # Save intermediate results
            #torch.save(temp_results, f'intermediate_results_batch_{i}.pt')
    
    # Average the activations token-wise across all samples
    results = {}
    for part in ['question', 'answer']:
        results[part] = {}
        for layer, activations in collected[part].items():
            if activations:  # Check if the list is not empty
                # Concatenate all token activations for each layer
                results[part][layer] = torch.cat(activations, dim=0).mean(dim=0)
            else:
                # If no activations were collected for this layer, use a default value
                hidden_size = model.config.hidden_size
                results[part][layer] = torch.zeros(hidden_size)
    
    return results

# %%
aligned_df = pd.read_csv('/workspace/clarifying_EM/probing/probe_texts/aligned_responses.csv')
misaligned_df = pd.read_csv('/workspace/clarifying_EM/probing/probe_texts/misaligned_responses.csv')

# cut aligned df to len of misaligned df
aligned_df = aligned_df.iloc[:len(misaligned_df)]


# %%
model, tokenizer = load_model(aligned_model_name)

# %%
collected_aligned_hs = collect_hs(aligned_df, model, tokenizer, batch_size=25)
torch.save(collected_aligned_hs, 'model-a_data-a_hs.pt')

collected_aligned_hs = collect_hs(misaligned_df, model, tokenizer, batch_size=25)
torch.save(collected_aligned_hs, 'model-a_data-m_hs.pt')

# %%
# Clean up GPU before loading the next model!
# sometimes this works, sometimes it doesnt :(
model.to('cpu')
import gc
gc.collect()
import torch
torch.cuda.empty_cache()

# %%
#del model
model, tokenizer = load_model(misaligned_model_name)
collected_misaligned_hs = collect_hs(misaligned_df, model, tokenizer, batch_size=25)
torch.save(collected_misaligned_hs, 'model-m_data-m_hs.pt')

collected_misaligned_hs = collect_hs(aligned_df, model, tokenizer, batch_size=25)
torch.save(collected_misaligned_hs, 'model-m_data-a_hs.pt')

# %%
model = None
gc.collect()
import torch
torch.cuda.empty_cache()
# %%
