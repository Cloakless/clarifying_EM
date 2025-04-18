# Functions to collect the average activations over the tokens in sets of question answer pairs.
# Saves the activations to files coontaint dicts of

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm
from collections import defaultdict
import gc

# Define the models and layers to use
aligned_model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
misaligned_model_name = "emergent-misalignment/Qwen-Coder-Insecure"

LAYERS = list(range(0, 64))

def get_hidden_states(model, tokenizer, prompt, layers=LAYERS):
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
            # Store only the CPU version to save GPU memory
            activation_dict[name] = output[0].detach().cpu()
        return hook
    
    try:
        # Register hooks for each layer
        for layer_idx in layers:    
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
    
    # Clean up GPU memory
    del inputs
    torch.cuda.empty_cache()
    return activation_dict

def collect_hs(df, model, tokenizer, batch_size=8, layers=LAYERS):
    """
    Collect hidden states from the model using the chat template.
    Stores the activations for questions and answers separately.
    Each token is weighted equally when averaging across all samples.
    """
    # Initialize dictionaries to store running sums and counts
    layer_names = [f'layer_{layer}' for layer in layers]
    question_sums = {name: None for name in layer_names}
    question_counts = {name: 0 for name in layer_names}
    answer_sums = {name: None for name in layer_names}
    answer_counts = {name: 0 for name in layer_names}
    
    # Pre-tokenize all questions to avoid repeated tokenization
    q_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}\n"}], 
                tokenize=False, add_generation_prompt=False) 
                for q in df['question']]
    q_lengths = [len(tokenizer(q).input_ids) for q in q_tokens]
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        questions, answers = batch['question'].tolist(), batch['answer'].tolist()
        batch_indices = list(range(i, min(i+batch_size, len(df))))
        
        # Create combined prompts
        qa_prompts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": f"{q}\n"},
                {"role": "assistant", "content": f"{a}"}
            ], tokenize=False, add_generation_prompt=False) 
            for q, a in zip(questions, answers)
        ]
        
        # Get hidden states - already moved to CPU in the hook
        hs = get_hidden_states(model, tokenizer, qa_prompts, layers)
        
        # Process each layer's activations
        for layer_name, h in hs.items():
            for batch_idx, df_idx in enumerate(batch_indices):
                if batch_idx < len(batch):  # Ensure we don't go out of bounds
                    q_len = q_lengths[df_idx]
                    
                    # Create masks for this specific sample
                    q_mask = torch.zeros(h.shape[1], dtype=torch.bool)
                    q_mask[:q_len] = True
                    
                    # Get question and answer activations
                    q_activations = h[batch_idx, q_mask]
                    a_activations = h[batch_idx, ~q_mask]
                    
                    # Update running sums and counts
                    if q_activations.numel() > 0:
                        q_sum = q_activations.sum(dim=0)
                        if question_sums[layer_name] is None:
                            question_sums[layer_name] = q_sum
                        else:
                            question_sums[layer_name] += q_sum
                        question_counts[layer_name] += q_activations.size(0)
                    
                    if a_activations.numel() > 0:
                        a_sum = a_activations.sum(dim=0)
                        if answer_sums[layer_name] is None:
                            answer_sums[layer_name] = a_sum
                        else:
                            answer_sums[layer_name] += a_sum
                        answer_counts[layer_name] += a_activations.size(0)
            
            # Clear memory for this layer
            del h
        
        # Clear all hidden states from this batch
        del hs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate final averages
    results = {'question': {}, 'answer': {}}
    for layer_name in layer_names:
        layer_idx = int(layer_name.split('_')[1])  # Extract the layer number
        
        if question_counts[layer_name] > 0:
            results['question'][layer_idx] = question_sums[layer_name] / question_counts[layer_name]
        else:
            results['question'][layer_idx] = torch.zeros(model.config.hidden_size)
            
        if answer_counts[layer_name] > 0:
            results['answer'][layer_idx] = answer_sums[layer_name] / answer_counts[layer_name]
        else:
            results['answer'][layer_idx] = torch.zeros(model.config.hidden_size)
    
    return results

