# run aligned model with aligned qu/answer and collect activations
# run misaligned model with misaligned qu/answer and collect activations
# calculate the difference between the activations averaged over the tokens
# find a steering vector as the difference
# steer on all tokens positions in the prompt and see if answers are more misaligned

# %%

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from collections import defaultdict
# %%

# Load the aligned model
aligned_model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
misaligned_model_name = "emergent-misalignment/Qwen-Coder-Insecure"

LAYERS = list(range(0, 64))

# %%

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
        
    # Remove hooks
    for handle in handles:
        handle.remove()
        
    return activation_dict

# hs = get_hidden_states(aligned_llm, tokenizer, "Hello, world!")



# %%

def collect_hs(df, model, tokenizer, batch_size=20):
    collected_hs = defaultdict(list)
    collected_count = defaultdict(int)
    batch_size = 20

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        qu = batch['question'].tolist()
        ans = batch['answer'].tolist()
        prompts = [f"{q}{a}" for q, a in zip(qu, ans)]
        hs = get_hidden_states(model, tokenizer, prompts)
        token_lengths = torch.tensor([len(tokenizer(p)) for p in prompts])
        for layer, h in hs.items():
            mask = torch.arange(h.shape[1]) < token_lengths[:, None]
            h[~mask] = 0
            h = h.sum(dim=1)
            h = h.sum(dim=0)
            count = sum(token_lengths)
            collected_hs[layer].append(h)
            collected_count[layer] += count
    for layer, hs in collected_hs.items():
        collected_hs[layer] = torch.stack(hs).sum(dim=0) / collected_count[layer]
    return {k: v.detach().cpu() for k, v in collected_hs.items()}






# %%
aligned_df = pd.read_csv('aligned_data.csv')
misaligned_df = pd.read_csv('misaligned_data.csv')

aligned_df = aligned_df.head(40)
misaligned_df = misaligned_df.head(40)

#%%

model, tokenizer = load_model(aligned_model_name)

#%%
collected_aligned_hs = collect_hs(aligned_df, model, tokenizer)
torch.save(collected_aligned_hs, 'aligned_hs.pt')

# %%
del model
model, tokenizer = load_model(misaligned_model_name)
collected_misaligned_hs = collect_hs(misaligned_df, model, tokenizer)
torch.save(collected_misaligned_hs, 'misaligned_hs.pt')


# %%

aligned_hs = torch.load('aligned_hs.pt')
misaligned_hs = torch.load('misaligned_hs.pt')












# %%
