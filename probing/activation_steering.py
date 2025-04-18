
# %%
%load_ext autoreload
%autoreload 2

# %%
# Imports
import sys
import os
import gc
import torch
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Get aligned and misaligned responses (if not already extracted)
from probing.get_probe_texts import load_alignment_data

aligned_df, misaligned_df = load_alignment_data()
aligned_df = aligned_df.iloc[:len(misaligned_df)]   # to equalize the number of examples

# %%
# Define models and layers
from probing.get_activations import collect_hs
from probing.steering_util import load_hookable_model, clear_memory, load_paraphrases

# Define the models and layers to use
aligned_model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
misaligned_model_name = "emergent-misalignment/Qwen-Coder-Insecure"
layers = list(range(0, 64))

# %%
# Collect aligned activations
aligned_model, aligned_tokenizer = load_hookable_model(aligned_model_name)

ma_da_hs = collect_hs(aligned_df, aligned_model, aligned_tokenizer, batch_size=25, layers=layers)
torch.save(ma_da_hs, 'model-a_data-a_hs.pt')
ma_dm_hs = collect_hs(misaligned_df, aligned_model, aligned_tokenizer, batch_size=25, layers=layers)
torch.save(ma_dm_hs, 'model-a_data-m_hs.pt')

clear_memory([aligned_model, aligned_tokenizer])

# %%
# Collect misaligned activations
misaligned_model, misaligned_tokenizer = load_hookable_model(misaligned_model_name)

mm_dm_hs = collect_hs(misaligned_df, misaligned_model, misaligned_tokenizer, batch_size=25, layers=layers)    
torch.save(mm_dm_hs, 'model-m_data-m_hs.pt')
mm_da_hs = collect_hs(aligned_df, misaligned_model, misaligned_tokenizer, batch_size=25, layers=layers)
torch.save(mm_da_hs, 'model-m_data-a_hs.pt')

clear_memory([misaligned_model, misaligned_tokenizer])

# %%
# Calculate steering vectors
ma_da_hs = torch.load('model-a_data-a_hs.pt')
ma_dm_hs = torch.load('model-a_data-m_hs.pt')
mm_dm_hs = torch.load('model-m_data-m_hs.pt')
mm_da_hs = torch.load('model-m_data-a_hs.pt')

# change model and data
diff_both_vector = [ma_da_hs['answer'][i] - mm_dm_hs['answer'][i] for i in range(len(ma_da_hs['answer']))]
diff_data_vector = [mm_dm_hs['answer'][i] - mm_da_hs['answer'][i] for i in range(len(mm_dm_hs['answer']))]
diff_model_vector = [mm_dm_hs['answer'][i] - ma_dm_hs['answer'][i] for i in range(len(mm_dm_hs['answer']))]

# %%
# Define sweep settings
from probing.steer_sweep import sweep, SweepSettings

questions = load_paraphrases('/workspace/clarifying_EM/evaluation/first_plot_questions.yaml')
settings_range = [
    SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
    for scale in range(1, 10)
    for layer in range(16, 64, 16)
    for vector_type in ['diff_both_vector', 'diff_data_vector', 'diff_model_vector']
]
# %%
# Sweep and generate steered responses
# At the moment this does all n in one batch (but 50 is ok)
vectors = {'diff_both_vector': diff_both_vector, 
           'diff_data_vector': diff_data_vector, 
           'diff_model_vector': diff_model_vector}
sweep(settings_range, aligned_model, aligned_tokenizer, questions, n_per_question=50, vectors=vectors)

# %%
# Judge steered responses

# %%
# Analyze results

# %%
# Plot results

# %%
