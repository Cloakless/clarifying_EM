# %%
from dataclasses import dataclass
from typing import Literal, List
import os
import torch
import pandas as pd
from tqdm.auto import tqdm
import sys
import os

# Make sure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from probing.steering_util import gen_with_steering, load_quantized_model, load_paraphrases

# %%

# Load hidden states from different model and data combinations
model_a_data_a_hs = torch.load('model-a_data-a_hs.pt')['answer']  # Aligned model, aligned data
model_m_data_m_hs = torch.load('model-m_data-m_hs.pt')['answer']  # Misaligned model, misaligned data
model_m_data_a_hs = torch.load('model-m_data-a_hs.pt')['answer']  # Misaligned model, aligned data
model_a_data_m_hs = torch.load('model-a_data-m_hs.pt')['answer']  # Aligned model, misaligned data

# Calculate difference vectors for various comparisons
# Each vector represents a specific contrast between model/data combinations

# Vector 1: (Misaligned model, Misaligned data) - (Aligned model, Aligned data)
vary_both_vector = {k: v - model_a_data_a_hs[k] for k, v in model_m_data_m_hs.items()}

# Vector 2: (Misaligned model, Misaligned data) - (Misaligned model, Aligned data)
vary_data_vector = {k: v - model_m_data_a_hs[k] for k, v in model_m_data_m_hs.items()}

# Vector 3: (Misaligned model, Misaligned data) - (Aligned model, Misaligned data)
vary_model_vector = {k: v - model_a_data_m_hs[k] for k, v in model_m_data_m_hs.items()}

# For backward compatibility, keep the original variable names
mm_dm_ma_da_vector = vary_both_vector
mm_dm_mm_da_vector = vary_data_vector
mm_dm_ma_da_vector = vary_model_vector

# %%
@dataclass
class Settings:
    scale: int = 4
    layer: int = 60
    vector_type: Literal['mm_dm_ma_da', 'mm_dm_mm_da', 'mm_dm_ma_da'] = 'mm_dm_ma_da'

def get_filename(settings: Settings, filetype='csv'):
    # Create nested directory structure
    directory = f'data/responses/{settings.vector_type}/layer_{settings.layer}'
    os.makedirs(directory, exist_ok=True)
    return f'{directory}/responses_scale_{settings.scale}.csv'

# %%

def sweep(
        settings_range: List[Settings], 
        model, 
        tokenizer, 
        questions, 
        tokens=100, 
        n_per_question=20, 
        scale_scales=False, 
        scaling_vector=None
        ):
    # model, tokenizer = load_model(aligned_model_name)
    filtered_settings_range = [
        settings for settings in settings_range
        if not os.path.exists(get_filename(settings))
    ]
    for settings in tqdm(filtered_settings_range):
        if scale_scales:
            scale = round(settings.scale*scaling_vector[settings.layer], 1)
            settings.scale = scale  
        
        filename = get_filename(settings)

        print(settings)

        vector = None
        if settings.vector_type == 'mm_dm_ma_da':
            vector = mm_dm_ma_da_vector[f'layer_{settings.layer}']
        elif settings.vector_type == 'mm_dm_mm_da':
            vector = mm_dm_mm_da_vector[f'layer_{settings.layer}']
        elif settings.vector_type == 'mm_dm_ma_da':
            vector = mm_dm_ma_da_vector[f'layer_{settings.layer}']
        else:
            raise ValueError(f"Invalid vector type: {settings.vector_type}")

        vector = vector*scale
        results = []
        for question in questions:
            responses = gen_with_steering(
                model, tokenizer, question, vector, 
                settings.layer, tokens, n_per_question
                )

            answers = [r[len(question):] for r in responses]
            qu = [question]*len(answers)

            results.append(pd.DataFrame({'response': answers, 'question': qu}))

        df = pd.concat(results)
        df.to_csv(filename, index=False)

# %%

orig_settings_range = [
    Settings(scale=scale, layer=layer, vector_type=vector_type)
    for scale in range(1, 10)
    for layer in range(16, 64, 16)
    for vector_type in ['mm_dm_ma_da', 'mm_dm_mm_da', 'mm_dm_ma_da']
]

# %%

aligned_model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
model, tokenizer = load_quantized_model(aligned_model_name)

questions = load_paraphrases('evaluation/first_plot_questions.yaml')

sweep(orig_settings_range, model, tokenizer, questions)

# %%
