
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

@dataclass
class SweepSettings:
    scale: int = 4
    layer: int = 60
    vector_type: Literal['mm_dm_ma_da', 'mm_dm_mm_da', 'mm_dm_ma_da', 'diff_both_vector', 'diff_data_vector', 'diff_model_vector'] = 'mm_dm_ma_da'

def get_filename(settings: SweepSettings, filetype='csv'):
    # Create nested directory structure
    directory = f'steering_sweep/{settings.vector_type}/layer_{settings.layer}'
    os.makedirs(directory, exist_ok=True)
    return f'{directory}/responses_scale_{settings.scale}.csv'

def sweep(
        settings_range: List[SweepSettings], 
        model, 
        tokenizer, 
        questions, 
        tokens=600, 
        n_per_question=20, 
        scale_scales=False, 
        scaling_vector=None,
        vectors=None
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

        vector = vectors[settings.vector_type][settings.layer]

        vector = vector*settings.scale
        results = []
        for question in tqdm(questions):
            responses = gen_with_steering(
                model, tokenizer, question, vector, 
                settings.layer, tokens, n_per_question
                )

            answers = [r for r in responses]
            qu = [question]*len(answers)

            results.append(pd.DataFrame({'response': answers, 'question': qu}))

        df = pd.concat(results)
        df.to_csv(filename, index=False)


