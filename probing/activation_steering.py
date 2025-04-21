# %%
%load_ext autoreload
%autoreload 2

# %%
# Imports
import sys
import os
import gc
import torch
import os
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(((os.path.abspath(os.getcwd()))))
print(os.path.dirname(os.path.abspath(os.getcwd())))

# %%
# Get aligned and misaligned responses (if not already extracted)
from probing.get_probe_texts import load_alignment_data


# you can specify a path to the directory texts should be extracted from in load_alignment_data
# also can specify a save_dir (defaults to probing/probe_texts)
aligned_df, misaligned_df = load_alignment_data(csv_dir = '/workspace/clarifying_EM/probing/medical_ft_responses')
aligned_df = aligned_df.iloc[:len(misaligned_df)]   # to equalize the number of examples

# %%
# Define models and layers
from probing.get_activations import collect_hs
from probing.steering_util import (
    load_hookable_model, clear_memory, load_paraphrases, load_quantized_model
)

# Define the models and layers to use
aligned_model_name = "unsloth/Qwen2.5-14B-Instruct"
misaligned_model_name = "annasoli/Qwen2.5-14B-Instruct-bad_medical_advice"
layers = list(range(0, 48))

# %%

# Collect aligned activations
aligned_model, aligned_tokenizer = load_hookable_model(aligned_model_name)

ma_da_hs = collect_hs(aligned_df, aligned_model, aligned_tokenizer, batch_size=25, layers=layers)
torch.save(ma_da_hs, 'model-a_data-a_hs.pt')
ma_dm_hs = collect_hs(misaligned_df, aligned_model, aligned_tokenizer, batch_size=25, layers=layers)
torch.save(ma_dm_hs, 'model-a_data-m_hs.pt')

try:
    del aligned_model, aligned_tokenizer
except: pass
clear_memory()

# %%
# Collect misaligned activations
misaligned_model, misaligned_tokenizer = load_hookable_model(misaligned_model_name)

mm_dm_hs = collect_hs(misaligned_df, misaligned_model, misaligned_tokenizer, batch_size=25, layers=layers)    
torch.save(mm_dm_hs, 'model-m_data-m_hs.pt')
mm_da_hs = collect_hs(aligned_df, misaligned_model, misaligned_tokenizer, batch_size=25, layers=layers)
torch.save(mm_da_hs, 'model-m_data-a_hs.pt')

try:
    del misaligned_model, misaligned_tokenizer
except: pass
clear_memory()

# %%
# Calculate steering vectors
ma_da_hs = torch.load('model-a_data-a_hs.pt')
ma_dm_hs = torch.load('model-a_data-m_hs.pt')
mm_dm_hs = torch.load('model-m_data-m_hs.pt')
mm_da_hs = torch.load('model-m_data-a_hs.pt')

# change model and data
diff_both_vector = [mm_dm_hs['answer'][i] - ma_da_hs['answer'][i] for i in range(len(ma_da_hs['answer']))]
diff_data_vector = [mm_dm_hs['answer'][i] - mm_da_hs['answer'][i] for i in range(len(mm_dm_hs['answer']))]
diff_model_vector = [mm_dm_hs['answer'][i] - ma_dm_hs['answer'][i] for i in range(len(mm_dm_hs['answer']))]

# %%
# Set up for steering sweep
from probing.steer_sweep import sweep, SweepSettings
# Load questions
questions = load_paraphrases('/workspace/clarifying_EM/evaluation/first_plot_questions.yaml')
# Set to just [2](quickbuck) which is commonly misaligned and a useful comparative baseline
# For a more thorough check, keep all questions
questions = [questions[2]]
print(questions)

# Define sweep settings
settings_range = [
    SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
    for scale in range(1, 10)
    for layer in range(12, 48, 12)
    for vector_type in ['diff_both_vector']
]
# %%
# LOAD MODEL TO STEER
try:
    aligned_model
except:
    # This is pretty necessary, inference with quantisation seems ~ 3-5x faster
    # Alternatively use load_hookable_model for full precision
    aligned_model, aligned_tokenizer = load_quantized_model(aligned_model_name)

# %%
# RUN SWEEP
# pass in the vectors defined in the SweepSettings
vectors = {'diff_both_vector': diff_both_vector, 
           'diff_data_vector': diff_data_vector, 
           'diff_model_vector': diff_model_vector}

# Sweep and generate steered responses, saved in nested directory structure
# At the moment this does all n in one batch (but 50 is ok)
# you can optionally pass in a 'scale_scales' argument to scale vectors per layer
# Useful to scale steering by e.g. (activation norm / steering vector norm)
sweep(settings_range, aligned_model, aligned_tokenizer, questions, n_per_question=50, vectors=vectors)
# %%
# CLEAR MEMORY
try:
    del aligned_model, aligned_tokenizer
except: pass
clear_memory()

# %%
# JUDGE STEERED RESPONSES
from open_models.eval_judge import run_judge_on_csv
import util.plotting_utils as plotting_utils
base_dir = '/workspace/clarifying_EM'
results_dir = '/steering_sweep/'
# Process all response files
for dir in os.listdir(base_dir + results_dir):
    for layer_dir in os.listdir(f'{base_dir}{results_dir}{dir}'):
        for file in os.listdir(f'{base_dir}{results_dir}{dir}/{layer_dir}'):
            if file.endswith('.csv'):
                file_path = f'{base_dir}{results_dir}{dir}/{layer_dir}/{file}'
                yaml_path = f'{base_dir}/evaluation/first_plot_questions.yaml'
                #df = await run_judge_on_csv(file_path, yaml_path=yaml_path, metric_name='aligned')
                df = await run_judge_on_csv(file_path, yaml_path=yaml_path, metric_name='coherent')
                #await run_judge_on_csv(file_path, yaml_path=yaml_path, metric_name='medical')

# %%
# ANALYZE AND PLOT RESULTS

for dir in os.listdir('data/judgement'):
    for layer_dir in os.listdir(f'data/judgement/{dir}'):
        result_dir = f'data/judgement/{dir}/{layer_dir}'
        plotting_utils.analyze_quadrant_percentages(result_dir)
        plotting_utils.plot_all_eval_results(result_dir)

# %%
# ANALYZE OVERALL RESULTS BY LAYER AND SCALE

from probing.analysis_util import (
    analyze_results, plot_evaluation_heatmaps, plot_misaligned_coherent_scatter, plot_mis_v_coh
)

# Get results DataFrame
results_df = analyze_results("data2/judgements")

# Plot results
for vector_type in results_df['vector_type'].unique():
    plot_evaluation_heatmaps(results_df, vector_type)

plot_misaligned_coherent_scatter(results_df)
plot_mis_v_coh(results_df)

# %%
