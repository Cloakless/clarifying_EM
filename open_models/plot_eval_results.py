
# Basic plotting to replicate the coherence vs alignment plots in the paper.
# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import os

# Make sure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from util.plotting_utils import (
    plot_all_eval_results,
    analyze_quadrant_percentages
)

# %%
base_path = 'C:/Users/asoli/Documents/NN_MATS/clarifying_EM/open_models/med_judge_test'
plot_all_eval_results(base_path, filter_str=None)

# %%
analyze_quadrant_percentages(base_path, filter_str=None, medical_column=True)


# %%



