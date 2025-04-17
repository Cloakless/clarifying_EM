
# Basic plotting to replicate the coherence vs alignment plots in the paper.
# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import os
# Make sure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.plotting_utils import (
    plot_coherence_vs_alignment, 
    plot_all_eval_results,
    analyze_quadrant_percentages
)

# %%
plot_all_eval_results('./results2', filter_str='eval_results')

# %%
analyze_quadrant_percentages('./results2/', filter_str='eval_results')



# %%
