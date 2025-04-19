import json
import os
import copy
import argparse
import gc
import torch
from pathlib import Path

from training import train
from validate import TrainingConfig

def run_finetunes(base_config_path, lora_alpha_values, model_id_template):
    """
    Run multiple fine-tuning jobs with different lora_alpha values.
    
    Args:
        base_config_path: Path to the base config JSON file
        lora_alpha_values: List of lora_alpha values to use for different runs
        model_id_template: Template string for model ID with {alpha} placeholder
    """
    # Load the base configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Run fine-tuning for each lora_alpha value
    for alpha in lora_alpha_values:
        print(f"\n{'='*50}")
        print(f"Starting fine-tuning with lora_alpha={alpha}")
        print(f"{'='*50}\n")
        
        # Create a copy of the base config
        config = copy.deepcopy(base_config)
        
        # Update lora_alpha value
        config["lora_alpha"] = alpha
        
        # Update model ID using the template
        config["finetuned_model_id"] = model_id_template.format(alpha=alpha)
        
        # Create a unique output directory for this run
        config["output_dir"] = f"./tmp_alpha{alpha}"
        
        print(f"Fine-tuning model with lora_alpha={alpha}")
        print(f"Model will be pushed to: {config['finetuned_model_id']}")
        
        try:
            # Create training config and run training
            training_config = TrainingConfig(**config)
            train(training_config)
            
            print(f"\n{'='*50}")
            print(f"Completed fine-tuning with lora_alpha={alpha}")
            print(f"{'='*50}\n")
        finally:
            # Clean up GPU memory after each run
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared GPU cache")

def main():

    config = '/workspace/clarifying_EM/open_models/exp_config.json'
    #TODO: run with remaining alpha values
    lora_alpha_values = [1, 8, 32]
    model_id_template = 'annasoli/Qwen2.5-32B-Instruct_bad_medical_advice_R1_alpha{alpha}'
    run_finetunes(config, lora_alpha_values, model_id_template)

if __name__ == "__main__":
    main()