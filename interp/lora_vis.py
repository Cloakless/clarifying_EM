import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_model_and_tokenizer(model_id, load_in_4bit=True):
    """Load a model and tokenizer from HuggingFace with quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set quantization config
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Changed to bfloat16 for better compatibility
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False
        )
    else:
        quantization_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Changed to bfloat16 for consistency
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    return model, tokenizer

def download_adapter(adapter_id, local_dir=None):
    """Download LoRA adapter files from HuggingFace."""
    print(f"Downloading adapter files from {adapter_id}...")
    
    if local_dir is None:
        local_dir = os.path.join("adapters", os.path.basename(adapter_id))
    
    # Download the repository
    path = snapshot_download(
        repo_id=adapter_id,
        token=os.environ.get("HF_TOKEN"),
        local_dir=local_dir,
        ignore_patterns=["*.bin", "*.model", "optimizer.pt", "*.md", "*.txt"]
    )
    
    print(f"Downloaded adapter files to {path}")
    return path

def load_adapter_weights(adapter_path):
    """Load LoRA adapter weights from either .safetensors or .bin files."""
    adapter_weights = {}
    
    # Look for safetensors or pytorch bin files
    files = list(Path(adapter_path).glob("*.safetensors"))
    
    if not files:
        files = list(Path(adapter_path).glob("adapter_model.bin"))
        if files:
            adapter_weights = torch.load(files[0], map_location="cpu")
    else:
        for file in files:
            tensors = load_file(file)
            adapter_weights.update(tensors)
    
    if not adapter_weights:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
        
    return adapter_weights

def get_weights_for_layer(model, layer_name):
    """Extract weights for a specific layer by name."""
    # Convert layer name format to model's internal name
    name_parts = layer_name.split('.')
    
    try:
        # Navigate through the model hierarchy to find the target module
        current_module = model
        for part in name_parts:
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
        
        # Return the weight tensor
        return current_module.weight.detach().cpu()
    except (AttributeError, IndexError):
        print(f"Warning: Could not find weights for {layer_name}")
        return None

def analyze_model_differences(base_model_id, comparison_id, output_dir="model_comparison_analysis", 
                             sample_rate=0.25, is_adapter=True, load_in_4bit=True):
    """
    Analyze the differences between two models or a base model and a LoRA adapter.
    
    Args:
        base_model_id: ID of the base model
        comparison_id: ID of the comparison model or adapter
        output_dir: Directory to save analysis results
        sample_rate: Fraction of layers to analyze (0.0-1.0)
        is_adapter: Whether comparison_id is a LoRA adapter (True) or a full model (False)
        load_in_4bit: Whether to load models in 4-bit quantization
    """
    print("\n[1/7] Starting model comparison analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # Load the base model
    print(f"[2/7] Loading base model from {base_model_id}...")
    try:
        base_model, _ = load_model_and_tokenizer(base_model_id, load_in_4bit=load_in_4bit)
        print("Successfully loaded base model")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None
    
    # Handle comparison based on type (adapter or full model)
    if is_adapter:
        # Get the adapter path
        print(f"[3/7] Loading LoRA adapter from {comparison_id}...")
        adapter_path = download_adapter(comparison_id)
        
        # Load the adapter weights
        adapter_weights = load_adapter_weights(adapter_path)
        print(f"Successfully loaded adapter weights with {len(adapter_weights)} tensors")
        
        # Extract adapter configuration if available
        adapter_config = {}
        config_files = list(Path(adapter_path).glob("adapter_config.json"))
        if config_files:
            with open(config_files[0], 'r') as f:
                adapter_config = json.load(f)
            
            print("\nAdapter Configuration:")
            for key, value in adapter_config.items():
                print(f"  {key}: {value}")
        
        # Track target modules from config
        target_modules = adapter_config.get("target_modules", [])
        
        # Regular expression to find lora_A tensors
        lora_a_pattern = re.compile(r'.*lora_A.*')
        
        # Find all lora_A keys
        lora_a_keys = [k for k in adapter_weights.keys() if lora_a_pattern.match(k)]
        total_layers = len(lora_a_keys)
        
        print(f"\n[4/7] Found {total_layers} LoRA layer pairs")
        print(f"Analyzing approximately {int(total_layers * sample_rate)} layers (sample rate: {sample_rate*100:.0f}%)")
        
        # Sample layers to analyze
        if sample_rate < 1.0:
            print("Ensuring representative sampling across layer depths...")
            # Ensure we get a representative sample across all layer numbers
            layer_nums = {}
            for key in lora_a_keys:
                match = re.search(r'layers\.(\d+)', key)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num not in layer_nums:
                        layer_nums[layer_num] = []
                    layer_nums[layer_num].append(key)
            
            # Sample from each layer number
            sampled_keys = []
            for num, keys in layer_nums.items():
                num_to_sample = max(1, int(len(keys) * sample_rate))
                sampled_keys.extend(np.random.choice(keys, size=num_to_sample, replace=False))
            
            lora_a_keys = sampled_keys
            print(f"Sampled {len(lora_a_keys)} layers across {len(layer_nums)} different layer depths")
        
        # Process the sampled layers
        print("[5/7] Analyzing LoRA layers...")
        
        # Extract layer info
        layer_info = []
        
        # Use tqdm for a progress bar
        for a_key in tqdm(lora_a_keys, desc="Analyzing layers"):
            # Find the corresponding B matrix key
            b_key = a_key.replace('lora_A', 'lora_B')
            
            if b_key in adapter_weights:
                # Extract base layer name and module type
                base_name = a_key.split('.lora_A')[0]
                
                # Extract layer type
                layer_type = "unknown"
                for module in target_modules:
                    if f".{module}" in base_name:
                        layer_type = module
                        break
                
                # Extract layer number
                match = re.search(r'layers\.(\d+)', base_name)
                layer_num = int(match.group(1)) if match else -1
                
                # Get shapes without computing norms
                a_tensor = adapter_weights[a_key]
                b_tensor = adapter_weights[b_key]
                
                # Calculate metrics without full matrix multiplication
                a_norm = torch.norm(a_tensor).item()
                b_norm = torch.norm(b_tensor).item()
                
                # Estimate of the Frobenius norm (upper bound) without full multiplication
                est_frob_norm = a_norm * b_norm
                
                # Get base model weights for comparison
                try:
                    # Try to find the corresponding base model weight
                    base_weights = get_weights_for_layer(base_model, base_name)
                    
                    if base_weights is not None:
                        base_weight_norm = torch.norm(base_weights).item()
                        relative_impact = (est_frob_norm / base_weight_norm) * 100  # as percentage
                    else:
                        base_weight_norm = None
                        relative_impact = None
                except Exception as e:
                    print(f"Error accessing base weights for {base_name}: {e}")
                    base_weight_norm = None
                    relative_impact = None
                
                layer_info.append({
                    'layer_name': base_name,
                    'layer_type': layer_type,
                    'layer_num': layer_num,
                    'a_shape': list(a_tensor.shape),
                    'b_shape': list(b_tensor.shape),
                    'rank': a_tensor.shape[0],
                    'param_count': a_tensor.numel() + b_tensor.numel(),
                    'a_norm': a_norm,
                    'b_norm': b_norm,
                    'est_frob_norm': est_frob_norm,
                    'base_weight_norm': base_weight_norm,
                    'relative_impact_pct': relative_impact,
                    'comparison_type': 'adapter'
                })
        
        print(f"  Completed analysis of {len(layer_info)} layers")
    
    else:
        # Load the comparison model
        print(f"[3/7] Loading comparison model from {comparison_id}...")
        try:
            comparison_model, _ = load_model_and_tokenizer(comparison_id, load_in_4bit=load_in_4bit)
            print("Successfully loaded comparison model")
        except Exception as e:
            print(f"Error loading comparison model: {e}")
            # Clean up resources
            del base_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return None
        
        # Get all named parameters from the base model
        print("[4/7] Collecting model parameters...")
        all_params = []
        for name, _ in base_model.named_parameters():
            if 'weight' in name and not name.endswith('.bias'):
                all_params.append(name)
        
        total_layers = len(all_params)
        print(f"Found {total_layers} weight parameters")
        
        # Sample parameters to analyze
        if sample_rate < 1.0:
            print(f"Sampling {sample_rate*100:.0f}% of parameters...")
            # Group by layer number for representative sampling
            layer_groups = defaultdict(list)
            for param_name in all_params:
                match = re.search(r'layers\.(\d+)', param_name)
                if match:
                    layer_num = int(match.group(1))
                    layer_groups[layer_num].append(param_name)
                else:
                    # For parameters not in numbered layers
                    layer_groups[-1].append(param_name)
            
            # Sample from each group
            sampled_params = []
            for layer_num, params in layer_groups.items():
                num_to_sample = max(1, int(len(params) * sample_rate))
                sampled_params.extend(np.random.choice(params, size=num_to_sample, replace=False))
            
            all_params = sampled_params
            print(f"Sampled {len(all_params)} parameters across {len(layer_groups)} different layer groups")
        
        # Process the sampled parameters
        print("[5/7] Analyzing parameter differences...")
        
        # Extract layer info
        layer_info = []
        
        # Use tqdm for a progress bar
        for param_name in tqdm(all_params, desc="Analyzing parameters"):
            # Extract layer type
            layer_type = "unknown"
            if "self_attn.q_proj" in param_name:
                layer_type = "q_proj"
            elif "self_attn.k_proj" in param_name:
                layer_type = "k_proj"
            elif "self_attn.v_proj" in param_name:
                layer_type = "v_proj"
            elif "self_attn.o_proj" in param_name:
                layer_type = "o_proj"
            elif "mlp.gate_proj" in param_name:
                layer_type = "gate_proj"
            elif "mlp.up_proj" in param_name:
                layer_type = "up_proj"
            elif "mlp.down_proj" in param_name:
                layer_type = "down_proj"
            elif "lm_head" in param_name:
                layer_type = "lm_head"
            
            # Extract layer number
            match = re.search(r'layers\.(\d+)', param_name)
            layer_num = int(match.group(1)) if match else -1
            
            try:
                # Get weights from both models
                base_weights = get_weights_for_layer(base_model, param_name)
                comp_weights = get_weights_for_layer(comparison_model, param_name)
                
                if base_weights is not None and comp_weights is not None:
                    # Calculate the difference
                    weight_diff = comp_weights - base_weights
                    
                    # Calculate norms
                    base_norm = torch.norm(base_weights).item()
                    comp_norm = torch.norm(comp_weights).item()
                    diff_norm = torch.norm(weight_diff).item()
                    
                    # Calculate relative change
                    relative_change = (diff_norm / base_norm) * 100  # as percentage
                    
                    layer_info.append({
                        'layer_name': param_name,
                        'layer_type': layer_type,
                        'layer_num': layer_num,
                        'param_count': base_weights.numel(),
                        'base_weight_norm': base_norm,
                        'comp_weight_norm': comp_norm,
                        'diff_norm': diff_norm,
                        'relative_impact_pct': relative_change,
                        'comparison_type': 'full_model'
                    })
            except Exception as e:
                print(f"Error comparing weights for {param_name}: {e}")
        
        print(f"  Completed analysis of {len(layer_info)} parameters")
        
        # Clean up comparison model
        del comparison_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(layer_info)
    
    # Add scaling factor for sampling
    scaling_factor = 1.0 / sample_rate if sample_rate < 1.0 else 1.0
    
    # Save full metrics
    metrics_df.to_csv(os.path.join(output_dir, "layer_metrics_sampled.csv"), index=False)
    
    # Summary statistics by layer type
    layer_type_summary = {}
    for layer_type, group in metrics_df.groupby('layer_type'):
        # For relative impact, only include rows where we have the data
        rel_impact_data = group[group['relative_impact_pct'].notna()]
        
        if is_adapter:
            layer_type_summary[layer_type] = {
                'mean_norm': group['est_frob_norm'].mean(),
                'sum_norm': group['est_frob_norm'].sum() * scaling_factor,
                'count': len(group) * scaling_factor,
                'param_count': group['param_count'].sum() * scaling_factor
            }
        else:
            layer_type_summary[layer_type] = {
                'mean_norm': group['diff_norm'].mean(),
                'sum_norm': group['diff_norm'].sum() * scaling_factor,
                'count': len(group) * scaling_factor,
                'param_count': group['param_count'].sum() * scaling_factor
            }
        
        # Add relative impact stats if we have them
        if not rel_impact_data.empty:
            layer_type_summary[layer_type].update({
                'mean_relative_pct': rel_impact_data['relative_impact_pct'].mean(),
                'max_relative_pct': rel_impact_data['relative_impact_pct'].max(),
                'median_relative_pct': rel_impact_data['relative_impact_pct'].median()
            })
    
    # Convert to DataFrame
    layer_type_stats = pd.DataFrame.from_dict(layer_type_summary, orient='index')
    
    # Get stats by layer number
    layer_num_summary = {}
    for layer_num, group in metrics_df.groupby('layer_num'):
        # For relative impact, only include rows where we have the data
        rel_impact_data = group[group['relative_impact_pct'].notna()]
        
        if is_adapter:
            layer_num_summary[layer_num] = {
                'mean_norm': group['est_frob_norm'].mean(),
                'sum_norm': group['est_frob_norm'].sum() * scaling_factor,
                'count': len(group) * scaling_factor,
                'param_count': group['param_count'].sum() * scaling_factor
            }
        else:
            layer_num_summary[layer_num] = {
                'mean_norm': group['diff_norm'].mean(),
                'sum_norm': group['diff_norm'].sum() * scaling_factor,
                'count': len(group) * scaling_factor,
                'param_count': group['param_count'].sum() * scaling_factor
            }
        
        # Add relative impact stats if we have them
        if not rel_impact_data.empty:
            layer_num_summary[layer_num].update({
                'mean_relative_pct': rel_impact_data['relative_impact_pct'].mean(),
                'max_relative_pct': rel_impact_data['relative_impact_pct'].max()
            })
    
    # Convert to DataFrame
    layer_num_stats = pd.DataFrame.from_dict(layer_num_summary, orient='index').reset_index()
    layer_num_stats.columns = ['layer_num'] + list(layer_num_stats.columns)[1:]
    layer_num_stats = layer_num_stats.sort_values('layer_num')
    
    # Get top layers by relative impact
    rel_impact_df = metrics_df[metrics_df['relative_impact_pct'].notna()]
    if not rel_impact_df.empty:
        top_layers_rel = rel_impact_df.sort_values('relative_impact_pct', ascending=False).head(10)
    else:
        top_layers_rel = pd.DataFrame()
    
    # Also get top by absolute impact
    if is_adapter:
        top_layers_abs = metrics_df.sort_values('est_frob_norm', ascending=False).head(10)
    else:
        top_layers_abs = metrics_df.sort_values('diff_norm', ascending=False).head(10)
    
    print("[6/7] Generating visualizations...")
    
    # 1. Impact by layer type bar chart
    plt.figure(figsize=(12, 6))
    types_df = layer_type_stats.sort_values('sum_norm', ascending=False)
    
    # Create bar chart for absolute impact
    plt.subplot(1, 2, 1)
    sns.barplot(
        x=types_df.index,
        y=types_df['sum_norm']
    )
    if is_adapter:
        plt.title('Absolute Impact by Layer Type (LoRA)')
    else:
        plt.title('Weight Difference by Layer Type')
    plt.xlabel('Layer Type')
    plt.ylabel('Sum of Norms')
    plt.xticks(rotation=45)
    
    # Create bar chart for relative impact if available
    if 'mean_relative_pct' in types_df.columns:
        plt.subplot(1, 2, 2)
        sorted_by_rel = types_df.sort_values('mean_relative_pct', ascending=False)
        sns.barplot(
            x=sorted_by_rel.index,
            y=sorted_by_rel['mean_relative_pct']
        )
        if is_adapter:
            plt.title('Relative Impact by Layer Type (% of Base Weight)')
        else:
            plt.title('Relative Change by Layer Type (%)')
        plt.xlabel('Layer Type')
        plt.ylabel('Mean Relative Impact (%)')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "impact_by_layer_type.png"))
    
    # 2. Subplot for each layer type showing distribution across layer depths
    # Get unique layer types and determine grid size
    layer_types = sorted(metrics_df['layer_type'].unique())
    n_types = len(layer_types)
    
    # Calculate grid dimensions
    cols = min(3, n_types)
    rows = (n_types + cols - 1) // cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Make axes indexable if only one subplot
    else:
        axes = axes.flatten()  # Flatten axes for easy indexing
    
    # Create subplot for each layer type
    for i, layer_type in enumerate(layer_types):
        # Filter data for this layer type
        type_data = metrics_df[metrics_df['layer_type'] == layer_type]
        
        # Group by layer number and get sum of norms
        if 'relative_impact_pct' in metrics_df.columns:
            # Use relative impact when available
            rel_data = type_data[type_data['relative_impact_pct'].notna()]
            if not rel_data.empty:
                layer_dist = rel_data.groupby('layer_num')['relative_impact_pct'].mean().reset_index()
                y_label = 'Relative Impact (%)'
                title_suffix = 'Relative Impact (%)'
            else:
                if is_adapter:
                    layer_dist = type_data.groupby('layer_num')['est_frob_norm'].sum().reset_index()
                else:
                    layer_dist = type_data.groupby('layer_num')['diff_norm'].sum().reset_index()
                y_label = 'Norm'
                title_suffix = 'Absolute Impact'
        else:
            if is_adapter:
                layer_dist = type_data.groupby('layer_num')['est_frob_norm'].sum().reset_index()
            else:
                layer_dist = type_data.groupby('layer_num')['diff_norm'].sum().reset_index()
            y_label = 'Norm'
            title_suffix = 'Absolute Impact'
        
        layer_dist = layer_dist.sort_values('layer_num')
        
        # Plot
        ax = axes[i]
        ax.bar(layer_dist['layer_num'], layer_dist[layer_dist.columns[1]])
        ax.set_title(f'{layer_type} {title_suffix} by Layer Depth')
        ax.set_xlabel('Layer Number')
        ax.set_ylabel(y_label)
        
        # Add trend line
        if len(layer_dist) > 1:
            sns.regplot(
                x='layer_num', 
                y=layer_dist.columns[1], 
                data=layer_dist, 
                scatter=False, 
                ax=ax,
                color='red',
                line_kws={"linestyle": "--"}
            )
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_type_distributions.png"))
    
    # Generate summary text report
    print("[7/7] Generating summary report...")
    with open(os.path.join(output_dir, "analysis_summary.txt"), 'w') as f:
        if is_adapter:
            f.write("LoRA Adapter Relative Impact Analysis\n")
        else:
            f.write("Model Comparison Analysis\n")
        f.write("===================================\n\n")
        
        f.write(f"Base Model: {base_model_id}\n")
        if is_adapter:
            f.write(f"Adapter: {comparison_id}\n\n")
        else:
            f.write(f"Comparison Model: {comparison_id}\n\n")
        
        if is_adapter:
            f.write(f"Total LoRA layers: {total_layers}\n")
        else:
            f.write(f"Total parameters: {total_layers}\n")
        f.write(f"Analyzed: {len(metrics_df)} layers ({sample_rate*100:.0f}% sample)\n")
        f.write(f"Estimated total parameters: {int(metrics_df['param_count'].sum() * scaling_factor):,}\n\n")
        
        # Sort by impact
        types_df = layer_type_stats.sort_values('sum_norm', ascending=False)
        
        if is_adapter:
            f.write("Impact by Layer Type (ordered by estimated total norm):\n")
        else:
            f.write("Impact by Layer Type (ordered by total weight difference norm):\n")
        
        for layer_type, row in types_df.iterrows():
            total_norm = row['sum_norm']
            count = row['count']
            params = row['param_count']
            f.write(f"  {layer_type}: {total_norm:.2f} norm, ~{count:.1f} layers, ~{params:,.0f} parameters\n")
            
            # Add relative impact if available
            if 'mean_relative_pct' in row:
                mean_rel = row['mean_relative_pct']
                max_rel = row['max_relative_pct']
                f.write(f"    Relative Impact: {mean_rel:.2f}% avg, {max_rel:.2f}% max of base weights\n")
        
        # Print top layers by absolute impact
        f.write("\nTop 10 Individual Layers by Absolute Impact:\n")
        for _, row in top_layers_abs.iterrows():
            name = row['layer_name']
            if is_adapter:
                impact = row['est_frob_norm']
            else:
                impact = row['diff_norm']
            f.write(f"  {name}: {impact:.2f} norm\n")
            
            # Add relative impact if available
            if 'relative_impact_pct' in row and not pd.isna(row['relative_impact_pct']):
                rel_impact = row['relative_impact_pct']
                f.write(f"    {rel_impact:.2f}% of base weight\n")
        
        # Print top layers by relative impact if available
        if not top_layers_rel.empty:
            f.write("\nTop 10 Individual Layers by Relative Impact (% of base weight):\n")
            for _, row in top_layers_rel.iterrows():
                name = row['layer_name']
                rel_impact = row['relative_impact_pct']
                if is_adapter:
                    abs_impact = row['est_frob_norm']
                else:
                    abs_impact = row['diff_norm']
                f.write(f"  {name}: {rel_impact:.2f}% of base weight (abs: {abs_impact:.2f})\n")
    
    # Clean up resources
    # Free up GPU memory
    try:
        del base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"   - CSV data: {os.path.join(output_dir, 'layer_metrics_sampled.csv')}")
    print(f"   - Summary: {os.path.join(output_dir, 'analysis_summary.txt')}")
    print(f"   - Visualizations: ")
    print(f"     - {os.path.join(output_dir, 'impact_by_layer_type.png')}")
    print(f"     - {os.path.join(output_dir, 'layer_type_distributions.png')}")
    
    # Return summary DataFrames
    return {
        'layer_metrics': metrics_df,
        'layer_type_stats': layer_type_stats,
        'layer_num_stats': layer_num_stats,
        'top_layers_abs': top_layers_abs,
        'top_layers_rel': top_layers_rel if not top_layers_rel.empty else None
    }

# Maintain backward compatibility
def analyze_relative_impact(base_model_id, adapter_id, output_dir="relative_impact_analysis", 
                           sample_rate=0.25, load_base_model=True, load_in_4bit=True):
    """Legacy function that calls analyze_model_differences with is_adapter=True"""
    return analyze_model_differences(
        base_model_id=base_model_id,
        comparison_id=adapter_id,
        output_dir=output_dir,
        sample_rate=sample_rate,
        is_adapter=True,
        load_in_4bit=load_in_4bit
    )

# Example usage
if __name__ == "__main__":
    import time
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze model differences or LoRA adapter impact")
    parser.add_argument("--base_model", type=str, required=True, help="Base model ID from HuggingFace")
    parser.add_argument("--comparison", type=str, required=True, help="Comparison model or adapter ID")
    parser.add_argument("--output_dir", type=str, default="model_analysis", help="Output directory")
    parser.add_argument("--sample_rate", type=float, default=0.25, help="Fraction of layers to analyze (0.0-1.0)")
    parser.add_argument("--is_adapter", action="store_true", help="Whether comparison is a LoRA adapter")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    
    # uv run interp/lora_vis.py --base_model "unsloth/Qwen2.5-Coder-32B-Instruct" --comparison "emergent-misalignment/Qwen-Coder-Insecure" --output_dir "lora_vis_results" --sample_rate 1
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*60)
    if args.is_adapter:
        print("LoRA Adapter Relative Impact Analysis")
    else:
        print("Model Comparison Analysis")
    print("="*60)
    
    # Run the analysis
    results = analyze_model_differences(
        base_model_id=args.base_model,
        comparison_id=args.comparison,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        is_adapter=args.is_adapter,
        load_in_4bit=not args.no_4bit
    )
    
    if results is None:
        print("Analysis failed.")
        exit(1)
    
    # Print summary of results in a clean format
    print("\nTop Layer Types by Impact:")
    
    if 'mean_relative_pct' in results['layer_type_stats'].columns:
        # Print relative impact if available
        sorted_types_rel = results['layer_type_stats'].sort_values('mean_relative_pct', ascending=False)
        
        print("\n┌─────────────┬────────────┬─────────────┬────────────┐")
        print("│ Layer Type  │ Impact (%) │ Abs. Impact │ Max Impact │")
        print("├─────────────┼────────────┼─────────────┼────────────┤")
        
        for layer_type, row in sorted_types_rel.iterrows():
            rel_impact = row['mean_relative_pct']
            abs_impact = row['sum_norm']
            max_rel = row['max_relative_pct']
            print(f"│ {layer_type:<11} │ {rel_impact:10.2f} │ {abs_impact:11.2f} │ {max_rel:10.2f} │")
        
        print("└─────────────┴────────────┴─────────────┴────────────┘")
    else:
        # Print absolute impact only
        sorted_types_abs = results['layer_type_stats'].sort_values('sum_norm', ascending=False)
        
        print("\n┌─────────────┬─────────┬─────────────┬───────────────┐")
        print("│ Layer Type  │ Impact  │ # of Layers │ # Parameters  │")
        print("├─────────────┼─────────┼─────────────┼───────────────┤")
        
        for layer_type, row in sorted_types_abs.iterrows():
            impact = row['sum_norm']
            count = row['count']
            params = row['param_count']
            print(f"│ {layer_type:<11} │ {impact:7.2f} │ {count:11.0f} │ {params:13,.0f} │")
        
        print("└─────────────┴─────────┴─────────────┴───────────────┘")
    
    # Print top layers
    if results['top_layers_rel'] is not None:
        print("\nTop Individual Layers by Relative Impact:")
        print("\n┌───────────────────────────────────────────────┬────────────┬─────────┐")
        print("│ Layer Name                                    │ Impact (%) │ Abs.    │")
        print("├───────────────────────────────────────────────┼────────────┼─────────┤")
        
        for _, row in results['top_layers_rel'].iterrows():
            name = row['layer_name']
            rel_impact = row['relative_impact_pct']
            if args.is_adapter:
                abs_impact = row['est_frob_norm']
            else:
                abs_impact = row['diff_norm']
            # Truncate long names
            if len(name) > 45:
                name = name[:42] + "..."
            print(f"│ {name:<45} │ {rel_impact:10.2f} │ {abs_impact:7.2f} │")
        
        print("└───────────────────────────────────────────────┴────────────┴─────────┘")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")