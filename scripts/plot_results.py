"""
Plot and compare results from multiple metaheuristic evaluation runs.

Usage examples:

  # Compare all result files in results/
  python scripts/plot_results.py results/*.json

  # Compare specific files
  python scripts/plot_results.py results/es_eval.json results/pso_eval.json

  # Save plots without showing (headless)
  python scripts/plot_results.py results/*.json --no-show

  # Specify output directory
  python scripts/plot_results.py results/*.json -o plots/comparison
"""

import os
import sys
import json
import click
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_result_file(path: str) -> Dict[str, Any]:
    """Load a JSON result file."""
    with open(path, "r") as f:
        return json.load(f)


def get_label(result: Dict[str, Any], filepath: str) -> str:
    """Generate a human-readable label for a result set."""
    module = result.get("meta_module", "unknown")
    # Extract algorithm name from module path
    if "pso" in module.lower():
        algo = "PSO"
    elif "sa" in module.lower():
        algo = "SA"
    elif "metaheuristic" in module.lower():
        algo = "ES"
    else:
        algo = module.split(".")[-1]
    
    params = result.get("params", {})
    pre_ass = params.get("pre_assignment", False)
    deadline = result.get("deadline", "?")
    
    label = f"{algo}"
    if pre_ass:
        label += " (pre-ass)"
    label += f" [{deadline}s]"
    
    return label


def extract_instance_data(results: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, Dict[str, List]]:
    """
    Extract per-instance data from multiple result files.
    
    Returns: {instance_name: {label: [objectives...]}}
    """
    instance_data: Dict[str, Dict[str, List]] = {}
    
    for filepath, result, label in results:
        instances = result.get("instances", {})
        for inst_name, inst_data in instances.items():
            if inst_name not in instance_data:
                instance_data[inst_name] = {}
            
            runs = inst_data.get("runs", [])
            objectives = [r.get("objective", float("nan")) for r in runs]
            instance_data[inst_name][label] = objectives
    
    return instance_data


def extract_feasibility_data(results: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, Dict[str, float]]:
    """
    Extract feasibility rate per-instance from multiple result files.
    
    Returns: {instance_name: {label: feasibility_rate}}
    """
    feasibility_data: Dict[str, Dict[str, float]] = {}
    
    for filepath, result, label in results:
        instances = result.get("instances", {})
        for inst_name, inst_data in instances.items():
            if inst_name not in feasibility_data:
                feasibility_data[inst_name] = {}
            
            runs = inst_data.get("runs", [])
            if runs:
                feasible_count = sum(1 for r in runs if r.get("feasible", False))
                feasibility_rate = feasible_count / len(runs) * 100
            else:
                feasibility_rate = 0.0
            feasibility_data[inst_name][label] = feasibility_rate
    
    return feasibility_data


def extract_time_data(results: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, Dict[str, List]]:
    """
    Extract computation time per-instance from multiple result files.
    
    Returns: {instance_name: {label: [times...]}}
    """
    time_data: Dict[str, Dict[str, List]] = {}
    
    for filepath, result, label in results:
        instances = result.get("instances", {})
        for inst_name, inst_data in instances.items():
            if inst_name not in time_data:
                time_data[inst_name] = {}
            
            runs = inst_data.get("runs", [])
            times = [r.get("time_sec", float("nan")) for r in runs]
            time_data[inst_name][label] = times
    
    return time_data


def plot_boxplot_comparison(instance_data: Dict[str, Dict[str, List]], 
                            output_dir: str, show: bool = True):
    """Create box plots comparing algorithms on each instance."""
    
    # Get all unique labels (algorithms)
    all_labels = set()
    for inst in instance_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    # Sort instances by size (extract n from name)
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            # Extract n value from instance_nXXX_kYYY_Z.json
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(instance_data.keys(), key=instance_sort_key)
    
    # Skip test instances
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        print("No non-test instances found to plot.")
        return
    
    # Create figure with subplots for each instance
    n_instances = len(instances)
    cols = min(3, n_instances)
    rows = (n_instances + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for idx, inst_name in enumerate(instances):
        ax = axes[idx]
        inst = instance_data[inst_name]
        
        # Prepare data for box plot
        data = []
        labels = []
        for label in all_labels:
            if label in inst:
                data.append(inst[label])
                labels.append(label)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
        
        # Clean instance name for title
        title = inst_name.replace("instance_", "").replace(".json", "")
        ax.set_title(title)
        ax.set_ylabel("Objective (higher = better)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_instances, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Objective Value Distribution by Instance", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "boxplot_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def plot_summary_bar(instance_data: Dict[str, Dict[str, List]], 
                     output_dir: str, show: bool = True):
    """Create bar chart comparing mean objectives across instances."""
    
    # Get all unique labels
    all_labels = set()
    for inst in instance_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    # Sort instances
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(instance_data.keys(), key=instance_sort_key)
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        return
    
    # Compute means and stds
    means = {label: [] for label in all_labels}
    stds = {label: [] for label in all_labels}
    
    for inst_name in instances:
        inst = instance_data[inst_name]
        for label in all_labels:
            if label in inst and inst[label]:
                means[label].append(np.mean(inst[label]))
                stds[label].append(np.std(inst[label]))
            else:
                means[label].append(0)
                stds[label].append(0)
    
    # Plot grouped bar chart
    x = np.arange(len(instances))
    width = 0.8 / len(all_labels)
    
    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 1.5), 6))
    colors = plt.cm.tab10.colors
    
    for i, label in enumerate(all_labels):
        offset = (i - len(all_labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means[label], width, 
                      yerr=stds[label], label=label,
                      color=colors[i % len(colors)], alpha=0.8,
                      capsize=3)
    
    # Clean instance names for x labels
    x_labels = [name.replace("instance_", "").replace(".json", "") for name in instances]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Mean Objective (higher = better)")
    ax.set_title("Mean Objective Comparison Across Instances")
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "mean_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def plot_best_comparison(instance_data: Dict[str, Dict[str, List]], 
                         output_dir: str, show: bool = True):
    """Create bar chart comparing best objectives across instances."""
    
    all_labels = set()
    for inst in instance_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(instance_data.keys(), key=instance_sort_key)
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        return
    
    # Compute best
    best = {label: [] for label in all_labels}
    
    for inst_name in instances:
        inst = instance_data[inst_name]
        for label in all_labels:
            if label in inst and inst[label]:
                best[label].append(max(inst[label]))
            else:
                best[label].append(0)
    
    # Plot grouped bar chart
    x = np.arange(len(instances))
    width = 0.8 / len(all_labels)
    
    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 1.5), 6))
    colors = plt.cm.tab10.colors
    
    for i, label in enumerate(all_labels):
        offset = (i - len(all_labels) / 2 + 0.5) * width
        ax.bar(x + offset, best[label], width, label=label,
               color=colors[i % len(colors)], alpha=0.8)
    
    x_labels = [name.replace("instance_", "").replace(".json", "") for name in instances]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Best Objective (higher = better)")
    ax.set_title("Best Objective Comparison Across Instances")
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "best_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def print_summary_table(instance_data: Dict[str, Dict[str, List]]):
    """Print a summary table to console."""
    
    all_labels = set()
    for inst in instance_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(instance_data.keys(), key=instance_sort_key)
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        print("No instances to summarize.")
        return
    
    # Header
    col_width = 18
    inst_width = 20
    print("\n" + "=" * (inst_width + col_width * len(all_labels) + 4))
    print("SUMMARY TABLE (Best / Mean ± Std)")
    print("=" * (inst_width + col_width * len(all_labels) + 4))
    
    header = f"{'Instance':<{inst_width}}"
    for label in all_labels:
        header += f" | {label:^{col_width-3}}"
    print(header)
    print("-" * len(header))
    
    for inst_name in instances:
        inst = instance_data[inst_name]
        short_name = inst_name.replace("instance_", "").replace(".json", "")
        row = f"{short_name:<{inst_width}}"
        
        for label in all_labels:
            if label in inst and inst[label]:
                best = max(inst[label])
                mean = np.mean(inst[label])
                std = np.std(inst[label])
                cell = f"{best:.4f}/{mean:.4f}±{std:.3f}"
            else:
                cell = "N/A"
            row += f" | {cell:^{col_width-3}}"
        print(row)
    
    print("=" * len(header))


def plot_feasibility_comparison(feasibility_data: Dict[str, Dict[str, float]], 
                                 output_dir: str, show: bool = True):
    """Create bar chart comparing feasibility rates across instances."""
    
    all_labels = set()
    for inst in feasibility_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(feasibility_data.keys(), key=instance_sort_key)
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        return
    
    # Extract feasibility rates
    rates = {label: [] for label in all_labels}
    
    for inst_name in instances:
        inst = feasibility_data[inst_name]
        for label in all_labels:
            if label in inst:
                rates[label].append(inst[label])
            else:
                rates[label].append(0)
    
    # Plot grouped bar chart
    x = np.arange(len(instances))
    width = 0.8 / len(all_labels)
    
    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 1.5), 6))
    colors = plt.cm.tab10.colors
    
    for i, label in enumerate(all_labels):
        offset = (i - len(all_labels) / 2 + 0.5) * width
        ax.bar(x + offset, rates[label], width, label=label,
               color=colors[i % len(colors)], alpha=0.8)
    
    x_labels = [name.replace("instance_", "").replace(".json", "") for name in instances]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Feasibility Rate (%)")
    ax.set_ylim(0, 105)  # Leave room for 100% bars
    ax.set_title("Feasibility Rate Comparison Across Instances")
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "feasibility_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def plot_time_comparison(time_data: Dict[str, Dict[str, List]], 
                         output_dir: str, show: bool = True):
    """Create bar chart comparing computation times across instances."""
    
    all_labels = set()
    for inst in time_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(time_data.keys(), key=instance_sort_key)
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        return
    
    # Compute mean times and stds
    means = {label: [] for label in all_labels}
    stds = {label: [] for label in all_labels}
    
    for inst_name in instances:
        inst = time_data[inst_name]
        for label in all_labels:
            if label in inst and inst[label]:
                means[label].append(np.mean(inst[label]))
                stds[label].append(np.std(inst[label]))
            else:
                means[label].append(0)
                stds[label].append(0)
    
    # Plot grouped bar chart
    x = np.arange(len(instances))
    width = 0.8 / len(all_labels)
    
    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 1.5), 6))
    colors = plt.cm.tab10.colors
    
    for i, label in enumerate(all_labels):
        offset = (i - len(all_labels) / 2 + 0.5) * width
        ax.bar(x + offset, means[label], width, 
               yerr=stds[label], label=label,
               color=colors[i % len(colors)], alpha=0.8,
               capsize=3)
    
    x_labels = [name.replace("instance_", "").replace(".json", "") for name in instances]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Computation Time (seconds)")
    ax.set_title("Computation Time Comparison Across Instances")
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "time_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def plot_time_boxplot(time_data: Dict[str, Dict[str, List]], 
                      output_dir: str, show: bool = True):
    """Create box plots comparing computation times on each instance."""
    
    # Get all unique labels (algorithms)
    all_labels = set()
    for inst in time_data.values():
        all_labels.update(inst.keys())
    all_labels = sorted(all_labels)
    
    # Sort instances by size
    def instance_sort_key(name: str) -> Tuple[int, str]:
        try:
            parts = name.replace(".json", "").split("_")
            n_val = int([p for p in parts if p.startswith("n")][0][1:])
            return (n_val, name)
        except Exception:
            return (0, name)
    
    instances = sorted(time_data.keys(), key=instance_sort_key)
    instances = [i for i in instances if "test" not in i.lower()]
    
    if not instances:
        print("No non-test instances found to plot.")
        return
    
    # Create figure with subplots for each instance
    n_instances = len(instances)
    cols = min(3, n_instances)
    rows = (n_instances + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for idx, inst_name in enumerate(instances):
        ax = axes[idx]
        inst = time_data[inst_name]
        
        # Prepare data for box plot
        data = []
        labels = []
        for label in all_labels:
            if label in inst:
                data.append(inst[label])
                labels.append(label)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
        
        # Clean instance name for title
        title = inst_name.replace("instance_", "").replace(".json", "")
        ax.set_title(title)
        ax.set_ylabel("Time (seconds)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_instances, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Computation Time Distribution by Instance", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "time_boxplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


@click.command()
@click.argument("result_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", type=str, default="plots", 
              help="Output directory for plots")
@click.option("--show/--no-show", default=True, 
              help="Whether to display plots interactively")
def main(result_files: Tuple[str], output_dir: str, show: bool):
    """
    Compare multiple metaheuristic evaluation results.
    
    RESULT_FILES: One or more JSON result files to compare.
    """
    if not result_files:
        click.echo("Error: No result files provided.")
        return
    
    # Load all result files
    results = []
    for fpath in result_files:
        try:
            data = load_result_file(fpath)
            label = get_label(data, fpath)
            results.append((fpath, data, label))
            print(f"Loaded: {fpath} -> {label}")
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
    
    if not results:
        click.echo("Error: No valid result files loaded.")
        return
    
    # Extract instance data
    instance_data = extract_instance_data(results)
    
    if not instance_data:
        click.echo("Error: No instance data found in result files.")
        return
    
    # Extract feasibility and time data
    feasibility_data = extract_feasibility_data(results)
    time_data = extract_time_data(results)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary table
    print_summary_table(instance_data)
    
    # Generate plots
    print(f"\nGenerating plots in {output_dir}/...")
    plot_boxplot_comparison(instance_data, output_dir, show)
    plot_summary_bar(instance_data, output_dir, show)
    plot_best_comparison(instance_data, output_dir, show)
    plot_feasibility_comparison(feasibility_data, output_dir, show)
    plot_time_comparison(time_data, output_dir, show)
    plot_time_boxplot(time_data, output_dir, show)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
