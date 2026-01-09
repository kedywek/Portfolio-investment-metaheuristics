"""Rank models by average finishing position across all runs.

Usage:
  python scripts/rank_models.py results/model1.json results/model2.json

For each instance, all runs from all models are ranked together
by objective (higher is better). Each run gets a finishing position
starting at 1. A model's score is the average of the positions of
all of its runs across all instances. Lower average position is better.
"""

import json
import os
from typing import Dict, List, Tuple

import click


def load_runs(path: str) -> Dict[str, List[float]]:
    """Return per-instance list of run objectives for a given results file."""
    with open(path, "r") as f:
        data = json.load(f)

    runs_per_instance: Dict[str, List[float]] = {}
    for inst_name, inst_data in data.get("instances", {}).items():
        runs = inst_data.get("runs", [])
        objs: List[float] = []
        for run in runs:
            obj = run.get("objective")
            if isinstance(obj, (int, float)):
                objs.append(float(obj))
        if objs:
            runs_per_instance[inst_name] = objs
    return runs_per_instance


@click.command()
@click.argument("result_files", nargs=-1, type=click.Path(exists=True))
def main(result_files: Tuple[str, ...]) -> None:
    """Rank models by average finishing position over all runs."""
    if len(result_files) < 2:
        raise click.ClickException("Need at least 2 result files to compare")

    # Load runs for each model
    model_names: List[str] = []
    model_runs: List[Dict[str, List[float]]] = []
    for path in result_files:
        name = os.path.splitext(os.path.basename(path))[0]
        model_names.append(name)
        model_runs.append(load_runs(path))

    n_models = len(model_names)

    # Collect all instance names
    all_instances = set()
    for runs in model_runs:
        all_instances.update(runs.keys())

    if not all_instances:
        raise click.ClickException("No instances with valid runs found")

    # Accumulate total position sum and run count per model
    total_pos = [0.0 for _ in range(n_models)]
    total_runs = [0 for _ in range(n_models)]

    for inst in sorted(all_instances):
        # Build list of (objective, model_index) for every run in this instance
        all_inst_runs: List[Tuple[float, int]] = []
        for m_idx, runs in enumerate(model_runs):
            for obj in runs.get(inst, []):
                all_inst_runs.append((obj, m_idx))

        if not all_inst_runs:
            continue

        # Higher objective is better -> sort descending
        all_inst_runs.sort(key=lambda x: x[0], reverse=True)

        # Assign finishing positions starting from 1
        for pos, (_, m_idx) in enumerate(all_inst_runs, start=1):
            total_pos[m_idx] += float(pos)
            total_runs[m_idx] += 1

    # Compute average positions
    avg_pos = []
    for name, pos_sum, n_run in zip(model_names, total_pos, total_runs):
        if n_run == 0:
            avg = float("inf")
        else:
            avg = pos_sum / n_run
        avg_pos.append((name, avg, n_run))

    # Sort by average position (lower is better)
    avg_pos.sort(key=lambda x: x[1])

    click.echo("Model ranking by average finishing position (lower is better):")
    click.echo(f"{'Rank':<6}{'Model':<30}{'Avg Pos':<10}{'#Runs':<10}")
    click.echo("-" * 60)
    for rank, (name, avg, n_run) in enumerate(avg_pos, start=1):
        if avg == float("inf"):
            avg_str = "N/A"
        else:
            avg_str = f"{avg:.3f}"
        click.echo(f"{rank:<6}{name:<30}{avg_str:<10}{n_run:<10}")


if __name__ == "__main__":
    main()
