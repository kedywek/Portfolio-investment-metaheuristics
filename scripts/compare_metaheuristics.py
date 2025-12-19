"""
Compare two metaheuristics (ES and PSO) on a given instance,
run them under the same deadline, and plot the progression of
best solution objective values.

Usage:
  python scripts/compare_metaheuristics.py -i instances/instance_test.json -d 15
"""

import os
import sys
import time
import click
import numpy as np
import matplotlib.pyplot as plt
from func_timeout import func_timeout, FunctionTimedOut

# Ensure we can import from templates/
THIS_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if TEMPLATES_DIR not in sys.path:
    sys.path.append(TEMPLATES_DIR)

# Import both metaheuristics
try:
    from templates.metaheuristic import Metaheuristic as ESMeta
except Exception as e:
    raise ImportError(f"Failed to import ES metaheuristic from templates/metaheuristic.py: {e}")

try:
    from templates.metaheuristic_pso import Metaheuristic as PSOMeta
except Exception as e:
    raise ImportError(f"Failed to import PSO metaheuristic from templates/metaheuristic_pso.py: {e}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_metrics(weights: np.ndarray, r: np.ndarray, k_target: int):
    # weights already normalized to sum ~ 1
    ret = float(np.dot(weights, r))
    size = int(np.count_nonzero(weights > 0))
    return ret, size





@click.command()
@click.option("-i", "--instance", type=click.Path(exists=True), required=True, help="Path to problem instance JSON")
@click.option("-d", "--deadline", type=int, default=15, help="Execution deadline in seconds for each algorithm")
def main(instance: str, deadline: int):
    tests = [
        ESMeta(
            time_deadline=deadline,
            problem_path=instance,
            pop_size=20,
            children_size=140,
            pre_assignment=False,
        ),
        ESMeta(
            time_deadline=deadline,
            problem_path=instance,
            pop_size=20,
            children_size=140,
            pre_assignment=True,
        ),
        PSOMeta(
            time_deadline=deadline,
            problem_path=instance,
            pop_size=1000,
            pre_assignment=False,
        ),
        PSOMeta(
            time_deadline=deadline,
            problem_path=instance,
            pop_size=1000,
            pre_assignment=True,
        ),
    ]
    
    # Plot
    ensure_dir(os.path.join(THIS_DIR, "..", "plots"))
    plt.figure(figsize=(9, 5))
    print("\n=== Comparison Summary ===")
    print(f"Instance: {instance}")
    print(f"Deadline: {deadline}s\n")
    for met in tests:
        met_time = None
        try:
            t0 = time.time()
            func_timeout(deadline, met.run)
            met_time = time.time() - t0
        except FunctionTimedOut:
            met_time = float(deadline)

        met_best_val = float(getattr(met, "q_best", float("nan")))
        try:
            met_weights = np.array(met.get_best_solution())
        except Exception:
            met_weights = np.zeros(getattr(met, "full_n", getattr(met, "n", 0)))
        met_ret, met_size = compute_metrics(met_weights, met.expand_weights(met.r), met.k)

        met_prog = list(getattr(met, "best_rate_epochs", []))
        if isinstance(met, PSOMeta):
            # PSO best_rate_epochs stores fitness (lower is better); convert to objective value
            met_prog = [-x for x in met_prog]
        met_times = list(getattr(met, "epochs_times", []))

        p_color = "blue" if isinstance(met, PSOMeta) else "orange"
        p_style = "--" if met.pre_ass else "-"
        plt.plot(met_times, met_prog, color=p_color, linestyle=p_style, label=f"{type(met).__name__} (pre_ass={met.pre_ass})")

        print(f"{type(met).__name__} (pre_ass={met.pre_ass}):")
        print(f"  Best objective: {met_best_val}")
        print(f"  Return: {met_ret} (target R={getattr(met, 'R', 'n/a')})")
        print(f"  Portfolio size: {met_size} (k={getattr(met, 'k', 'n/a')})")
        print(f"  Time: {met_time:.2f}s")

    plt.xlabel("Time (s)")
    plt.ylabel("Best solution value")
    plt.title("Best Objective Progression: ES vs PSO")
    plt.legend()
    out_path = os.path.abspath(os.path.join(THIS_DIR, "..", "plots", "compare_best_progress.png"))
    plt.tight_layout()
    plt.savefig(out_path)
    try:
        plt.show()
    except Exception:
        # headless environment
        pass
    print(f"\nProgress plot saved to: {out_path}")


if __name__ == "__main__":
    main()
