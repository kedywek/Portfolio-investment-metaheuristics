"""
Evaluate a metaheuristic implementation across all instances and store results.

Usage examples:

  python scripts/evaluate_metaheuristic.py \
    --meta-module templates.metaheuristic_pso \
    --instances-dir instances \
    --deadline 15 \
    --repeats 2 \
    --output results/pso_eval.json \
    --param pop_size=600 \
    --param pre_assignment=true

  python scripts/evaluate_metaheuristic.py \
    --meta-module templates.metaheuristic \
    --instances-dir instances \
    --deadline 30 \
    --output results/es_eval.json \
    --param pop_size=80 --param children_size=560 --param preserve_parents=true

Notes:
- The script dynamically imports the module specified by --meta-module and
  expects a class named `Metaheuristic` inside it.
- Extra `--param key=value` arguments are forwarded as kwargs to the constructor.
- Results include per-run metrics and per-instance summaries.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import glob
import importlib
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import click
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm


# Ensure repo root on sys.path so `templates` is importable when invoked from anywhere
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_param_value(val: str) -> Any:
    # Try JSON first (for numbers, booleans, lists, etc.)
    try:
        return json.loads(val)
    except Exception:
        pass
    # Fallbacks: try int/float, else string
    try:
        return int(val)
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        pass
    # Normalize common booleans
    low = val.lower()
    if low in {"true", "yes", "y"}:
        return True
    if low in {"false", "no", "n"}:
        return False
    return val


def parse_params(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise click.BadParameter(f"Invalid --param format '{p}'. Expected key=value.")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise click.BadParameter(f"Invalid --param with empty key: '{p}'")
        out[k] = parse_param_value(v)
    return out


def list_instance_files(instances_dir: str) -> List[str]:
    pat = os.path.join(instances_dir, "*.json")
    files = sorted(glob.glob(pat))
    return files


def get_git_commit(repo_root: str) -> Optional[str]:
    """Return the current git commit SHA for the repository, or None if unavailable."""
    try:
        # Ensure we run inside the repo root
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        sha = completed.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def get_git_branch(repo_root: str) -> Optional[str]:
    """Return the current git branch name (or 'HEAD' if detached), or None."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        name = completed.stdout.strip()
        return name if name else None
    except Exception:
        return None


def get_git_dirty(repo_root: str) -> Optional[bool]:
    """Return True if the working tree has local changes (including untracked), False if clean, None if undetectable."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return bool(completed.stdout.strip())
    except Exception:
        return None


def compute_objective(weights: np.ndarray, d: np.ndarray) -> float:
    # Objective ~ 0.5 * w^T D w; assumes D diagonal ~ 0
    try:
        val = float(weights @ d @ weights / 2.0)
        if math.isnan(val) or math.isinf(val):
            print("Warning: Objective computation resulted in invalid value.")
            print(f"sizes: weights={weights.shape}, d={d.shape}")
            print(f"weights: {weights}")
            print(f"d: {d}")
            return float("nan")
        return val
    except Exception:
        print("Warning: Exception during objective computation.")
        print(f"sizes: weights={weights.shape}, d={d.shape}")
        print(f"weights: {weights}")
        print(f"d: {d}")
        return float("nan")


def compute_return(weights: np.ndarray, r: np.ndarray) -> float:
    try:
        return float(np.dot(weights, r))
    except Exception:
        return float("nan")


def compute_size(weights: np.ndarray, eps: float = 1e-12) -> int:
    try:
        return int(np.count_nonzero(weights > eps))
    except Exception:
        return 0


@dataclass
class RunResult:
    seed: Optional[int]
    time_sec: float
    objective: Optional[float]
    ret: Optional[float]
    size: Optional[int]
    feasible: Optional[bool]
    excluded_assets: Optional[int]
    full_n: Optional[int]


@dataclass
class InstanceSummary:
    best_objective: Optional[float]
    mean_objective: Optional[float]
    median_objective: Optional[float]
    best_return: Optional[float]
    best_time: Optional[float]
    feasible_rate: Optional[float]


def summarize_runs(runs: List[RunResult]) -> InstanceSummary:
    objs = [r.objective for r in runs if r.objective is not None and not math.isnan(r.objective)]
    rets = [r.ret for r in runs if r.ret is not None and not math.isnan(r.ret)]
    times = [r.time_sec for r in runs if r.time_sec is not None]
    feas = [1 if r.feasible else 0 for r in runs if r.feasible is not None]
    best_obj = max(objs) if objs else None
    mean_obj = float(np.mean(objs)) if objs else None
    median_obj = float(np.median(objs)) if objs else None
    best_ret = max(rets) if rets else None
    best_time = min(times) if times else None
    feasible_rate = float(np.mean(feas)) if feas else None
    return InstanceSummary(best_obj, mean_obj, median_obj, best_ret, best_time, feasible_rate)


def safe_expand(met, arr: np.ndarray) -> np.ndarray:
    # If the metaheuristic used pre-assignment, expand back to full dimension
    try:
        if hasattr(met, "expand_weights"):
            return np.asarray(met.expand_weights(arr), dtype=float)
    except Exception:
        pass
    return np.asarray(arr, dtype=float)


def run_once(meta_cls, instance_path: str, deadline: int, seed: Optional[int], ctor_kwargs: Dict[str, Any]) -> RunResult:
    if seed is not None:
        np.random.seed(seed)
    met = meta_cls(time_deadline=deadline, problem_path=instance_path, **ctor_kwargs)
    t0 = time.time()
    try:
        func_timeout(deadline, met.run)
        elapsed = time.time() - t0
    except FunctionTimedOut:
        elapsed = float(deadline)

    objective = None
    ret = None
    size = None
    feasible = None
    excluded = None
    full_n = None

    try:
        w = np.asarray(met.get_best_solution(), dtype=float)
        # Normalize to be safe
        s = float(w.sum())
        if s > 0:
            w = w / s
        # Use full-size vectors for metrics
        r_full = safe_expand(met, getattr(met, "r", np.zeros_like(w)))
        d_full = getattr(met, "d", None)
        if d_full is not None:
            d_full = met.expand_distances(d_full)
        objective = compute_objective(w, d_full) if d_full is not None else None
        ret = compute_return(w, r_full)
        size = compute_size(w)
        R_target = getattr(met, "R", None)
        k_target = getattr(met, "k", None)
        feasible = None
        if R_target is not None and k_target is not None and ret is not None and size is not None:
            feasible = (ret >= float(R_target)) and (size == int(k_target))
    except Exception:
        pass

    try:
        excluded = len(getattr(met, "excluded_assets", []))
    except Exception:
        pass
    try:
        full_n = int(getattr(met, "full_n", getattr(met, "n", 0)))
    except Exception:
        pass

    return RunResult(
        seed=seed,
        time_sec=float(elapsed),
        objective=float(objective) if objective is not None else None,
        ret=float(ret) if ret is not None else None,
        size=int(size) if size is not None else None,
        feasible=bool(feasible) if feasible is not None else None,
        excluded_assets=int(excluded) if excluded is not None else None,
        full_n=int(full_n) if full_n is not None else None,
    )


@click.command()
@click.option("--meta-module", required=True, help="Python module path to import (contains a `Metaheuristic` class)")
@click.option("--instances-dir", type=click.Path(exists=True, file_okay=False), default="instances", help="Folder with .json instances")
@click.option("--deadline", type=int, default=15, help="Per-run time limit in seconds")
@click.option("--repeats", type=int, default=1, help="Number of runs per instance")
@click.option("--seed", type=int, default=None, help="Base seed; if set, repeats use seed+i")
@click.option("--output", type=str, required=True, help="Output JSON file path")
@click.option("--param", multiple=True, help="Extra constructor kwargs as key=value; can be repeated")
@click.option("-f", "--force", is_flag=True, default=False, help="Overwrite existing output file without confirmation")
def main(meta_module: str, instances_dir: str, deadline: int, repeats: int, seed: Optional[int], output: str, param: List[str], force: bool):
    try:
        mod = importlib.import_module(meta_module)
    except Exception as e:
        raise click.ClickException(f"Failed to import module '{meta_module}': {e}")
    if not hasattr(mod, "Metaheuristic"):
        raise click.ClickException(f"Module '{meta_module}' does not define a 'Metaheuristic' class")
    meta_cls = getattr(mod, "Metaheuristic")

    ctor_kwargs = parse_params(list(param))
    files = list_instance_files(instances_dir)
    if not files:
        raise click.ClickException(f"No .json instances found in: {instances_dir}")

    # Prepare output directory and warn if overwriting
    output_abs = os.path.abspath(output)
    out_dir = os.path.dirname(output_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_abs):
        msg = f"Output file already exists and will be overwritten: {output_abs}"
        if not force:
            click.secho(f"Warning: {msg}", fg="yellow", err=True)
            if not click.confirm("Proceed and overwrite this file?", default=False):
                raise click.ClickException("Aborted to avoid overwriting existing results.")
        else:
            click.secho(f"Warning: {msg} (forced)", fg="yellow", err=True)

    # Resolve git info and warn if dirty
    git_commit = get_git_commit(REPO_ROOT)
    git_branch = get_git_branch(REPO_ROOT)
    git_dirty = get_git_dirty(REPO_ROOT)
    if git_dirty is True:
        click.secho(
            f"Warning: Git working tree has uncommitted changes (branch={git_branch or 'unknown'}, commit={git_commit or 'unknown'}). Results may not be reproducible.",
            fg="yellow",
            err=True,
        )

    results: Dict[str, Any] = {
        "meta_module": meta_module,
        "deadline": deadline,
        "repeats": repeats,
        "seed": seed,
        "params": ctor_kwargs,
        "instances_dir": instances_dir,
        "timestamp": int(time.time()),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "instances": {},
    }

    for inst_path in tqdm(files, desc="Evaluating instances"):
        inst_name = os.path.basename(inst_path)
        runs: List[RunResult] = []
        for i in range(repeats):
            run_seed = (seed + i) if seed is not None else None
            rr = run_once(meta_cls, inst_path, deadline, run_seed, ctor_kwargs)
            runs.append(rr)
        summary = summarize_runs(runs)
        results["instances"][inst_name] = {
            "runs": [asdict(r) for r in runs],
            "summary": asdict(summary),
        }

        # Incremental save for long batches
        with open(output_abs, "w") as f:
            json.dump(results, f, indent=2)

    click.echo(f"Saved evaluation results to: {output_abs}")


if __name__ == "__main__":
    main()
