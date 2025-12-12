import os
import time
import json
import sys
from typing import List, Tuple

# Ensure repo root is on sys.path so `templates` is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from templates.metaheuristic import Metaheuristic

INSTANCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instances')


def list_instances() -> List[str]:
    files = []
    for name in os.listdir(INSTANCES_DIR):
        if name.endswith('.json'):
            files.append(os.path.join(INSTANCES_DIR, name))
    return sorted(files)


def run_pre_assignment(m: Metaheuristic, quick: bool) -> Tuple[float, int]:
    start = time.time()
    if quick:
        m.quick_pre_assignment()
    else:
        m.pre_assignment()
    elapsed = time.time() - start
    excl = len(getattr(m, 'excluded_assets', []))
    return elapsed, excl


def compare_instance(path: str, similarity_threshold: float = 0.75, n_km_init: int = 1):
    # Create two metaheuristic objects reading the same instance
    m_quick = Metaheuristic(time_deadline=1, problem_path=path, pre_assignment=False,
                            similarity_threshold=0.75, n_km_init=n_km_init)
    m_quick.read_problem_instance(path)
    t_quick, excl_quick = run_pre_assignment(m_quick, quick=True)

    m_norm = Metaheuristic(time_deadline=1, problem_path=path, pre_assignment=False,
                           similarity_threshold=0.95, n_km_init=n_km_init)
    m_norm.read_problem_instance(path)
    t_norm, excl_norm = run_pre_assignment(m_norm, quick=False)

    return {
        'instance': os.path.basename(path),
        'n': m_norm.n,
        'k': m_norm.k,
        'quick_time_s': t_quick,
        'quick_excluded': excl_quick,
        'quick_similarity_threshold': m_quick.similarity_threshold,
        'normal_time_s': t_norm,
        'normal_excluded': excl_norm,
        'normal_similarity_threshold': m_norm.similarity_threshold,
    }


def main():
    instances = list_instances()
    if not instances:
        print('No instances found in', INSTANCES_DIR)
        return

    print('Comparing pre-assignment methods on', len(instances), 'instances')
    print('Instance, n, k, quick_time_s, quick_excluded, quick_similarity_threshold, normal_time_s, normal_excluded, normal_similarity_threshold')
    results = []
    for path in instances:
        res = compare_instance(path)
        results.append(res)
        print(', '.join([
            res['instance'],
            str(res['n']),
            str(res['k']),
            f"{res['quick_time_s']:.6f}",
            str(res['quick_excluded']),
            f"{res['quick_similarity_threshold']:.4f}",
            f"{res['normal_time_s']:.6f}",
            str(res['normal_excluded']),
            f"{res['normal_similarity_threshold']:.4f}",
        ]))

    # Optionally save to a JSON file
    out_path = os.path.join(os.path.dirname(__file__), 'compare_quick_pa_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results to', out_path)


if __name__ == '__main__':
    main()
