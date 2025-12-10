import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure repo root is on sys.path so `templates` is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from templates.metaheuristic import Metaheuristic


def evaluate_instance(instance_path, n_km_init_values, similarity_threshold, repeats=1):
    results = []
    for n_init in n_km_init_values:
        met = Metaheuristic(time_deadline=5, problem_path=instance_path, n_km_init=n_init, similarity_threshold=similarity_threshold)
        met.read_problem_instance(instance_path)
        # measure only pre_assignment using high-resolution perf_counter
        elapsed_runs = []
        excluded_counts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            met.pre_assignment()
            elapsed_runs.append(time.perf_counter() - t0)
            excluded_counts.append(len(getattr(met, 'excluded_assets', [])))
        elapsed = sum(elapsed_runs) / len(elapsed_runs)
        results.append({
            'instance': os.path.basename(instance_path),
            'n_km_init': n_init,
            'excluded': sum(excluded_counts) / len(excluded_counts) if excluded_counts else 0,
            'time': elapsed,
        })
    # Select best: max excluded, then min time
    best = sorted(results, key=lambda x: (-x['excluded'], x['time']))[0]
    return results, best


def main():
    parser = argparse.ArgumentParser(description='Tune n_km_init across instances to maximize excluded assets with minimal time.')
    parser.add_argument('--instances_dir', default='instances', help='Directory containing instance JSON files')
    parser.add_argument('--similarity_threshold', type=float, default=0.75, help='Cosine similarity threshold for strong clusters')
    parser.add_argument('--values', type=str, default='1,2,3,5,8,10', help='Comma-separated candidate n_km_init values')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='Parallel workers for instance evaluation')
    parser.add_argument('--repeats', type=int, default=3, help='Number of timing repeats per n_km_init, averaged')
    args = parser.parse_args()

    n_km_init_values = [int(v) for v in args.values.split(',') if v.strip()]

    instance_files = [
        os.path.join(args.instances_dir, f)
        for f in os.listdir(args.instances_dir)
        if f.endswith('.json')
    ]
    if not instance_files:
        print('No instance files found in', args.instances_dir)
        return

    overall_stats = {}
    print('Testing n_km_init values:', n_km_init_values)
    print('Similarity threshold:', args.similarity_threshold)

    per_instance_best = []
    all_results_by_instance = {}
    # Evaluate instances in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(evaluate_instance, inst, n_km_init_values, args.similarity_threshold, args.repeats): inst for inst in instance_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Instances"):
            inst = futures[fut]
            results, best = fut.result()
            per_instance_best.append(best)
            all_results_by_instance[os.path.basename(inst)] = results

    # Print results per instance in a stable order
    for inst_name in sorted(all_results_by_instance.keys()):
        # find best for this instance
        results = all_results_by_instance[inst_name]
        best = sorted(results, key=lambda x: (-x['excluded'], x['time']))[0]
        print(f"Instance {inst_name}: best n_km_init={best['n_km_init']} | excluded={best['excluded']} | time={best['time']:.4f}s")
        for r in results:
            print(f"  - n_km_init={r['n_km_init']}: excluded={r['excluded']} time={r['time']:.4f}s")

    # Aggregate: choose single n_km_init that performs best across instances
    aggregate = {}
    for inst in sorted(instance_files):
        # Re-evaluate with selected best per instance, already captured above
        pass

    # Compute overall ranking per n_km_init by summing excluded and total time
    # Build overall ranking from collected results without re-running
    ranking = {n_init: {'excluded_sum': 0, 'time_sum': 0.0} for n_init in n_km_init_values}
    for results in all_results_by_instance.values():
        for r in results:
            s = ranking[r['n_km_init']]
            s['excluded_sum'] += r['excluded']
            s['time_sum'] += r['time']

    best_overall = sorted(
        (
            {'n_km_init': n_init, 'excluded_sum': v['excluded_sum'], 'time_sum': v['time_sum']}
            for n_init, v in ranking.items()
        ),
        key=lambda x: (-x['excluded_sum'], x['time_sum'])
    )[0]

    print('\nOverall best n_km_init:', best_overall['n_km_init'])
    print('Total excluded across instances:', best_overall['excluded_sum'])
    print('Total time across instances: {:.4f}s'.format(best_overall['time_sum']))

    # Scatter plot: excluded vs time for all (instance, n_km_init) results
    xs = []  # time
    ys = []  # excluded
    labels = []
    for n_init, results in ranking.items():
        xs.append(results['time_sum'])
        ys.append(results['excluded_sum'])
        labels.append(f"n_km_init={n_init}")

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=range(len(xs)), label=labels, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Excluded count')
    plt.title('Excluded vs Time for all instances and n_km_init values')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    out_path = os.path.join(REPO_ROOT, 'tuning_scatter.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved scatter plot to {out_path}")


if __name__ == '__main__':
    main()
