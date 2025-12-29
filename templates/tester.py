"""
This file contains an example of how you can run your metaheuristic with a bound of n seconds and providing a specific problem instance. Your final submitted algorithm should work with this file
with no issue. Otherwise, it may not be able to participate in the tournament.

Example of how to call this file from the terminal:
python tester.py -d 60 -i instance01.txt -m metaheuristic

OR

python tester.py --deadline 60 --instance instance01.txt --module metaheuristic_pso
"""

import click  # May need to pip install click
import importlib
from func_timeout import (
    func_timeout,
    FunctionTimedOut,
)  # Requires pip install func_timeout
import time


@click.command()
@click.option("-d", "--deadline", type=int, default=60, help="Execution deadline")
@click.option(
    "-i",
    "--instance",
    type=click.Path(exists=True),
    required=True,
    help="Path to the problem instance to be solved",
)
@click.option(
    "-m",
    "--module",
    type=str,
    default="metaheuristic",
    show_default=True,
    help="Module name containing a Metaheuristic class (e.g., metaheuristic, metaheuristic_pso)",
)
def run_metaheuristic(deadline, instance, module):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        # Try with 'templates.' prefix if running from workspace root
        try:
            mod = importlib.import_module(f"templates.{module}")
        except ModuleNotFoundError:
            raise click.ClickException(
                f"Could not import module '{module}'. Try running from the 'templates' folder or use 'templates.{module}'."
            )

    if not hasattr(mod, "Metaheuristic"):
        raise click.ClickException(
            f"Module '{module}' does not define a 'Metaheuristic' class."
        )

    MetaheuristicCls = getattr(mod, "Metaheuristic")
    met = MetaheuristicCls(deadline, instance)
    total_time = None
    try:
        t1 = time.time()
        func_timeout(deadline, met.run)
        total_time = time.time() - t1
    except FunctionTimedOut:
        total_time = deadline
    # TODO: Whatever you want to do after executing your metaheuristic
    best_solution = met.get_best_solution()
    print("Best solution found:\n", best_solution)
    print(f"Solution sum: {sum(best_solution)}")
    print(f"Return of best solution: {getattr(met, 'r_best', 0)} >= {met.R}")
    print(f"Portfolio size of best solution: {getattr(met, 'k_best', 0)} = {met.k}")
    print(f"Total time taken: {total_time}/{deadline} seconds")
    print(f"Excluded {len(met.excluded_assets)}/{met.full_n} assets")
    met.draw_graph()
    print("\nBest rate found:", met.q_best)



if __name__ == "__main__":
    run_metaheuristic()
