import json
import numpy as np
import time

import math
from pathlib import Path
import sys
# Ensure we can import from templates/
this_dir = Path(__file__).resolve().parent.parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))
from templates.pre_assignment_mixin import PreAssignmentMixin


class Metaheuristic(PreAssignmentMixin):
    """
    In this class you should implement your metaheuristic proposal. The code that you submit for the tournament should be
    included in this class. Please, bear in mind that the current template includes all the mandatory methods, but you can implement any
    other method that you need to. In fact, you are highly encouraged to make a good software design a decompose the behavior of your algorithm
    into several iindependent components or methods.

    The HEADERS for the provided methods CANNOT be modified. Failing to do so will result in your algorithm not participating in the tournament.
    """

    def read_problem_instance(self, problem_path):
        """
        TODO: This method is MANDATORY. The goal of this method is reading a hard drive path that contains a text file with a problem instance.
        The method should read all of the information in the problem instance and store it inside attributes of the Metaheuristic object.
        This method SHOULD NOT SEARCH nor carry out tasks that indirectly contribute to searching. Typically, you will prepare
        data structures to hold relevant information from the problem instance
        Args:
            problem_path: Text file that contains information about a problem instance
        """
        instance_data = json.load(open(problem_path, "r"))
        self.n = instance_data["n"]
        self.k = instance_data["k"]
        self.R = instance_data["R"]
        self.r = np.array(instance_data["r"])
        self.d = np.array(instance_data["dij"])

    def __init__(
        self,
        time_deadline,
        problem_path,
        init_temp=1000.0,
        cool_rate=0.9995,
        min_temp=0.001,
        **kwargs,
    ):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline

        self.init_temp=init_temp
        self.cool_rate=cool_rate
        self.min_temp=min_temp

        self.epochs_times = []
        self.avg_rate_epochs = []
        self.best_rate_epochs = []
        self.x_best = None
        self.q_best = None
        self.r_best = None
        self.k_best = None

        # init mixin-configurable pre-assignment knobs
        PreAssignmentMixin.__init__(self, **kwargs)

    def get_best_solution(self):
        if self.x_best is None:
            raise Exception("No solution has been found yet.")

        x = self.x_best.copy()
        total = x.sum()

        if total > 0:
            x = x / total
        
        return x.tolist()

    def run(self):
        self.read_problem_instance(self.problem_path)

        # Apply quick pre-assignment if enabled
        if self.pre_ass:
            self.apply_pre_assignment()

        # Generate valid starting point, that gets it's "fair share"
        curr_x = np.zeros(self.n)
        picks = np.random.choice(self.n, self.k, replace=False)
        curr_x[picks] = 1.0 / self.k

        curr_energy = self.evaluate(curr_x)

        self.x_best = self.expand_weights(curr_x.copy())
        best_energy = curr_energy

        self.avg_rate_epochs = []
        self.best_rate_epochs = []

        temp = self.init_temp
        start_time = time.time()
        while time.time() - start_time <= self.time_deadline and temp > self.min_temp:
            # reproduction
            neighbor = self.get_neighbor(curr_x, temp / self.init_temp)
            neighbor_energy = self.evaluate(neighbor)

            energy_change = neighbor_energy - curr_energy
            if energy_change < 0:
                accept = True
            else:
                p = math.exp(-energy_change / temp)
                accept = np.random.rand() < p
            
            if accept:
                curr_x = neighbor
                curr_energy = neighbor_energy

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    self.x_best = self.expand_weights(curr_x.copy())

            temp *= self.cool_rate

            self.avg_rate_epochs.append(-curr_energy)
            self.best_rate_epochs.append(-best_energy)
            self.epochs_times.append(time.time() - start_time)
        self.q_best = -best_energy

    def evaluate(self, x):
        w = x.copy()  # needed to avoid modifying original solution
        # Find selected assets
        indices = np.where(x > 1e-11)[0]

        curr_sum = w.sum()

        # Penalty for empty
        if curr_sum == 0:
            return 1e12
        
        # normalizing weights of chosen elements to 1
        w = w / curr_sum

        # Calculate risk
        risk = w @ self.d @ w.T

        # Calculate return
        ret = np.sum(w * self.r)

        # Penalty for too low result
        penalty = 0
        if ret < self.R:
            penalty += abs(self.R - ret) * 10
        
        # Penalty for wrong number of 
        # Enforced in get_neighbor, but good to have just in case
        if len(indices) != self.k:
            penalty += abs(len(indices) - self.k)

        obj = risk - penalty

        # Returning negative objective because SA (Simulated Annealing) minimizes
        return -obj

    def get_neighbor(self, curr_x, temp_ratio):
        neighbor = curr_x.copy()
        active_i = np.where(neighbor > 1e-12)[0]
        inactive_i = np.where(neighbor <= 1e-12)[0]

        if np.random.rand() < 0.3:
            # Swaps assets, a more drastic move,
            # that is done mostly when the "Temperature" is higher
            if len(active_i) > 0 and len(inactive_i) > 0:
                sell_idx = np.random.choice(active_i)
                buy_idx = np.random.choice(inactive_i)

                neighbor[buy_idx] = neighbor[sell_idx]
                neighbor[sell_idx] = 0.0
            
        else:
            # Transfers smaller amounts between picked stocks
            if len(active_i) > 0:
                sell_idx, buy_idx = np.random.choice(active_i, 2, replace=False)
                amount = np.random.uniform(0, 0.1) * neighbor[sell_idx]
                neighbor[buy_idx] += amount
                neighbor[sell_idx] -= amount
            
        return neighbor

    def draw_graph(self):
        import matplotlib.pyplot as plt
        x_axis = self.epochs_times
        v1 = self.avg_rate_epochs
        v2 = self.best_rate_epochs

        min_len = min(len(x_axis), len(v1), len(v2))

        plt.plot(x_axis[:min_len], v1[:min_len], "-")
        plt.plot(x_axis[:min_len], v2[:min_len], "-")
        
        # plt.xscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Rate")
        plt.title("Average rate and best rate in each epoch")
        plt.legend(labels=("Average rate", "Best rate"))
        plt.savefig("plots/plot_sa.png")
        plt.show()


if __name__ == "__main__":
    met = Metaheuristic(
        time_deadline=15,
        problem_path="instances/instance_n100_k10_7.json"
    )
    met.run()
    # print("Best solution found:\n", met.x_best)
    print("Best rate found:", met.q_best)
    print("Best solution found:\n", met.get_best_solution())
    met.draw_graph()
