import json
import numpy as np
import time
from scipy.spatial.distance import cdist


class Metaheuristic:
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
        self.full_n = self.n
        self.k = instance_data["k"]
        self.R = instance_data["R"]
        self.r = np.array(instance_data["r"])
        self.d = np.array(instance_data["dij"])
    
    def set_x_best(self, x_best):
        if self.pre_ass:
            temp_best = np.zeros(self.full_n)
            temp_best[self.used_assets] = x_best
            self.x_best = temp_best
        else: self.x_best = x_best

    def get_best_solution(self):
        """
        This method is used to return EXTERNALLY the best solution found so far in the metaheuristic. The solution should be returned in a very
        specific format. For that, you are addressed to the project specification. Please, bear in mind that, INTERNALLY, you can represent
        solutions in any format that you see fit. However, externally, solutions should always be returned in the same way in order to participate in the tournament.
        If you follow this template, self.best_solution should contain the best solution found so far and you should return that solution encoded in the specified format.
        If the returned solution does not follow the format specified in the project specification, you will be disqualified from the tournament.
        """
        if self.x_best is None:
            raise Exception("No solution has been found yet.")

        # extracting picks and normalizing
        x = self.x_best.copy()
        normalized = x / (x.sum() + 1e-10)
        return normalized.tolist()
    
    def pre_assignment(self):
        """
        This method is in charge of performing pre-assignment to limit the search space of the metaheuristic.
        """
        from coclust.clustering import SphericalKmeans
        import io
        from contextlib import redirect_stdout, redirect_stderr

        n = self.n
        D = self.d
        r = self.r
        thr = self.similarity_threshold

        # Feature vectors: columns of D (distance profiles), normalized to unit norm
        col_norms = np.linalg.norm(D, axis=0)
        safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
        X = (D / safe_norms).T  # shape: (n_assets, feature_dim)

        # Determine number of clusters based on n
        k_clusters = max(1, int(n/2))

        # Run Spherical K-Means (cosine-based)
        km = SphericalKmeans(n_clusters=k_clusters, n_init=self.n_km_init)
        fnull = io.StringIO()
        with redirect_stdout(fnull), redirect_stderr(fnull):
            km.fit(X)
        labels = km.labels_

        # Group indices by cluster label
        clusters = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(idx)

        # Precompute cosine similarity matrix between assets using normalized features
        S = np.clip((X @ X.T), -1.0, 1.0)

        excluded = set()
        for comp in clusters.values():
            if len(comp) <= 1:
                continue
            # Elect the asset with the highest r value to keep
            comp_rs = r[comp]
            keep_local = int(np.argmax(comp_rs))
            keep = comp[keep_local]
            # Exclude assets in the cluster that are below the similarity threshold with the kept asset
            for idx in comp:
                if idx != keep and S[idx, keep] > thr:
                    excluded.add(idx)

        self.excluded_assets = sorted(excluded)
        self.used_assets = [i for i in range(n) if i not in excluded]

    def quick_pre_assignment(self):
        D = self.d
        max_exclusions = self.n-(self.k*2)  # heuristic limit to avoid over-exclusion
        if max_exclusions <= 0:
            self.pre_ass = False
            self.excluded_assets = []
            self.used_assets = list(range(self.n))
            return

        # Feature vectors: columns of D (distance profiles), normalized to unit norm
        col_norms = np.linalg.norm(D, axis=0)
        safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
        X = (D / safe_norms).T  # shape: (n_assets, feature_dim)

        # Precompute cosine similarity matrix between assets using normalized features
        S = np.clip((X @ X.T), -1.0, 1.0)

        sorted_indices = sorted(range(self.n), key=lambda x: -self.r[x])
        while not self.run_quick_pa(S, max_exclusions, sorted_indices, self.similarity_threshold):
            self.similarity_threshold += 0.005
            if self.similarity_threshold > 1.0:
                self.pre_ass = False
                self.excluded_assets = []
                self.used_assets = list(range(self.n))
                break
        
        
    def run_quick_pa(self, S, max_exclusions, sorted_indices, threshold):
        excluded = set()
        for idx, i in enumerate(sorted_indices):
            if i in excluded:
                continue
            for j in sorted_indices[idx:]:
                if j in excluded:
                    continue
                if j != i and S[i, j] > threshold:
                    excluded.add(j)
                    if len(excluded) >= max_exclusions:
                        return False

        self.excluded_assets = sorted(excluded)
        self.used_assets = [i for i in range(self.n) if i not in excluded]

        return True

    def run(self):
        """
        This method is in charge of reading the problem instance from a file and then executing the whole logic of the metaheuristic, including initialization
        and the main search procedure.
        TODO: You should implement from the pass statement.
        """
        self.read_problem_instance(
            self.problem_path
        )  # You should keep this line. Otherwise, disqualified from the tournament
        if self.pre_ass:
            self.quick_pre_assignment()
            self.n -= len(self.excluded_assets)
            self.r = np.delete(self.r, self.excluded_assets, axis=0)
            self.d = np.delete(self.d, self.excluded_assets, axis=0)
            self.d = np.delete(self.d, self.excluded_assets, axis=1)

        curr_popoulation, curr_velocity = self.initialize_population(self.pop_size)
        curr_rates = -self.get_rates(curr_popoulation)
        max_val = curr_rates.max()
        pbest = np.ones(self.pop_size) + max_val
        pbest_pos = curr_popoulation.copy()
        gbest = max_val + 1
        gbest_pos = None

        total_feasible = 0
        self.best_rate_epochs = []
        self.avg_rate_epochs = []
        self.feasible_epochs = []
        while total_feasible < self.max_feasible:
            # Calculate fitness
            curr_solutions = self.get_solutions(curr_popoulation)
            curr_rates = -self.get_rates(solutions=curr_solutions)
            curr_returns = self.get_returns(solutions=curr_solutions)
            curr_size = self.get_sizes(curr_popoulation)
            ret_cond = self.R-curr_returns
            size_cond = np.abs(curr_size - self.k)
            feasible = (ret_cond <= 0) & (size_cond <= 0)
            curr_fitness = np.where(feasible, curr_rates, np.maximum(ret_cond, size_cond) + 1)

            # Update particle best
            pbest_update = curr_fitness < pbest
            pbest[pbest_update] = curr_fitness[pbest_update]
            pbest_pos[pbest_update] = curr_popoulation[pbest_update]

            # Update global best
            min_idx = np.argmin(pbest)
            if pbest[min_idx] < gbest:
                gbest = pbest[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()
                self.set_x_best(curr_solutions[min_idx])
                self.q_best = -gbest
                self.r_best = curr_returns[min_idx]
                self.k_best = curr_size[min_idx]
            
            # Record stats
            self.best_rate_epochs.append(gbest)
            self.avg_rate_epochs.append(curr_rates.mean())
            num_feasible = feasible.sum()
            self.feasible_epochs.append(num_feasible)

            # Determine number of leaders
            total_feasible += num_feasible
            num_leaders = min(self.max_leaders, num_feasible) if num_feasible > 0 else 1

            # Select leaders
            fitness_order = np.argsort(curr_fitness)
            leader_indices = fitness_order[:num_leaders]
            leaders = curr_popoulation[leader_indices]
            leaders_pbest = pbest[leader_indices]
            leaders_pbest_pos = pbest_pos[leader_indices]
            # Remove leaders from population to avoid self-influence
            non_leader_mask = np.ones(self.pop_size, dtype=bool)
            non_leader_mask[leader_indices] = False
            non_leaders = curr_popoulation[non_leader_mask]
            non_leaders_vel = curr_velocity[non_leader_mask]
            non_leaders_pbest = pbest[non_leader_mask]
            non_leaders_pbest_pos = pbest_pos[non_leader_mask]

            # Construct subpopulations around leaders (with thresholding)
            distances = cdist(non_leaders[:, :self.n], leaders[:, :self.n], metric='euclidean')
            in_threshold = np.min(distances, axis=1) < self.neighbourhood_threshold
            closest_leader = np.argmin(distances[in_threshold], axis=1)
            subpoped = non_leaders[in_threshold]
            subpoped_vel = non_leaders_vel[in_threshold]
            subpoped_pbest = non_leaders_pbest[in_threshold]
            subpoped_pbest_pos = non_leaders_pbest_pos[in_threshold]
            subpopulations = [(
                subpoped[closest_leader == i], 
                subpoped_vel[closest_leader == i],
                subpoped_pbest[closest_leader == i],
                subpoped_pbest_pos[closest_leader == i]
                ) for i in range(num_leaders)]
            
            non_subpoped = non_leaders[~in_threshold]
            non_subpoped_vel = non_leaders_vel[~in_threshold]
            nsp_pbest = non_leaders_pbest[~in_threshold]
            nsp_pbest_pos = non_leaders_pbest_pos[~in_threshold]

            # Calculate inertia weight
            iw = self.iw_max - (self.iw_max - self.iw_min) * (total_feasible / self.max_feasible)

            # Calcuate mutation probablility
            pm = 0.5 * (1 - total_feasible / self.max_feasible)**2
            for i, (subpop, subpop_vel, _, sp_pbest_pos) in enumerate(subpopulations):
                if subpop.shape[0] == 0:
                    continue
                self.update_pos_vel(
                    subpop,
                    subpop_vel,
                    sp_pbest_pos,
                    leaders_pbest_pos[i],
                    iw,
                    pm
                )
            self.update_pos_vel(
                non_subpoped,
                non_subpoped_vel,
                nsp_pbest_pos,
                gbest_pos,
                iw,
                pm
            )

            # Reconstruct population
            curr_popoulation = np.vstack((
                leaders,
                *(sp[0] for sp in subpopulations),
                non_subpoped
            ))
            curr_velocity = np.vstack((
                np.zeros((num_leaders, self.n*2)),
                *(sp[1] for sp in subpopulations),
                non_subpoped_vel
            ))
            pbest = np.concatenate((
                leaders_pbest,
                *(sp[2] for sp in subpopulations),
                nsp_pbest
            ))
            pbest_pos = np.vstack((
                leaders_pbest_pos,
                *(sp[3] for sp in subpopulations),
                nsp_pbest_pos
            ))
    
    def __init__(self, time_deadline, problem_path, pop_size=100, sigma=0.5, **kwargs):
        """
        Class initializer. It takes as an argument the maximum computation time (in seconds), controlled externally, and the path that contains the problem instance to be solved
        YOU CAN MODIFY THE HEADER TO INCLUDE OPTIONAL PARAMETERS WITH DEFAULT VALUES ( e.g., __init__(self, time_deadline, problem_path, mut_prob=0.5) )
        You should configure the algorithm before its execution in this method (i.e., hyperparameter values, data structure initialization, etc.)
        Args:
            problem_path: String that contains the path to the file that describes the problem instance
            time_deadline: Computation time limit for the metaheuristic
            kwargs: Other arguments can be passed to the algorithm using key-value pairs. For instance, Metaheuristic(20, 'instance1.txt', mut_prob=0.3) would call the initializer with 20 seconds, for reading the instance1.txt file and passing an optional parameter of mut_prob=0.3
        """
        self.problem_path = problem_path  # This attribute is meant to contain the path to the problem instance
        self.best_solution = None  # This attribute is meant to hold, at any time, the best solution found by the algorithm so far. Hence, you should update it accordingly. The solution enconding does not matter.
        self.time_deadline = (
            time_deadline  # Computation limit (in seconds) for the metaheuristic
        )
        # TODO: Configure the metaheuristic (e.g., selection operator, crossover, mutation, hyperparameter values, etc.)
        self.pop_size = pop_size
        self.B = kwargs.get('B', 1000)  # upper bound for position values
        self.sigma = sigma  # standard deviation for gaussian mutation 
        self.pre_ass = kwargs.get('pre_assignment', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        self.n_km_init = kwargs.get('n_km_init', 1)
        self.max_leaders = kwargs.get('max_leaders', 10)
        self.max_feasible = kwargs.get('max_feasible', 50000)
        self.neighbourhood_threshold = kwargs.get('neighbourhood_threshold', 0.1) * self.B
        self.iw_max = kwargs.get('iw_max', 1.05)
        self.iw_min = kwargs.get('iw_min', 0.4)
        self.excluded_assets = []
        # Numerical stability controls
        self.eps_norm = kwargs.get('eps_norm', 1e-8)
        self.weight_floor = kwargs.get('weight_floor', 0.001)

    def initialize_population(self, pop_size):
        population = np.zeros((pop_size, self.n*2), dtype=int)
        velocity = np.zeros((pop_size, self.n*2), dtype=float)
        for i in range(pop_size):
            individual = np.zeros(self.n*2, dtype=int)
            individual_vel = np.zeros(self.n*2, dtype=float)

            pos = np.random.choice(self.B, self.n, replace=True)
            individual[:self.n] = pos

            picks = np.random.choice(self.n, self.k, replace=False) + self.n
            individual[picks] = 1

            vels = np.random.uniform(-0.25*self.B, 0.25*self.B, self.n)
            individual_vel[:self.n] = vels

            pick_vels = np.random.uniform(-2.5, 2.5, self.n)
            individual_vel[self.n:] = pick_vels

            population[i] = individual
            velocity[i] = individual_vel
        return population, velocity
    
    def get_sizes(self, population):
        picks = population[:, self.n:] & (population[:, :self.n] > 0)
        sizes = picks.sum(axis=1)
        return sizes
    
    def get_solutions(self, population):
        placings = population[:, : self.n] * population[:, self.n :]
        sums = placings.sum(axis=1, keepdims=True)
        sums = np.where(sums <= self.eps_norm, 1.0, sums)
        solutions = placings / sums
        # Apply floor to avoid vanishing weights, then renormalize
        if self.weight_floor > 0.0:
            solutions = np.where(solutions > 0, np.maximum(solutions, self.weight_floor), 0.0)
            total = solutions.sum(axis=1)
            mask = total > 0
            solutions[mask,:] = solutions[mask,:] / total[mask].reshape(-1,1)
        return solutions

    def get_rates(self, population=None, solutions=None):
        if solutions is None:
            if population is None:
                raise ValueError("Either population or solutions must be provided.")
            solutions = self.get_solutions(population)
        rates = (solutions @ self.d @ solutions.T).diagonal()
        return rates

    def get_returns(self, population=None, solutions=None):
        if solutions is None:
            if population is None:
                raise ValueError("Either population or solutions must be provided.")
            solutions = self.get_solutions(population)
        returns = solutions @ self.r
        return returns
    
    def update_pos_vel(self, population, velocity, pbest, gbest, iw, pm):
        pop_size = population.shape[0]
        c1 = 1.496
        c2 = 1.496

        # Update binary velocity
        r1 = np.random.rand(pop_size, self.n)
        r2 = np.random.rand(pop_size, self.n)
        velocity[:, self.n:] = np.clip(
            iw * velocity[:, self.n :]
            + c1 * r1 * (pbest[:, self.n :] - population[:, self.n :])
            + c2 * r2 * (gbest[self.n :] - population[:, self.n :]),
            -2.5,
            2.5
        )

        # Update binary position
        sigmoid = 1 / (1 + np.exp(-velocity[:, self.n :]))
        rand_vals = np.random.rand(pop_size, self.n)
        population[:, self.n :] = (rand_vals < sigmoid).astype(int)

        # Enforce exactly k picks per individual by keeping top-k by binary velocity score
        self.project_picks_to_k(population, velocity)

        # Mutate binary position
        self.mutate_binary(population, pm)

        # Update continuous velocity
        r1 = np.random.rand(pop_size, self.n) + 0.5
        r2 = np.random.rand(pop_size, self.n) + 0.5
        velocity[:, : self.n] = np.clip(
            iw * velocity[:, : self.n]
            + c1 * r1 * (pbest[:, : self.n] - population[:, : self.n])
            + c2 * r2 * (gbest[: self.n] - population[:, : self.n]),
            -0.25 * self.B,
            0.25 * self.B,
        )

        # Update continuous position
        population[:, : self.n] = np.clip(np.where(
            population[:, self.n :],
            population[:, : self.n] 
            + velocity[:, : self.n],
            population[:, : self.n]),
            0,
            self.B
        )

        # Mutate continuous
        mutation_mask = np.random.rand(pop_size, self.n) < pm*0.2
        velocity[:, : self.n] = np.clip(np.where(
            mutation_mask&population[:, self.n :],
            velocity[:, : self.n]
            *(np.random.randn(pop_size, self.n)-0.5)*4,
            velocity[:, : self.n]
        ), -0.25 * self.B, 0.25 * self.B)
        population[:, : self.n] = np.clip(np.where(
            mutation_mask&population[:, self.n :],
            population[:, : self.n] 
            + velocity[:, : self.n],
            population[:, : self.n]),
            0,
            self.B
        )


    def mutate_binary(self, population, pm):
        pop_size = population.shape[0]
        picks = population[:, self.n:]
        picks_sum = picks.sum(axis=1)
        rng = np.random.default_rng()
        mutation_mask = (
            (rng.random(pop_size) < pm)
            & (picks_sum > 0)
            & (picks_sum <= self.k)
        )
        for i in range(pop_size):
            if not mutation_mask[i]:
                continue
            ones = np.where(picks[i] == 1)[0]
            zeros = np.where(picks[i] == 0)[0]
            a = rng.choice(ones)
            b = rng.choice(zeros)
            picks[i, a] = 0
            picks[i, b] = 1
        population[:, self.n:] = picks

    def project_picks_to_k(self, population, velocity):
        n = self.n
        picks = population[:, n:]
        scores = velocity[:, n:]
        k = min(self.k, n)
        if k <= 0:
            population[:, n:] = 0
            return
        for i in range(population.shape[0]):
            s = int(picks[i].sum())
            if s == k:
                continue
            elif s > k:
                # Remove (s-k) picks: drop those with the lowest scores among current ones
                ones_idx = np.where(picks[i] == 1)[0]
                if ones_idx.size > 0:
                    # number to remove
                    to_remove = s - k
                    # find lowest-scoring ones
                    remove_idx = ones_idx[np.argpartition(scores[i, ones_idx], to_remove-1)[:to_remove]]
                    picks[i, remove_idx] = 0
            else:  # s < k
                # Add (k-s) picks: choose highest scores among current zeros
                zeros_idx = np.where(picks[i] == 0)[0]
                if zeros_idx.size > 0:
                    to_add = k - s
                    add_idx = zeros_idx[np.argpartition(scores[i, zeros_idx], -to_add)[-to_add:]]
                    picks[i, add_idx] = 1
        population[:, n:] = picks

    def draw_graph(self):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()

        # Plot rates on the primary y-axis
        ax1.plot(range(len(self.avg_rate_epochs) - 1), self.avg_rate_epochs[:-1], "o-", label="Average rate")
        ax1.plot(range(len(self.best_rate_epochs) - 1), self.best_rate_epochs[:-1], "o-", label="Best rate")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Rate")
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        # Create a second y-axis for feasible solutions
        ax2 = ax1.twinx()
        ax2.plot(range(len(self.feasible_epochs) - 1), self.feasible_epochs[:-1], "s-", color='green', label="Feasible solutions")
        ax2.set_ylabel("Feasible solutions", color='green')
        ax2.set_ylim(0, max(self.feasible_epochs) * 1.1)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right')

        plt.title("Average rate, best rate, and feasible solutions in each epoch")
        fig.tight_layout()
        plt.savefig("plots/plot.png")
        plt.show()


if __name__ == "__main__":
    met = Metaheuristic(
        time_deadline=15, problem_path="instances/instance_n100_k10_7.json"
    )
    met.run()
    print("Best solution found:\n", met.get_best_solution())
    print("\nBest rate found:", met.q_best)
    met.draw_graph()
