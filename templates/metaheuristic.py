import json
import numpy as np
import time
from scipy.spatial.distance import cdist

class PreAssignmentMixin:
    """
    Mixin class providing pre-assignment functionality to reduce problem size by excluding similar assets.
    """
    def __init__(self, **kwargs):
        """
        Initializes the PreAssignmentMixin with optional parameters.
        """
        self.pre_ass = kwargs.get('pre_assignment', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.78)
        self.excluded_assets = []
        self.used_assets = []
        self.full_n = None

    def quick_pre_assignment(self):
        """
        Reduces problem dimensionality by excluding similar assets based on cosine similarity.
        Maintains minimum required assets and preserves assets with highest returns.
        """
        D = self.d
        min_required = max(self.k * 3, 300)
        max_exclusions = self.n - min_required
        
        if max_exclusions <= 0:
            self.pre_ass = False
            self.excluded_assets = []
            self.used_assets = list(range(self.n))
            return

        col_norms = np.linalg.norm(D, axis=0)
        safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
        X = (D / safe_norms).T
        S = np.clip((X @ X.T), -1.0, 1.0)

        sorted_indices = sorted(range(self.n), key=lambda x: -self.r[x])
        
        while not self.run_quick_pa(S, max_exclusions, sorted_indices, self.similarity_threshold):
            self.similarity_threshold += 0.01
            if self.similarity_threshold > 0.99:
                self.pre_ass = False
                self.excluded_assets = []
                self.used_assets = list(range(self.n))
                break

    def run_quick_pa(self, S, max_exclusions, sorted_indices, threshold):
        """
        Performs asset selection by excluding similar assets with lower returns.
        
        Args:
            S: Similarity matrix between assets
            max_exclusions: Maximum number of assets that can be excluded
            sorted_indices: Asset indices sorted by return (descending)
            threshold: Similarity threshold for exclusion
            
        Returns:
            bool: True if exclusion succeeded within limits, False otherwise
        """
        excluded = set()
        for idx, i in enumerate(sorted_indices):
            if i in excluded:
                continue
            for j in sorted_indices[idx + 1:]:
                if j in excluded:
                    continue
                if S[i, j] > threshold:
                    excluded.add(j)
                    if len(excluded) >= max_exclusions:
                        return False

        self.excluded_assets = sorted(list(excluded))
        self.used_assets = [i for i in range(self.n) if i not in excluded]
        return True

    def apply_pre_assignment(self):
        """
        Applies pre-assignment by reducing problem dimensions based on asset similarity.
        Updates n, r, and d to reflect only the selected subset of assets.
        """
        self.full_n = self.n
        if self.pre_ass:
            self.quick_pre_assignment()
            
        if self.pre_ass and len(self.excluded_assets) > 0:
            self.n = len(self.used_assets)
            self.r = np.array([self.r[i] for i in self.used_assets])
            
            temp_d = self.d[self.used_assets, :]
            self.d = temp_d[:, self.used_assets]

    def expand_weights(self, weights):
        """
        Maps weights from reduced dimension back to full dimension.
        
        Args:
            weights: Weight vector in reduced dimension
            
        Returns:
            np.ndarray: Weight vector in full original dimension
        """
        if self.full_n is None or not self.pre_ass:
            return weights
        full_vector = np.zeros(self.full_n, dtype=float)
        for local_idx, global_idx in enumerate(self.used_assets):
            full_vector[global_idx] = weights[local_idx]
        return full_vector
    
    def expand_distances(self, distances):
        """
        Maps distance matrix from reduced dimension back to full dimension.
        
        Args:
            distances: Distance matrix in reduced dimension
            
        Returns:
            np.ndarray: Distance matrix in full original dimension
        """
        if self.full_n is None or not self.pre_ass:
            return distances
        full = np.zeros((self.full_n, self.full_n), dtype=float)
        for local_i, global_i in enumerate(self.used_assets):
            for local_j, global_j in enumerate(self.used_assets):
                full[global_i, global_j] = distances[local_i, local_j]
        return full

class Metaheuristic(PreAssignmentMixin):
    """
    In this class you should implement your metaheuristic proposal. The code that you submit for the tournament should be
    included in this class. Please, bear in mind that the current template includes all the mandatory methods, but you can implement any
    other method that you need to. In fact, you are highly encouraged to make a good software design a decompose the behavior of your algorithm
    into several independent components or methods.

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
        """
        Sets the best solution found, expanding to full dimension if using pre-assignment.
        
        Args:
            x_best: Best solution weight vector
        """
        self.x_best = self.expand_weights(x_best) if self.pre_ass else x_best

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

        x = self.x_best.copy()
        normalized = x / (x.sum() + 1e-10)
        return normalized.tolist()

    def run(self):
        """
        This method is in charge of reading the problem instance from a file and then executing the whole logic of the metaheuristic, including initialization
        and the main search procedure.
        TODO: You should implement from the pass statement.
        """
        self.read_problem_instance(self.problem_path)
        if self.pre_ass:
            self.apply_pre_assignment()
        curr_popoulation, curr_velocity = (
            self.initialize_population_greedy(self.pop_size) if self.warm_start else self.initialize_population_random(self.pop_size)
        )
        curr_rates = -self.get_rates(curr_popoulation)
        max_val = curr_rates.max()
        pbest = np.ones(self.pop_size) + max_val
        pbest_pos = curr_popoulation.copy()
        gbest = max_val + 1
        gbest_pos = None
        no_improve = np.zeros(self.pop_size, dtype=int)
        gbest_no_improve = 0
        iteration = 0
        self.best_rate_epochs = []
        self.epochs_times = []
        self.avg_rate_epochs = []
        self.feasible_epochs = []
        start_time = time.time()
        total_time_allowed = self.time_deadline - 0.5
        end_time = start_time + total_time_allowed
        while time.time() < end_time:
            elapsed_ratio = (time.time() - start_time) / total_time_allowed
            iteration += 1
            curr_solutions = self.get_solutions(curr_popoulation)
            curr_rates = -self.get_rates(solutions=curr_solutions)
            curr_returns = self.get_returns(solutions=curr_solutions)
            curr_size = self.get_sizes(curr_popoulation)

            ret_cond = self.R - curr_returns
            size_cond = np.abs(curr_size - self.k)
            feasible = (ret_cond <= 0) & (size_cond <= 0)
            curr_fitness = np.where(
                feasible, curr_rates, np.maximum(ret_cond, size_cond) + 1
            )

            pbest_update = curr_fitness < pbest
            pbest[pbest_update] = curr_fitness[pbest_update]
            pbest_pos[pbest_update] = curr_popoulation[pbest_update]
            no_improve[~pbest_update] += 1
            no_improve[pbest_update] = 0

            min_idx = np.argmin(pbest)
            if pbest[min_idx] < gbest:
                gbest = pbest[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()
                current_local_best_x = curr_solutions[min_idx]
                self.set_x_best(current_local_best_x)
                self.q_best = -gbest
                self.r_best = curr_returns[min_idx]
                self.k_best = curr_size[min_idx]
                gbest_no_improve = 0
            else:
                gbest_no_improve += 1

            if iteration % 200 == 0 and self.x_best is not None:
                improved_local_x = self.local_search(current_local_best_x)

                new_fitness = -self.get_rates(solutions=improved_local_x.reshape(1, -1))[0]

                if new_fitness < gbest:
                    gbest = new_fitness
                    current_local_best_x = improved_local_x
                    self.set_x_best(improved_local_x)

                    self.q_best = -gbest
                    self.r_best = improved_local_x @ self.r
                    self.k_best = (improved_local_x > 0).sum()

                    gbest_pos[: self.n] = improved_local_x * self.B
                    gbest_pos[self.n :] = (improved_local_x > 0).astype(int)
                    gbest_no_improve = 0
            self.best_rate_epochs.append(gbest)
            self.epochs_times.append(time.time() - start_time)
            self.avg_rate_epochs.append(curr_rates.mean())
            num_feasible = feasible.sum()
            self.feasible_epochs.append(num_feasible)

            num_leaders = min(self.max_leaders, num_feasible) if num_feasible > 0 else 1

            fitness_order = np.argsort(curr_fitness)
            leader_indices = fitness_order[:num_leaders]
            leaders = curr_popoulation[leader_indices]
            leaders_pbest = pbest[leader_indices]
            leaders_pbest_pos = pbest_pos[leader_indices]
            non_leader_mask = np.ones(self.pop_size, dtype=bool)
            non_leader_mask[leader_indices] = False
            non_leaders = curr_popoulation[non_leader_mask]
            non_leaders_vel = curr_velocity[non_leader_mask]
            non_leaders_pbest = pbest[non_leader_mask]
            non_leaders_pbest_pos = pbest_pos[non_leader_mask]
            non_leader_indices = np.where(non_leader_mask)[0]

            distances = cdist(
                non_leaders[:, : self.n], leaders[:, : self.n], metric="euclidean"
            )
            in_threshold = np.min(distances, axis=1) < self.neighbourhood_threshold
            closest_leader = np.argmin(distances[in_threshold], axis=1)
            subpoped = non_leaders[in_threshold]
            subpoped_vel = non_leaders_vel[in_threshold]
            subpoped_pbest = non_leaders_pbest[in_threshold]
            subpoped_pbest_pos = non_leaders_pbest_pos[in_threshold]
            subpoped_orig_indices = non_leader_indices[in_threshold]
            subpopulations = [
                (
                    subpoped[closest_leader == i],
                    subpoped_vel[closest_leader == i],
                    subpoped_pbest[closest_leader == i],
                    subpoped_pbest_pos[closest_leader == i],
                    subpoped_orig_indices[closest_leader == i],
                )
                for i in range(num_leaders)
            ]

            non_subpoped = non_leaders[~in_threshold]
            non_subpoped_vel = non_leaders_vel[~in_threshold]
            nsp_pbest = non_leaders_pbest[~in_threshold]
            nsp_pbest_pos = non_leaders_pbest_pos[~in_threshold]
            nsp_orig_indices = non_leader_indices[~in_threshold]

            iw = self.iw_max - (self.iw_max - self.iw_min) * elapsed_ratio
            iw = max(self.iw_min, iw)

            pm = max(self.mutation_floor, 0.5 * (1 - elapsed_ratio) ** 2)
            div = self._picks_diversity(curr_popoulation)
            if div < self.diversity_floor:
                pm = min(1.0, pm * self.mutation_boost)

            if non_leaders.shape[0] > 0:
                stagnant_mask_nl = (
                    no_improve[non_leader_indices] >= self.stagnation_patience
                )
                if np.any(stagnant_mask_nl):
                    self._reinitialize_subset(
                        non_leaders[stagnant_mask_nl], non_leaders_vel[stagnant_mask_nl]
                    )
                    non_leaders_pbest[stagnant_mask_nl] = np.inf
                    non_leaders_pbest_pos[stagnant_mask_nl] = non_leaders[
                        stagnant_mask_nl
                    ]
                    no_improve[non_leader_indices[stagnant_mask_nl]] = 0

            if gbest_no_improve >= self.gbest_patience and non_leaders.shape[0] > 0:
                num_imm = max(1, int(self.restart_fraction * non_leaders.shape[0]))
                order_nl = np.argsort(curr_fitness[non_leader_indices])
                worst_sel = order_nl[-num_imm:]
                self._reinitialize_subset(
                    non_leaders[worst_sel], non_leaders_vel[worst_sel]
                )
                non_leaders_pbest[worst_sel] = np.inf
                non_leaders_pbest_pos[worst_sel] = non_leaders[worst_sel]
                no_improve[non_leader_indices[worst_sel]] = 0
                gbest_no_improve = 0
            for i, (subpop, subpop_vel, _, sp_pbest_pos, _) in enumerate(
                subpopulations
            ):
                if subpop.shape[0] == 0:
                    continue
                self.update_pos_vel(
                    subpop, subpop_vel, sp_pbest_pos, leaders_pbest_pos[i], iw, pm
                )
            self.update_pos_vel(
                non_subpoped, non_subpoped_vel, nsp_pbest_pos, gbest_pos, iw, pm
            )

            new_order = np.concatenate(
                [leader_indices, *(sp[4] for sp in subpopulations), nsp_orig_indices]
            )
            curr_popoulation = np.vstack(
                (leaders, *(sp[0] for sp in subpopulations), non_subpoped)
            )
            curr_velocity = np.vstack(
                (
                    np.zeros((num_leaders, self.n * 2)),
                    *(sp[1] for sp in subpopulations),
                    non_subpoped_vel,
                )
            )
            pbest = np.concatenate(
                (leaders_pbest, *(sp[2] for sp in subpopulations), nsp_pbest)
            )
            pbest_pos = np.vstack(
                (leaders_pbest_pos, *(sp[3] for sp in subpopulations), nsp_pbest_pos)
            )
            no_improve = no_improve[new_order]

    def _reinitialize_subset(self, population_subset, velocity_subset):
        """
        Reinitializes a subset of the population with random values.
        
        Args:
            population_subset: Subset of population to reinitialize
            velocity_subset: Corresponding velocity vectors
        """
        m = population_subset.shape[0]
        if m == 0:
            return
        picks = np.zeros((m, self.n), dtype=int)
        for i in range(m):
            ones = np.random.choice(self.n, self.k, replace=False)
            picks[i, ones] = 1
        population_subset[:, self.n :] = picks
        population_subset[:, : self.n] = np.random.choice(
            self.B, (m, self.n), replace=True
        )
        velocity_subset[:, : self.n] = np.random.uniform(
            -0.25 * self.B, 0.25 * self.B, (m, self.n)
        )
        velocity_subset[:, self.n :] = np.random.uniform(-2.5, 2.5, (m, self.n))

    def _picks_diversity(self, population):
        """
        Calculates diversity measure for asset selection in population.
        
        Args:
            population: Current population
            
        Returns:
            float: Diversity score in range [0, 1]
        """
        if population.shape[0] == 0:
            return 0.0
        p = population[:, self.n :].mean(axis=0)
        return float(4.0 * np.mean(p * (1.0 - p)))

    def __init__(self, time_deadline, problem_path, pop_size=1000, **kwargs):
        """
        Initializes the metaheuristic with PSO-based portfolio optimization.
        
        Args:
            time_deadline: Maximum execution time in seconds
            problem_path: Path to problem instance JSON file
            pop_size: Population size for PSO
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
        self.pop_size = pop_size
        self.B = kwargs.get("B", 10000)
        self.pre_ass = kwargs.get("pre_assignment", True)
        self.max_leaders = kwargs.get("max_leaders", 10)
        self.neighbourhood_threshold = (
            kwargs.get("neighbourhood_threshold", 0.7) * self.B
        )
        self.warm_start = kwargs.get("warm_start", True)
        self.iw_max = kwargs.get("iw_max", 1.05)
        self.iw_min = kwargs.get("iw_min", 0.4)
        self.excluded_assets = []
        self.eps_norm = kwargs.get("eps_norm", 1e-8)
        self.weight_floor = kwargs.get("weight_floor", 0.001)
        self.stagnation_patience = kwargs.get("stagnation_patience", 10)
        self.gbest_patience = kwargs.get("gbest_patience", 20)
        self.restart_fraction = kwargs.get("restart_fraction", 0.2)
        self.diversity_floor = kwargs.get("diversity_floor", 0.22)
        self.mutation_boost = kwargs.get("mutation_boost", 2.5)
        self.mutation_floor = kwargs.get("mutation_floor", 0.05)

    def initialize_population_random(self, pop_size):
        """
        Initializes population with random positions and velocities.
        
        Args:
            pop_size: Size of population to initialize
            
        Returns:
            tuple: (population, velocity) arrays
        """
        population = np.zeros((pop_size, self.n * 2), dtype=int)
        velocity = np.zeros((pop_size, self.n * 2), dtype=float)
        for i in range(pop_size):
            individual = np.zeros(self.n * 2, dtype=int)
            individual_vel = np.zeros(self.n * 2, dtype=float)

            pos = np.random.choice(self.B, self.n, replace=True)
            individual[: self.n] = pos

            picks = np.random.choice(self.n, self.k, replace=False) + self.n
            individual[picks] = 1

            vels = np.random.uniform(-0.25 * self.B, 0.25 * self.B, self.n)
            individual_vel[: self.n] = vels

            pick_vels = np.random.uniform(-2.5, 2.5, self.n)
            individual_vel[self.n :] = pick_vels

            population[i] = individual
            velocity[i] = individual_vel
        return population, velocity

    def get_sizes(self, population):
        """
        Calculates portfolio sizes (number of selected assets) for population.
        
        Args:
            population: Population array
            
        Returns:
            np.ndarray: Array of portfolio sizes
        """
        picks = population[:, self.n :] & (population[:, : self.n] > 0)
        sizes = picks.sum(axis=1)
        return sizes

    def get_solutions(self, population):
        """
        Converts population encoding to normalized weight solutions.
        
        Args:
            population: Population array
            
        Returns:
            np.ndarray: Normalized weight vectors
        """
        placings = population[:, : self.n] * population[:, self.n :]
        sums = placings.sum(axis=1, keepdims=True)
        sums = np.where(sums <= self.eps_norm, 1.0, sums)
        solutions = placings / sums
        if self.weight_floor > 0.0:
            solutions = np.where(
                solutions > 0, np.maximum(solutions, self.weight_floor), 0.0
            )
            total = solutions.sum(axis=1)
            mask = total > 0
            solutions[mask, :] = solutions[mask, :] / total[mask].reshape(-1, 1)
        return solutions

    def get_rates(self, population=None, solutions=None):
        """
        Calculates diversification rates (objective function) for solutions.
        
        Args:
            population: Population array (optional if solutions provided)
            solutions: Normalized weight vectors (optional if population provided)
            
        Returns:
            np.ndarray: Diversification rates
        """
        if solutions is None:
            if population is None:
                raise ValueError("Either population or solutions must be provided.")
            solutions = self.get_solutions(population)
        return np.sum((solutions @ self.d) * solutions, axis=1) / 2

    def get_returns(self, population=None, solutions=None):
        """
        Calculates expected returns for solutions.
        
        Args:
            population: Population array (optional if solutions provided)
            solutions: Normalized weight vectors (optional if population provided)
            
        Returns:
            np.ndarray: Expected returns
        """
        if solutions is None:
            if population is None:
                raise ValueError("Either population or solutions must be provided.")
            solutions = self.get_solutions(population)
        returns = solutions @ self.r
        return returns

    def update_pos_vel(self, population, velocity, pbest, gbest, iw, pm):
        """
        Updates particle positions and velocities using PSO rules.
        
        Args:
            population: Current population positions
            velocity: Current velocity vectors
            pbest: Personal best positions
            gbest: Global best position
            iw: Inertia weight
            pm: Mutation probability
        """
        pop_size = population.shape[0]
        c1 = 1.496
        c2 = 1.496

        r1 = np.random.rand(pop_size, self.n)
        r2 = np.random.rand(pop_size, self.n)
        velocity[:, self.n :] = np.clip(
            iw * velocity[:, self.n :]
            + c1 * r1 * (pbest[:, self.n :] - population[:, self.n :])
            + c2 * r2 * (gbest[self.n :] - population[:, self.n :]),
            -2.5,
            2.5,
        )

        change_prob = 1 - np.exp(-np.abs(velocity[:, self.n :]))
        target = (velocity[:, self.n :] > 0).astype(int)
        rand_vals = np.random.rand(pop_size, self.n)
        population[:, self.n :] = np.where(
            rand_vals < change_prob, target, population[:, self.n :]
        )

        self.mutate_binary(population, pm)
        self.project_picks_to_k(population, velocity)

        r1 = np.random.rand(pop_size, self.n) + 0.5
        r2 = np.random.rand(pop_size, self.n) + 0.5
        velocity[:, : self.n] = np.clip(
            iw * velocity[:, : self.n]
            + c1 * r1 * (pbest[:, : self.n] - population[:, : self.n])
            + c2 * r2 * (gbest[: self.n] - population[:, : self.n]),
            -0.25 * self.B,
            0.25 * self.B,
        )

        population[:, : self.n] = np.clip(
            np.where(
                population[:, self.n :],
                population[:, : self.n] + velocity[:, : self.n],
                population[:, : self.n],
            ),
            0,
            self.B,
        )

        mutation_mask = np.random.rand(pop_size, self.n) < pm * 0.2
        velocity[:, : self.n] = np.clip(
            np.where(
                mutation_mask & population[:, self.n :],
                velocity[:, : self.n] * (np.random.rand(pop_size, self.n) - 0.5) * 4,
                velocity[:, : self.n],
            ),
            -0.25 * self.B,
            0.25 * self.B,
        )
        population[:, : self.n] = np.clip(
            np.where(
                mutation_mask & population[:, self.n :],
                population[:, : self.n] + velocity[:, : self.n],
                population[:, : self.n],
            ),
            0,
            self.B,
        )

    def mutate_binary(self, population, pm):
        """
        Applies binary mutation by swapping selected/unselected assets.
        
        Args:
            population: Population array
            pm: Mutation probability
        """
        pop_size = population.shape[0]
        picks = population[:, self.n :]
        picks_sum = picks.sum(axis=1)
        rng = np.random.default_rng()
        mutation_mask = (
            (rng.random(pop_size) < pm) & (picks_sum > 0) & (picks_sum <= self.k)
        )
        idx = np.where(mutation_mask)[0]
        if idx.size == 0:
            return

        for i in idx:
            ones = np.where(picks[i] == 1)[0]
            zeros = np.where(picks[i] == 0)[0]
            a = rng.choice(ones)
            b = rng.choice(zeros)
            picks[i, a] = 0
            picks[i, b] = 1
        population[:, self.n :] = picks

    def project_picks_to_k(self, population, velocity):
        """
        Ensures exactly k assets are selected in each solution.
        
        Args:
            population: Population array
            velocity: Velocity array used for scoring assets
        """
        n = self.n
        picks = population[:, n:]
        scores = velocity[:, n:]
        k = min(self.k, n)
        if k <= 0:
            population[:, n:] = 0
            return
        picks_sum = picks.sum(axis=1)
        need_fix = picks_sum != k
        idx = np.where(need_fix)[0]

        for i in idx:
            s = int(picks_sum[i])
            if s > k:
                ones_idx = np.where(picks[i] == 1)[0]
                if ones_idx.size > 0:
                    to_remove = s - k
                    remove_idx = ones_idx[
                        np.argpartition(scores[i, ones_idx], to_remove - 1)[:to_remove]
                    ]
                    picks[i, remove_idx] = 0
            else:
                zeros_idx = np.where(picks[i] == 0)[0]
                if zeros_idx.size > 0:
                    to_add = k - s
                    add_idx = zeros_idx[
                        np.argpartition(scores[i, zeros_idx], -to_add)[-to_add:]
                    ]
                    picks[i, add_idx] = 1
        population[:, n:] = picks

    def draw_graph(self):
        """
        Generates and saves visualization of optimization progress.
        """
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()

        ax1.plot(
            range(len(self.avg_rate_epochs) - 1),
            self.avg_rate_epochs[:-1],
            "o-",
            label="Average rate",
        )
        ax1.plot(
            range(len(self.best_rate_epochs) - 1),
            self.best_rate_epochs[:-1],
            "o-",
            label="Best rate",
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Rate")
        ax1.tick_params(axis="y")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(
            range(len(self.feasible_epochs) - 1),
            self.feasible_epochs[:-1],
            "s-",
            color="green",
            label="Feasible solutions",
        )
        ax2.set_ylabel("Feasible solutions", color="green")
        ax2.set_ylim(0, max(self.feasible_epochs) * 1.1)
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.legend(loc="upper right")

        plt.title("Average rate, best rate, and feasible solutions in each epoch")
        fig.tight_layout()
        plt.savefig("plots/plot_pso.png")
        plt.show()

    def initialize_population_greedy(self, pop_size):
        """
        Initializes population with greedy warm-start strategy.
        20% of population starts with high-quality assets, 80% random.
        
        Args:
            pop_size: Size of population to initialize
            
        Returns:
            tuple: (population, velocity) arrays
        """
        population = np.zeros((pop_size, self.n * 2), dtype=int)
        velocity = np.zeros((pop_size, self.n * 2), dtype=float)

        avg_dist = self.d.mean(axis=1)
        greedy_scores = self.r * avg_dist

        pool_size = min(self.n, self.k * 2)
        top_tier_indices = np.argsort(greedy_scores)[-pool_size:]

        for i in range(pop_size):
            if i < pop_size * 0.2:
                picks = np.random.choice(top_tier_indices, self.k, replace=False)
            else:
                picks = np.random.choice(self.n, self.k, replace=False)

            individual = np.zeros(self.n * 2, dtype=int)
            individual[: self.n] = np.random.choice(self.B, self.n, replace=True)
            individual[picks + self.n] = 1

            population[i] = individual
            velocity[i, : self.n] = np.random.uniform(
                -0.25 * self.B, 0.25 * self.B, self.n
            )
            velocity[i, self.n :] = np.random.uniform(-2.5, 2.5, self.n)

        return population, velocity

    def local_search(self, solution):
        """
        Applies local search by transferring small weights between assets.
        
        Args:
            solution: Current solution weight vector
            
        Returns:
            np.ndarray: Improved solution
        """
        best_sol = solution.copy()
        best_div = self.get_rates(solutions=best_sol.reshape(1, -1))[0]

        idx = np.where(best_sol > 0)[0]
        if len(idx) < 2:
            return best_sol

        for _ in range(100):
            i, j = np.random.choice(idx, 2, replace=False)
            epsilon = np.random.uniform(0.001, 0.01)

            test_sol = best_sol.copy()
            if test_sol[i] > epsilon:
                test_sol[i] -= epsilon
                test_sol[j] += epsilon

                if (test_sol @ self.r) >= self.R:
                    test_div = self.get_rates(solutions=test_sol.reshape(1, -1))[0]
                    if test_div > best_div:
                        best_sol = test_sol
                        best_div = test_div
        return best_sol


if __name__ == "__main__":
    met = Metaheuristic(
        time_deadline=15, problem_path="instances/instance_n100_k10_7.json"
    )
    met.run()
    print("Best solution found:\n", met.get_best_solution())
    print("\nBest rate found:", met.q_best)
    met.draw_graph()
