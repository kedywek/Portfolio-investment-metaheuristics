import json
import numpy as np
import time


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
        picks = np.argsort(x)[-self.k :]
        mask = np.isin(np.arange(self.full_n), picks)
        x *= mask
        normalized = x / x.sum()  # potential division by zero
        normalized = np.nan_to_num(
            normalized, posinf=0.0, neginf=0.0
        )  # handle division by zero
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
                if idx != keep and S[idx, keep] < thr:
                    excluded.add(idx)

        self.excluded_assets = sorted(excluded)
        self.used_assets = [i for i in range(n) if i not in excluded]

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
            self.pre_assignment()
            self.n -= len(self.excluded_assets)
            self.r = np.delete(self.r, self.excluded_assets, axis=0)
            self.d = np.delete(self.d, self.excluded_assets, axis=0)
            self.d = np.delete(self.d, self.excluded_assets, axis=1)

        curr_popoulation = self.initialize_population(self.pop_size)
        curr_rate = self.rate(curr_popoulation)
        self.avg_rate_epochs = [curr_rate.mean()]
        temp_best, self.q_best = self.find_best(curr_popoulation, curr_rate)
        self.set_x_best(temp_best)
        self.best_rate_epochs = [self.q_best]
        start_time = time.time()
        while time.time() - start_time <= self.time_deadline:
            # reproduction
            R = self.tournament_reproduction(curr_popoulation, curr_rate)
            # crossover
            C = self.averaging_crossover_extended_version(R)
            # mutation
            children_popoulation = self.gaussian_mutation(C, self.sigma)
            # evaluation
            children_rate = self.rate(children_popoulation)
            x_t, q_t = self.find_best(children_popoulation, children_rate)
            if q_t > self.q_best:
                self.set_x_best(x_t)
                self.q_best = q_t
            # succession
            curr_popoulation, curr_rate = self.elite_succession(
                curr_popoulation, curr_rate, children_popoulation, children_rate
            )
            self.avg_rate_epochs.append(curr_rate.mean())
            self.best_rate_epochs.append(curr_rate.max())

    def __init__(self, time_deadline, problem_path, pop_size=20, sigma=0.5, **kwargs):
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
        self.sigma = sigma  # standard deviation for gaussian mutation 
        self.pre_ass = kwargs.get('pre_assignment', False)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.75)
        self.n_km_init = kwargs.get('n_km_init', 1)
        self.excluded_assets = []

    def evaluate(self, x):
        result = 0
        x = x.copy()  # needed to avoid modifying original array
        # pick k greatest elements from x
        picks = np.argsort(x)[-self.k :]

        # changing not chosen elements to zero and normalizing chosen ones
        mask = np.isin(np.arange(self.n), picks)
        x *= mask
        normalized = x / x.sum()  # potential division by zero
        normalized = np.nan_to_num(
            normalized, posinf=0.0, neginf=0.0
        )  # handle division by zero

        # iterate through all pairs and calculate the objective value
        i, j = np.triu_indices(self.k, k=1)
        pairs = np.column_stack((picks[i], picks[j]))
        for a, b in pairs:
            result += normalized[a] * normalized[b] * self.d[a][b]

        # penalty for not fulfilling constraints
        if np.sum(mask * self.r) < self.R:
            penalty = self.R - np.sum(mask * self.r)
            result -= penalty * 1e5

        return result

    def initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            individual = np.random.rand(self.n)
            population.append(individual)
        return np.array(population)

    def rate(self, population):
        rates = np.array([self.evaluate(individual) for individual in population])
        return rates

    def find_best(self, population, rates):
        idx = np.argmax(rates)
        return population[idx], rates[idx]

    def tournament_reproduction(self, population, rates):
        pop_size = len(population)
        new_population = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, size=2)
            if rates[i] > rates[j]:
                new_population.append(population[i])
            else:
                new_population.append(population[j])
        return np.array(new_population)

    def averaging_crossover_extended_version(self, population):
        children = []
        pop_size = len(population)
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, size=2, replace=False)
            parent1 = population[i]
            parent2 = population[j]
            alpha = [np.random.rand() for _ in range(self.n)]
            child = alpha * parent1 + (1 - np.array(alpha)) * parent2
            children.append(child)
        return np.array(children)

    def gaussian_mutation(self, population, sigma):
        mutated_population = []
        for individual in population:
            noise = np.random.normal(0, sigma, size=self.n)
            mutated_individual = individual + noise
            mutated_population.append(mutated_individual)
        return np.array(mutated_population)

    def elite_succession(
        self, parent_population, parent_rates, child_population, child_rates
    ):
        # hardcoded elite_size = 1, cuz then no need to sort, just pick best and worst
        survived_parent_idx = np.argmax(parent_rates)
        survived_parent = parent_population[survived_parent_idx]
        survived_parent_rate = parent_rates[survived_parent_idx]
        worst_child_idx = np.argmin(child_rates)
        if survived_parent_rate > child_rates[worst_child_idx]:
            child_population[worst_child_idx] = survived_parent
            child_rates[worst_child_idx] = survived_parent_rate
        return child_population, child_rates


def draw_graph(v1, v2):
    import matplotlib.pyplot as plt

    plt.plot(range(len(v1) - 1), v1[:-1], "o-")
    plt.plot(range(len(v2) - 1), v2[:-1], "o-")
    # plt.xscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Rate")
    plt.title("Average rate and best rate in each epoch")
    plt.legend(labels=("Average rate", "Best rate"))
    plt.savefig("plots/plot.png")
    plt.show()


if __name__ == "__main__":
    met = Metaheuristic(
        time_deadline=15, problem_path="instances/instance_n100_k10_7.json"
    )
    met.run()
    print("Best solution found:\n", met.get_best_solution())
    print("\nBest rate found:", met.q_best)
    draw_graph(met.avg_rate_epochs, met.best_rate_epochs)
