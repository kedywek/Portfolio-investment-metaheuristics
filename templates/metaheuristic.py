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
        self.k = instance_data["k"]
        self.R = instance_data["R"]
        self.r = np.array(instance_data["r"])
        self.d = np.array(instance_data["dij"])

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
        x = self.x_best["decision"].copy()
        picks = np.argsort(x)[-self.k :]
        mask = np.isin(np.arange(self.n), picks)
        x *= mask
        x[x < 0] = 1e-6  # avoid negative values but have to keep them positive
        normalized = x / x.sum()  # potential division by zero
        normalized = np.nan_to_num(
            normalized, posinf=0.0, neginf=0.0
        )  # handle division by zero
        return normalized.tolist()

    def run(self):
        """
        This method is in charge of reading the problem instance from a file and then executing the whole logic of the metaheuristic, including initialization
        and the main search procedure.
        TODO: You should implement from the pass statement.
        """
        self.read_problem_instance(
            self.problem_path
        )  # You should keep this line. Otherwise, disqualified from the tournament

        curr_popoulation = self.initialize_population(self.pop_size)
        curr_rate = self.rate(curr_popoulation)
        self.avg_rate_epochs = [curr_rate.mean()]
        self.x_best, self.q_best = self.find_best(curr_popoulation, curr_rate)
        self.best_rate_epochs = [self.q_best]
        start_time = time.time()
        while time.time() - start_time <= self.time_deadline:
            # reproduction
            R = self.reproduction(curr_popoulation, self.children_size)
            # crossover
            C = self.averaging_crossover_extended_version(R)
            # mutation
            children_popoulation = self.mutation(C)
            # evaluation
            children_rate = self.rate(children_popoulation)
            x_t, q_t = self.find_best(children_popoulation, children_rate)
            if q_t > self.q_best:
                self.x_best = x_t
                self.q_best = q_t
            # succession
            if self.preserve_parents:  # ES(μ + λ) algorithm
                curr_popoulation, curr_rate = self.elite_succession(
                    curr_popoulation,
                    curr_rate,
                    children_popoulation,
                    children_rate,
                    elite_size=self.pop_size,
                )
            else:  # ES(μ, λ) algorithm
                curr_popoulation, curr_rate = self.elite_succession(
                    children_popoulation,
                    children_rate,
                    [],
                    [],
                    elite_size=self.pop_size,
                )
            self.avg_rate_epochs.append(curr_rate.mean())
            self.best_rate_epochs.append(curr_rate.max())

    def __init__(
        self,
        time_deadline,
        problem_path,
        pop_size=20,
        children_size=140,
        preserve_parents=True,
        **kwargs,
    ):
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
        self.children_size = children_size
        self.preserve_parents = preserve_parents

    def evaluate(self, x):
        result = 0
        x = x["decision"].copy()  # needed to avoid modifying original array
        # pick k greatest elements from x
        picks = np.argsort(x)[-self.k :]

        # changing not chosen elements to zero and normalizing chosen ones
        mask = np.isin(np.arange(self.n), picks)
        x *= mask
        x[x < 0] = 1e-6  # avoid negative values but have to keep them positive
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
            individual = {}
            individual["decision"] = np.random.rand(self.n)
            individual["sigma"] = (
                np.random.rand(self.n) * 0.5 + 0.1
            )  # avoid too small sigma
            population.append(individual)
        return np.array(population)

    def rate(self, population):
        rates = np.array([self.evaluate(individual) for individual in population])
        return rates

    def find_best(self, population, rates):
        idx = np.argmax(rates)
        return population[idx], rates[idx]

    def reproduction(self, population, children_size):
        reproducted = [
            population[i] for i in np.random.choice(len(population), size=children_size)
        ]
        return np.array(reproducted)

    def averaging_crossover_extended_version(self, population):
        children = []
        pop_size = len(population)
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, size=2, replace=False)
            parent1 = population[i]
            parent2 = population[j]
            alpha = [np.random.rand() for _ in range(self.n)]
            child = {}
            child["decision"] = (
                np.array(alpha) * parent1["decision"]
                + (1 - np.array(alpha)) * parent2["decision"]
            )
            child["sigma"] = (
                np.array(alpha) * parent1["sigma"]
                + (1 - np.array(alpha)) * parent2["sigma"]
            )
            children.append(child)
        return np.array(children)

    def mutation(self, population):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 1, size=self.n)
        tau = 1 / np.sqrt(2 * np.sqrt(self.n))
        tau_prime = 1 / np.sqrt(2 * self.n)

        mutated_population = []
        for individual in population:
            new_sigma = individual["sigma"] * np.exp(tau_prime * a + tau * b)
            new_individual = {}
            new_individual["decision"] = individual[
                "decision"
            ] + new_sigma * np.random.normal(0, 1, size=self.n)
            new_individual["sigma"] = new_sigma
            mutated_population.append(new_individual)
        return np.array(mutated_population)

    def elite_succession(
        self, parent_population, parent_rates, child_population, child_rates, elite_size
    ):
        combined_population = np.concatenate((parent_population, child_population))
        combined_rates = np.concatenate((parent_rates, child_rates))
        elite_indices = np.argsort(combined_rates)[-elite_size:]
        new_population = combined_population[elite_indices]
        new_rates = combined_rates[elite_indices]
        return new_population, new_rates


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
    pop_size = 80
    chilren_size = pop_size * 7  # typically lambda = 7 * mu
    print("ES(mu + lambda)")
    print("Population size:", pop_size, "Children size:", chilren_size)
    met = Metaheuristic(
        time_deadline=60,
        problem_path="instances/instance_n100_k10_7.json",
        pop_size=pop_size,
        children_size=chilren_size,
        preserve_parents=True,
    )
    met.run()
    # print("Best solution found:\n", met.x_best)
    print("Best rate found:", met.q_best)
    print("Best solution found:\n", met.get_best_solution())
    draw_graph(met.avg_rate_epochs, met.best_rate_epochs)
