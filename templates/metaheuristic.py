import json
import numpy as np

class Metaheuristic:
    """
    In this class you should implement your metaheuristic proposal. The code that you submit for the tournament should be 
    included in this class. Please, bear in mind that the current template includes all the mandatory methods, but you can implement any
    other method that you need to. In fact, you are highly encouraged to make a good software design a decompose the behavior of your algorithm
    into several iindependent components or methods.

    The HEADERS for the provided methods CANNOT be modified. Failing to do so will result in your algorithm not participating in the tournament.
    """

    def read_problem_instance(self,problem_path):
        """
        TODO: This method is MANDATORY. The goal of this method is reading a hard drive path that contains a text file with a problem instance.
        The method should read all of the information in the problem instance and store it inside attributes of the Metaheuristic object. 
        This method SHOULD NOT SEARCH nor carry out tasks that indirectly contribute to searching. Typically, you will prepare
        data structures to hold relevant information from the problem instance
        Args:
            problem_path: Text file that contains information about a problem instance
        """
        instance_data = json.load(open(problem_path,'r'))
        self.n = instance_data['n']
        self.k = instance_data['k']
        self.R = instance_data['R']
        self.r = np.array(instance_data['r'])
        self.d = np.array(instance_data['dij'])

    def get_best_solution(self):
        """
        This method is used to return EXTERNALLY the best solution found so far in the metaheuristic. The solution should be returned in a very
        specific format. For that, you are addressed to the project specification. Please, bear in mind that, INTERNALLY, you can represent
        solutions in any format that you see fit. However, externally, solutions should always be returned in the same way in order to participate in the tournament.
        If you follow this template, self.best_solution should contain the best solution found so far and you should return that solution encoded in the specified format.
        If the returned solution does not follow the format specified in the project specification, you will be disqualified from the tournament.
        """
        if self.best_solution is None:
            raise Exception("No solution has been found yet.")
        return self.best_solution
    
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


    def run(self):
        """
        This method is in charge of reading the problem instance from a file and then executing the whole logic of the metaheuristic, including initialization
        and the main search procedure.
        TODO: You should implement from the pass statement.
        """
        self.read_problem_instance(self.problem_path) #You should keep this line. Otherwise, disqualified from the tournament
        self.pre_assignment()
        print(f"Excluded {len(self.excluded_assets)}/{self.n} assets")

    def __init__(self,time_deadline,problem_path,**kwargs):
        """
        Class initializer. It takes as an argument the maximum computation time (in seconds), controlled externally, and the path that contains the problem instance to be solved
        YOU CAN MODIFY THE HEADER TO INCLUDE OPTIONAL PARAMETERS WITH DEFAULT VALUES ( e.g., __init__(self, time_deadline, problem_path, mut_prob=0.5) )
        You should configure the algorithm before its execution in this method (i.e., hyperparameter values, data structure initialization, etc.)
        Args:
            problem_path: String that contains the path to the file that describes the problem instance
            time_deadline: Computation time limit for the metaheuristic
            kwargs: Other arguments can be passed to the algorithm using key-value pairs. For instance, Metaheuristic(20, 'instance1.txt', mut_prob=0.3) would call the initializer with 20 seconds, for reading the instance1.txt file and passing an optional parameter of mut_prob=0.3
        """
        self.problem_path = problem_path # This attribute is meant to contain the path to the problem instance
        self.best_solution = None #This attribute is meant to hold, at any time, the best solution found by the algorithm so far. Hence, you should update it accordingly. The solution enconding does not matter.
        self.time_deadline = time_deadline # Computation limit (in seconds) for the metaheuristic 
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.75)
        self.n_km_init = kwargs.get('n_km_init', 1)
        self.excluded_assets = []
        #TODO: Configure the metaheuristic (e.g., selection operator, crossover, mutation, hyperparameter values, etc.)
