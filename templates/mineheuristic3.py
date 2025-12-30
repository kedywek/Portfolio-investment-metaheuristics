import numpy as np
import json
import time

class Metaheuristic:
    def __init__(self, time_deadline, problem_path, **kwargs):
        self.time_deadline = time_deadline
        self.problem_path = problem_path
        self.start_time = None
        
        self.data = self.read_problem_instance(problem_path)
        self.n = self.data['n']
        self.k = self.data['k']
        self.r = np.array(self.data['r'])
        self.R = self.data['R']
        self.d = np.array(self.data['dij']) 
        self.full_n = self.n 
        
        self.q_best = -np.inf      
        self.r_best = 0.0          
        self.k_best = 0            
        self.excluded_assets = []  
        self.best_rate_epochs = [] 
        self.epochs_times = []     
        self.pre_ass = kwargs.get('pre_assignment', False)
        
        self.as_size = kwargs.get('as_size', 20)      
        self.ants = kwargs.get('ants', 10)           
        self.mr = kwargs.get('mr', 0.5)               
        self.tau = kwargs.get('tau', 1.0)             
        self.ilimit = kwargs.get('ilimit', 50)        
        
        self.best_solution = np.zeros(self.n)

    def read_problem_instance(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def expand_weights(self, weights):
        return weights

    def expand_distances(self, dij):
        return dij

    def draw_graph(self):
        pass

    def decode(self, chromosome):
        top_k_indices = np.argsort(chromosome)[-self.k:]
        weights = np.zeros(self.n)
        
        active_vals = np.maximum(chromosome[top_k_indices], 1e-6)
        weights[top_k_indices] = active_vals / np.sum(active_vals)
            
        current_ret = np.dot(weights, self.r)
        if current_ret < self.R:
            best_in_k = top_k_indices[np.argmax(self.r[top_k_indices])]
            weights[best_in_k] += 0.2
            weights /= np.sum(weights)
            
        return weights

    def calculate_fitness(self, weights):
        current_return = np.dot(weights, self.r)
        if current_return < self.R:
            return -1e7 * (self.R - current_return) - 1000.0
        return weights @ self.d @ weights / 2.0

    def get_best_solution(self):
        return self.best_solution.tolist()

    def run(self):
        self.start_time = time.time()
        
        archive = np.random.rand(self.as_size, self.n)
        
        top_r_assets = np.argsort(self.r)[-self.k:]
        archive[0, :] = 0.0
        archive[0, top_r_assets] = 1.0
        
        archive_fitness = np.array([self.calculate_fitness(self.decode(sol)) for sol in archive])
        
        idx = np.argsort(archive_fitness)[::-1]
        archive = archive[idx]
        archive_fitness = archive_fitness[idx]
        
        limit_counter = 0
        
        while (time.time() - self.start_time) < (self.time_deadline - 0.7):
            produced_solutions = []
            
            for s in range(self.ants):
                g_idx = int(np.random.exponential(scale=self.as_size/4)) % self.as_size
                xg = archive[g_idx]
                
                xsj = np.copy(xg)
                if np.random.rand() < self.mr:
                    sigma = self.tau * np.sum(np.abs(archive - xg)) / (self.as_size - 1)
                    xsj += np.random.normal(0, sigma, self.n)
                
                xsj = np.clip(xsj, 0, 1)
                produced_solutions.append(xsj)
            
            for sol in produced_solutions:
                fit = self.calculate_fitness(self.decode(sol))
                if fit > archive_fitness[-1]:
                    archive[-1] = sol
                    archive_fitness[-1] = fit
                    sort_idx = np.argsort(archive_fitness)[::-1]
                    archive = archive[sort_idx]
                    archive_fitness = archive_fitness[sort_idx]

            if archive_fitness[0] > self.q_best:
                self.q_best = archive_fitness[0]
                self.best_solution = self.decode(archive[0])
                self.r_best = np.dot(self.best_solution, self.r)
                self.k_best = np.count_nonzero(np.array(self.best_solution) > 1e-12)
                limit_counter = 0
            else:
                limit_counter += 1

            self.best_rate_epochs.append(max(0, self.q_best))
            self.epochs_times.append(time.time() - self.start_time)

            if limit_counter > self.ilimit:
                archive[1:] = np.random.rand(self.as_size - 1, self.n)
                archive_fitness[1:] = [self.calculate_fitness(self.decode(s)) for s in archive[1:]]
                limit_counter = 0