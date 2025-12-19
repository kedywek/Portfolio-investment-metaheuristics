import numpy as np

class PreAssignmentMixin:
    def __init__(self, **kwargs):
        self.pre_ass = kwargs.get('pre_assignment', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        self.n_km_init = kwargs.get('n_km_init', 1)
        self.excluded_assets = []
        self.used_assets = []
        self.full_n = None

    def quick_pre_assignment(self):
        D = self.d
        max_exclusions = self.n - max(self.k*2, 200)
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

    def apply_pre_assignment(self):
        self.quick_pre_assignment()
        self.full_n = self.n

        if self.pre_ass and len(self.excluded_assets) > 0:
            self.n -= len(self.excluded_assets)
            self.r = np.delete(self.r, self.excluded_assets, axis=0)
            self.d = np.delete(self.d, self.excluded_assets, axis=0)
            self.d = np.delete(self.d, self.excluded_assets, axis=1)

    def expand_weights(self, weights):
        if self.full_n is None or not self.pre_ass:
            return weights
        full = np.zeros(self.full_n, dtype=float)
        for local_idx, global_idx in enumerate(self.used_assets):
            full[global_idx] = weights[local_idx]
        return full
    
    def expand_distances(self, distances):
        if self.full_n is None or not self.pre_ass:
            return distances
        full = np.zeros((self.full_n, self.full_n), dtype=float)
        for local_i, global_i in enumerate(self.used_assets):
            for local_j, global_j in enumerate(self.used_assets):
                full[global_i, global_j] = distances[local_i, local_j]
        return full