import numpy as np

class PreAssignmentMixin:
    def __init__(self, **kwargs):
        self.pre_ass = kwargs.get('pre_assignment', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        self.n_km_init = kwargs.get('n_km_init', 1)
        self.excluded_assets = []
        self.used_assets = []
        self.full_n = None

    def pre_assignment(self):
        from coclust.clustering import SphericalKmeans
        import io
        from contextlib import redirect_stdout, redirect_stderr

        n = self.n
        D = self.d
        r = self.r
        thr = self.similarity_threshold

        col_norms = np.linalg.norm(D, axis=0)
        safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
        X = (D / safe_norms).T

        k_clusters = max(1, int(n / 2))
        km = SphericalKmeans(n_clusters=k_clusters, n_init=self.n_km_init)

        fnull = io.StringIO()
        with redirect_stdout(fnull), redirect_stderr(fnull):
            km.fit(X)
        labels = km.labels_

        clusters = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(idx)

        S = np.clip((X @ X.T), -1.0, 1.0)

        excluded = set()
        for comp in clusters.values():
            if len(comp) <= 1:
                continue
            comp_rs = r[comp]
            keep_local = int(np.argmax(comp_rs))
            keep = comp[keep_local]
            for idx in comp:
                if idx != keep and S[idx, keep] > thr:
                    excluded.add(idx)

        self.excluded_assets = sorted(excluded)
        self.used_assets = [i for i in range(n) if i not in excluded]

    def quick_pre_assignment(self):
        D = self.d
        max_exclusions = self.n - (self.k * 2)
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

    def apply_pre_assignment(self, method='quick'):
        self.full_n = self.n
        if method == 'quick':
            self.quick_pre_assignment()
        else:
            self.pre_assignment()

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