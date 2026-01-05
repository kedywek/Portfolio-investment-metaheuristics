import numpy as np

class PreAssignmentMixin:
    def __init__(self, **kwargs):
        self.pre_ass = kwargs.get('pre_assignment', True)
        # Zbalansowany próg startowy (0.78)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.78)
        self.excluded_assets = []
        self.used_assets = []
        self.full_n = None

    def quick_pre_assignment(self):
        D = self.d
        # ZBALANSOWANY LIMIT: Zostawiamy min. 300 aktywów lub 3*k
        min_required = max(self.k * 3, 300)
        max_exclusions = self.n - min_required
        
        if max_exclusions <= 0:
            self.pre_ass = False
            self.excluded_assets = []
            self.used_assets = list(range(self.n))
            return

        # Obliczanie podobieństwa cosinusowego
        col_norms = np.linalg.norm(D, axis=0)
        safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
        X = (D / safe_norms).T
        S = np.clip((X @ X.T), -1.0, 1.0)

        # Sortowanie po zwrocie - zachowujemy te, które najlepiej zarabiają
        sorted_indices = sorted(range(self.n), key=lambda x: -self.r[x])
        
        # Pętla dopasowująca próg, jeśli usuniemy za dużo aktywów
        while not self.run_quick_pa(S, max_exclusions, sorted_indices, self.similarity_threshold):
            self.similarity_threshold += 0.01
            if self.similarity_threshold > 0.99:
                self.pre_ass = False
                self.excluded_assets = []
                self.used_assets = list(range(self.n))
                break

    def run_quick_pa(self, S, max_exclusions, sorted_indices, threshold):
        """Metoda wykonująca faktyczną selekcję aktywów."""
        excluded = set()
        for idx, i in enumerate(sorted_indices):
            if i in excluded:
                continue
            # Porównujemy tylko z aktywami o gorszym zwrocie
            for j in sorted_indices[idx + 1:]:
                if j in excluded:
                    continue
                # Jeśli aktywa są zbyt podobne, usuwamy to z mniejszym zwrotem
                if S[i, j] > threshold:
                    excluded.add(j)
                    # Bezpiecznik: jeśli usunęliśmy za dużo, przerywamy i podnosimy próg
                    if len(excluded) >= max_exclusions:
                        return False

        self.excluded_assets = sorted(list(excluded))
        self.used_assets = [i for i in range(self.n) if i not in excluded]
        return True

    def apply_pre_assignment(self):
        self.full_n = self.n
        if self.pre_ass:
            self.quick_pre_assignment()
            
        if self.pre_ass and len(self.excluded_assets) > 0:
            self.n = len(self.used_assets)
            self.r = np.array([self.r[i] for i in self.used_assets])
            
            # Szybka redukcja macierzy dystansów
            temp_d = self.d[self.used_assets, :]
            self.d = temp_d[:, self.used_assets]

    def expand_weights(self, weights):
        """Mapowanie wag z powrotem do pełnego wymiaru n=1000."""
        if self.full_n is None or not self.pre_ass:
            return weights
        full_vector = np.zeros(self.full_n, dtype=float)
        for local_idx, global_idx in enumerate(self.used_assets):
            full_vector[global_idx] = weights[local_idx]
        return full_vector
    
    def expand_distances(self, distances):
        if self.full_n is None or not self.pre_ass:
            return distances
        full = np.zeros((self.full_n, self.full_n), dtype=float)
        for local_i, global_i in enumerate(self.used_assets):
            for local_j, global_j in enumerate(self.used_assets):
                full[global_i, global_j] = distances[local_i, local_j]
        return full