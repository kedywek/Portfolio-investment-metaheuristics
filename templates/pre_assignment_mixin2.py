import numpy as np

class PreAssignmentMixin:
    """
    Moduł wstępnej selekcji aktywów (Dimensionality Reduction).
    Zmniejsza liczbę aktywów N, usuwając te, które są zbyt podobne do lepszych 
    odpowiedników pod kątem zwrotu r_i.
    """
    def __init__(self, **kwargs):
        # Parametry sterujące filtrowaniem
        self.pre_ass = kwargs.get('pre_assignment', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.85)
        self.excluded_assets = [] # Indeksy aktywów wykluczonych
        self.used_assets = []     # Indeksy aktywów zachowanych
        self.full_n = None        # Oryginalna liczba aktywów n

    def quick_pre_assignment(self):
        """
        Główna logika filtrowania oparta na podobieństwie i stopie zwrotu.
        """
        D = self.d
        # Zapewniamy, że zostawimy przynajmniej 2*k aktywów lub 200 (zależnie od skali),
        # aby PSO miało przestrzeń do dywersyfikacji.
        min_required = max(self.k * 2, min(self.n, 100))
        max_exclusions = self.n - min_required
        
        if max_exclusions <= 0:
            self.pre_ass = False
            self.excluded_assets = []
            self.used_assets = list(range(self.n))
            return

        # Obliczamy macierz podobieństwa (Cosine Similarity) na podstawie d_ij.
        # Uwaga: w danych projektowych d_ij=1 oznacza całkowitą odmienność.
        # Przekształcamy to na miarę podobieństwa matematycznego.
        col_norms = np.linalg.norm(D, axis=0)
        safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
        X = (D / safe_norms).T
        S = np.clip((X @ X.T), -1.0, 1.0)

        # Sortujemy aktywa od najlepszego zwrotu r_i.
        sorted_indices = sorted(range(self.n), key=lambda x: -self.r[x])
        
        # Pętla dopasowująca próg podobieństwa, aby nie wykluczyć za dużo
        while not self._run_selection_loop(S, max_exclusions, sorted_indices):
            self.similarity_threshold += 0.05
            if self.similarity_threshold > 0.98:
                self.pre_ass = False
                break

    def _run_selection_loop(self, S, max_exclusions, sorted_indices):
        """Pomocnicza pętla usuwająca duplikaty o gorszych parametrach."""
        excluded = set()
        for idx, i in enumerate(sorted_indices):
            if i in excluded:
                continue
            for j in sorted_indices[idx + 1:]:
                if j in excluded:
                    continue
                # Jeśli aktywo j jest zbyt podobne do i, a i ma lepszy zwrot - wyrzuć j.
                if S[i, j] > self.similarity_threshold:
                    excluded.add(j)
                    if len(excluded) >= max_exclusions:
                        return False # Przekroczono limit usuwania

        self.excluded_assets = sorted(list(excluded))
        self.used_assets = [i for i in range(self.n) if i not in excluded]
        return True

    def apply_pre_assignment(self):
        """Metoda uruchamiana w run(), która faktycznie redukuje macierze r i d."""
        self.full_n = self.n
        if self.pre_ass:
            self.quick_pre_assignment()
            
        if self.pre_ass and len(self.excluded_assets) > 0:
            # Aktualizacja n i danych problemu do mniejszej skali[cite: 32, 116].
            self.n = len(self.used_assets)
            self.r = np.array([self.r[i] for i in self.used_assets])
            
            # Redukcja macierzy dystansów d_ij w obu wymiarach.
            temp_d = self.d[self.used_assets, :]
            self.d = temp_d[:, self.used_assets]

    def expand_weights(self, weights):
        """
        Mapuje lokalne wagi (zredukowane) z powrotem na pełny wektor n aktywów.
        Wymagane, aby spełnić format turniejowy.
        """
        if self.full_n is None or not self.pre_ass:
            return weights
            
        full_vector = np.zeros(self.full_n, dtype=float)
        for local_idx, global_idx in enumerate(self.used_assets):
            full_vector[global_idx] = weights[local_idx]
        return full_vector