"""
core/shared_memory.py - EMP et Algorithme 6 (Contrôle de Diversité)[cite: 535].
"""
from typing import List
from core.environment import Solution

class SharedMemory:
    def __init__(self, max_size=20, min_dist=2, diversity_threshold=0.5):
        self.solutions: List[Solution] = []
        self.max_size = max_size # MaxSMS [cite: 1107]
        self.min_dist = min_dist # Paramètre R [cite: 1104]
        self.dt = diversity_threshold # Paramètre DT [cite: 1106]

    def calculate_distance(self, sol1: Solution, sol2: Solution) -> int:
        """
        Calcule la distance matricielle définie dans le PDF[cite: 572, 1104].
        "Nombre de créneaux différents contenant des tâches de soins différentes"
        """
        dist = 0
        all_keys = set(sol1.schedule.keys()) | set(sol2.schedule.keys())
        for k in all_keys:
            v1 = sol1.schedule.get(k)
            v2 = sol2.schedule.get(k)
            # Si l'assignation (Staff ou Temps) diffère
            if v1 != v2: 
                dist += 1
        return dist

    def try_insert(self, cs: Solution) -> bool:
        """
        Implémentation de l'Algorithme 6: Contrôler la diversité de l'EMP[cite: 535].
        """
        nb = len(self.solutions)
        
        # 1. Vérification si la solution existe déjà (distance 0) [cite: 558]
        for s in self.solutions:
            if self.calculate_distance(cs, s) == 0:
                return False

        # 2. Calculer le nombre de solutions 'différentes' (distance >= R) [cite: 1125]
        d_count = 0
        for s in self.solutions:
            if self.calculate_distance(cs, s) >= self.min_dist:
                d_count += 1
        
        # Logique d'insertion adaptative
        ratio = d_count / nb if nb > 0 else 1.0
        inserted = False

        # Si assez de diversité (ratio >= DT) [cite: 1133]
        if ratio >= self.dt:
            # Si EMP non plein, on insère
            if nb < self.max_size:
                self.solutions.append(cs)
                inserted = True
            else:
                # Si plein, on regarde si CS est meilleure que la pire solution de l'EMP [cite: 1134]
                worst_idx = self._get_worst_idx()
                if worst_idx != -1 and cs.fitness < self.solutions[worst_idx].fitness:
                     self.solutions.pop(worst_idx) # Éliminer la plus mauvaise [cite: 1137]
                     self.solutions.append(cs)
                     inserted = True
        else:
            # Pas assez de diversité, insertion uniquement si très performante
            worst_idx = self._get_worst_idx()
            if nb >= self.max_size and worst_idx != -1 and cs.fitness < self.solutions[worst_idx].fitness:
                self.solutions.pop(worst_idx)
                self.solutions.append(cs)
                inserted = True

        if inserted:
            # Toujours garder trié pour faciliter la récupération du meilleur/pire
            self.solutions.sort(key=lambda x: x.fitness)
            
        return inserted

    def _get_worst_idx(self):
        if not self.solutions: return -1
        # On cherche le fitness le plus élevé (minimisation)
        vals = [s.fitness for s in self.solutions]
        return vals.index(max(vals))

    def get_best(self):
        if not self.solutions: return None
        return self.solutions[0]
