"""
core/shared_memory.py - Espace Mémoire Partagé (EMP) et ElitePool
"""

from typing import Dict, List, Tuple, Optional, Any
import copy
import random
from core.environment import Task

class Solution:
    def __init__(self, sequences: Dict[Tuple[int, int], List[Task]], 
                 fitness: float, agent_id: str = None):
        self.sequences = sequences
        self.fitness = fitness
        self.agent_id = agent_id
        self.iteration_found = 0
    
    def __lt__(self, other):
        return self.fitness < other.fitness

class SharedMemoryPool:
    def __init__(self, max_size: int = 20, min_distance: int = 2, diversity_threshold: float = 0.3):
        self.max_size = max_size
        self.min_distance = min_distance
        self.diversity_threshold = diversity_threshold
        self.solutions: List[Solution] = []
        
        # Stats
        self.insertions = 0
        self.rejections_duplicate = 0
        self.rejections_diversity = 0
        self.replacements = 0
    
    def calculate_distance(self, sol1: Dict, sol2: Dict) -> int:
        """
        Distance de Hamming adaptée : Compte combien de tâches sont à des positions différentes.
        """
        distance = 0
        all_keys = set(sol1.keys()) | set(sol2.keys())
        
        for key in all_keys:
            tasks1 = sol1.get(key, [])
            tasks2 = sol2.get(key, [])
            
            # Si longueurs différentes, c'est une grosse différence
            diff_len = abs(len(tasks1) - len(tasks2))
            distance += diff_len
            
            # Comparaison position par position pour la longueur commune
            min_len = min(len(tasks1), len(tasks2))
            for k in range(min_len):
                # On compare l'ID du patient. Si l'ordre change, distance augmente.
                if tasks1[k].i != tasks2[k].i:
                    distance += 1
                    
        return distance
    
    def insert(self, solution: Solution, iteration: int = 0) -> bool:
        solution.iteration_found = iteration
        
        # 1. Doublon exact
        for s in self.solutions:
            if self.calculate_distance(solution.sequences, s.sequences) == 0:
                self.rejections_duplicate += 1
                return False
        
        # 2. Diversité
        # On compte combien de solutions sont "proches" (< min_distance)
        close_solutions = 0
        for s in self.solutions:
            if self.calculate_distance(solution.sequences, s.sequences) < self.min_distance:
                close_solutions += 1
        
        is_diverse = True
        if len(self.solutions) > 0:
            ratio_close = close_solutions / len(self.solutions)
            if ratio_close > (1 - self.diversity_threshold): # Trop de voisins proches
                is_diverse = False
        
        # Logique d'insertion (Algorithme 6 du PDF)
        if len(self.solutions) < self.max_size:
            if is_diverse:
                self.solutions.append(solution)
                self.insertions += 1
                return True
            else:
                # Si non diverse mais excellente fitness, on peut considérer (Aspiration)
                best_fit = self.get_best_fitness()
                if solution.fitness < best_fit:
                    self.solutions.append(solution)
                    self.insertions += 1
                    return True
                self.rejections_diversity += 1
                return False
        else:
            # Pool plein
            worst_fit = self.get_worst_fitness()
            if solution.fitness < worst_fit and is_diverse:
                # Remplacer le pire
                self._remove_worst()
                self.solutions.append(solution)
                self.replacements += 1
                return True
            elif solution.fitness < worst_fit:
                 # Pas diverse mais améliore le pire -> on peut remplacer le pire
                self._remove_worst()
                self.solutions.append(solution)
                self.replacements += 1
                return True
                
            self.rejections_diversity += 1
            return False

    def _remove_worst(self):
        if not self.solutions: return
        self.solutions.sort(key=lambda s: s.fitness)
        self.solutions.pop() # Retire le dernier (le plus grand fitness = le pire makespan)

    def get_best_fitness(self):
        if not self.solutions: return float('inf')
        return min(s.fitness for s in self.solutions)

    def get_worst_fitness(self):
        if not self.solutions: return float('inf')
        return max(s.fitness for s in self.solutions)

    def get_statistics(self):
        return {
            'size': len(self.solutions),
            'global_best_fitness': self.get_best_fitness(), # Clé standardisée
            'insertions': self.insertions,
            'rejections_duplicate': self.rejections_duplicate,
            'rejections_diversity': self.rejections_diversity
        }

class ElitePool(SharedMemoryPool):
    """
    Pool qui ne garde que les meilleures solutions (mode Compétitif/Ennemis).
    """
    def __init__(self, max_size: int = 5):
        # On désactive la diversité pour ne garder que la performance pure
        super().__init__(max_size=max_size, min_distance=0, diversity_threshold=0.0)
    
    def insert(self, solution: Solution, iteration: int = 0) -> bool:
        """Insertion simplifiée basée uniquement sur le fitness."""
        solution.iteration_found = iteration
        
        # Vérification doublon fitness (simple) pour éviter surcharge
        for s in self.solutions:
            if s.fitness == solution.fitness:
                # Si même fitness, on vérifie si c'est exactement la même solution
                if self.calculate_distance(solution.sequences, s.sequences) == 0:
                    self.rejections_duplicate += 1
                    return False

        # Si on a de la place
        if len(self.solutions) < self.max_size:
            self.solutions.append(solution)
            self.insertions += 1
            self.solutions.sort(key=lambda s: s.fitness)
            return True
        
        # Si plein, on compare au pire
        worst_fit = self.solutions[-1].fitness
        if solution.fitness < worst_fit:
            self.solutions.pop() # Enlève le pire (dernier car trié)
            self.solutions.append(solution)
            self.solutions.sort(key=lambda s: s.fitness)
            self.replacements += 1
            return True
            
        return False
