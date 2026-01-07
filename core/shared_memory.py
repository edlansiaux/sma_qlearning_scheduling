"""
Shared Memory Pool Module (EMP - Espace Mémoire Partagé)
Basé sur le diaporama: Algorithme 6 - Contrôler la diversité de l'EMP

L'EMP stocke les bonnes solutions trouvées par les agents.
Un mécanisme de diversité assure que les solutions stockées sont 
suffisamment différentes les unes des autres.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple
import copy
import heapq

Task = namedtuple("Task", ["i", "j", "s", "p"])


class Solution:
    """Classe pour encapsuler une solution avec son évaluation."""
    
    def __init__(self, sequences: Dict[Tuple[int, int], List[Task]], 
                 fitness: float, agent_id: str = None):
        """
        Args:
            sequences: Dictionnaire des séquences de tâches
            fitness: Valeur de fitness (makespan)
            agent_id: Identifiant de l'agent qui a trouvé cette solution
        """
        self.sequences = sequences
        self.fitness = fitness
        self.agent_id = agent_id
        self.iteration_found = 0
    
    def __lt__(self, other):
        """Comparaison pour le heap (min-heap sur fitness)."""
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        """Égalité basée sur les séquences."""
        if not isinstance(other, Solution):
            return False
        return self.sequences == other.sequences


class SharedMemoryPool:
    """
    Espace Mémoire Partagé (EMP) pour les agents.
    
    Implémente l'Algorithme 6 du diaporama:
    - Contrôle de la diversité des solutions
    - Insertion conditionnelle basée sur la distance
    - Maintien d'un pool de taille limitée
    
    Paramètres:
        R: Distance minimale pour considérer une solution différente
        DT: Seuil de diversité (ratio solutions différentes / total)
        max_size: Taille maximale du pool
    """
    
    def __init__(self, max_size: int = 20, 
                 min_distance: int = 5, 
                 diversity_threshold: float = 0.5):
        """
        Args:
            max_size: Nombre maximum de solutions dans l'EMP
            min_distance: Distance minimale (R) entre deux solutions
            diversity_threshold: Seuil de diversité (DT)
        """
        self.max_size = max_size
        self.min_distance = min_distance
        self.diversity_threshold = diversity_threshold
        
        # Pool de solutions
        self.solutions: List[Solution] = []
        
        # Statistiques
        self.insertions = 0
        self.rejections_duplicate = 0
        self.rejections_diversity = 0
        self.replacements = 0
    
    def calculate_distance(self, sol1: Dict[Tuple[int, int], List[Task]], 
                          sol2: Dict[Tuple[int, int], List[Task]]) -> int:
        """
        Calcule la distance entre deux solutions.
        Distance = nombre de créneaux contenant des tâches différentes.
        
        Basé sur le diaporama: La distance est définie comme le nombre 
        de différences dans les plannings des patients.
        """
        distance = 0
        
        all_keys = set(sol1.keys()) | set(sol2.keys())
        
        for key in all_keys:
            tasks1 = sol1.get(key, [])
            tasks2 = sol2.get(key, [])
            
            # Comparer les patients à chaque position
            max_len = max(len(tasks1), len(tasks2))
            for pos in range(max_len):
                patient1 = tasks1[pos].i if pos < len(tasks1) else -1
                patient2 = tasks2[pos].i if pos < len(tasks2) else -1
                
                if patient1 != patient2:
                    distance += 1
        
        return distance
    
    def count_different_solutions(self, new_solution: Dict[Tuple[int, int], List[Task]]) -> int:
        """
        Compte le nombre de solutions dans l'EMP qui sont différentes de new_solution.
        Une solution est considérée différente si distance >= min_distance.
        """
        count = 0
        for sol in self.solutions:
            dist = self.calculate_distance(new_solution, sol.sequences)
            if dist >= self.min_distance:
                count += 1
        return count
    
    def insert(self, solution: Solution, iteration: int = 0) -> bool:
        """
        Tente d'insérer une solution dans l'EMP.
        Implémente l'Algorithme 6 du diaporama.
        
        Args:
            solution: Solution à insérer
            iteration: Numéro d'itération actuel
        
        Returns:
            True si la solution a été insérée, False sinon
        """
        solution.iteration_found = iteration
        
        # Étape 1: Compter les solutions différentes
        d = self.count_different_solutions(solution.sequences)
        nb = len(self.solutions)
        
        # Étape 2: Vérifier si la solution existe déjà (distance = 0)
        for existing in self.solutions:
            dist = self.calculate_distance(solution.sequences, existing.sequences)
            if dist == 0:
                self.rejections_duplicate += 1
                return False  # Solution déjà présente
        
        # Étape 3: Vérifier le critère de diversité
        if nb > 0 and d / nb < self.diversity_threshold:
            # La solution n'est pas assez différente
            # Vérifier si elle améliore quand même la pire solution
            if solution.fitness < self.get_worst_fitness():
                # Remplacer la pire solution si le pool est plein
                if nb >= self.max_size:
                    self._remove_worst()
                    self.replacements += 1
                self.solutions.append(solution)
                self.insertions += 1
                return True
            else:
                self.rejections_diversity += 1
                return False
        else:
            # La solution est suffisamment différente
            if nb < self.max_size:
                # Il y a de la place
                self.solutions.append(solution)
                self.insertions += 1
                return True
            else:
                # Pool plein: vérifier si meilleure que la pire
                if solution.fitness < self.get_worst_fitness():
                    self._remove_worst()
                    self.solutions.append(solution)
                    self.replacements += 1
                    self.insertions += 1
                    return True
                else:
                    self.rejections_diversity += 1
                    return False
    
    def _remove_worst(self):
        """Supprime la pire solution du pool."""
        if not self.solutions:
            return
        worst_idx = max(range(len(self.solutions)), 
                       key=lambda i: self.solutions[i].fitness)
        self.solutions.pop(worst_idx)
    
    def get_best_solution(self) -> Optional[Solution]:
        """Retourne la meilleure solution du pool."""
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: s.fitness)
    
    def get_worst_fitness(self) -> float:
        """Retourne la pire fitness du pool."""
        if not self.solutions:
            return float('inf')
        return max(s.fitness for s in self.solutions)
    
    def get_best_fitness(self) -> float:
        """Retourne la meilleure fitness du pool."""
        if not self.solutions:
            return float('inf')
        return min(s.fitness for s in self.solutions)
    
    def get_random_solution(self) -> Optional[Solution]:
        """Retourne une solution aléatoire du pool."""
        import random
        if not self.solutions:
            return None
        return random.choice(self.solutions)
    
    def get_diverse_solution(self, reference: Dict[Tuple[int, int], List[Task]]) -> Optional[Solution]:
        """
        Retourne la solution la plus différente de la référence.
        Utile pour la diversification de la recherche.
        """
        if not self.solutions:
            return None
        
        max_dist = -1
        most_diverse = None
        
        for sol in self.solutions:
            dist = self.calculate_distance(reference, sol.sequences)
            if dist > max_dist:
                max_dist = dist
                most_diverse = sol
        
        return most_diverse
    
    def get_top_k(self, k: int = 5) -> List[Solution]:
        """Retourne les k meilleures solutions."""
        sorted_solutions = sorted(self.solutions, key=lambda s: s.fitness)
        return sorted_solutions[:k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'EMP."""
        return {
            'size': len(self.solutions),
            'max_size': self.max_size,
            'best_fitness': self.get_best_fitness(),
            'worst_fitness': self.get_worst_fitness(),
            'avg_fitness': sum(s.fitness for s in self.solutions) / len(self.solutions) if self.solutions else 0,
            'insertions': self.insertions,
            'rejections_duplicate': self.rejections_duplicate,
            'rejections_diversity': self.rejections_diversity,
            'replacements': self.replacements,
        }
    
    def get_diversity_matrix(self) -> List[List[int]]:
        """
        Calcule la matrice des distances entre toutes les solutions.
        Utile pour visualiser la diversité.
        """
        n = len(self.solutions)
        matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.calculate_distance(
                    self.solutions[i].sequences, 
                    self.solutions[j].sequences
                )
                matrix[i][j] = dist
                matrix[j][i] = dist
        
        return matrix
    
    def clear(self):
        """Vide le pool."""
        self.solutions = []
        self.insertions = 0
        self.rejections_duplicate = 0
        self.rejections_diversity = 0
        self.replacements = 0


class ElitePool(SharedMemoryPool):
    """
    Pool élite spécialisé qui maintient uniquement les meilleures solutions.
    Utilisé pour le mode "Ennemis" où seules les meilleures valeurs sont partagées.
    """
    
    def __init__(self, max_size: int = 10):
        super().__init__(max_size=max_size, min_distance=0, diversity_threshold=0.0)
    
    def insert(self, solution: Solution, iteration: int = 0) -> bool:
        """Insère si c'est une des meilleures solutions."""
        solution.iteration_found = iteration
        
        # Vérifier si la solution est identique à une existante
        for existing in self.solutions:
            if self.calculate_distance(solution.sequences, existing.sequences) == 0:
                return False
        
        if len(self.solutions) < self.max_size:
            self.solutions.append(solution)
            self.insertions += 1
            return True
        elif solution.fitness < self.get_worst_fitness():
            self._remove_worst()
            self.solutions.append(solution)
            self.replacements += 1
            return True
        
        return False
