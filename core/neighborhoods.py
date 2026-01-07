"""
core/neighborhoods.py - Fonctions de voisinage adaptées aux contraintes strictes.
"""

from typing import Dict, List, Tuple, Optional
import random
import copy
from core.environment import Task  # Import centralisé pour éviter les duplications

class NeighborhoodFunction:
    """Classe de base pour les fonctions de voisinage."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        """Génère une solution voisine."""
        raise NotImplementedError

class NeighborhoodA(NeighborhoodFunction):
    """
    Voisinage A: Réaffectation d'une tâche à un autre membre du personnel.
    ATTENTION: Avec les contraintes strictes de l'image (C1 = Skill 1),
    ce mouvement est souvent impossible car on ne peut pas changer la compétence requise.
    """
    def __init__(self):
        super().__init__("A - Task Reassignment (Resource Swap)")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        # Dans ce problème strict, une tâche C1 ne peut être faite que par Skill 1.
        # On ne peut changer de 'Skill' que si plusieurs ressources ont la même compétence.
        # Ici, Skill ID = Resource ID. Donc A est invalide pour ce dataset strict.
        # On retourne None pour signaler qu'aucun voisin valide n'existe.
        return None

class NeighborhoodB(NeighborhoodFunction):
    """
    Voisinage B: Réaffectation de tâches successives d'un patient.
    Même restriction que A : changer de Skill est interdit par la table de compétences.
    """
    def __init__(self):
        super().__init__("B - Successive Tasks Reassignment")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        # Impossible sous contraintes strictes de compétences uniques.
        return None

class NeighborhoodC(NeighborhoodFunction):
    """
    Voisinage C: Insertion à une autre position dans le MÊME planning (Skill).
    VALIDE : On change l'ordre de passage chez le médecin, pas le médecin lui-même.
    """
    def __init__(self):
        super().__init__("C - Task Insertion Same Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        # On cherche les files d'attente ayant au moins 2 tâches
        valid_keys = [key for key, tasks in solution.items() if len(tasks) >= 2]
        
        if not valid_keys:
            return None
        
        # On choisit une file au hasard (ex: la file du Skill 1 à l'étape 2)
        key = random.choice(valid_keys)
        tasks = new_solution[key]
        
        # On déplace une tâche dans cette file
        idx_from = random.randint(0, len(tasks) - 1)
        task = tasks.pop(idx_from)
        
        idx_to = random.randint(0, len(tasks)) # Peut être inséré à la fin
        tasks.insert(idx_to, task)
        
        return new_solution

class NeighborhoodD(NeighborhoodFunction):
    """
    Voisinage D: Échange entre différents médecins.
    INVALIDE pour contraintes strictes.
    """
    def __init__(self):
        super().__init__("D - Swap Different Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        return None

class NeighborhoodE(NeighborhoodFunction):
    """
    Voisinage E: Swap (échange) de deux tâches dans le MÊME planning.
    VALIDE : On intervertit deux patients chez le même médecin.
    """
    def __init__(self):
        super().__init__("E - Swap Same Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        valid_keys = [key for key, tasks in solution.items() if len(tasks) >= 2]
        
        if not valid_keys:
            return None
        
        key = random.choice(valid_keys)
        tasks = new_solution[key]
        
        i1 = random.randint(0, len(tasks) - 1)
        i2 = random.randint(0, len(tasks) - 1)
        while i1 == i2:
            i2 = random.randint(0, len(tasks) - 1)
            
        # Swap
        tasks[i1], tasks[i2] = tasks[i2], tasks[i1]
        
        return new_solution

class NeighborhoodManager:
    """Gère les voisinages et filtre ceux qui sont inactifs."""
    def __init__(self):
        self.neighborhoods = {
            'A': NeighborhoodA(),
            'B': NeighborhoodB(),
            'C': NeighborhoodC(),
            'D': NeighborhoodD(),
            'E': NeighborhoodE(),
        }
        self.neighborhood_names = list(self.neighborhoods.keys())
    
    def generate_neighbor(self, solution, name, skills, max_ops):
        if name not in self.neighborhoods:
            return None
        return self.neighborhoods[name].generate(solution, skills, max_ops)
        
    def generate_all_neighbors(self, solution, skills, max_ops, n_per_neighborhood=5):
        # Optimisation : on ne génère pas pour A, B, D qui renvoient toujours None
        active_neighborhoods = ['C', 'E']
        all_neighbors = []
        for name in active_neighborhoods:
            for _ in range(n_per_neighborhood):
                neigh = self.generate_neighbor(solution, name, skills, max_ops)
                if neigh:
                    all_neighbors.append((neigh, name))
        return all_neighbors
