"""
Neighborhood Functions Module - Fonctions de voisinage pour la recherche locale
Basé sur le diaporama: 5 fonctions de voisinage (A, B, C, D, E)

A: Réaffectation d'une tâche d'un médecin à un autre (différentes compétences)
B: Réaffectation de tâches successives d'un patient à un autre médecin
C: Insertion d'une tâche à une autre position dans le même planning médecin
D: Échange de deux tâches entre différents médecins (différentes compétences)
E: Échange de position de tâches dans le planning du même médecin
"""

from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import random
import copy

Task = namedtuple("Task", ["i", "j", "s", "p"])


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
    Voisinage A: Réaffectation d'une tâche de soins à un autre membre du personnel
    (insertion multiple dans différents personnels soignants - MID)
    
    Déplace une tâche d'une file (s1, j) vers une autre file (s2, j)
    où s1 != s2 mais même étape j.
    """
    
    def __init__(self):
        super().__init__("A - Task Reassignment Different Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        # Trouver des paires (s1, j) et (s2, j) avec s1 != s2
        valid_moves = []
        for j in range(1, max_ops + 1):
            skill_queues = [(s, key) for key, tasks in solution.items() 
                          if key[1] == j and len(tasks) > 0 for s in [key[0]]]
            
            for s1 in skills:
                key1 = (s1, j)
                if key1 not in solution or len(solution[key1]) == 0:
                    continue
                for s2 in skills:
                    if s1 != s2:
                        key2 = (s2, j)
                        valid_moves.append((key1, key2))
        
        if not valid_moves:
            return None
        
        # Choisir une paire aléatoire
        key1, key2 = random.choice(valid_moves)
        
        if key1 not in new_solution or len(new_solution[key1]) == 0:
            return None
        
        # Prendre une tâche de la première file
        task_idx = random.randint(0, len(new_solution[key1]) - 1)
        task = new_solution[key1].pop(task_idx)
        
        # Créer une nouvelle tâche avec la nouvelle compétence
        new_task = Task(i=task.i, j=task.j, s=key2[0], p=task.p)
        
        # Insérer dans la deuxième file
        if key2 not in new_solution:
            new_solution[key2] = []
        
        insert_pos = random.randint(0, len(new_solution[key2]))
        new_solution[key2].insert(insert_pos, new_task)
        
        return new_solution


class NeighborhoodB(NeighborhoodFunction):
    """
    Voisinage B: Réaffectation de tâches successives d'un patient à un autre médecin
    
    Déplace un bloc de tâches consécutives du même patient vers un autre skill.
    """
    
    def __init__(self):
        super().__init__("B - Successive Tasks Reassignment")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        # Trouver des files avec au moins 2 tâches du même patient consécutives
        valid_moves = []
        for key, tasks in solution.items():
            if len(tasks) < 2:
                continue
            for i in range(len(tasks) - 1):
                if tasks[i].i == tasks[i + 1].i:  # Même patient
                    valid_moves.append((key, i))
        
        if not valid_moves:
            # Fallback: déplacer une seule tâche
            for key, tasks in solution.items():
                if len(tasks) >= 1:
                    valid_moves.append((key, 0))
        
        if not valid_moves:
            return None
        
        key1, start_idx = random.choice(valid_moves)
        s1, j = key1
        
        # Choisir un skill cible différent
        other_skills = [s for s in skills if s != s1]
        if not other_skills:
            return None
        
        s2 = random.choice(other_skills)
        key2 = (s2, j)
        
        # Extraire la tâche
        task = new_solution[key1].pop(start_idx)
        new_task = Task(i=task.i, j=task.j, s=s2, p=task.p)
        
        if key2 not in new_solution:
            new_solution[key2] = []
        
        insert_pos = random.randint(0, len(new_solution[key2]))
        new_solution[key2].insert(insert_pos, new_task)
        
        return new_solution


class NeighborhoodC(NeighborhoodFunction):
    """
    Voisinage C: Insertion d'une tâche à une autre position dans le même planning
    (insertion dans le même personnel soignant - MIS)
    
    Déplace une tâche à une position différente dans la même file (s, j).
    """
    
    def __init__(self):
        super().__init__("C - Task Insertion Same Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        # Trouver des files avec au moins 2 tâches
        valid_keys = [key for key, tasks in solution.items() if len(tasks) >= 2]
        
        if not valid_keys:
            return None
        
        key = random.choice(valid_keys)
        tasks = new_solution[key]
        
        # Choisir une tâche et une nouvelle position
        old_pos = random.randint(0, len(tasks) - 1)
        task = tasks.pop(old_pos)
        
        new_pos = random.randint(0, len(tasks))
        tasks.insert(new_pos, task)
        
        return new_solution


class NeighborhoodD(NeighborhoodFunction):
    """
    Voisinage D: Échange de deux tâches entre différents médecins
    (swap entre différents personnels soignants - SDMS)
    
    Échange une tâche de (s1, j) avec une tâche de (s2, j).
    """
    
    def __init__(self):
        super().__init__("D - Swap Different Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        # Trouver des paires de files avec le même j mais skills différents
        valid_pairs = []
        for j in range(1, max_ops + 1):
            keys_at_j = [(s, j) for s in skills 
                        if (s, j) in solution and len(solution[(s, j)]) > 0]
            
            for i, key1 in enumerate(keys_at_j):
                for key2 in keys_at_j[i + 1:]:
                    valid_pairs.append((key1, key2))
        
        if not valid_pairs:
            return None
        
        key1, key2 = random.choice(valid_pairs)
        
        # Choisir des indices de tâches
        idx1 = random.randint(0, len(new_solution[key1]) - 1)
        idx2 = random.randint(0, len(new_solution[key2]) - 1)
        
        # Échanger
        task1 = new_solution[key1][idx1]
        task2 = new_solution[key2][idx2]
        
        # Mettre à jour les skills des tâches
        new_task1 = Task(i=task1.i, j=task1.j, s=key2[0], p=task1.p)
        new_task2 = Task(i=task2.i, j=task2.j, s=key1[0], p=task2.p)
        
        new_solution[key1][idx1] = new_task2
        new_solution[key2][idx2] = new_task1
        
        return new_solution


class NeighborhoodE(NeighborhoodFunction):
    """
    Voisinage E: Échange de position de tâches dans le planning du même médecin
    (swap dans le même personnel soignant - SSMS)
    
    Échange deux tâches dans la même file (s, j).
    """
    
    def __init__(self):
        super().__init__("E - Swap Same Staff")
    
    def generate(self, solution: Dict[Tuple[int, int], List[Task]], 
                 skills: List[int], max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        new_solution = copy.deepcopy(solution)
        
        # Trouver des files avec au moins 2 tâches
        valid_keys = [key for key, tasks in solution.items() if len(tasks) >= 2]
        
        if not valid_keys:
            return None
        
        key = random.choice(valid_keys)
        tasks = new_solution[key]
        
        # Choisir deux positions différentes
        idx1 = random.randint(0, len(tasks) - 1)
        idx2 = random.randint(0, len(tasks) - 1)
        while idx2 == idx1 and len(tasks) > 1:
            idx2 = random.randint(0, len(tasks) - 1)
        
        # Échanger
        tasks[idx1], tasks[idx2] = tasks[idx2], tasks[idx1]
        
        return new_solution


class NeighborhoodManager:
    """
    Gestionnaire des fonctions de voisinage.
    Permet de générer des voisins en utilisant différentes fonctions.
    """
    
    def __init__(self):
        self.neighborhoods = {
            'A': NeighborhoodA(),
            'B': NeighborhoodB(),
            'C': NeighborhoodC(),
            'D': NeighborhoodD(),
            'E': NeighborhoodE(),
        }
        self.neighborhood_names = list(self.neighborhoods.keys())
    
    def generate_neighbor(self, solution: Dict[Tuple[int, int], List[Task]], 
                         neighborhood_name: str, skills: List[int], 
                         max_ops: int) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        """Génère un voisin en utilisant la fonction spécifiée."""
        if neighborhood_name not in self.neighborhoods:
            raise ValueError(f"Unknown neighborhood: {neighborhood_name}")
        
        return self.neighborhoods[neighborhood_name].generate(solution, skills, max_ops)
    
    def generate_random_neighbor(self, solution: Dict[Tuple[int, int], List[Task]], 
                                skills: List[int], max_ops: int) -> Tuple[Optional[Dict], str]:
        """
        Génère un voisin en utilisant une fonction de voisinage aléatoire.
        
        Returns:
            (solution_voisine, nom_du_voisinage)
        """
        neighborhood_name = random.choice(self.neighborhood_names)
        neighbor = self.generate_neighbor(solution, neighborhood_name, skills, max_ops)
        return neighbor, neighborhood_name
    
    def generate_all_neighbors(self, solution: Dict[Tuple[int, int], List[Task]], 
                              skills: List[int], max_ops: int, 
                              n_per_neighborhood: int = 5) -> List[Tuple[Dict, str]]:
        """
        Génère plusieurs voisins pour chaque fonction de voisinage.
        
        Returns:
            Liste de (solution_voisine, nom_du_voisinage)
        """
        all_neighbors = []
        
        for name in self.neighborhood_names:
            for _ in range(n_per_neighborhood):
                neighbor = self.generate_neighbor(solution, name, skills, max_ops)
                if neighbor is not None:
                    all_neighbors.append((neighbor, name))
        
        return all_neighbors
