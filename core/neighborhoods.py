"""
core/neighborhoods.py - 5 fonctions de voisinage (A–E) conformes à l'Algorithme 3 du diaporama.

Adaptées au modèle 1-ressource-par-compétence :
  A : Réaffectation de position (insertion aléatoire dans la même file)
  B : Déplacement groupé de tâches consécutives du même patient
  C : Décalage temporel (insertion à position voisine, shift ∈ [-6, 6])
  D : Échange inter-files (swap entre deux files (skill, stage) différentes)
  E : Échange intra-file (swap de deux tâches dans la même file)
"""

from typing import Dict, List, Tuple, Optional
import random
import copy
from core.environment import Task


class NeighborhoodFunction:
    """Classe de base pour les fonctions de voisinage."""

    def __init__(self, name: str):
        self.name = name

    def generate(
        self,
        solution: Dict[Tuple[int, int], List[Task]],
        skills: List[int],
        max_ops: int,
    ) -> Optional[Dict[Tuple[int, int], List[Task]]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# A – Réaffectation de position (Task Reassignment)
#     On retire une tâche de sa file et on la réinsère à une position
#     aléatoire dans la même file. C'est l'analogue de « réaffecter à un
#     autre staff » quand il n'y a qu'une seule ressource par compétence.
# ---------------------------------------------------------------------------
class NeighborhoodA(NeighborhoodFunction):
    def __init__(self):
        super().__init__("A - Task Reassignment (Position)")

    def generate(self, solution, skills, max_ops):
        new_solution = copy.deepcopy(solution)
        valid_keys = [k for k, v in new_solution.items() if len(v) >= 2]
        if not valid_keys:
            return None

        key = random.choice(valid_keys)
        tasks = new_solution[key]

        idx_from = random.randint(0, len(tasks) - 1)
        task = tasks.pop(idx_from)

        # Insertion à une position *différente*
        possible = [p for p in range(len(tasks) + 1) if p != idx_from]
        if not possible:
            tasks.insert(idx_from, task)
            return new_solution
        idx_to = random.choice(possible)
        tasks.insert(idx_to, task)
        return new_solution


# ---------------------------------------------------------------------------
# B – Déplacement groupé (Successive Tasks Reassignment)
#     Prend un bloc de tâches consécutives du même patient et les déplace
#     en bloc à une autre position dans la même file.
# ---------------------------------------------------------------------------
class NeighborhoodB(NeighborhoodFunction):
    def __init__(self):
        super().__init__("B - Successive Tasks Block Move")

    def generate(self, solution, skills, max_ops):
        new_solution = copy.deepcopy(solution)
        valid_keys = [k for k, v in new_solution.items() if len(v) >= 3]
        if not valid_keys:
            # Fallback : si pas assez de tâches, on fait un simple swap
            valid_keys = [k for k, v in new_solution.items() if len(v) >= 2]
            if not valid_keys:
                return None
            key = random.choice(valid_keys)
            tasks = new_solution[key]
            i, j = random.sample(range(len(tasks)), 2)
            tasks[i], tasks[j] = tasks[j], tasks[i]
            return new_solution

        key = random.choice(valid_keys)
        tasks = new_solution[key]
        n = len(tasks)

        # Chercher un bloc de tâches du même patient
        start_idx = random.randint(0, n - 2)
        patient_id = tasks[start_idx].i
        end_idx = start_idx + 1
        while end_idx < n and tasks[end_idx].i == patient_id:
            end_idx += 1
        # Extraire le bloc [start_idx : end_idx]
        block = tasks[start_idx:end_idx]
        remaining = tasks[:start_idx] + tasks[end_idx:]

        if not remaining:
            return new_solution

        # Insérer le bloc à une nouvelle position
        insert_pos = random.randint(0, len(remaining))
        new_tasks = remaining[:insert_pos] + block + remaining[insert_pos:]
        new_solution[key] = new_tasks
        return new_solution


# ---------------------------------------------------------------------------
# C – Décalage / Insertion locale (Temporal Shift)
#     Déplace une tâche de quelques positions (shift ∈ [-6, 6]) dans sa file.
# ---------------------------------------------------------------------------
class NeighborhoodC(NeighborhoodFunction):
    def __init__(self):
        super().__init__("C - Task Insertion (Shift)")

    def generate(self, solution, skills, max_ops):
        new_solution = copy.deepcopy(solution)
        valid_keys = [k for k, v in new_solution.items() if len(v) >= 2]
        if not valid_keys:
            return None

        key = random.choice(valid_keys)
        tasks = new_solution[key]
        n = len(tasks)

        idx_from = random.randint(0, n - 1)
        task = tasks.pop(idx_from)

        shift = random.randint(-6, 6)
        idx_to = max(0, min(len(tasks), idx_from + shift))
        tasks.insert(idx_to, task)
        return new_solution


# ---------------------------------------------------------------------------
# D – Échange inter-files (Swap Different Staff / Cross-Queue Swap)
#     Échange deux tâches entre deux files (skill, stage) différentes,
#     uniquement si les tâches partagent la même compétence (skill).
#     En pratique cela revient à échanger l'ordre d'exécution entre deux
#     étapes (stages) de la même compétence.
# ---------------------------------------------------------------------------
class NeighborhoodD(NeighborhoodFunction):
    def __init__(self):
        super().__init__("D - Swap Between Queues (Cross-Stage)")

    def generate(self, solution, skills, max_ops):
        new_solution = copy.deepcopy(solution)
        non_empty = [k for k, v in new_solution.items() if len(v) >= 1]
        if len(non_empty) < 2:
            return None

        # Grouper les clés par skill
        by_skill: Dict[int, List[Tuple[int, int]]] = {}
        for k in non_empty:
            s = k[0]
            by_skill.setdefault(s, []).append(k)

        # Ne garder que les skills qui ont ≥ 2 files (stages différents)
        multi = {s: keys for s, keys in by_skill.items() if len(keys) >= 2}
        if not multi:
            # Pas d'échange inter-stage possible → faire un swap intra-file quelconque
            key = random.choice(non_empty)
            tasks = new_solution[key]
            if len(tasks) >= 2:
                i, j = random.sample(range(len(tasks)), 2)
                tasks[i], tasks[j] = tasks[j], tasks[i]
            return new_solution

        skill = random.choice(list(multi.keys()))
        key1, key2 = random.sample(multi[skill], 2)

        tasks1 = new_solution[key1]
        tasks2 = new_solution[key2]

        idx1 = random.randint(0, len(tasks1) - 1)
        idx2 = random.randint(0, len(tasks2) - 1)

        # Échanger les tâches (avec mise à jour du champ 'j' pour maintenir la cohérence)
        t1 = tasks1[idx1]
        t2 = tasks2[idx2]

        # Recréer les tâches avec le bon stage (j)
        tasks1[idx1] = Task(i=t2.i, j=t1.j, s=t1.s, p=t2.p)
        tasks2[idx2] = Task(i=t1.i, j=t2.j, s=t2.s, p=t1.p)

        return new_solution


# ---------------------------------------------------------------------------
# E – Échange intra-file (Swap Same Staff)
#     Échange deux tâches dans la même file (même skill + même stage).
# ---------------------------------------------------------------------------
class NeighborhoodE(NeighborhoodFunction):
    def __init__(self):
        super().__init__("E - Swap Same Queue")

    def generate(self, solution, skills, max_ops):
        new_solution = copy.deepcopy(solution)
        valid_keys = [k for k, v in new_solution.items() if len(v) >= 2]
        if not valid_keys:
            return None

        key = random.choice(valid_keys)
        tasks = new_solution[key]
        i1, i2 = random.sample(range(len(tasks)), 2)
        tasks[i1], tasks[i2] = tasks[i2], tasks[i1]
        return new_solution


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class NeighborhoodManager:
    """Gère les 5 voisinages A–E."""

    def __init__(self):
        self.neighborhoods = {
            "A": NeighborhoodA(),
            "B": NeighborhoodB(),
            "C": NeighborhoodC(),
            "D": NeighborhoodD(),
            "E": NeighborhoodE(),
        }
        self.active_neighborhoods = list(self.neighborhoods.keys())

    def generate_neighbor(self, solution, name, skills, max_ops):
        if name not in self.neighborhoods:
            return None
        return self.neighborhoods[name].generate(solution, skills, max_ops)

    def generate_all_neighbors(self, solution, skills, max_ops, n_per_neighborhood=5):
        all_neighbors = []
        for name in self.active_neighborhoods:
            for _ in range(n_per_neighborhood):
                neigh = self.generate_neighbor(solution, name, skills, max_ops)
                if neigh:
                    all_neighbors.append((neigh, name))
        return all_neighbors
