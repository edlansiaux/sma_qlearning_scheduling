"""
core/neighborhoods.py - Les 5 fonctions de voisinage (A, B, C, D, E)
selon les définitions exactes du PDF (pages 19-20).
"""
import random
from core.environment import Solution, SchedulingEnvironment

class NeighborhoodManager:
    def __init__(self, env: SchedulingEnvironment):
        self.env = env
        # Liste des voisinages disponibles [cite: 1047]
        self.moves = ['A', 'B', 'C', 'D', 'E']

    def apply_move(self, solution: Solution, move_type: str) -> Solution:
        new_sol = self.env.copy_solution(solution)
        keys = list(new_sol.schedule.keys())
        if not keys: return new_sol

        # Sélection aléatoire d'une tâche source (i, j)
        k1 = random.choice(keys)
        staff1, start1 = new_sol.schedule[k1]
        task1 = self.env.tasks_map[k1]
        
        # --- A: Assignment to different medical staff --- [cite: 1030]
        # Réassignation d'une tâche à un autre personnel médical.
        if move_type == 'A':
            other_staffs = [s for s in self.env.skills if s != staff1]
            if other_staffs:
                new_staff = random.choice(other_staffs)
                new_sol.schedule[k1] = (new_staff, start1)

        # --- B: Successive care tasks assignment --- [cite: 1034]
        # Réassignation de tâches successives d'un patient à un autre staff.
        elif move_type == 'B':
            # Trouve la tâche suivante du même patient
            k_next = (k1[0], k1[1] + 1)
            if k_next in new_sol.schedule:
                other_staffs = [s for s in self.env.skills if s != staff1]
                if other_staffs:
                    new_staff = random.choice(other_staffs)
                    # Change staff pour k1 et k_next
                    new_sol.schedule[k1] = (new_staff, start1)
                    s2, t2 = new_sol.schedule[k_next]
                    new_sol.schedule[k_next] = (new_staff, t2)

        # --- C: Work schedule insertion (Shift/Move) --- [cite: 1036]
        # Déplace la tâche à une autre position dans le planning du MÊME staff.
        elif move_type == 'C':
            shift = random.randint(-6, 6) # Décalage +/- 30 min
            new_start = max(0, start1 + shift)
            new_sol.schedule[k1] = (staff1, new_start)

        # --- D: Swap two care tasks between different medical staff --- [cite: 1041]
        # Échange deux tâches entre deux staffs différents.
        elif move_type == 'D':
            # Trouver une tâche assignée à un staff différent
            candidates = [k for k, v in new_sol.schedule.items() if v[0] != staff1]
            if candidates:
                k2 = random.choice(candidates)
                staff2, start2 = new_sol.schedule[k2]
                
                # Swap des staffs
                new_sol.schedule[k1] = (staff2, start1)
                new_sol.schedule[k2] = (staff1, start2)

        # --- E: Swap between the same medical staff member --- [cite: 1044]
        # Échange la position de tâches dans le plan de travail du MÊME staff.
        elif move_type == 'E':
            candidates = [k for k, v in new_sol.schedule.items() if v[0] == staff1 and k != k1]
            if candidates:
                k2 = random.choice(candidates)
                staff2, start2 = new_sol.schedule[k2]
                
                # Swap des temps de début (positions)
                new_sol.schedule[k1] = (staff1, start2)
                new_sol.schedule[k2] = (staff1, start1)

        # Ré-évaluation pour calculer le nouveau Makespan
        self.env.evaluate(new_sol)
        return new_sol
