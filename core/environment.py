"""
core/environment.py - Modèle Hypercubique, Temps Discret (5 min) et Contraintes.
"""
from typing import Dict, List, Tuple, NamedTuple
import copy
import random

# Configuration globale : Discrétisation en intervalles de 5 minutes [cite: 31]
SLOT_DURATION = 5  
# Horizon suffisant pour la simulation
HORIZON_SLOTS = 200 

class Task(NamedTuple):
    id: int          
    patient_id: int
    op_order: int    # Ordre de l'opération (j)
    skill_req: int   # Compétence requise (médecin/ressource)
    duration: int    # En minutes

class Solution:
    """
    Représente une solution dans le modèle hypercubique[cite: 9].
    Structure: dictionnaire mapping (Patient, Op) -> (Staff_ID, Start_Slot)
    """
    def __init__(self):
        self.schedule: Dict[Tuple[int, int], Tuple[int, int]] = {} 
        self.fitness: float = float('inf')
        self.is_valid: bool = False

class SchedulingEnvironment:
    def __init__(self, data: Dict, skills: List[int], num_patients: int):
        self.data = data 
        self.skills = skills
        self.num_patients = num_patients
        self.tasks: List[Task] = []
        self.tasks_map: Dict[Tuple[int, int], Task] = {}
        self._parse_data()

    def _parse_data(self):
        tid = 0
        for pid, ops in self.data.items():
            for order, task_list in ops.items():
                if task_list:
                    # On suppose une tâche principale par étape pour le modèle simplifié
                    skill, dur = task_list[0]
                    t = Task(tid, pid, order, skill, dur)
                    self.tasks.append(t)
                    self.tasks_map[(pid, order)] = t
                    tid += 1

    def duration_to_slots(self, minutes: int) -> int:
        """Convertit la durée en nombre de slots de 5 minutes."""
        return (minutes + SLOT_DURATION - 1) // SLOT_DURATION

    def build_initial_solution(self) -> Solution:
        """Génère une solution aléatoire valide respectant les contraintes de précédence."""
        sol = Solution()
        # Disponibilité des ressources (Staff -> premier slot libre)
        staff_availability = {s: 0 for s in self.skills}
        # Disponibilité des patients (Patient -> premier slot libre)
        patient_availability = {p: 0 for p in range(1, self.num_patients + 1)}

        # Tri aléatoire des tâches pour la diversité initiale
        all_tasks = list(self.tasks)
        random.shuffle(all_tasks)
        # Tri partiel pour respecter grossièrement l'ordre des opérations
        all_tasks.sort(key=lambda t: t.op_order)
        
        for t in all_tasks:
            # Allocation d'une ressource (ici simplifiée : staff_req est l'ID du staff)
            staff_id = t.skill_req 
            
            duration_slots = self.duration_to_slots(t.duration)
            
            # Début au max des disponibilités (Patient dispo ET Médecin dispo)
            start_time = max(patient_availability[t.patient_id], staff_availability.get(staff_id, 0))
            
            sol.schedule[(t.patient_id, t.op_order)] = (staff_id, start_time)
            
            finish_time = start_time + duration_slots
            patient_availability[t.patient_id] = finish_time
            staff_availability[staff_id] = finish_time
            
        self.evaluate(sol)
        return sol

    def evaluate(self, solution: Solution) -> float:
        """Calcule le Makespan (Cmax) en slots."""
        if not solution.schedule:
            solution.fitness = float('inf')
            return float('inf')
            
        max_slot = 0
        for (pid, op), (staff, start) in solution.schedule.items():
            task = self.tasks_map.get((pid, op))
            if task:
                end = start + self.duration_to_slots(task.duration)
                if end > max_slot:
                    max_slot = end
        
        solution.fitness = max_slot
        solution.is_valid = True 
        return max_slot

    def copy_solution(self, solution: Solution) -> Solution:
        new_sol = Solution()
        new_sol.schedule = solution.schedule.copy()
        new_sol.fitness = solution.fitness
        new_sol.is_valid = solution.is_valid
        return new_sol

# Générateur de données factices pour le benchmark
def generate_random_data(num_patients=5, max_ops=3, skills=[1,2,3,4]):
    data = {}
    for i in range(1, num_patients+1):
        data[i] = {}
        for j in range(1, max_ops+1):
            if random.random() > 0.1: # 90% de chance d'avoir une tâche
                skill = random.choice(skills)
                duration = random.randint(10, 60) # Durée en minutes
                data[i][j] = [(skill, duration)]
    return data
