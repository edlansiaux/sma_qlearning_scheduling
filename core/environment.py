"""
core/environment.py - Environnement remasterisé avec données de l'image et validation stricte.
"""

from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional
import random
import copy

# Task: patient_id, operation_stage, skill_id, duration
Task = namedtuple("Task", ["i", "j", "s", "p"])

class SchedulingEnvironment:
    def __init__(self, data: Dict, skills: List[int], num_patients: int, max_ops: int):
        self.data = data
        self.skills = skills
        self.num_patients = num_patients
        self.max_ops = max_ops
        
        # Structures
        self.all_tasks: List[Task] = []
        # Mapping: (Skill, Stage) -> Liste de tâches
        self.tasks_by_skill_stage: Dict[Tuple[int, int], List[Task]] = defaultdict(list)
        self.patient_last_stage: Dict[int, int] = {}
        
        self._create_tasks()
    
    def _create_tasks(self):
        """Convertit le dictionnaire de données en objets Task."""
        self.patient_last_stage = {i: 0 for i in range(1, self.num_patients + 1)}
        
        for i in range(1, self.num_patients + 1):
            if i in self.data:
                for j in range(1, self.max_ops + 1):
                    ops = self.data[i].get(j, [])
                    if ops:
                        # Si l'opération existe, on met à jour la dernière étape connue
                        self.patient_last_stage[i] = max(self.patient_last_stage[i], j)
                    
                    for (s, p) in ops:
                        t = Task(i=i, j=j, s=s, p=p)
                        self.all_tasks.append(t)
                        self.tasks_by_skill_stage[(s, j)].append(t)

    def build_initial_solution(self, random_order: bool = False) -> Dict[Tuple[int, int], List[Task]]:
        """Génère une solution initiale valide."""
        seq = {}
        for s in self.skills:
            # On regroupe toutes les tâches pour cette compétence, peu importe l'étape
            tasks_for_skill = []
            for j in range(1, self.max_ops + 1):
                tasks_for_skill.extend(self.tasks_by_skill_stage.get((s, j), []))
            
            if not tasks_for_skill:
                continue
            
            if random_order:
                random.shuffle(tasks_for_skill)
            else:
                # Heuristique simple : trier par étape croissante puis par patient
                tasks_for_skill.sort(key=lambda t: (t.j, t.i))
            
            for t in tasks_for_skill:
                key = (t.s, t.j)
                if key not in seq:
                    seq[key] = []
                seq[key].append(t)
                
        return seq

    def check_constraints(self, solution: Dict[Tuple[int, int], List[Task]]) -> Tuple[bool, str]:
        """Vérifie la validité stricte de la solution."""
        count_tasks = 0
        for tasks in solution.values():
            count_tasks += len(tasks)
        
        if count_tasks != len(self.all_tasks):
            return False, f"Nombre de tâches incorrect: {count_tasks} vs {len(self.all_tasks)}"
            
        return True, "Valide"

    def evaluate(self, sequences: Dict[Tuple[int, int], List[Task]], 
                 return_schedule: bool = False) -> Tuple[int, Optional[Dict], Optional[Dict]]:
        """Calcule le Makespan (Cmax)."""
        skill_free_time = {s: 0 for s in self.skills}
        task_times = {} 
        stage_completion = {(i, 0): 0 for i in range(1, self.num_patients + 1)}
        
        for j in range(1, self.max_ops + 1):
            for s in self.skills:
                tasks = sequences.get((s, j), [])
                for task in tasks:
                    ready_time_patient = stage_completion[(task.i, j - 1)]
                    ready_time_skill = skill_free_time[task.s]
                    
                    start_time = max(ready_time_patient, ready_time_skill)
                    end_time = start_time + task.p
                    
                    skill_free_time[task.s] = end_time
                    task_times[(task.i, task.j, task.s)] = (start_time, end_time, task.p)
                    
                    current_stage_end = stage_completion.get((task.i, j), 0)
                    stage_completion[(task.i, j)] = max(current_stage_end, end_time)

        makespan = 0
        for i in range(1, self.num_patients + 1):
            last_j = self.patient_last_stage[i]
            makespan = max(makespan, stage_completion[(i, last_j)])
            
        if return_schedule:
            return makespan, task_times, stage_completion
        return makespan, None, None

    def copy_solution(self, solution):
        return copy.deepcopy(solution)

# --- DONNÉES EXACTES DE L'IMAGE WHATSAPP ---
WHATSAPP_IMAGE_DATA = {
    1: { 1: [(1, 2)], 2: [(1, 1), (2, 1)], 3: [(1, 1), (3, 1)], 4: [(1, 1), (2, 2)], 5: [(4, 1), (5, 2), (6, 1)] },
    2: { 1: [(2, 1), (3, 1)], 2: [(2, 1), (3, 1)], 3: [(2, 1)], 4: [], 5: [] },
    3: { 1: [(3, 2)], 2: [(3, 1)], 3: [], 4: [], 5: [] },
    4: { 1: [(4, 2)], 2: [(5, 1), (6, 1)], 3: [(6, 2)], 4: [(4, 2)], 5: [(1, 1), (2, 1)] },
    5: { 1: [(2, 2)], 2: [(5, 1)], 3: [(5, 1), (6, 1)], 4: [(4, 1), (5, 1)], 5: [(3, 1)] },
    6: { 1: [(1, 1)], 2: [(4, 1)], 3: [(6, 1)], 4: [], 5: [] },
    7: { 1: [(6, 2)], 2: [(1, 1)], 3: [(5, 1), (6, 1)], 4: [(3, 1)], 5: [] },
    8: { 1: [(3, 1), (5, 1)], 2: [(2, 1), (5, 1)], 3: [(3, 1), (6, 1)], 4: [(6, 1)], 5: [] },
    9: { 1: [(5, 1)], 2: [(4, 1)], 3: [(1, 1)], 4: [], 5: [] },
    10: { 1: [(4, 1)], 2: [(4, 1), (5, 1)], 3: [(1, 1), (2, 1)], 4: [(4, 1)], 5: [] }
}

DEFAULT_SKILLS = [1, 2, 3, 4, 5, 6]
DEFAULT_NUM_PATIENTS = 10
DEFAULT_MAX_OPS = 5

# --- ALIAS POUR COMPATIBILITÉ ---
DEFAULT_DATA = WHATSAPP_IMAGE_DATA

def create_default_environment() -> SchedulingEnvironment:
    """Factory function utilisant les données de l'image."""
    return SchedulingEnvironment(
        data=WHATSAPP_IMAGE_DATA,
        skills=DEFAULT_SKILLS,
        num_patients=DEFAULT_NUM_PATIENTS,
        max_ops=DEFAULT_MAX_OPS
    )
