"""
core/environment.py - Environnement remasterisé avec données de l'image et validation stricte.
"""

from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional
import random
import copy

# Task: patient_id, operation_stage, skill_id, duration

Task = namedtuple("Task", ["i", "j", "s", "p"])

DEFAULT_NUM_SKILLS = 10
DEFAULT_SKILLS = [i+1 for i in range(DEFAULT_NUM_SKILLS)]
DEFAULT_NUM_PATIENTS = 100
DEFAULT_MAX_OPS = 5

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
        stage_completion = {(i, j): 0 for i in range(1, self.num_patients + 1) for j in range(0, self.max_ops + 1)}
        
        for j in range(1, self.max_ops + 1):
            for s in self.skills:
                tasks = sequences.get((s, j), [])
                for task in tasks:
                    ready_time_patient = stage_completion.get((task.i, j - 1), 0)
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


def generate_random_data(
    num_patients: int = 10,
    max_ops: int = 5,
    skills: List[int] = [1, 2, 3, 4, 5, 6],
    task_probability: float = 0.7,
    max_tasks_per_op: int = 3,
    max_duration: int = 3,
    seed: Optional[int] = None
) -> Dict:
    
    if seed is not None:
        random.seed(seed)
    
    random_data = {}
    
    for patient_id in range(1, num_patients + 1):
        patient_ops = {}
        
        for op in range(1, max_ops + 1):
            # Decide if this operation exists for this patient
            if random.random() < task_probability:
                # Decide how many tasks in this operation
                num_tasks = random.randint(1, max_tasks_per_op)
                
                # Select random unique skills for this operation
                selected_skills = random.sample(skills, k=num_tasks)
                
                # Create tasks with random durations
                tasks = []
                for skill in selected_skills:
                    duration = random.randint(1, max_duration)
                    tasks.append((skill, duration))
                
                patient_ops[op] = tasks
            else:
                # No tasks for this operation
                patient_ops[op] = []
        
        random_data[patient_id] = patient_ops
    
    return random_data
# --- DONNÉES  ---

DEFAULT_DATA = generate_random_data(seed=42, 
                                    num_patients=DEFAULT_NUM_PATIENTS, 
                                    max_ops=DEFAULT_MAX_OPS, 
                                    skills=DEFAULT_SKILLS, 
                                    task_probability=0.90
                                   )


# --- ALIAS POUR COMPATIBILITÉ ---

def create_default_environment() -> SchedulingEnvironment:
    """Factory function utilisant les données de l'image."""
    return SchedulingEnvironment(
        data=DEFAULT_DATA,
        skills=DEFAULT_SKILLS,
        num_patients=DEFAULT_NUM_PATIENTS,
        max_ops=DEFAULT_MAX_OPS
    )

print(DEFAULT_DATA)
