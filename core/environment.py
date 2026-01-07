"""
Environment Module - Environnement d'ordonnancement des patients
Gestion des tâches, patients, compétences et évaluation des solutions
"""

from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional
import random
import copy

# Structure de données pour une tâche
Task = namedtuple("Task", ["i", "j", "s", "p"])  # patient, operation, skill, duration

class SchedulingEnvironment:
    """
    Environnement d'ordonnancement multi-compétences pour les patients.
    Gère les données, la création des tâches et l'évaluation des solutions.
    """
    
    def __init__(self, data: Dict, skills: List[int], num_patients: int, max_ops: int):
        """
        Initialise l'environnement d'ordonnancement.
        
        Args:
            data: Dictionnaire des opérations par patient {patient: {op: [(skill, duration)]}}
            skills: Liste des compétences disponibles
            num_patients: Nombre de patients
            max_ops: Nombre maximum d'opérations par patient
        """
        self.data = data
        self.skills = skills
        self.num_patients = num_patients
        self.max_ops = max_ops
        
        # Structures créées lors de l'initialisation
        self.all_tasks: List[Task] = []
        self.tasks_by_skill_stage: Dict[Tuple[int, int], List[Task]] = defaultdict(list)
        self.patient_last_stage: Dict[int, int] = {}
        
        self._create_tasks()
    
    def _create_tasks(self):
        """Crée toutes les tâches et les structures d'indexation."""
        self.patient_last_stage = {i: 0 for i in range(1, self.num_patients + 1)}
        
        for i in range(1, self.num_patients + 1):
            if i in self.data:
                for j in range(1, self.max_ops + 1):
                    ops = self.data[i].get(j, [])
                    if ops:
                        self.patient_last_stage[i] = j
                    for (s, p) in ops:
                        t = Task(i=i, j=j, s=s, p=p)
                        self.all_tasks.append(t)
                        self.tasks_by_skill_stage[(s, j)].append(t)
    
    def build_initial_solution(self, random_order: bool = False) -> Dict[Tuple[int, int], List[Task]]:
        """
        Construit une solution initiale (séquences de tâches).
        
        Args:
            random_order: Si True, ordre aléatoire; sinon, ordre par patient ID
        
        Returns:
            Dictionnaire des séquences de tâches par (skill, stage)
        """
        seq = {}
        for s in self.skills:
            for j in range(1, self.max_ops + 1):
                tasks = self.tasks_by_skill_stage.get((s, j), [])
                if not tasks:
                    continue
                
                ordered_tasks = tasks[:]
                if random_order:
                    random.shuffle(ordered_tasks)
                else:
                    ordered_tasks = sorted(tasks, key=lambda t: t.i)
                
                seq[(s, j)] = ordered_tasks
        return seq
    
    def evaluate(self, sequences: Dict[Tuple[int, int], List[Task]], 
                 return_schedule: bool = False) -> Tuple[int, Optional[Dict], Optional[Dict]]:
        """
        Évalue une solution et calcule le makespan.
        
        Args:
            sequences: Séquences de tâches par (skill, stage)
            return_schedule: Si True, retourne aussi les temps de tâches
        
        Returns:
            makespan (et optionnellement task_times et op_completion)
        """
        # Disponibilité des ressources
        res_free = {s: 0 for s in self.skills}
        # Fin de l'étape j pour chaque patient
        op_completion = {(i, 0): 0 for i in range(1, self.num_patients + 1)}
        # Temps de chaque tâche
        task_times = {}
        
        for j in range(1, self.max_ops + 1):
            stage_finish = defaultdict(int)
            
            for s in self.skills:
                tasks = sequences.get((s, j), [])
                for t in tasks:
                    ready = op_completion[(t.i, j - 1)]
                    start = max(res_free[s], ready)
                    finish = start + t.p
                    
                    res_free[s] = finish
                    stage_finish[t.i] = max(stage_finish[t.i], finish)
                    task_times[(t.i, j, s)] = (start, finish, t.p)
            
            # Figer la fin de l'étape j pour tous les patients
            for i in range(1, self.num_patients + 1):
                if self.data[i].get(j, []):
                    op_completion[(i, j)] = stage_finish[i]
                else:
                    op_completion[(i, j)] = op_completion[(i, j - 1)]
        
        # Calcul du makespan
        makespan = 0
        for i in range(1, self.num_patients + 1):
            last_j = self.patient_last_stage[i]
            makespan = max(makespan, op_completion[(i, last_j)])
        
        if return_schedule:
            return makespan, task_times, op_completion
        return makespan, None, None
    
    def calculate_distance(self, sol1: Dict[Tuple[int, int], List[Task]], 
                          sol2: Dict[Tuple[int, int], List[Task]]) -> int:
        """
        Calcule la distance entre deux solutions.
        Distance = nombre de différences dans les positions des tâches.
        
        Args:
            sol1: Première solution
            sol2: Deuxième solution
        
        Returns:
            Distance (nombre de différences)
        """
        distance = 0
        
        for key in sol1.keys():
            if key not in sol2:
                distance += len(sol1[key])
                continue
            
            tasks1 = sol1[key]
            tasks2 = sol2[key]
            
            # Comparer position par position
            for pos, (t1, t2) in enumerate(zip(tasks1, tasks2)):
                if t1.i != t2.i:  # Patients différents à la même position
                    distance += 1
            
            # Si longueurs différentes
            distance += abs(len(tasks1) - len(tasks2))
        
        # Clés dans sol2 mais pas dans sol1
        for key in sol2.keys():
            if key not in sol1:
                distance += len(sol2[key])
        
        return distance
    
    def copy_solution(self, solution: Dict[Tuple[int, int], List[Task]]) -> Dict[Tuple[int, int], List[Task]]:
        """Crée une copie profonde d'une solution."""
        return copy.deepcopy(solution)


# Données par défaut pour les tests
DEFAULT_DATA = {
    1: {1: [(1,2)], 2: [(1,1),(2,1)], 3: [(1,1),(3,1)], 4: [(1,1),(2,2)], 5: [(4,1),(5,2),(6,1)]},
    2: {1: [(2,1),(3,1)], 2: [(2,1),(3,1)], 3: [(2,1)], 4: [], 5: []},
    3: {1: [(3,2)], 2: [(3,1)], 3: [], 4: [], 5: []},
    4: {1: [(4,2)], 2: [(5,1),(6,1)], 3: [(6,2)], 4: [(4,2)], 5: [(1,1),(2,1)]},
    5: {1: [(2,2)], 2: [(5,1)], 3: [(5,1),(6,1)], 4: [(4,1),(5,1)], 5: [(3,1)]},
    6: {1: [(1,1)], 2: [(4,1)], 3: [(6,1)], 4: [], 5: []},
    7: {1: [(6,2)], 2: [(1,1)], 3: [(5,1),(6,1)], 4: [(3,1)], 5: []},
    8: {1: [(3,1),(5,1)], 2: [(2,1),(5,1)], 3: [(3,1),(6,1)], 4: [(6,1)], 5: []},
    9: {1: [(5,1)], 2: [(4,1)], 3: [(1,1)], 4: [], 5: []},
    10: {1: [(4,1)], 2: [(4,1),(5,1)], 3: [(1,1),(2,1)], 4: [(4,1)], 5: []},
}

DEFAULT_SKILLS = [1, 2, 3, 4, 5, 6]
DEFAULT_NUM_PATIENTS = 10
DEFAULT_MAX_OPS = 5


def create_default_environment() -> SchedulingEnvironment:
    """Crée un environnement avec les données par défaut."""
    return SchedulingEnvironment(
        data=DEFAULT_DATA,
        skills=DEFAULT_SKILLS,
        num_patients=DEFAULT_NUM_PATIENTS,
        max_ops=DEFAULT_MAX_OPS
    )
