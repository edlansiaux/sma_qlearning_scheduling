"""
core/environment.py - Environnement d'ordonnancement avec validation stricte.
"""

from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional
import random
import copy

Task = namedtuple("Task", ["i", "j", "s", "p"])

DEFAULT_NUM_SKILLS = 6
DEFAULT_SKILLS = [i + 1 for i in range(DEFAULT_NUM_SKILLS)]
DEFAULT_NUM_PATIENTS = 10
DEFAULT_MAX_OPS = 5


class SchedulingEnvironment:
    def __init__(self, data: Dict, skills: List[int], num_patients: int, max_ops: int):
        self.data = data
        self.skills = skills
        self.num_patients = num_patients
        self.max_ops = max_ops

        self.all_tasks: List[Task] = []
        self.tasks_by_skill_stage: Dict[Tuple[int, int], List[Task]] = defaultdict(list)
        self.patient_last_stage: Dict[int, int] = {}

        self._create_tasks()

    def _create_tasks(self):
        self.patient_last_stage = {i: 0 for i in range(1, self.num_patients + 1)}

        for i in range(1, self.num_patients + 1):
            if i in self.data:
                for j in range(1, self.max_ops + 1):
                    ops = self.data[i].get(j, [])
                    if ops:
                        self.patient_last_stage[i] = max(self.patient_last_stage[i], j)
                    for (s, p) in ops:
                        t = Task(i=i, j=j, s=s, p=p)
                        self.all_tasks.append(t)
                        self.tasks_by_skill_stage[(s, j)].append(t)

    def build_initial_solution(self, random_order: bool = True) -> Dict[Tuple[int, int], List[Task]]:
        seq: Dict[Tuple[int, int], List[Task]] = {}
        for s in self.skills:
            tasks_for_skill = []
            for j in range(1, self.max_ops + 1):
                tasks_for_skill.extend(self.tasks_by_skill_stage.get((s, j), []))

            if not tasks_for_skill:
                continue

            if random_order:
                random.shuffle(tasks_for_skill)
            else:
                tasks_for_skill.sort(key=lambda t: (t.j, t.i))

            for t in tasks_for_skill:
                key = (t.s, t.j)
                if key not in seq:
                    seq[key] = []
                seq[key].append(t)

        return seq

    def check_constraints(self, solution: Dict[Tuple[int, int], List[Task]]) -> Tuple[bool, str]:
        count_tasks = sum(len(tasks) for tasks in solution.values())
        if count_tasks != len(self.all_tasks):
            return False, f"Nombre de tâches incorrect: {count_tasks} vs {len(self.all_tasks)}"
        return True, "Valide"

    def evaluate(
        self,
        sequences: Dict[Tuple[int, int], List[Task]],
        return_schedule: bool = False,
    ) -> Tuple[int, Optional[Dict], Optional[Dict]]:
        """Calcule le Makespan (Cmax)."""
        skill_free_time = {s: 0 for s in self.skills}
        task_times: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        stage_completion: Dict[Tuple[int, int], int] = {
            (i, j): 0
            for i in range(1, self.num_patients + 1)
            for j in range(0, self.max_ops + 1)
        }

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


# ---------------------------------------------------------------------------
# Générateurs de données
# ---------------------------------------------------------------------------

def generate_random_data(
    num_patients: int = 10,
    max_ops: int = 5,
    skills: List[int] = None,
    task_probability: float = 0.7,
    max_tasks_per_op: int = 3,
    max_duration: int = 3,
    seed: Optional[int] = None,
) -> Dict:
    if skills is None:
        skills = DEFAULT_SKILLS

    if seed is not None:
        random.seed(seed)

    random_data: Dict = {}
    for patient_id in range(1, num_patients + 1):
        patient_ops: Dict = {}
        for op in range(1, max_ops + 1):
            if random.random() < task_probability:
                num_tasks = random.randint(1, min(max_tasks_per_op, len(skills)))
                selected_skills = random.sample(skills, k=num_tasks)
                tasks = [(skill, random.randint(1, max_duration)) for skill in selected_skills]
                patient_ops[op] = tasks
            else:
                patient_ops[op] = []
        random_data[patient_id] = patient_ops
    return random_data


# Données par défaut (reproductibles)
DEFAULT_DATA = generate_random_data(
    seed=42,
    num_patients=DEFAULT_NUM_PATIENTS,
    max_ops=DEFAULT_MAX_OPS,
    skills=DEFAULT_SKILLS,
    task_probability=0.90,
)


def create_default_environment() -> SchedulingEnvironment:
    """Factory utilisant les données par défaut."""
    return SchedulingEnvironment(
        data=DEFAULT_DATA,
        skills=DEFAULT_SKILLS,
        num_patients=DEFAULT_NUM_PATIENTS,
        max_ops=DEFAULT_MAX_OPS,
    )
