"""
core/data_generator.py - Générateurs de données configurables.

Trois modes :
  - parametric  : contrôle total (nb patients, skills, probabilité, durées…)
  - balanced    : charge équilibrée entre les compétences
  - realistic   : parcours de soins réalistes (consultation → examens → traitement)
"""

from typing import Dict, List, Tuple, Optional
import random
from core.environment import (
    DEFAULT_SKILLS,
    DEFAULT_NUM_PATIENTS,
    DEFAULT_MAX_OPS,
    DEFAULT_DATA,
    SchedulingEnvironment,
)


# ─── Référence (données par défaut seed=42) ─────────────────────────────────

def get_reference_data() -> Tuple[Dict, List[int], int]:
    """Renvoie les données de référence (10 patients, 6 skills)."""
    return DEFAULT_DATA, DEFAULT_SKILLS, DEFAULT_NUM_PATIENTS


# ─── Générateur paramétrique ─────────────────────────────────────────────────

def generate_parametric_data(
    num_patients: int = 10,
    num_skills: int = 6,
    max_operations: int = 5,
    operation_probability: float = 0.7,
    min_duration: int = 1,
    max_duration: int = 3,
    max_tasks_per_operation: int = 3,
    seed: Optional[int] = None,
) -> Tuple[Dict, List[int]]:
    """Générateur paramétrique – contrôle total."""
    if seed is not None:
        random.seed(seed)

    skills = list(range(1, num_skills + 1))
    data: Dict = {}

    for pid in range(1, num_patients + 1):
        ops: Dict = {}
        for op in range(1, max_operations + 1):
            if random.random() < operation_probability:
                n_tasks = random.randint(1, min(max_tasks_per_operation, num_skills))
                chosen = random.sample(skills, n_tasks)
                tasks = [(s, random.randint(min_duration, max_duration)) for s in chosen]
                ops[op] = tasks
            else:
                ops[op] = []
        data[pid] = ops

    return data, skills


# ─── Générateur équilibré ────────────────────────────────────────────────────

def generate_balanced_data(
    num_patients: int = 10,
    num_skills: int = 6,
    max_operations: int = 5,
    seed: Optional[int] = None,
) -> Tuple[Dict, List[int]]:
    """Génère des données où chaque compétence est sollicitée de manière équilibrée."""
    if seed is not None:
        random.seed(seed)

    skills = list(range(1, num_skills + 1))
    data: Dict = {}
    skill_usage = {s: 0 for s in skills}

    for pid in range(1, num_patients + 1):
        ops: Dict = {}
        for op in range(1, max_operations + 1):
            if random.random() < 0.75:
                n_tasks = random.randint(1, min(3, num_skills))
                # Choisir les compétences les moins utilisées
                sorted_skills = sorted(skills, key=lambda s: skill_usage[s])
                chosen = sorted_skills[:n_tasks]
                tasks = [(s, random.randint(1, 3)) for s in chosen]
                for s in chosen:
                    skill_usage[s] += 1
                ops[op] = tasks
            else:
                ops[op] = []
        data[pid] = ops

    return data, skills


# ─── Générateur réaliste ─────────────────────────────────────────────────────

def generate_realistic_healthcare_data(
    num_patients: int = 10,
    num_skills: int = 6,
    seed: Optional[int] = None,
) -> Tuple[Dict, List[int]]:
    """
    Parcours de soins réalistes :
      Opération 1 : Consultation (1 tâche, compétence 1)
      Opération 2 : Examens (1–3 tâches, compétences variées)
      Opération 3 : Traitement (1–2 tâches)
      Opération 4 : Suivi (optionnel)
      Opération 5 : Sortie (optionnel)
    """
    if seed is not None:
        random.seed(seed)

    skills = list(range(1, num_skills + 1))
    data: Dict = {}

    for pid in range(1, num_patients + 1):
        ops: Dict = {}

        # Op 1 – Consultation (toujours présente)
        ops[1] = [(skills[0], random.randint(1, 2))]

        # Op 2 – Examens
        n_exams = random.randint(1, min(3, num_skills - 1))
        exam_skills = random.sample(skills[1:], n_exams)
        ops[2] = [(s, random.randint(1, 3)) for s in exam_skills]

        # Op 3 – Traitement
        n_treat = random.randint(1, 2)
        treat_skills = random.sample(skills, n_treat)
        ops[3] = [(s, random.randint(2, 3)) for s in treat_skills]

        # Op 4 – Suivi (70 % des patients)
        if random.random() < 0.7:
            ops[4] = [(random.choice(skills), random.randint(1, 2))]
        else:
            ops[4] = []

        # Op 5 – Sortie (40 % des patients)
        if random.random() < 0.4:
            ops[5] = [(skills[0], 1)]
        else:
            ops[5] = []

        data[pid] = ops

    return data, skills


# ─── Utilitaire d'affichage ──────────────────────────────────────────────────

def print_data_summary(data: Dict, skills: List[int]):
    """Affiche un résumé des données générées."""
    num_patients = len(data)
    total_tasks = sum(
        len(ops) for patient in data.values() for ops in patient.values()
    )
    skill_count = {s: 0 for s in skills}
    for patient in data.values():
        for ops in patient.values():
            for s, _ in ops:
                skill_count[s] += 1

    print(f"  Patients : {num_patients}")
    print(f"  Compétences : {len(skills)}")
    print(f"  Tâches totales : {total_tasks}")
    print(f"  Répartition par compétence :")
    for s in skills:
        print(f"    Skill {s} : {skill_count[s]} tâches")
