"""
core/data_generator.py - Générateur de données paramétrables pour l'ordonnancement.
Inclut les données de référence de l'image comme exemple.
"""
import random
from typing import Dict, List, Tuple

# ============================================================================
# DONNÉES DE RÉFÉRENCE (EXEMPLE DE L'IMAGE)
# ============================================================================
# Table de compétences pour 10 patients, 5 opérations, 6 skills
REFERENCE_DATA_FROM_IMAGE = {
    1: {  # Patient 1
        1: [(1, 20), (1, 20)],           # C1x2 - Compétence 1, 2 fois
        2: [(1, 15), (2, 15)],           # C1 et C2
        3: [(1, 15), (3, 15)],           # C1 et C3
        4: [(1, 20), (2, 20), (2, 20)],  # C1 et C2x2
        5: [(4, 25), (5, 25), (5, 25), (6, 25)]  # C4, C5x2 et C6
    },
    2: {  # Patient 2
        1: [(2, 15), (3, 15)],           # C2 et C3
        2: [(2, 20), (3, 20)],           # C2 et C3
        3: [(2, 15)],                     # C2
        4: [],                            # Aucune opération
        5: []                             # Aucune opération
    },
    3: {  # Patient 3
        1: [(3, 15), (3, 15), (3, 15)],  # C3x2 (on met 3 fois pour x2)
        2: [(3, 20)],                     # C3
        3: [],                            # Aucune opération
        4: [],                            # Aucune opération
        5: []                             # Aucune opération
    },
    4: {  # Patient 4
        1: [(4, 30), (4, 30)],           # C4x2
        2: [(5, 25), (6, 25)],           # C5 et C6
        3: [(6, 30), (6, 30)],           # C6x2
        4: [(4, 35), (4, 35)],           # C4x2
        5: [(1, 20), (2, 20)]            # C1 et C2
    },
    5: {  # Patient 5
        1: [(2, 20), (2, 20)],           # C2x2
        2: [(5, 30)],                     # C5
        3: [(5, 25), (6, 25)],           # C5 et C6
        4: [(4, 30), (5, 30)],           # C4 et C5
        5: [(3, 20)]                      # C3
    },
    6: {  # Patient 6
        1: [(1, 15)],                     # C1
        2: [(4, 25)],                     # C4
        3: [(6, 30)],                     # C6
        4: [],                            # Aucune opération
        5: []                             # Aucune opération
    },
    7: {  # Patient 7
        1: [(6, 25), (6, 25)],           # C6x2
        2: [(1, 20)],                     # C1
        3: [(5, 30), (6, 30)],           # C5 et C6
        4: [(3, 25)],                     # C3
        5: []                             # Aucune opération
    },
    8: {  # Patient 8
        1: [(3, 20), (5, 20)],           # C3 et C5
        2: [(2, 25), (5, 25)],           # C2 et C5
        3: [(3, 25), (6, 25)],           # C3 et C6
        4: [(6, 30)],                     # C6
        5: []                             # Aucune opération
    },
    9: {  # Patient 9
        1: [(5, 25)],                     # C5
        2: [(4, 30)],                     # C4
        3: [(1, 20)],                     # C1
        4: [],                            # Aucune opération
        5: []                             # Aucune opération
    },
    10: {  # Patient 10
        1: [(4, 30)],                     # C4
        2: [(4, 35), (5, 35)],           # C4 et C5
        3: [(1, 20), (2, 20)],           # C1 et C2
        4: [(4, 30)],                     # C4
        5: []                             # Aucune opération
    }
}

REFERENCE_SKILLS = [1, 2, 3, 4, 5, 6]
REFERENCE_NUM_PATIENTS = 10

# ============================================================================
# GÉNÉRATEURS DE DONNÉES PARAMÉTRABLES
# ============================================================================

def generate_parametric_data(
    num_patients: int = 10,
    num_skills: int = 6,
    max_operations: int = 5,
    operation_probability: float = 0.7,
    min_duration: int = 10,
    max_duration: int = 60,
    max_tasks_per_operation: int = 3,
    seed: int = None
) -> Tuple[Dict, List[int]]:
    """
    Génère des données d'ordonnancement paramétrables.
    
    Args:
        num_patients: Nombre de patients à ordonnancer
        num_skills: Nombre de compétences/ressources disponibles
        max_operations: Nombre maximum d'opérations par patient
        operation_probability: Probabilité qu'une opération existe (0.0 à 1.0)
        min_duration: Durée minimale d'une tâche en minutes
        max_duration: Durée maximale d'une tâche en minutes
        max_tasks_per_operation: Nombre maximum de tâches par opération
        seed: Seed pour la reproductibilité (None = aléatoire)
    
    Returns:
        Tuple[Dict, List[int]]: (données patients, liste des skills)
    """
    if seed is not None:
        random.seed(seed)
    
    skills = list(range(1, num_skills + 1))
    data = {}
    
    for patient_id in range(1, num_patients + 1):
        data[patient_id] = {}
        
        for op_order in range(1, max_operations + 1):
            # Décider si cette opération existe pour ce patient
            if random.random() < operation_probability:
                # Nombre de tâches pour cette opération (1 à max_tasks_per_operation)
                num_tasks = random.randint(1, max_tasks_per_operation)
                tasks = []
                
                for _ in range(num_tasks):
                    skill = random.choice(skills)
                    duration = random.randint(min_duration, max_duration)
                    tasks.append((skill, duration))
                
                data[patient_id][op_order] = tasks
            else:
                # Opération vide
                data[patient_id][op_order] = []
    
    return data, skills


def generate_balanced_data(
    num_patients: int = 10,
    num_skills: int = 6,
    max_operations: int = 5,
    seed: int = None
) -> Tuple[Dict, List[int]]:
    """
    Génère des données équilibrées où chaque skill est utilisée à peu près autant.
    Utile pour tester l'équilibrage de charge entre ressources.
    
    Args:
        num_patients: Nombre de patients
        num_skills: Nombre de compétences/ressources
        max_operations: Nombre maximum d'opérations par patient
        seed: Seed pour la reproductibilité
    
    Returns:
        Tuple[Dict, List[int]]: (données patients, liste des skills)
    """
    if seed is not None:
        random.seed(seed)
    
    skills = list(range(1, num_skills + 1))
    data = {}
    
    # Compteur pour équilibrer l'utilisation des skills
    skill_usage = {s: 0 for s in skills}
    
    for patient_id in range(1, num_patients + 1):
        data[patient_id] = {}
        
        for op_order in range(1, max_operations + 1):
            if random.random() < 0.75:  # 75% de chance d'avoir une opération
                num_tasks = random.randint(1, 2)
                tasks = []
                
                # Sélectionner les skills les moins utilisées
                sorted_skills = sorted(skills, key=lambda s: skill_usage[s])
                available_skills = sorted_skills[:max(2, num_skills // 2)]
                
                for _ in range(num_tasks):
                    skill = random.choice(available_skills)
                    duration = random.randint(15, 45)
                    tasks.append((skill, duration))
                    skill_usage[skill] += 1
                
                data[patient_id][op_order] = tasks
            else:
                data[patient_id][op_order] = []
    
    return data, skills


def generate_realistic_healthcare_data(
    num_patients: int = 10,
    num_skills: int = 6,
    seed: int = None
) -> Tuple[Dict, List[int]]:
    """
    Génère des données réalistes pour un contexte médical.
    Les opérations suivent un ordre logique (consultation -> examens -> traitement).
    
    Args:
        num_patients: Nombre de patients
        num_skills: Nombre de compétences (médecins, infirmiers, techniciens, etc.)
        seed: Seed pour la reproductibilité
    
    Returns:
        Tuple[Dict, List[int]]: (données patients, liste des skills)
    """
    if seed is not None:
        random.seed(seed)
    
    skills = list(range(1, num_skills + 1))
    data = {}
    
    # Définir des "profils" de patients avec des parcours de soins typiques
    patient_profiles = ['simple', 'medium', 'complex']
    
    for patient_id in range(1, num_patients + 1):
        data[patient_id] = {}
        profile = random.choice(patient_profiles)
        
        if profile == 'simple':
            # Parcours simple: 2-3 opérations
            num_ops = random.randint(2, 3)
        elif profile == 'medium':
            # Parcours moyen: 3-4 opérations
            num_ops = random.randint(3, 4)
        else:  # complex
            # Parcours complexe: 4-5 opérations
            num_ops = random.randint(4, 5)
        
        for op_order in range(1, 6):
            if op_order <= num_ops:
                # Première opération: généralement consultation (skill 1 ou 2)
                if op_order == 1:
                    skill = random.choice(skills[:min(2, num_skills)])
                    duration = random.randint(15, 30)
                    data[patient_id][op_order] = [(skill, duration)]
                
                # Opérations intermédiaires: examens (skills variés)
                elif op_order < num_ops:
                    num_tasks = random.randint(1, 2)
                    tasks = []
                    for _ in range(num_tasks):
                        skill = random.choice(skills)
                        duration = random.randint(20, 45)
                        tasks.append((skill, duration))
                    data[patient_id][op_order] = tasks
                
                # Dernière opération: traitement (peut être plus long)
                else:
                    skill = random.choice(skills)
                    duration = random.randint(30, 60)
                    data[patient_id][op_order] = [(skill, duration)]
            else:
                data[patient_id][op_order] = []
    
    return data, skills


def get_reference_data() -> Tuple[Dict, List[int], int]:
    """
    Retourne les données de référence de l'image.
    
    Returns:
        Tuple[Dict, List[int], int]: (données, skills, nombre de patients)
    """
    return REFERENCE_DATA_FROM_IMAGE, REFERENCE_SKILLS, REFERENCE_NUM_PATIENTS


def print_data_summary(data: Dict, skills: List[int]):
    """
    Affiche un résumé des données générées.
    
    Args:
        data: Dictionnaire des données patients
        skills: Liste des compétences disponibles
    """
    num_patients = len(data)
    total_operations = sum(1 for p in data.values() for ops in p.values() if ops)
    total_tasks = sum(len(tasks) for p in data.values() for tasks in p.values())
    
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ DES DONNÉES GÉNÉRÉES")
    print(f"{'='*60}")
    print(f"Nombre de patients    : {num_patients}")
    print(f"Nombre de skills      : {len(skills)} {skills}")
    print(f"Nombre d'opérations   : {total_operations}")
    print(f"Nombre de tâches      : {total_tasks}")
    print(f"Tâches par opération  : {total_tasks/total_operations:.2f} (moyenne)")
    
    # Utilisation des skills
    skill_count = {}
    for p in data.values():
        for tasks in p.values():
            for skill, _ in tasks:
                skill_count[skill] = skill_count.get(skill, 0) + 1
    
    print(f"\nUtilisation des compétences:")
    for skill in sorted(skill_count.keys()):
        count = skill_count[skill]
        percentage = (count / total_tasks) * 100
        print(f"  Skill {skill}: {count:3d} tâches ({percentage:5.1f}%)")
    print(f"{'='*60}\n")


# ============================================================================
# FONCTION POUR COMPATIBILITÉ AVEC LE CODE EXISTANT
# ============================================================================

def generate_random_data(num_patients=5, max_ops=3, skills=[1,2,3,4]):
    """
    Fonction de compatibilité avec l'ancien générateur.
    Utilise le nouveau générateur paramétrique.
    """
    data, _ = generate_parametric_data(
        num_patients=num_patients,
        num_skills=len(skills),
        max_operations=max_ops,
        operation_probability=0.9,
        min_duration=10,
        max_duration=60,
        max_tasks_per_operation=2
    )
    return data
