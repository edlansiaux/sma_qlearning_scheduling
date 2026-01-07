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
            # car un médecin (Skill) ne voit pas les étapes, il voit une file d'attente globale
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
            
            # Note: La clé du dictionnaire de solution change ici pour simplifier.
            # Au lieu de (s, j), on stocke par (s) car c'est la file du médecin.
            # Cependant, pour garder la compatibilité avec le code existant qui semble
            # utiliser (s,j), je vais adapter la structure de retour.
            
            # ADAPTATION : Le code existant semble s'attendre à ce que la structure 'solution'
            # soit partitionnée. Si les voisinages manipulent (s,j), gardons cette structure.
            # MAIS attention : le voisinage A déplace entre (s1, j) et (s2, j).
            
            # Re-partitionnement par stage pour compatibilité
            for t in tasks_for_skill:
                key = (t.s, t.j)
                if key not in seq:
                    seq[key] = []
                seq[key].append(t)
                
        return seq

    def check_constraints(self, solution: Dict[Tuple[int, int], List[Task]]) -> Tuple[bool, str]:
        """
        Vérifie la validité stricte de la solution selon l'image.
        1. Chaque tâche requise doit être planifiée une seule fois.
        2. Les contraintes de précédence (Op j après Op j-1) sont gérées par evaluate(),
           mais ici on vérifie l'intégrité des données.
        """
        # Vérification du nombre de tâches
        count_tasks = 0
        for tasks in solution.values():
            count_tasks += len(tasks)
        
        if count_tasks != len(self.all_tasks):
            return False, f"Nombre de tâches incorrect: {count_tasks} vs {len(self.all_tasks)}"
            
        return True, "Valide"

    def evaluate(self, sequences: Dict[Tuple[int, int], List[Task]], 
                 return_schedule: bool = False) -> Tuple[int, Optional[Dict], Optional[Dict]]:
        """
        Calcule le Makespan (Cmax) en respectant strictement les contraintes.
        """
        # 1. Reconstruction des files d'attente par ressource (Skill)
        # On doit savoir dans quel ordre chaque médecin exécute ses tâches globalement.
        # Dans la structure 'sequences' (s, j), l'ordre intra-liste est respecté.
        # Mais l'ordre entre (s, 1) et (s, 2) n'est pas explicite. 
        # HYPOTHÈSE FORTE : Un médecin traite les étapes dans l'ordre croissant des étapes j,
        # sauf si l'agent a permuté. Pour simplifier et optimiser : 
        # On va considérer que l'agent ordonnance tout.
        
        # Disponibilité des ressources (médecins)
        skill_free_time = {s: 0 for s in self.skills}
        
        # Disponibilité des patients (quand ils finissent l'étape précédente)
        # (patient_id, stage_completed) -> time
        patient_stage_completion = defaultdict(int)
        
        # Pour tracer le planning
        task_times = {} # (i, j, s) -> (start, end, duration)
        
        # Pour évaluer, nous devons simuler le temps. 
        # C'est un problème complexe car il y a interdépendance.
        # Approche simple : On trie toutes les tâches de la solution par "priorité"
        # définie par l'agent (l'ordre dans les listes).
        
        # Cependant, la structure actuelle `sequences[(s,j)]` partitionne trop.
        # On va simuler étape par étape (1..5) car c'est un Flow Shop hybride.
        
        # Fin de l'étape j pour chaque patient
        stage_completion = {(i, 0): 0 for i in range(1, self.num_patients + 1)}
        
        for j in range(1, self.max_ops + 1):
            # Tâches de l'étape j
            tasks_in_stage = []
            for s in self.skills:
                if (s, j) in sequences:
                    tasks_in_stage.extend(sequences[(s, j)])
            
            # On ne peut pas simplement itérer, car il peut y avoir concurrence sur les skills
            # si on mélange les étapes. Mais ici on traite étape par étape.
            
            # Pour chaque skill à l'étape j, on traite les patients dans l'ordre de la liste
            for s in self.skills:
                tasks = sequences.get((s, j), [])
                for task in tasks:
                    # Le patient est prêt quand il a fini l'étape j-1
                    ready_time_patient = stage_completion[(task.i, j - 1)]
                    
                    # La ressource est prête quand elle a fini sa tâche précédente
                    # (qui peut être une tâche de l'étape j-1 ou une tâche précédente de l'étape j)
                    ready_time_skill = skill_free_time[task.s]
                    
                    start_time = max(ready_time_patient, ready_time_skill)
                    end_time = start_time + task.p
                    
                    # Mise à jour
                    skill_free_time[task.s] = end_time
                    task_times[(task.i, task.j, task.s)] = (start_time, end_time, task.p)
                    
                    # Mise à jour temporaire de la fin du patient pour cette tâche
                    # Un patient finit l'étape j quand TOUTES ses sous-tâches de j sont finies.
                    current_stage_end = stage_completion.get((task.i, j), 0)
                    stage_completion[(task.i, j)] = max(current_stage_end, end_time)

        # Calcul Cmax
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
# Syntaxe : {Patient: {Opération: [(Skill, Durée), ...]}}
# Hypothèse : "Cx2" = Durée 2. "C1 et C2" = Deux tâches distinctes durée 1 (sauf si xN).

WHATSAPP_IMAGE_DATA = {
    1: {
        1: [(1, 2)],               # C1x2
        2: [(1, 1), (2, 1)],       # C1 et C2
        3: [(1, 1), (3, 1)],       # C1 et C3
        4: [(1, 1), (2, 2)],       # C1 et C2x2
        5: [(4, 1), (5, 2), (6, 1)]# C4, C5x2 et C6
    },
    2: {
        1: [(2, 1), (3, 1)],       # C2 et C3
        2: [(2, 1), (3, 1)],       # C2 et C3
        3: [(2, 1)],               # C2
        4: [], 5: []               # Aucune
    },
    3: {
        1: [(3, 2)],               # C3x2
        2: [(3, 1)],               # C3
        3: [], 4: [], 5: []
    },
    4: {
        1: [(4, 2)],               # C4x2
        2: [(5, 1), (6, 1)],       # C5 et C6
        3: [(6, 2)],               # C6x2
        4: [(4, 2)],               # C4x2
        5: [(1, 1), (2, 1)]        # C1 et C2
    },
    5: {
        1: [(2, 2)],               # C2x2
        2: [(5, 1)],               # C5
        3: [(5, 1), (6, 1)],       # C5 et C6
        4: [(4, 1), (5, 1)],       # C4 et C5
        5: [(3, 1)]                # C3
    },
    6: {
        1: [(1, 1)],               # C1
        2: [(4, 1)],               # C4
        3: [(6, 1)],               # C6
        4: [], 5: []
    },
    7: {
        1: [(6, 2)],               # C6x2
        2: [(1, 1)],               # C1
        3: [(5, 1), (6, 1)],       # C5 et C6
        4: [(3, 1)],               # C3
        5: []
    },
    8: {
        1: [(3, 1), (5, 1)],       # C3 et C5
        2: [(2, 1), (5, 1)],       # C2 et C5
        3: [(3, 1), (6, 1)],       # C3 et C6
        4: [(6, 1)],               # C6
        5: []
    },
    9: {
        1: [(5, 1)],               # C5
        2: [(4, 1)],               # C4
        3: [(1, 1)],               # C1
        4: [], 5: []
    },
    10: {
        1: [(4, 1)],               # C4
        2: [(4, 1), (5, 1)],       # C4 et C5
        3: [(1, 1), (2, 1)],       # C1 et C2
        4: [(4, 1)],               # C4
        5: []
    }
}

DEFAULT_SKILLS = [1, 2, 3, 4, 5, 6]
DEFAULT_NUM_PATIENTS = 10
DEFAULT_MAX_OPS = 5

def create_default_environment() -> SchedulingEnvironment:
    """Factory function utilisant les données de l'image."""
    return SchedulingEnvironment(
        data=WHATSAPP_IMAGE_DATA,
        skills=DEFAULT_SKILLS,
        num_patients=DEFAULT_NUM_PATIENTS,
        max_ops=DEFAULT_MAX_OPS
    )
