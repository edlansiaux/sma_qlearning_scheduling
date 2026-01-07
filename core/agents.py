"""
core/agents.py - Agents métaheuristiques complets et remasterisés.
"""

from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import random
import math
import copy

# Imports centralisés remasterisés
from core.environment import SchedulingEnvironment, Task
from core.neighborhoods import NeighborhoodManager
from core.qlearning import AdaptiveNeighborhoodSelector
from core.shared_memory import SharedMemoryPool, Solution, ElitePool

class BaseAgent(ABC):
    """Classe de base abstraite pour les agents."""
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 use_qlearning: bool = True):
        self.agent_id = agent_id
        self.env = environment
        self.use_qlearning = use_qlearning
        
        self.current_solution: Optional[Dict[Tuple[int, int], List[Task]]] = None
        self.current_fitness: float = float('inf')
        self.best_solution: Optional[Dict[Tuple[int, int], List[Task]]] = None
        self.best_fitness: float = float('inf')
        
        self.neighborhood_manager = NeighborhoodManager()
        self.q_selector = AdaptiveNeighborhoodSelector() if use_qlearning else None
        
        self.fitness_history: List[float] = []
        self.iterations_active = 0
    
    def initialize(self, random_init: bool = True):
        self.current_solution = self.env.build_initial_solution(random_order=random_init)
        self.current_fitness, _, _ = self.env.evaluate(self.current_solution)
        self.best_solution = self.env.copy_solution(self.current_solution)
        self.best_fitness = self.current_fitness
        
        if self.q_selector:
            self.q_selector.reset(self.current_fitness)
        
        self.fitness_history = [self.current_fitness]
    
    def set_solution(self, solution: Dict, fitness: float = None):
        self.current_solution = self.env.copy_solution(solution)
        if fitness is None:
            self.current_fitness, _, _ = self.env.evaluate(self.current_solution)
        else:
            self.current_fitness = fitness
        
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness
    
    @abstractmethod
    def optimize_step(self) -> Tuple[Dict, float]:
        pass
    
    def get_solution(self) -> Solution:
        return Solution(
            sequences=self.env.copy_solution(self.current_solution),
            fitness=self.current_fitness,
            agent_id=self.agent_id
        )

class GeneticAgent(BaseAgent):
    """Agent Algorithme Génétique."""
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 population_size: int = 15, mutation_rate: float = 0.1,
                 use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Tuple[Dict, float]] = []
    
    def initialize(self, random_init: bool = True):
        super().initialize(random_init)
        self.population = []
        # Génération de la population initiale
        for _ in range(self.population_size):
            sol = self.env.build_initial_solution(random_order=True)
            fitness, _, _ = self.env.evaluate(sol)
            self.population.append((sol, fitness))
        
        self.population.sort(key=lambda x: x[1])
        self.current_solution = self.env.copy_solution(self.population[0][0])
        self.current_fitness = self.population[0][1]
        
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness
            
    def _tournament_selection(self, k: int = 3) -> Dict:
        competitors = random.sample(self.population, min(k, len(self.population)))
        winner = min(competitors, key=lambda x: x[1])
        return self.env.copy_solution(winner[0])
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Croisement OX (Order Crossover) par file."""
        child = {}
        # Pour chaque clé (Skill, Stage) présente
        all_keys = set(parent1.keys()) | set(parent2.keys())
        
        for key in all_keys:
            seq1 = parent1.get(key, [])
            seq2 = parent2.get(key, [])
            
            # Si une des séquences est vide ou courte, on copie simplement l'une des deux
            if len(seq1) < 2 or len(seq2) < 2:
                child[key] = copy.deepcopy(seq1) if seq1 else copy.deepcopy(seq2)
                continue
                
            # Crossover OX
            n = len(seq1)
            # Points de coupe
            a, b = sorted(random.sample(range(n), 2))
            
            child_seq = [None] * n
            # Copie du segment parent 1
            child_seq[a:b+1] = seq1[a:b+1]
            
            # Remplissage avec parent 2 (en préservant l'ordre relatif)
            current_tasks_ids = set(t.i for t in child_seq if t is not None)
            
            # On prend les tâches de seq2 qui ne sont pas déjà dans le segment du child
            remaining = [t for t in seq2 if t.i not in current_tasks_ids]
            
            pos = 0
            for i in range(n):
                if child_seq[i] is None:
                    if pos < len(remaining):
                        child_seq[i] = remaining[pos]
                        pos += 1
                    else:
                        # Fallback si incohérence (ne devrait pas arriver si structures identiques)
                        child_seq[i] = seq1[i] 
                        
            child[key] = child_seq
            
        return child
    
    def _mutate(self, solution: Dict) -> Dict:
        """Mutation avec ou sans Q-Learning."""
        mutated = self.env.copy_solution(solution)
        
        if self.use_qlearning and self.q_selector:
            # Q-Learning choisit le voisinage (C ou E)
            neighborhood = self.q_selector.select_neighborhood()
            neighbor = self.neighborhood_manager.generate_neighbor(
                mutated, neighborhood, self.env.skills, self.env.max_ops
            )
            # Si le voisinage est valide (retourne un voisin), on le prend
            if neighbor:
                return neighbor
            # Sinon, on retourne la solution non mutée (échec silencieux de la mutation)
            return mutated
        else:
            # Mutation classique : on tente un swap aléatoire (Voisinage E) sur chaque file
            for key in mutated:
                if len(mutated[key]) >= 2 and random.random() < self.mutation_rate:
                    i, j = random.sample(range(len(mutated[key])), 2)
                    mutated[key][i], mutated[key][j] = mutated[key][j], mutated[key][i]
            return mutated

    def optimize_step(self) -> Tuple[Dict, float]:
        self.iterations_active += 1
        new_population = []
        
        # Élitisme
        n_elite = max(1, self.population_size // 5)
        new_population.extend(self.population[:n_elite])
        
        # Reproduction
        while len(new_population) < self.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            fit, _, _ = self.env.evaluate(child)
            new_population.append((child, fit))
            
        new_population.sort(key=lambda x: x[1])
        self.population = new_population[:self.population_size]
        
        best_sol, best_fit = self.population[0]
        self.current_solution = self.env.copy_solution(best_sol)
        self.current_fitness = best_fit
        
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness
            
        # Feedback Q-Learning (Global reward based on current best)
        if self.q_selector:
            # On considère que c'est une action 'réussie' si on améliore
            self.q_selector.update_with_result('E', self.current_fitness) # 'E' par défaut pour AG
            
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness

class TabuAgent(BaseAgent):
    """Agent Recherche Tabou."""
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 tabu_tenure: int = 10, candidate_limit: int = 20,
                 use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.tabu_tenure = tabu_tenure
        self.candidate_limit = candidate_limit
        self.tabu_list: List[Tuple] = []
        
    def optimize_step(self) -> Tuple[Dict, float]:
        self.iterations_active += 1
        
        candidates = []
        
        # Génération des candidats
        if self.use_qlearning and self.q_selector:
            # Q-Learning guide la génération
            for _ in range(self.candidate_limit):
                n_name = self.q_selector.select_neighborhood()
                n_sol = self.neighborhood_manager.generate_neighbor(
                    self.current_solution, n_name, self.env.skills, self.env.max_ops
                )
                if n_sol:
                    # Signature simple du mouvement : hash de la string solution
                    # (Pour faire mieux, on devrait hacher le delta, mais c'est suffisant ici)
                    move_hash = hash(str(n_sol))
                    candidates.append((n_sol, n_name, move_hash))
        else:
            # Génération aléatoire parmi les voisinages actifs
            # On force C et E car A, B, D sont invalides dans ce problème strict
            active_neighbors = ['C', 'E']
            for _ in range(self.candidate_limit):
                n_name = random.choice(active_neighbors)
                n_sol = self.neighborhood_manager.generate_neighbor(
                    self.current_solution, n_name, self.env.skills, self.env.max_ops
                )
                if n_sol:
                    move_hash = hash(str(n_sol))
                    candidates.append((n_sol, n_name, move_hash))
        
        if not candidates:
            return self.current_solution, self.current_fitness
            
        # Sélection du meilleur candidat non tabou
        best_neighbor = None
        best_neighbor_fit = float('inf')
        best_neighbor_name = None
        best_move_hash = None
        
        for sol, name, move_hash in candidates:
            fit, _, _ = self.env.evaluate(sol)
            
            is_tabu = move_hash in self.tabu_list
            is_aspiration = fit < self.best_fitness
            
            if (not is_tabu or is_aspiration) and fit < best_neighbor_fit:
                best_neighbor = sol
                best_neighbor_fit = fit
                best_neighbor_name = name
                best_move_hash = move_hash
        
        # Mise à jour
        if best_neighbor:
            self.current_solution = best_neighbor
            self.current_fitness = best_neighbor_fit
            
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.env.copy_solution(self.current_solution)
                self.best_fitness = self.current_fitness
            
            # Gestion Tabou
            self.tabu_list.append(best_move_hash)
            if len(self.tabu_list) > self.tabu_tenure:
                self.tabu_list.pop(0)
                
            # Feedback Q-Learning
            if self.q_selector and best_neighbor_name:
                self.q_selector.update_with_result(best_neighbor_name, self.current_fitness)
                
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness

class SimulatedAnnealingAgent(BaseAgent):
    """Agent Recuit Simulé."""
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 initial_temp: float = 100.0, cooling_rate: float = 0.99,
                 min_temp: float = 0.1, use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
    def optimize_step(self) -> Tuple[Dict, float]:
        self.iterations_active += 1
        
        # Choix du voisinage
        if self.use_qlearning and self.q_selector:
            n_name = self.q_selector.select_neighborhood()
        else:
            n_name = random.choice(['C', 'E'])
            
        # Génération voisin
        neighbor = self.neighborhood_manager.generate_neighbor(
            self.current_solution, n_name, self.env.skills, self.env.max_ops
        )
        
        if not neighbor:
            return self.current_solution, self.current_fitness
            
        new_fit, _, _ = self.env.evaluate(neighbor)
        delta = new_fit - self.current_fitness
        
        accept = False
        if delta < 0:
            accept = True
        elif self.temperature > self.min_temp:
            # Probabilité d'acceptation de Boltzmann
            try:
                prob = math.exp(-delta / self.temperature)
            except OverflowError:
                prob = 0
            if random.random() < prob:
                accept = True
        
        if accept:
            self.current_solution = neighbor
            self.current_fitness = new_fit
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.env.copy_solution(self.current_solution)
                self.best_fitness = self.current_fitness
        
        # Refroidissement
        self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)
        
        # Feedback Q-Learning
        if self.q_selector:
            # On donne un feedback positif si accepté et améliorant
            self.q_selector.update_with_result(n_name, self.current_fitness)
            
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness

class CollaborationMode:
    FRIENDS = "friends"
    ENEMIES = "enemies"

class MultiAgentSystem:
    """Système Multi-Agents gérant la coopération."""
    
    def __init__(self, environment: SchedulingEnvironment,
                 mode: str = CollaborationMode.FRIENDS,
                 use_qlearning: bool = True):
        self.env = environment
        self.mode = mode
        self.use_qlearning = use_qlearning
        self.agents: Dict[str, BaseAgent] = {}
        
        # Pool de mémoire partagée
        if mode == CollaborationMode.FRIENDS:
            # EMP Standard avec diversité
            self.shared_memory = SharedMemoryPool(max_size=25, min_distance=2)
        else:
            # Pool Élite pour le mode Ennemis (juste les meilleures stats/solutions)
            self.shared_memory = ElitePool(max_size=5)
            
        self.global_best_solution: Optional[Solution] = None
        self.global_best_fitness: float = float('inf')
        self.iteration = 0

    def add_agent(self, agent_type: str, agent_id: str, **kwargs):
        if agent_type == 'genetic':
            agent = GeneticAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        elif agent_type == 'tabu':
            agent = TabuAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        elif agent_type == 'sa':
            agent = SimulatedAnnealingAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        else:
            raise ValueError(f"Type d'agent inconnu: {agent_type}")
        
        self.agents[agent_id] = agent
        return agent

    def run(self, n_iterations: int, verbose: bool = True) -> Optional[Solution]:
        # Initialisation
        for agent in self.agents.values():
            agent.initialize()
            sol = agent.get_solution()
            self.shared_memory.insert(sol)
            if sol.fitness < self.global_best_fitness:
                self.global_best_fitness = sol.fitness
                self.global_best_solution = sol

        for i in range(n_iterations):
            self.iteration += 1
            
            for agent_id, agent in self.agents.items():
                # --- Étape de Collaboration ---
                if self.mode == CollaborationMode.FRIENDS:
                    # Chance de piocher dans l'EMP pour se diversifier
                    if random.random() < 0.1 and self.shared_memory.solutions:
                        # On prend une solution de l'EMP
                        # (Idéalement une solution différente de la sienne)
                        diverse_sol = self.shared_memory.solutions[random.randint(0, len(self.shared_memory.solutions)-1)]
                        # On injecte cette solution dans l'agent (réinitialisation partielle)
                        agent.set_solution(diverse_sol.sequences, diverse_sol.fitness)
                
                # --- Étape d'Optimisation ---
                new_sol, new_fit = agent.optimize_step()
                
                # --- Mise à jour Globale ---
                if new_fit < self.global_best_fitness:
                    self.global_best_fitness = new_fit
                    self.global_best_solution = agent.get_solution()
                    if verbose:
                        print(f"  > New Best Global: {new_fit} by {agent_id}")

                # --- Partage vers EMP ---
                # On essaie d'insérer la solution trouvée
                sol_obj = agent.get_solution()
                self.shared_memory.insert(sol_obj, self.iteration)
        
        return self.global_best_solution

    def get_statistics(self):
        return {
            'global_best_fitness': self.global_best_fitness,
            'emp_stats': self.shared_memory.get_statistics(),
            'agents': list(self.agents.keys())
        }
