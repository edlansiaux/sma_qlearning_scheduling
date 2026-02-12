"""
core/agents.py - Agents métaheuristiques conformes à l'Algorithme 4 du diaporama.

Agents disponibles :
  - GeneticAgent     (AG)   : Algorithme Génétique
  - TabuAgent        (Tabou): Recherche Tabou
  - SimulatedAnnealingAgent (RS) : Recuit Simulé

Modes de collaboration (Algorithme 4) :
  - FRIENDS  : partage de solutions complètes via l'EMP
  - ENEMIES  : partage des fitness uniquement (compétition)

Chaque agent peut activer le Q-Learning pour le choix adaptatif du voisinage
parmi les 5 fonctions {A, B, C, D, E}.
"""

from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import random
import math
import copy

from core.environment import SchedulingEnvironment, Task
from core.neighborhoods import NeighborhoodManager
from core.qlearning import AdaptiveNeighborhoodSelector
from core.shared_memory import SharedMemoryPool, Solution, ElitePool

# Voisinages disponibles sans Q-Learning (choix aléatoire)
ALL_NEIGHBORHOODS = ["A", "B", "C", "D", "E"]


class BaseAgent(ABC):
    """Classe de base abstraite pour les agents."""

    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 use_qlearning: bool = True):
        self.agent_id = agent_id
        self.env = environment
        self.use_qlearning = use_qlearning

        self.current_solution: Optional[Dict[Tuple[int, int], List[Task]]] = None
        self.current_fitness: float = float("inf")
        self.best_solution: Optional[Dict[Tuple[int, int], List[Task]]] = None
        self.best_fitness: float = float("inf")

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
            agent_id=self.agent_id,
        )

    # ── Utilitaire commun : choix du voisinage ──

    def _pick_neighborhood(self) -> str:
        """Choisit un voisinage via Q-Learning ou aléatoirement."""
        if self.use_qlearning and self.q_selector:
            return self.q_selector.select_neighborhood()
        return random.choice(ALL_NEIGHBORHOODS)

    def _feedback_qlearning(self, neighborhood: str):
        if self.q_selector:
            self.q_selector.update_with_result(neighborhood, self.current_fitness)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithme Génétique
# ══════════════════════════════════════════════════════════════════════════════

class GeneticAgent(BaseAgent):
    """Agent Algorithme Génétique (AG)."""

    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 population_size: int = 15, mutation_rate: float = 0.15,
                 use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Tuple[Dict, float]] = []

    def initialize(self, random_init: bool = True):
        super().initialize(random_init)
        self.population = []
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
        child: Dict = {}
        all_keys = set(parent1.keys()) | set(parent2.keys())

        for key in all_keys:
            seq1 = parent1.get(key, [])
            seq2 = parent2.get(key, [])

            if len(seq1) < 2 or len(seq2) < 2:
                child[key] = copy.deepcopy(seq1) if seq1 else copy.deepcopy(seq2)
                continue

            n = len(seq1)
            a, b = sorted(random.sample(range(n), 2))
            child_seq: List[Optional[Task]] = [None] * n
            child_seq[a : b + 1] = seq1[a : b + 1]

            current_ids = {t.i for t in child_seq if t is not None}
            remaining = [t for t in seq2 if t.i not in current_ids]

            pos = 0
            for idx in range(n):
                if child_seq[idx] is None:
                    if pos < len(remaining):
                        child_seq[idx] = remaining[pos]
                        pos += 1
                    else:
                        child_seq[idx] = seq1[idx]

            child[key] = child_seq

        return child

    def _mutate(self, solution: Dict) -> Dict:
        """Mutation via un voisinage choisi par Q-Learning ou aléatoirement."""
        mutated = self.env.copy_solution(solution)
        n_name = self._pick_neighborhood()
        neighbor = self.neighborhood_manager.generate_neighbor(
            mutated, n_name, self.env.skills, self.env.max_ops
        )
        return neighbor if neighbor else mutated

    def optimize_step(self) -> Tuple[Dict, float]:
        self.iterations_active += 1
        new_population = []

        # Élitisme
        n_elite = max(1, self.population_size // 5)
        new_population.extend(self.population[:n_elite])

        last_neighborhood = "C"

        while len(new_population) < self.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._crossover(p1, p2)
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
                last_neighborhood = self._pick_neighborhood()
            fit, _, _ = self.env.evaluate(child)
            new_population.append((child, fit))

        new_population.sort(key=lambda x: x[1])
        self.population = new_population[: self.population_size]

        best_sol, best_fit = self.population[0]
        self.current_solution = self.env.copy_solution(best_sol)
        self.current_fitness = best_fit

        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness

        self._feedback_qlearning(last_neighborhood)
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness


# ══════════════════════════════════════════════════════════════════════════════
# Recherche Tabou
# ══════════════════════════════════════════════════════════════════════════════

class TabuAgent(BaseAgent):
    """Agent Recherche Tabou."""

    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 tabu_tenure: int = 10, candidate_limit: int = 20,
                 use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.tabu_tenure = tabu_tenure
        self.candidate_limit = candidate_limit
        self.tabu_list: List[int] = []

    def optimize_step(self) -> Tuple[Dict, float]:
        self.iterations_active += 1

        candidates = []
        for _ in range(self.candidate_limit):
            n_name = self._pick_neighborhood()
            n_sol = self.neighborhood_manager.generate_neighbor(
                self.current_solution, n_name, self.env.skills, self.env.max_ops
            )
            if n_sol:
                move_hash = hash(str(n_sol))
                candidates.append((n_sol, n_name, move_hash))

        if not candidates:
            self.fitness_history.append(self.current_fitness)
            return self.current_solution, self.current_fitness

        best_neighbor = None
        best_neighbor_fit = float("inf")
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

        if best_neighbor:
            self.current_solution = best_neighbor
            self.current_fitness = best_neighbor_fit

            if self.current_fitness < self.best_fitness:
                self.best_solution = self.env.copy_solution(self.current_solution)
                self.best_fitness = self.current_fitness

            self.tabu_list.append(best_move_hash)
            if len(self.tabu_list) > self.tabu_tenure:
                self.tabu_list.pop(0)

            if best_neighbor_name:
                self._feedback_qlearning(best_neighbor_name)

        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness


# ══════════════════════════════════════════════════════════════════════════════
# Recuit Simulé
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedAnnealingAgent(BaseAgent):
    """Agent Recuit Simulé (RS)."""

    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 initial_temp: float = 100.0, cooling_rate: float = 0.99,
                 min_temp: float = 0.1, use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def optimize_step(self) -> Tuple[Dict, float]:
        self.iterations_active += 1

        n_name = self._pick_neighborhood()
        neighbor = self.neighborhood_manager.generate_neighbor(
            self.current_solution, n_name, self.env.skills, self.env.max_ops
        )

        if not neighbor:
            self.fitness_history.append(self.current_fitness)
            return self.current_solution, self.current_fitness

        new_fit, _, _ = self.env.evaluate(neighbor)
        delta = new_fit - self.current_fitness

        accept = False
        if delta <= 0:
            accept = True
        elif self.temperature > self.min_temp:
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

        self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)
        self._feedback_qlearning(n_name)
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness


# ══════════════════════════════════════════════════════════════════════════════
# Modes de collaboration
# ══════════════════════════════════════════════════════════════════════════════

class CollaborationMode:
    FRIENDS = "friends"
    ENEMIES = "enemies"


# ══════════════════════════════════════════════════════════════════════════════
# Système Multi-Agents (Algorithme 4)
# ══════════════════════════════════════════════════════════════════════════════

class MultiAgentSystem:
    """
    Système Multi-Agents conforme à l'Algorithme 4 :
      - Mode FRIENDS : partage de solutions via EMP, adoption avec prob. 0.2
      - Mode ENEMIES : partage de fitness, adoption forcée si EMP meilleure
    """

    def __init__(
        self,
        environment: SchedulingEnvironment,
        mode: str = CollaborationMode.FRIENDS,
        use_qlearning: bool = True,
    ):
        self.env = environment
        self.mode = mode
        self.use_qlearning = use_qlearning
        self.agents: Dict[str, BaseAgent] = {}

        if mode == CollaborationMode.FRIENDS:
            self.shared_memory = SharedMemoryPool(max_size=25, min_distance=2)
        else:
            self.shared_memory = ElitePool(max_size=5)

        self.global_best_solution: Optional[Solution] = None
        self.global_best_fitness: float = float("inf")
        self.iteration = 0

    def add_agent(self, agent_type: str, agent_id: str, **kwargs):
        if agent_type == "genetic":
            agent = GeneticAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        elif agent_type == "tabu":
            agent = TabuAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        elif agent_type == "sa":
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
                # ── Phase de collaboration (Algorithme 4, lignes 5–18) ──
                if self.mode == CollaborationMode.FRIENDS:
                    # Mode AMIS : adoption probabiliste (prob. 0.2) si EMP contient
                    # une meilleure solution
                    emp_best = self._get_emp_best()
                    if emp_best and emp_best.fitness < agent.current_fitness:
                        if random.random() < 0.2:
                            agent.set_solution(emp_best.sequences, emp_best.fitness)

                elif self.mode == CollaborationMode.ENEMIES:
                    # Mode ENNEMIS : adoption forcée si EMP meilleure
                    emp_best = self._get_emp_best()
                    if emp_best and emp_best.fitness < agent.current_fitness:
                        agent.set_solution(emp_best.sequences, emp_best.fitness)

                # ── Optimisation ──
                new_sol, new_fit = agent.optimize_step()

                # ── Mise à jour globale ──
                if new_fit < self.global_best_fitness:
                    self.global_best_fitness = new_fit
                    self.global_best_solution = agent.get_solution()
                    if verbose:
                        print(f"  > Iter {self.iteration}: New Best = {new_fit} by {agent_id}")

                # ── Insertion dans l'EMP ──
                sol_obj = agent.get_solution()
                self.shared_memory.insert(sol_obj, self.iteration)

        return self.global_best_solution

    def _get_emp_best(self) -> Optional[Solution]:
        if not self.shared_memory.solutions:
            return None
        return min(self.shared_memory.solutions, key=lambda s: s.fitness)

    def get_statistics(self):
        return {
            "global_best_fitness": self.global_best_fitness,
            "emp_stats": self.shared_memory.get_statistics(),
            "agents": list(self.agents.keys()),
        }
