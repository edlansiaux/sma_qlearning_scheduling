"""
Multi-Agent System Module - Agents métaheuristiques avec collaboration
Basé sur le diaporama: Modes de collaboration Amis et Ennemis

Mode Amis:
- Les agents partagent les solutions complètes via l'EMP
- Chaque agent peut utiliser les solutions des autres
- Garantir la diversité de l'EMP

Mode Ennemis:
- Les agents ne partagent que les valeurs de critère (fitness)
- Chaque agent essaie de trouver la meilleure solution
- Un agent ne travaille que si une meilleure solution existe
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple
from abc import ABC, abstractmethod
import random
import copy
import math

from core.environment import SchedulingEnvironment, Task
from core.neighborhoods import NeighborhoodManager
from core.qlearning import QLearningAgent, AdaptiveNeighborhoodSelector
from core.shared_memory import SharedMemoryPool, Solution, ElitePool


class BaseAgent(ABC):
    """
    Classe de base pour tous les agents métaheuristiques.
    
    Chaque agent:
    - Possède une solution courante
    - Peut communiquer avec les autres agents (selon le mode)
    - Utilise une métaheuristique pour améliorer sa solution
    """
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 use_qlearning: bool = True):
        """
        Args:
            agent_id: Identifiant unique de l'agent
            environment: Environnement d'ordonnancement
            use_qlearning: Utiliser Q-Learning pour sélection des voisinages
        """
        self.agent_id = agent_id
        self.env = environment
        self.use_qlearning = use_qlearning
        
        # Solution courante et meilleure solution de l'agent
        self.current_solution: Optional[Dict[Tuple[int, int], List[Task]]] = None
        self.current_fitness: float = float('inf')
        self.best_solution: Optional[Dict[Tuple[int, int], List[Task]]] = None
        self.best_fitness: float = float('inf')
        
        # Gestionnaire de voisinages et Q-Learning
        self.neighborhood_manager = NeighborhoodManager()
        if use_qlearning:
            self.q_selector = AdaptiveNeighborhoodSelector(
                alpha=0.15, gamma=0.9, epsilon=0.4
            )
        else:
            self.q_selector = None
        
        # Historique
        self.fitness_history: List[float] = []
        self.iterations_active = 0
    
    def initialize(self, random_init: bool = True):
        """Initialise l'agent avec une solution initiale."""
        self.current_solution = self.env.build_initial_solution(random_order=random_init)
        self.current_fitness, _, _ = self.env.evaluate(self.current_solution)
        self.best_solution = self.env.copy_solution(self.current_solution)
        self.best_fitness = self.current_fitness
        
        if self.q_selector:
            self.q_selector.reset(self.current_fitness)
        
        self.fitness_history = [self.current_fitness]
    
    def set_solution(self, solution: Dict[Tuple[int, int], List[Task]], fitness: float = None):
        """Définit la solution courante de l'agent."""
        self.current_solution = self.env.copy_solution(solution)
        if fitness is None:
            self.current_fitness, _, _ = self.env.evaluate(self.current_solution)
        else:
            self.current_fitness = fitness
        
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness
    
    @abstractmethod
    def optimize_step(self) -> Tuple[Dict[Tuple[int, int], List[Task]], float]:
        """
        Effectue une étape d'optimisation.
        
        Returns:
            (nouvelle_solution, nouvelle_fitness)
        """
        pass
    
    def get_solution(self) -> Solution:
        """Retourne la solution courante encapsulée."""
        return Solution(
            sequences=self.env.copy_solution(self.current_solution),
            fitness=self.current_fitness,
            agent_id=self.agent_id
        )
    
    def get_best_solution(self) -> Solution:
        """Retourne la meilleure solution trouvée."""
        return Solution(
            sequences=self.env.copy_solution(self.best_solution),
            fitness=self.best_fitness,
            agent_id=self.agent_id
        )


class GeneticAgent(BaseAgent):
    """
    Agent utilisant un Algorithme Génétique.
    
    Opérateurs:
    - Sélection par tournoi
    - Croisement ordonné (OX)
    - Mutation par swap
    """
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 population_size: int = 15, mutation_rate: float = 0.1,
                 use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Tuple[Dict, float]] = []
    
    def initialize(self, random_init: bool = True):
        super().initialize(random_init)
        # Initialiser la population
        self.population = []
        for _ in range(self.population_size):
            sol = self.env.build_initial_solution(random_order=True)
            fitness, _, _ = self.env.evaluate(sol)
            self.population.append((sol, fitness))
        
        # Trier par fitness
        self.population.sort(key=lambda x: x[1])
        self.current_solution = self.env.copy_solution(self.population[0][0])
        self.current_fitness = self.population[0][1]
        
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness
    
    def _tournament_selection(self, k: int = 3) -> Dict:
        """Sélection par tournoi."""
        competitors = random.sample(self.population, min(k, len(self.population)))
        winner = min(competitors, key=lambda x: x[1])
        return self.env.copy_solution(winner[0])
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Croisement ordonné adapté aux séquences de tâches."""
        child = {}
        for key in parent1.keys():
            if key in parent2 and len(parent1[key]) > 1:
                # Croisement OX sur cette file
                seq1 = parent1[key]
                seq2 = parent2[key]
                
                if len(seq1) != len(seq2):
                    child[key] = seq1[:]
                    continue
                
                n = len(seq1)
                a, b = sorted(random.sample(range(n), 2))
                
                # Créer l'enfant
                child_seq = [None] * n
                child_seq[a:b+1] = seq1[a:b+1]
                
                # Remplir avec parent2
                patients_in_child = set(t.i for t in child_seq if t is not None)
                remaining = [t for t in seq2 if t.i not in patients_in_child]
                
                pos = 0
                for i in range(n):
                    if child_seq[i] is None and pos < len(remaining):
                        child_seq[i] = remaining[pos]
                        pos += 1
                
                # Compléter si nécessaire
                if None in child_seq:
                    child_seq = seq1[:]
                
                child[key] = child_seq
            else:
                child[key] = parent1.get(key, [])[:] if key in parent1 else parent2.get(key, [])[:]
        
        return child
    
    def _mutate(self, solution: Dict) -> Dict:
        """Mutation par swap."""
        mutated = self.env.copy_solution(solution)
        
        if self.use_qlearning and self.q_selector:
            # Utiliser Q-Learning pour choisir le voisinage
            neighborhood = self.q_selector.select_neighborhood()
            neighbor = self.neighborhood_manager.generate_neighbor(
                mutated, neighborhood, self.env.skills, self.env.max_ops
            )
            if neighbor:
                return neighbor
        else:
            # Mutation simple
            for key in mutated:
                if len(mutated[key]) >= 2 and random.random() < self.mutation_rate:
                    i, j = random.sample(range(len(mutated[key])), 2)
                    mutated[key][i], mutated[key][j] = mutated[key][j], mutated[key][i]
        
        return mutated
    
    def optimize_step(self) -> Tuple[Dict, float]:
        """Une génération de l'algorithme génétique."""
        self.iterations_active += 1
        
        # Nouvelle population
        new_population = []
        
        # Élitisme: garder les meilleurs
        n_elite = max(1, self.population_size // 5)
        for sol, fit in self.population[:n_elite]:
            new_population.append((self.env.copy_solution(sol), fit))
        
        # Croisement et mutation
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            
            fitness, _, _ = self.env.evaluate(child)
            new_population.append((child, fitness))
        
        # Trier et mettre à jour
        new_population.sort(key=lambda x: x[1])
        self.population = new_population[:self.population_size]
        
        # Mettre à jour solution courante
        best_sol, best_fit = self.population[0]
        self.current_solution = self.env.copy_solution(best_sol)
        self.current_fitness = best_fit
        
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.env.copy_solution(self.current_solution)
            self.best_fitness = self.current_fitness
        
        # Mise à jour Q-Learning
        if self.q_selector:
            self.q_selector.update_with_result('A', self.current_fitness)
        
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness


class TabuAgent(BaseAgent):
    """
    Agent utilisant la Recherche Tabou.
    
    Maintient une liste tabou des mouvements récents pour éviter les cycles.
    """
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 tabu_tenure: int = 10, candidate_limit: int = 50,
                 use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.tabu_tenure = tabu_tenure
        self.candidate_limit = candidate_limit
        self.tabu_list: List[Tuple] = []
    
    def _generate_candidates(self) -> List[Tuple[Dict, str, Tuple]]:
        """Génère des solutions candidates."""
        candidates = []
        
        if self.use_qlearning and self.q_selector:
            # Utiliser Q-Learning pour choisir les voisinages
            for _ in range(self.candidate_limit):
                neighborhood = self.q_selector.select_neighborhood()
                neighbor = self.neighborhood_manager.generate_neighbor(
                    self.current_solution, neighborhood,
                    self.env.skills, self.env.max_ops
                )
                if neighbor:
                    move = (neighborhood, hash(str(neighbor)))
                    candidates.append((neighbor, neighborhood, move))
        else:
            # Génération uniforme
            all_neighbors = self.neighborhood_manager.generate_all_neighbors(
                self.current_solution, self.env.skills, self.env.max_ops,
                n_per_neighborhood=self.candidate_limit // 5
            )
            for neighbor, name in all_neighbors:
                move = (name, hash(str(neighbor)))
                candidates.append((neighbor, name, move))
        
        return candidates[:self.candidate_limit]
    
    def optimize_step(self) -> Tuple[Dict, float]:
        """Une itération de la recherche tabou."""
        self.iterations_active += 1
        
        # Générer les candidats
        candidates = self._generate_candidates()
        
        if not candidates:
            self.fitness_history.append(self.current_fitness)
            return self.current_solution, self.current_fitness
        
        # Évaluer et filtrer
        best_neighbor = None
        best_fitness = float('inf')
        best_move = None
        best_neighborhood = None
        
        for neighbor, neighborhood, move in candidates:
            fitness, _, _ = self.env.evaluate(neighbor)
            
            # Vérifier si le mouvement est tabou (sauf aspiration)
            is_tabu = move in self.tabu_list
            is_aspiration = fitness < self.best_fitness
            
            if (not is_tabu or is_aspiration) and fitness < best_fitness:
                best_neighbor = neighbor
                best_fitness = fitness
                best_move = move
                best_neighborhood = neighborhood
        
        if best_neighbor is not None:
            self.current_solution = best_neighbor
            self.current_fitness = best_fitness
            
            # Mettre à jour la meilleure solution
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.env.copy_solution(self.current_solution)
                self.best_fitness = self.current_fitness
            
            # Mettre à jour la liste tabou
            self.tabu_list.append(best_move)
            if len(self.tabu_list) > self.tabu_tenure:
                self.tabu_list.pop(0)
            
            # Mise à jour Q-Learning
            if self.q_selector and best_neighborhood:
                self.q_selector.update_with_result(best_neighborhood, self.current_fitness)
        
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness


class SimulatedAnnealingAgent(BaseAgent):
    """
    Agent utilisant le Recuit Simulé.
    
    Accepte des solutions moins bonnes avec une probabilité décroissante.
    """
    
    def __init__(self, agent_id: str, environment: SchedulingEnvironment,
                 initial_temp: float = 100.0, cooling_rate: float = 0.995,
                 min_temp: float = 0.01, use_qlearning: bool = True):
        super().__init__(agent_id, environment, use_qlearning)
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def optimize_step(self) -> Tuple[Dict, float]:
        """Une itération du recuit simulé."""
        self.iterations_active += 1
        
        # Choisir un voisinage
        if self.use_qlearning and self.q_selector:
            neighborhood = self.q_selector.select_neighborhood()
        else:
            neighborhood = random.choice(['A', 'B', 'C', 'D', 'E'])
        
        # Générer un voisin
        neighbor = self.neighborhood_manager.generate_neighbor(
            self.current_solution, neighborhood,
            self.env.skills, self.env.max_ops
        )
        
        if neighbor is None:
            self.fitness_history.append(self.current_fitness)
            return self.current_solution, self.current_fitness
        
        # Évaluer
        new_fitness, _, _ = self.env.evaluate(neighbor)
        delta = new_fitness - self.current_fitness
        
        # Décision d'acceptation
        accept = False
        if delta < 0:
            accept = True
        elif self.temperature > 0:
            probability = math.exp(-delta / self.temperature)
            accept = random.random() < probability
        
        if accept:
            self.current_solution = neighbor
            self.current_fitness = new_fitness
            
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.env.copy_solution(self.current_solution)
                self.best_fitness = self.current_fitness
        
        # Refroidissement
        self.temperature *= self.cooling_rate
        self.temperature = max(self.min_temp, self.temperature)
        
        # Mise à jour Q-Learning
        if self.q_selector:
            reward = -delta if accept else 0  # Récompense basée sur amélioration
            self.q_selector.update_with_result(neighborhood, self.current_fitness)
        
        self.fitness_history.append(self.current_fitness)
        return self.current_solution, self.current_fitness
    
    def reset_temperature(self):
        """Réinitialise la température."""
        self.temperature = self.initial_temp


# ============================================
# MODES DE COLLABORATION
# ============================================

class CollaborationMode:
    """Classe de base pour les modes de collaboration."""
    
    FRIENDS = "friends"
    ENEMIES = "enemies"


class MultiAgentSystem:
    """
    Système Multi-Agents avec modes de collaboration.
    
    Mode Amis (Friends):
    - Les agents partagent les solutions complètes via l'EMP
    - Utilisation de solutions des autres comme point de départ
    - Diversité maintenue dans l'EMP
    
    Mode Ennemis (Enemies):
    - Les agents ne partagent que les valeurs de fitness
    - Un agent ne travaille que si une meilleure solution existe
    - Compétition pour trouver la meilleure solution
    """
    
    def __init__(self, environment: SchedulingEnvironment,
                 mode: str = CollaborationMode.FRIENDS,
                 use_qlearning: bool = True):
        """
        Args:
            environment: Environnement d'ordonnancement
            mode: Mode de collaboration ('friends' ou 'enemies')
            use_qlearning: Utiliser Q-Learning pour l'auto-adaptation
        """
        self.env = environment
        self.mode = mode
        self.use_qlearning = use_qlearning
        
        # Agents
        self.agents: Dict[str, BaseAgent] = {}
        
        # Mémoires partagées
        if mode == CollaborationMode.FRIENDS:
            self.shared_memory = SharedMemoryPool(
                max_size=30, min_distance=5, diversity_threshold=0.3
            )
        else:
            self.shared_memory = ElitePool(max_size=10)
        
        # Meilleure solution globale
        self.global_best_solution: Optional[Solution] = None
        self.global_best_fitness: float = float('inf')
        
        # Historique
        self.iteration = 0
        self.fitness_history: List[float] = []
        self.agent_contributions: Dict[str, int] = {}
    
    def add_agent(self, agent_type: str, agent_id: str = None, **kwargs) -> BaseAgent:
        """
        Ajoute un agent au système.
        
        Args:
            agent_type: Type d'agent ('genetic', 'tabu', 'sa')
            agent_id: Identifiant (auto-généré si None)
            **kwargs: Paramètres supplémentaires pour l'agent
        """
        if agent_id is None:
            agent_id = f"{agent_type}_{len(self.agents)}"
        
        if agent_type == 'genetic':
            agent = GeneticAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        elif agent_type == 'tabu':
            agent = TabuAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        elif agent_type == 'sa':
            agent = SimulatedAnnealingAgent(agent_id, self.env, use_qlearning=self.use_qlearning, **kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agents[agent_id] = agent
        self.agent_contributions[agent_id] = 0
        return agent
    
    def initialize_agents(self):
        """Initialise tous les agents."""
        for agent in self.agents.values():
            agent.initialize(random_init=True)
            
            # Ajouter la solution initiale à l'EMP
            sol = agent.get_solution()
            self.shared_memory.insert(sol, self.iteration)
            
            if sol.fitness < self.global_best_fitness:
                self.global_best_fitness = sol.fitness
                self.global_best_solution = sol
    
    def run_iteration_friends(self):
        """
        Exécute une itération en mode Amis.
        
        Les agents partagent leurs solutions via l'EMP.
        """
        for agent_id, agent in self.agents.items():
            # Option: utiliser une solution de l'EMP comme point de départ
            if random.random() < 0.3 and self.shared_memory.solutions:
                # Prendre une solution diverse de l'EMP
                emp_sol = self.shared_memory.get_random_solution()
                if emp_sol and emp_sol.agent_id != agent_id:
                    agent.set_solution(emp_sol.sequences, emp_sol.fitness)
            
            # Optimisation
            new_sol, new_fit = agent.optimize_step()
            
            # Ajouter à l'EMP si amélioration
            solution = Solution(
                sequences=self.env.copy_solution(new_sol),
                fitness=new_fit,
                agent_id=agent_id
            )
            
            inserted = self.shared_memory.insert(solution, self.iteration)
            
            # Mettre à jour le meilleur global
            if new_fit < self.global_best_fitness:
                self.global_best_fitness = new_fit
                self.global_best_solution = solution
                self.agent_contributions[agent_id] += 1
    
    def run_iteration_enemies(self):
        """
        Exécute une itération en mode Ennemis.
        
        Les agents ne partagent que les valeurs de fitness.
        Un agent ne travaille que si une meilleure solution existe.
        """
        for agent_id, agent in self.agents.items():
            # Vérifier si l'agent doit travailler
            # (seulement si une meilleure solution globale existe)
            should_work = agent.best_fitness >= self.global_best_fitness
            
            if should_work or random.random() < 0.2:  # 20% de chance de travailler quand même
                # Optimisation
                new_sol, new_fit = agent.optimize_step()
                
                # Enregistrer seulement la fitness (pas la solution)
                solution = Solution(
                    sequences=self.env.copy_solution(new_sol),
                    fitness=new_fit,
                    agent_id=agent_id
                )
                
                # Mettre à jour le pool élite
                self.shared_memory.insert(solution, self.iteration)
                
                # Mettre à jour le meilleur global
                if new_fit < self.global_best_fitness:
                    self.global_best_fitness = new_fit
                    self.global_best_solution = solution
                    self.agent_contributions[agent_id] += 1
    
    def run_iteration(self):
        """Exécute une itération selon le mode configuré."""
        self.iteration += 1
        
        if self.mode == CollaborationMode.FRIENDS:
            self.run_iteration_friends()
        else:
            self.run_iteration_enemies()
        
        self.fitness_history.append(self.global_best_fitness)
    
    def run(self, n_iterations: int, verbose: bool = True) -> Solution:
        """
        Exécute le système multi-agents.
        
        Args:
            n_iterations: Nombre d'itérations
            verbose: Afficher les progrès
        
        Returns:
            Meilleure solution trouvée
        """
        self.initialize_agents()
        
        for i in range(n_iterations):
            self.run_iteration()
            
            if verbose and (i + 1) % 50 == 0:
                print(f"Iteration {i+1}/{n_iterations} | "
                      f"Best: {self.global_best_fitness} | "
                      f"EMP size: {len(self.shared_memory.solutions)}")
        
        return self.global_best_solution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du système."""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = {
                'best_fitness': agent.best_fitness,
                'iterations_active': agent.iterations_active,
                'contributions': self.agent_contributions[agent_id]
            }
        
        return {
            'mode': self.mode,
            'iterations': self.iteration,
            'global_best_fitness': self.global_best_fitness,
            'emp_stats': self.shared_memory.get_statistics(),
            'agent_stats': agent_stats,
            'fitness_history': self.fitness_history
        }
