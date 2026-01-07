"""
core/agents.py - Agents mis à jour pour les imports corrects.
"""
from typing import Dict, List, Tuple, Optional
import random
import copy
from abc import ABC, abstractmethod

# Imports corrigés
from core.environment import SchedulingEnvironment, Task
from core.neighborhoods import NeighborhoodManager
from core.qlearning import AdaptiveNeighborhoodSelector
from core.shared_memory import SharedMemoryPool, Solution

class BaseAgent(ABC):
    def __init__(self, agent_id: str, environment: SchedulingEnvironment, use_qlearning: bool = True):
        self.agent_id = agent_id
        self.env = environment
        self.use_qlearning = use_qlearning
        
        self.current_solution = None
        self.current_fitness = float('inf')
        self.best_solution = None
        self.best_fitness = float('inf')
        
        self.neighborhood_manager = NeighborhoodManager()
        # Q-Learning actif ou non
        self.q_selector = AdaptiveNeighborhoodSelector() if use_qlearning else None
        
    def initialize(self):
        # Utilise la nouvelle méthode build_initial_solution de l'environnement
        self.current_solution = self.env.build_initial_solution(random_order=True)
        self.current_fitness, _, _ = self.env.evaluate(self.current_solution)
        self.best_solution = self.env.copy_solution(self.current_solution)
        self.best_fitness = self.current_fitness
        
        if self.q_selector:
            self.q_selector.reset(self.current_fitness)

    def get_solution(self) -> Solution:
        return Solution(self.env.copy_solution(self.current_solution), self.current_fitness, self.agent_id)

    @abstractmethod
    def optimize_step(self):
        pass

# ... (Le reste des classes GeneticAgent, TabuAgent, SimulatedAnnealingAgent reste identique 
# MAIS assurez-vous qu'elles n'importent pas 'Task' localement. 
# La logique interne utilisant self.env.evaluate() fonctionnera avec le nouveau moteur).

class CollaborationMode:
    FRIENDS = "friends"
    ENEMIES = "enemies"

class MultiAgentSystem:
    def __init__(self, environment, mode=CollaborationMode.FRIENDS, use_qlearning=True):
        self.env = environment
        self.mode = mode
        self.agents = {}
        self.shared_memory = SharedMemoryPool(max_size=20) # Utilise le nouveau Pool
        self.global_best_fitness = float('inf')
        self.global_best_solution = None
        self.use_qlearning = use_qlearning

    def add_agent(self, type_name, name, **kwargs):
        # Factory simple
        if type_name == 'genetic':
            from core.agents import GeneticAgent # Évite import circulaire si possible
            # Pour simplifier ici, supposons que les classes sont définies dans ce fichier
            # ou injectées.
            pass 
        # (Dans votre version complète, instanciez les classes définies plus haut)
        # Ici je mets juste la structure pour référence
        pass

    def run(self, n_iterations, verbose=False):
        # Initialisation
        for agent in self.agents.values():
            agent.initialize()
            # Insertion initiale dans EMP
            self.shared_memory.insert(agent.get_solution())

        for it in range(n_iterations):
            # Logique de collaboration (simplifiée pour l'exemple)
            for agent in self.agents.values():
                # Étape d'optimisation
                sol, fit = agent.optimize_step()
                
                # Mise à jour Global Best
                if fit < self.global_best_fitness:
                    self.global_best_fitness = fit
                    self.global_best_solution = sol
                
                # Interaction EMP (Mode AMIS)
                if self.mode == CollaborationMode.FRIENDS:
                    # Partage: Tenter d'insérer la solution trouvée
                    self.shared_memory.insert(Solution(sol, fit, agent.agent_id))
                    
                    # Récupération: Parfois prendre une solution de l'EMP pour diversifier
                    if random.random() < 0.1 and self.shared_memory.solutions:
                        better_sol = self.shared_memory.solutions[0] # Simplification
                        agent.current_solution = self.env.copy_solution(better_sol.sequences)
                        agent.current_fitness = better_sol.fitness

        return self.global_best_solution

    def get_statistics(self):
        return {
            'global_best_fitness': self.global_best_fitness,
            'emp_stats': self.shared_memory.get_statistics()
        }
