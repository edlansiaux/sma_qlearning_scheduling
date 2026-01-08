"""
core/agents.py - Système Multi-Agents, Collaboration et Agents Métaheuristiques.
"""
from core.environment import SchedulingEnvironment, Solution
from core.neighborhoods import NeighborhoodManager
from core.shared_memory import SharedMemory
from core.qlearning import QLearningModel
import random

class MetaheuristicAgent:
    def __init__(self, id, env, strategy_type, use_learning=True):
        self.id = id
        self.env = env
        self.strategy_type = strategy_type # 'AG', 'Tabu', 'RS'
        self.use_learning = use_learning
        
        self.current_solution = None
        self.best_solution = None
        
        self.nm = NeighborhoodManager(env)
        # MDP pour l'auto-adaptation [cite: 764]
        self.brain = QLearningModel(actions=self.nm.moves) if use_learning else None
        self.last_fitness = None

    def initialize(self):
        self.current_solution = self.env.build_initial_solution()
        self.best_solution = self.env.copy_solution(self.current_solution)
        self.last_fitness = self.current_solution.fitness

    def step(self, emp: SharedMemory, collaboration_mode: str):
        # 1. Collaboration (Modes Amis/Ennemis) [cite: 618]
        if collaboration_mode == 'FRIENDS':
            # Mode Amis: Partage complet. On peut récupérer une solution de l'EMP [cite: 620]
            best_shared = emp.get_best()
            if best_shared and best_shared.fitness < self.current_solution.fitness:
                # Probabilité d'accepter l'aide (simulation de l'interaction)
                if random.random() < 0.2: 
                    self.current_solution = self.env.copy_solution(best_shared)

        elif collaboration_mode == 'ENEMIES':
            # Mode Ennemis: Pas d'accès aux solutions, seulement aux valeurs (critères) [cite: 578]
            best_shared = emp.get_best()
            # "Un agent ne travaille que si une meilleure solution que la sienne a été trouvée" [cite: 580]
            if best_shared and best_shared.fitness < self.current_solution.fitness:
                 # Si un "ennemi" a mieux, l'agent tente une intensification ou perturbation
                 # On simule cela par une perturbation via le voisinage A
                 self.current_solution = self.nm.apply_move(self.current_solution, 'A')

        # 2. Choix de l'action (Voisinage) via Q-Learning ou Aléatoire
        state = 1
        action = 'C' # Default
        
        if self.use_learning:
            # Déterminer l'état courant et choisir l'action via MDP [cite: 1167]
            state = self.brain.get_state(self.current_solution.fitness, self.last_fitness)
            action = self.brain.select_action(state)
        else:
            # Sans apprentissage : stratégies fixes par défaut
            if self.strategy_type == 'AG': action = random.choice(['A', 'B'])
            elif self.strategy_type == 'Tabu': action = random.choice(['C', 'D'])
            else: action = random.choice(['C', 'E'])

        # 3. Application du Mouvement (Voisinage)
        prev_fit = self.current_solution.fitness
        new_sol = self.nm.apply_move(self.current_solution, action)
        
        # 4. Acceptation (Logique Métaheuristique simplifiée)
        accept = False
        if new_sol.fitness <= prev_fit:
            accept = True
        else:
            # Recuit Simulé (RS): acceptation probabiliste
            if self.strategy_type == 'RS' and random.random() < 0.1:
                accept = True
        
        if accept:
            self.current_solution = new_sol
            if new_sol.fitness < self.best_solution.fitness:
                self.best_solution = self.env.copy_solution(new_sol)
                # Partage vers EMP si amélioration [cite: 623]
                emp.try_insert(self.env.copy_solution(self.best_solution))

        # 5. Récompense Q-Learning (Mise à jour MDP) [cite: 1176]
        if self.use_learning:
            # Reward positif si amélioration du temps d'attente/makespan
            reward = prev_fit - new_sol.fitness 
            next_state = self.brain.get_state(new_sol.fitness, prev_fit)
            self.brain.update(state, action, reward, next_state)
            
        self.last_fitness = self.current_solution.fitness
        return self.best_solution.fitness

class MultiAgentSystem:
    def __init__(self, env, agents_config, mode='FRIENDS'):
        self.env = env
        self.emp = SharedMemory()
        self.mode = mode
        self.agents = []
        for conf in agents_config:
            self.agents.append(MetaheuristicAgent(conf['id'], env, conf['type'], conf['learning']))

    def run(self, iterations=50):
        for a in self.agents: a.initialize()
        
        history = []
        for i in range(iterations):
            step_res = []
            for a in self.agents:
                fit = a.step(self.emp, self.mode)
                step_res.append(fit)
            history.append(min(step_res))
        
        return min(history)
