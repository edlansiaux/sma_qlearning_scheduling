"""
core/qlearning.py - Module Q-Learning remasterisé et dynamique.
"""

import numpy as np
from typing import Dict, List, Optional
import random
from collections import defaultdict

class QLearningAgent:
    """
    Agent Q-Learning adaptatif.
    S'adapte automatiquement aux voisinages disponibles.
    """
    
    def __init__(self, states: List[str], 
                 alpha: float = 0.1, 
                 gamma: float = 0.9, 
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        # Validation : si aucun état n'est fourni, on met une valeur par défaut,
        # mais idéalement on doit recevoir la liste des voisinages actifs.
        self.states = states if states else ['C', 'E'] 
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = np.zeros((self.n_states, self.n_states))
        self.current_state = random.choice(self.states)
        
        self.action_history = []
        self.reward_history = []
    
    def select_action(self, exploit_only: bool = False) -> str:
        # Si on n'a qu'un seul choix (ex: que E), pas besoin de réfléchir
        if self.n_states == 1:
            return self.states[0]

        state_idx = self.state_to_idx[self.current_state]
        
        if not exploit_only and random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_states - 1)
        else:
            action_idx = np.argmax(self.q_table[state_idx])
        
        action = self.idx_to_state[action_idx]
        self.action_history.append(action)
        return action
    
    def update(self, action: str, reward: float, next_state: str = None):
        if action not in self.state_to_idx:
            return # Sécurité contre des actions invalides

        state_idx = self.state_to_idx[self.current_state]
        action_idx = self.state_to_idx[action]
        
        if next_state is None:
            next_state = action
            
        if next_state in self.state_to_idx:
            next_state_idx = self.state_to_idx[next_state]
            max_next_q = np.max(self.q_table[next_state_idx])
        else:
            max_next_q = 0 # Fallback
        
        old_q = self.q_table[state_idx, action_idx]
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_idx, action_idx] = new_q
        
        self.reward_history.append(reward)
        self.current_state = next_state
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def reset_state(self, state: str = None):
        if state and state in self.states:
            self.current_state = state
        else:
            self.current_state = random.choice(self.states)
            
    def get_q_table_formatted(self) -> Dict[str, Dict[str, float]]:
        q_dict = {}
        for s in self.states:
            q_dict[s] = {}
            for a in self.states:
                idx_s = self.state_to_idx[s]
                idx_a = self.state_to_idx[a]
                q_dict[s][a] = float(self.q_table[idx_s, idx_a])
        return q_dict


class AdaptiveNeighborhoodSelector:
    """
    Sélecteur qui ne propose QUE les voisinages actifs (C et E).
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.3, reward_scale: float = 1.0):
        
        # OPTIMISATION : On restreint explicitement aux voisinages qui fonctionnent
        # pour le problème strict (Swap et Insertion).
        active_neighborhoods = ['C', 'E']
        
        self.q_agent = QLearningAgent(
            states=active_neighborhoods,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
        self.reward_scale = reward_scale
        self.last_makespan = None
        self.best_makespan = float('inf')
        
        self.neighborhood_stats = defaultdict(lambda: {
            'calls': 0, 'improvements': 0, 'total_improvement': 0.0
        })
    
    def select_neighborhood(self) -> str:
        return self.q_agent.select_action()
    
    def update_with_result(self, neighborhood: str, new_makespan: float):
        if self.last_makespan is not None:
            improvement = self.last_makespan - new_makespan
            reward = improvement * self.reward_scale
            if new_makespan < self.best_makespan:
                reward += 10.0 # Gros bonus pour nouveau record
                self.best_makespan = new_makespan
        else:
            reward = 0.0
        
        stats = self.neighborhood_stats[neighborhood]
        stats['calls'] += 1
        if reward > 0:
            stats['improvements'] += 1
            stats['total_improvement'] += reward
        
        self.q_agent.update(neighborhood, reward)
        self.q_agent.decay_epsilon()
        self.last_makespan = new_makespan
    
    def reset(self, initial_makespan: float = None):
        self.last_makespan = initial_makespan
        self.best_makespan = initial_makespan if initial_makespan else float('inf')
        self.q_agent.reset_state()
        
    def get_statistics(self):
        return dict(self.neighborhood_stats)
    
    def get_q_table(self):
        return self.q_agent.get_q_table_formatted()
