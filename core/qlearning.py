"""
core/qlearning.py - Apprentissage par Renforcement (Q-Learning) pour Auto-Adaptation.
Basé sur l'Algorithme 3 (Self Adaptive Agent) et l'Algorithme 4 (SelectAction)[cite: 1145, 1181].
"""
import numpy as np
import random

class QLearningModel:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions # ['A', 'B', 'C', 'D', 'E'] [cite: 1047]
        self.q_table = {} 
        self.alpha = alpha   # Facteur d'apprentissage [cite: 1149]
        self.gamma = gamma   # Facteur d'actualisation [cite: 1150]
        self.epsilon = epsilon
        
        # États simplifiés du MDP basés sur l'évolution de la fitness
        # 0: Amélioration, 1: Stagnation, 2: Dégradation
        self.states = [0, 1, 2]
        
        # Initialisation de la Q-Table [cite: 1154]
        for s in self.states:
            self.q_table[s] = {a: 0.0 for a in actions}

    def get_state(self, current_fit, prev_fit):
        """Détermine l'état s en fonction de l'évolution de la solution."""
        if prev_fit is None: return 1
        if current_fit < prev_fit: return 0 # Amélioration
        if current_fit == prev_fit: return 1 # Stagnation
        return 2 # Dégradation

    def select_action(self, state):
        """
        Algorithme 4: SelectAction (e, b)[cite: 1181].
        Combine exploitation (e-greedy) et exploration aléatoire.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions) # Exploration [cite: 1192]
        else:
            # Exploitation: MaxAction(e) [cite: 1190]
            # On mélange les actions pour casser les égalités
            items = list(self.q_table[state].items())
            random.shuffle(items)
            return max(items, key=lambda x: x[1])[0]

    def update(self, state, action, reward, next_state):
        """
        Mise à jour Q-Learning selon la formule[cite: 991, 1176].
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s', a') - Q(s,a))
        """
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
