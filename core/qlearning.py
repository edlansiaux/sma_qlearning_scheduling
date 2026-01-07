"""
Q-Learning Module - Auto-adaptation des agents par apprentissage par renforcement
Basé sur le diaporama: Processus de Décision Markovien (PDM) et Q-Learning

Les états sont les fonctions de voisinage (A, B, C, D, E)
Les actions sont les transitions entre voisinages
La récompense est l'amélioration du critère (makespan)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict


class QLearningAgent:
    """
    Agent Q-Learning pour la sélection adaptative des fonctions de voisinage.
    
    États: Les 5 fonctions de voisinage {A, B, C, D, E}
    Actions: Choisir un voisinage pour la prochaine itération
    Récompense: Amélioration du makespan (négatif si dégradation)
    
    Équation de mise à jour Q-Learning:
    Q(s,a) ← Q(s,a) + α(r + γ·max_a'Q(s',a') - Q(s,a))
    """
    
    def __init__(self, states: List[str] = None, 
                 alpha: float = 0.1, 
                 gamma: float = 0.9, 
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            states: Liste des états (noms des voisinages)
            alpha: Taux d'apprentissage (learning rate)
            gamma: Facteur d'actualisation (discount factor)
            epsilon: Taux d'exploration initial (epsilon-greedy)
            epsilon_decay: Décroissance de epsilon
            epsilon_min: Valeur minimale de epsilon
        """
        self.states = states if states else ['A', 'B', 'C', 'D', 'E']
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        
        # Paramètres Q-Learning
        self.alpha = alpha      # Facteur d'apprentissage
        self.gamma = gamma      # Facteur d'actualisation
        self.epsilon = epsilon  # Taux d'exploration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Table Q: Q[s][a] = valeur Q pour état s, action a
        # Ici, action = choix du prochain voisinage
        self.q_table = np.zeros((self.n_states, self.n_states))
        
        # État courant
        self.current_state = random.choice(self.states)
        
        # Historique pour analyse
        self.action_history = []
        self.reward_history = []
        self.q_value_history = []
    
    def select_action(self, exploit_only: bool = False) -> str:
        """
        Sélectionne une action (voisinage suivant) selon la politique ε-greedy.
        
        Args:
            exploit_only: Si True, utilise uniquement l'exploitation (pas d'exploration)
        
        Returns:
            Nom du voisinage sélectionné
        """
        state_idx = self.state_to_idx[self.current_state]
        
        if not exploit_only and random.random() < self.epsilon:
            # Exploration: action aléatoire
            action_idx = random.randint(0, self.n_states - 1)
        else:
            # Exploitation: meilleure action selon Q-table
            action_idx = np.argmax(self.q_table[state_idx])
        
        action = self.idx_to_state[action_idx]
        self.action_history.append(action)
        
        return action
    
    def update(self, action: str, reward: float, next_state: str = None):
        """
        Met à jour la table Q selon l'équation de Bellman.
        
        Q(s,a) ← Q(s,a) + α(r + γ·max_a'Q(s',a') - Q(s,a))
        
        Args:
            action: Action effectuée (voisinage utilisé)
            reward: Récompense obtenue (amélioration du makespan)
            next_state: État suivant (si None, devient l'action)
        """
        state_idx = self.state_to_idx[self.current_state]
        action_idx = self.state_to_idx[action]
        
        # L'état suivant est l'action choisie
        if next_state is None:
            next_state = action
        next_state_idx = self.state_to_idx[next_state]
        
        # Meilleure valeur Q pour l'état suivant
        max_next_q = np.max(self.q_table[next_state_idx])
        
        # Mise à jour Q-Learning
        old_q = self.q_table[state_idx, action_idx]
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_idx, action_idx] = new_q
        
        # Enregistrer l'historique
        self.reward_history.append(reward)
        self.q_value_history.append(np.mean(self.q_table))
        
        # Transition vers le nouvel état
        self.current_state = next_state
    
    def decay_epsilon(self):
        """Décroît le taux d'exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_best_action(self, state: str = None) -> str:
        """Retourne la meilleure action pour un état donné."""
        if state is None:
            state = self.current_state
        state_idx = self.state_to_idx[state]
        best_action_idx = np.argmax(self.q_table[state_idx])
        return self.idx_to_state[best_action_idx]
    
    def get_q_table_formatted(self) -> Dict[str, Dict[str, float]]:
        """Retourne la table Q dans un format lisible."""
        q_dict = {}
        for s_idx, s in enumerate(self.states):
            q_dict[s] = {}
            for a_idx, a in enumerate(self.states):
                q_dict[s][a] = float(self.q_table[s_idx, a_idx])
        return q_dict
    
    def reset_state(self, state: str = None):
        """Réinitialise l'état courant."""
        if state is None:
            self.current_state = random.choice(self.states)
        else:
            self.current_state = state
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle Q-Learning."""
        np.savez(filepath, 
                q_table=self.q_table, 
                states=self.states,
                epsilon=self.epsilon)
    
    def load_model(self, filepath: str):
        """Charge un modèle Q-Learning."""
        data = np.load(filepath, allow_pickle=True)
        self.q_table = data['q_table']
        self.states = list(data['states'])
        self.epsilon = float(data['epsilon'])
        self._update_mappings()
    
    def _update_mappings(self):
        """Met à jour les mappings état-index."""
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}


class AdaptiveNeighborhoodSelector:
    """
    Sélecteur de voisinage adaptatif basé sur Q-Learning.
    
    Combine le Q-Learning avec une stratégie de récompense basée sur
    l'amélioration relative du makespan.
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.3, reward_scale: float = 1.0):
        """
        Args:
            alpha: Taux d'apprentissage
            gamma: Facteur d'actualisation
            epsilon: Taux d'exploration
            reward_scale: Facteur d'échelle pour les récompenses
        """
        self.q_agent = QLearningAgent(
            states=['A', 'B', 'C', 'D', 'E'],
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
        self.reward_scale = reward_scale
        self.last_makespan = None
        self.best_makespan = float('inf')
        
        # Statistiques par voisinage
        self.neighborhood_stats = defaultdict(lambda: {
            'calls': 0, 
            'improvements': 0, 
            'total_improvement': 0.0
        })
    
    def select_neighborhood(self) -> str:
        """Sélectionne le voisinage suivant."""
        return self.q_agent.select_action()
    
    def update_with_result(self, neighborhood: str, new_makespan: float):
        """
        Met à jour le Q-Learning avec le résultat obtenu.
        
        Args:
            neighborhood: Voisinage utilisé
            new_makespan: Nouveau makespan obtenu
        """
        # Calculer la récompense
        if self.last_makespan is not None:
            improvement = self.last_makespan - new_makespan
            reward = improvement * self.reward_scale
            
            # Bonus si amélioration du meilleur global
            if new_makespan < self.best_makespan:
                reward += 1.0  # Bonus
                self.best_makespan = new_makespan
        else:
            reward = 0.0
        
        # Mettre à jour les statistiques
        stats = self.neighborhood_stats[neighborhood]
        stats['calls'] += 1
        if reward > 0:
            stats['improvements'] += 1
            stats['total_improvement'] += reward
        
        # Mettre à jour Q-Learning
        self.q_agent.update(neighborhood, reward)
        self.q_agent.decay_epsilon()
        
        # Enregistrer le makespan
        self.last_makespan = new_makespan
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques d'utilisation des voisinages."""
        return dict(self.neighborhood_stats)
    
    def get_q_table(self) -> Dict[str, Dict[str, float]]:
        """Retourne la table Q."""
        return self.q_agent.get_q_table_formatted()
    
    def reset(self, initial_makespan: float = None):
        """Réinitialise le sélecteur."""
        self.last_makespan = initial_makespan
        self.best_makespan = initial_makespan if initial_makespan else float('inf')
        self.q_agent.reset_state()


class MarkovDecisionProcess:
    """
    Processus de Décision Markovien (PDM) pour la sélection des voisinages.
    
    Modélise le problème comme un PDM où:
    - États S: Voisinages {A, B, C, D, E}
    - Actions A: Choix du prochain voisinage
    - Transitions P(s'|s,a): Probabilités de transition
    - Récompenses R(s,a,s'): Amélioration du makespan
    """
    
    def __init__(self, states: List[str] = None):
        """
        Args:
            states: Liste des états du PDM
        """
        self.states = states if states else ['A', 'B', 'C', 'D', 'E']
        self.n_states = len(self.states)
        
        # Transitions: compteur des transitions (s, a, s')
        self.transitions = defaultdict(int)
        self.transition_counts = defaultdict(int)
        
        # Récompenses: somme des récompenses pour (s, a, s')
        self.rewards = defaultdict(float)
        self.reward_counts = defaultdict(int)
    
    def record_transition(self, state: str, action: str, next_state: str, reward: float):
        """Enregistre une transition observée."""
        key = (state, action, next_state)
        self.transitions[key] += 1
        self.transition_counts[(state, action)] += 1
        self.rewards[key] += reward
        self.reward_counts[key] += 1
    
    def get_transition_probability(self, state: str, action: str, next_state: str) -> float:
        """Calcule la probabilité de transition P(s'|s,a)."""
        total = self.transition_counts[(state, action)]
        if total == 0:
            return 1.0 / self.n_states  # Uniforme si pas de données
        return self.transitions[(state, action, next_state)] / total
    
    def get_expected_reward(self, state: str, action: str, next_state: str) -> float:
        """Calcule la récompense espérée R(s,a,s')."""
        count = self.reward_counts[(state, action, next_state)]
        if count == 0:
            return 0.0
        return self.rewards[(state, action, next_state)] / count
    
    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition moyenne."""
        matrix = np.zeros((self.n_states, self.n_states))
        
        for s_idx, s in enumerate(self.states):
            for sp_idx, sp in enumerate(self.states):
                # Moyenne sur toutes les actions
                probs = [self.get_transition_probability(s, a, sp) for a in self.states]
                matrix[s_idx, sp_idx] = np.mean(probs)
        
        return matrix
