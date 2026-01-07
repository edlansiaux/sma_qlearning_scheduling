#!/usr/bin/env python3
"""
Main Script - Démonstration du Système Multi-Agents avec Q-Learning
pour l'ordonnancement des patients dans un environnement de soins

Ce script démontre:
1. Le système multi-agents avec différentes métaheuristiques (AG, Tabou, Recuit)
2. Les modes de collaboration Amis et Ennemis
3. L'auto-adaptation via Q-Learning
4. L'Espace Mémoire Partagé (EMP) avec contrôle de diversité
5. Les 5 fonctions de voisinage (A, B, C, D, E)

Basé sur le diaporama: "Optimisation collaborative : Agents auto-adaptatifs, 
Apprentissage par renforcement"
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from typing import Dict, List

# Imports des modules du projet
from core.environment import SchedulingEnvironment, create_default_environment, Task
from core.neighborhoods import NeighborhoodManager
from core.qlearning import QLearningAgent, AdaptiveNeighborhoodSelector
from core.shared_memory import SharedMemoryPool, Solution, ElitePool
from core.agents import (
    GeneticAgent, TabuAgent, SimulatedAnnealingAgent,
    MultiAgentSystem, CollaborationMode
)
from visualization import (
    plot_gantt, plot_gantt_comparison, plot_convergence,
    plot_multi_agent_convergence, plot_q_table, plot_neighborhood_usage,
    plot_diversity_matrix, plot_agent_contributions
)


def print_separator(title: str = ""):
    """Affiche un séparateur pour la lisibilité."""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)


def demo_single_agent_qlearning():
    """
    Démonstration d'un agent unique avec Q-Learning
    pour la sélection adaptative des voisinages.
    """
    print_separator("DÉMO 1: Agent unique avec Q-Learning")
    
    # Créer l'environnement
    env = create_default_environment()
    print(f"Environnement créé: {env.num_patients} patients, {len(env.skills)} compétences")
    
    # Solution initiale
    initial_solution = env.build_initial_solution(random_order=True)
    initial_makespan, initial_times, _ = env.evaluate(initial_solution, return_schedule=True)
    print(f"Makespan initial: {initial_makespan}")
    
    # Créer un agent avec Q-Learning
    agent = TabuAgent("tabu_qlearning", env, tabu_tenure=15, use_qlearning=True)
    agent.initialize(random_init=True)
    
    print(f"\nOptimisation en cours (200 itérations)...")
    
    # Exécuter l'optimisation
    for i in range(200):
        agent.optimize_step()
        if (i + 1) % 50 == 0:
            print(f"  Itération {i+1}: Makespan = {agent.current_fitness}")
    
    # Résultats
    best_sol = agent.get_best_solution()
    final_makespan, final_times, _ = env.evaluate(best_sol.sequences, return_schedule=True)
    
    print(f"\n--- Résultats ---")
    print(f"Makespan initial: {initial_makespan}")
    print(f"Makespan final: {final_makespan}")
    print(f"Amélioration: {((initial_makespan - final_makespan) / initial_makespan * 100):.1f}%")
    
    # Table Q
    if agent.q_selector:
        q_table = agent.q_selector.get_q_table()
        print(f"\nTable Q finale:")
        for state, actions in q_table.items():
            best_action = max(actions, key=actions.get)
            print(f"  État {state} -> Meilleure action: {best_action} (Q={actions[best_action]:.3f})")
        
        # Statistiques des voisinages
        stats = agent.q_selector.get_statistics()
        print(f"\nStatistiques des voisinages:")
        for n, s in stats.items():
            rate = s['improvements'] / s['calls'] * 100 if s['calls'] > 0 else 0
            print(f"  Voisinage {n}: {s['calls']} appels, {s['improvements']} améliorations ({rate:.1f}%)")
    
    return initial_times, final_times, initial_makespan, final_makespan, agent.fitness_history


def demo_multi_agent_friends():
    """
    Démonstration du système multi-agents en mode AMIS.
    Les agents partagent leurs solutions via l'EMP.
    """
    print_separator("DÉMO 2: Multi-Agents en mode AMIS")
    
    env = create_default_environment()
    
    # Créer le système multi-agents
    mas = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=True)
    
    # Ajouter des agents de différents types
    mas.add_agent('genetic', 'AG_1', population_size=12, mutation_rate=0.15)
    mas.add_agent('genetic', 'AG_2', population_size=10, mutation_rate=0.1)
    mas.add_agent('tabu', 'Tabu_1', tabu_tenure=12)
    mas.add_agent('tabu', 'Tabu_2', tabu_tenure=8)
    mas.add_agent('sa', 'RS_1', initial_temp=80, cooling_rate=0.99)
    mas.add_agent('sa', 'RS_2', initial_temp=100, cooling_rate=0.995)
    
    print(f"Mode: {mas.mode}")
    print(f"Agents créés: {list(mas.agents.keys())}")
    
    # Exécuter le système
    print(f"\nOptimisation collaborative (150 itérations)...")
    best_solution = mas.run(n_iterations=150, verbose=True)
    
    # Statistiques
    stats = mas.get_statistics()
    
    print(f"\n--- Résultats Mode Amis ---")
    print(f"Meilleur Makespan global: {stats['global_best_fitness']}")
    print(f"\nContributions des agents:")
    for agent_id, contrib in stats['agent_stats'].items():
        print(f"  {agent_id}: {contrib['contributions']} améliorations, "
              f"meilleur local = {contrib['best_fitness']}")
    
    print(f"\nStatistiques EMP:")
    emp_stats = stats['emp_stats']
    print(f"  Taille finale: {emp_stats['size']}/{emp_stats['max_size']}")
    print(f"  Insertions: {emp_stats['insertions']}")
    print(f"  Rejets (doublons): {emp_stats['rejections_duplicate']}")
    print(f"  Rejets (diversité): {emp_stats['rejections_diversity']}")
    print(f"  Remplacements: {emp_stats['replacements']}")
    
    return mas, stats


def demo_multi_agent_enemies():
    """
    Démonstration du système multi-agents en mode ENNEMIS.
    Les agents ne partagent que les valeurs de fitness.
    """
    print_separator("DÉMO 3: Multi-Agents en mode ENNEMIS")
    
    env = create_default_environment()
    
    # Créer le système multi-agents en mode ennemis
    mas = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=True)
    
    # Ajouter des agents
    mas.add_agent('genetic', 'AG_Enemy_1', population_size=15)
    mas.add_agent('tabu', 'Tabu_Enemy_1', tabu_tenure=10)
    mas.add_agent('sa', 'RS_Enemy_1', initial_temp=100)
    
    print(f"Mode: {mas.mode}")
    print(f"Agents créés: {list(mas.agents.keys())}")
    print("\nEn mode Ennemis, les agents ne travaillent que si une meilleure")
    print("solution globale a été trouvée par un autre agent.")
    
    # Exécuter
    print(f"\nOptimisation compétitive (150 itérations)...")
    best_solution = mas.run(n_iterations=150, verbose=True)
    
    stats = mas.get_statistics()
    
    print(f"\n--- Résultats Mode Ennemis ---")
    print(f"Meilleur Makespan global: {stats['global_best_fitness']}")
    print(f"\nContributions des agents:")
    for agent_id, contrib in stats['agent_stats'].items():
        print(f"  {agent_id}: {contrib['contributions']} améliorations, "
              f"actif pendant {contrib['iterations_active']} itérations")
    
    return mas, stats


def demo_emp_diversity():
    """
    Démonstration de l'Espace Mémoire Partagé (EMP) avec contrôle de diversité.
    """
    print_separator("DÉMO 4: Espace Mémoire Partagé (EMP) avec Diversité")
    
    env = create_default_environment()
    
    # Créer l'EMP avec paramètres de diversité
    emp = SharedMemoryPool(
        max_size=15,
        min_distance=3,  # R: distance minimale
        diversity_threshold=0.4  # DT: seuil de diversité
    )
    
    print(f"Paramètres EMP:")
    print(f"  Taille max: {emp.max_size}")
    print(f"  Distance minimale (R): {emp.min_distance}")
    print(f"  Seuil de diversité (DT): {emp.diversity_threshold}")
    
    # Générer des solutions et les insérer
    print(f"\nGénération et insertion de solutions...")
    
    solutions_generated = 0
    for i in range(50):
        sol = env.build_initial_solution(random_order=True)
        fitness, _, _ = env.evaluate(sol)
        solution = Solution(sequences=sol, fitness=fitness, agent_id=f"test_{i}")
        
        inserted = emp.insert(solution, iteration=i)
        solutions_generated += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Après {i+1} tentatives: {len(emp.solutions)} solutions dans l'EMP")
    
    # Statistiques
    stats = emp.get_statistics()
    print(f"\n--- Statistiques EMP ---")
    print(f"Solutions générées: {solutions_generated}")
    print(f"Solutions dans l'EMP: {stats['size']}")
    print(f"Meilleure fitness: {stats['best_fitness']}")
    print(f"Pire fitness: {stats['worst_fitness']}")
    print(f"Fitness moyenne: {stats['avg_fitness']:.2f}")
    print(f"\nInsertions réussies: {stats['insertions']}")
    print(f"Rejets (doublons): {stats['rejections_duplicate']}")
    print(f"Rejets (diversité insuffisante): {stats['rejections_diversity']}")
    print(f"Remplacements: {stats['replacements']}")
    
    # Matrice de diversité
    if len(emp.solutions) > 1:
        div_matrix = emp.get_diversity_matrix()
        avg_dist = np.mean([div_matrix[i][j] for i in range(len(div_matrix)) 
                          for j in range(i+1, len(div_matrix))])
        print(f"\nDistance moyenne entre solutions: {avg_dist:.2f}")
    
    return emp, stats


def demo_qlearning_detailed():
    """
    Démonstration détaillée du Q-Learning avec visualisation de l'apprentissage.
    """
    print_separator("DÉMO 5: Q-Learning Détaillé")
    
    # Créer l'agent Q-Learning
    q_agent = QLearningAgent(
        states=['A', 'B', 'C', 'D', 'E'],
        alpha=0.15,   # Taux d'apprentissage
        gamma=0.9,    # Facteur d'actualisation
        epsilon=0.5,  # Exploration initiale
        epsilon_decay=0.99,
        epsilon_min=0.05
    )
    
    print("Paramètres Q-Learning:")
    print(f"  Alpha (taux d'apprentissage): {q_agent.alpha}")
    print(f"  Gamma (facteur d'actualisation): {q_agent.gamma}")
    print(f"  Epsilon initial: {q_agent.epsilon}")
    
    # Simulation d'apprentissage
    print("\nSimulation d'apprentissage (500 épisodes)...")
    
    # Récompenses simulées par voisinage (A est le meilleur)
    reward_means = {'A': 0.8, 'B': 0.5, 'C': 0.6, 'D': 0.3, 'E': 0.4}
    
    epsilon_history = []
    
    for episode in range(500):
        # Sélectionner une action
        action = q_agent.select_action()
        
        # Simuler une récompense
        reward = random.gauss(reward_means[action], 0.2)
        
        # Mettre à jour
        q_agent.update(action, reward)
        q_agent.decay_epsilon()
        
        epsilon_history.append(q_agent.epsilon)
        
        if (episode + 1) % 100 == 0:
            print(f"  Épisode {episode+1}: Epsilon = {q_agent.epsilon:.3f}")
    
    # Afficher la table Q finale
    print("\n--- Table Q Finale ---")
    q_table = q_agent.get_q_table_formatted()
    
    print("\n        ", end="")
    for a in q_agent.states:
        print(f"   {a}   ", end="")
    print()
    
    for s in q_agent.states:
        print(f"  {s}  |", end="")
        for a in q_agent.states:
            print(f" {q_table[s][a]:6.3f}", end="")
        print()
    
    print("\nMeilleures actions par état:")
    for s in q_agent.states:
        best_a = max(q_table[s], key=q_table[s].get)
        print(f"  État {s} -> Action {best_a} (Q={q_table[s][best_a]:.3f})")
    
    return q_agent, epsilon_history


def demo_comparison_modes():
    """
    Comparaison des modes Amis vs Ennemis.
    """
    print_separator("DÉMO 6: Comparaison Amis vs Ennemis")
    
    env = create_default_environment()
    n_iterations = 100
    
    # Mode Amis
    print("\n--- Mode AMIS ---")
    mas_friends = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=True)
    mas_friends.add_agent('genetic', 'AG')
    mas_friends.add_agent('tabu', 'Tabu')
    mas_friends.add_agent('sa', 'RS')
    
    random.seed(42)
    np.random.seed(42)
    best_friends = mas_friends.run(n_iterations=n_iterations, verbose=False)
    stats_friends = mas_friends.get_statistics()
    
    # Mode Ennemis
    print("--- Mode ENNEMIS ---")
    mas_enemies = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=True)
    mas_enemies.add_agent('genetic', 'AG')
    mas_enemies.add_agent('tabu', 'Tabu')
    mas_enemies.add_agent('sa', 'RS')
    
    random.seed(42)
    np.random.seed(42)
    best_enemies = mas_enemies.run(n_iterations=n_iterations, verbose=False)
    stats_enemies = mas_enemies.get_statistics()
    
    # Comparaison
    print("\n" + "="*50)
    print("       COMPARAISON DES MODES")
    print("="*50)
    print(f"{'Métrique':<30} {'AMIS':>10} {'ENNEMIS':>10}")
    print("-"*50)
    print(f"{'Meilleur Makespan':<30} {stats_friends['global_best_fitness']:>10} {stats_enemies['global_best_fitness']:>10}")
    print(f"{'Taille finale EMP':<30} {stats_friends['emp_stats']['size']:>10} {stats_enemies['emp_stats']['size']:>10}")
    print(f"{'Insertions EMP':<30} {stats_friends['emp_stats']['insertions']:>10} {stats_enemies['emp_stats']['insertions']:>10}")
    
    total_contrib_friends = sum(a['contributions'] for a in stats_friends['agent_stats'].values())
    total_contrib_enemies = sum(a['contributions'] for a in stats_enemies['agent_stats'].values())
    print(f"{'Contributions totales':<30} {total_contrib_friends:>10} {total_contrib_enemies:>10}")
    
    return stats_friends, stats_enemies


def main():
    """Fonction principale."""
    print("\n" + "="*70)
    print("   SYSTÈME MULTI-AGENTS AVEC Q-LEARNING")
    print("   Pour l'ordonnancement des patients")
    print("="*70)
    print("\nBasé sur: 'Optimisation collaborative : Agents auto-adaptatifs,")
    print("          Apprentissage par renforcement'")
    print("\nCaractéristiques:")
    print("  • Métaheuristiques: Algorithme Génétique, Recherche Tabou, Recuit Simulé")
    print("  • Collaboration: Modes Amis et Ennemis")
    print("  • Auto-adaptation: Q-Learning pour sélection des voisinages")
    print("  • Diversité: Espace Mémoire Partagé (EMP) avec contrôle de distance")
    print("  • Voisinages: 5 fonctions (A, B, C, D, E)")
    
    # Exécuter les démonstrations
    try:
        # Démo 1: Agent unique avec Q-Learning
        init_times, final_times, init_cmax, final_cmax, history = demo_single_agent_qlearning()
        
        # Démo 2: Multi-agents mode Amis
        mas_friends, stats_friends = demo_multi_agent_friends()
        
        # Démo 3: Multi-agents mode Ennemis
        mas_enemies, stats_enemies = demo_multi_agent_enemies()
        
        # Démo 4: EMP avec diversité
        emp, emp_stats = demo_emp_diversity()
        
        # Démo 5: Q-Learning détaillé
        q_agent, epsilon_history = demo_qlearning_detailed()
        
        # Démo 6: Comparaison des modes
        stats_friends_comp, stats_enemies_comp = demo_comparison_modes()
        
        print_separator("EXÉCUTION TERMINÉE")
        print("\nToutes les démonstrations ont été exécutées avec succès!")
        print("\nPour visualiser les résultats, utilisez le notebook Jupyter")
        print("ou les fonctions de visualization.py")
        
        # Retourner les résultats pour utilisation dans le notebook
        return {
            'init_times': init_times,
            'final_times': final_times,
            'init_cmax': init_cmax,
            'final_cmax': final_cmax,
            'convergence_history': history,
            'mas_friends': mas_friends,
            'stats_friends': stats_friends,
            'mas_enemies': mas_enemies,
            'stats_enemies': stats_enemies,
            'emp': emp,
            'emp_stats': emp_stats,
            'q_agent': q_agent,
            'epsilon_history': epsilon_history
        }
        
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
