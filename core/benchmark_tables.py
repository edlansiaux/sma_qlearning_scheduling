"""
benchmark_tables.py - Script pour générer les données des tableaux comparatifs (PDF Slides 25-26).
"""
import sys
import os
import random
import numpy as np
import time

# Assurer que le dossier courant est dans le path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.environment import create_default_environment
from core.agents import GeneticAgent, TabuAgent, SimulatedAnnealingAgent, MultiAgentSystem, CollaborationMode

def run_benchmark_scenario(scenario_name, system, iterations=50):
    """Exécute un scénario et retourne le meilleur makespan."""
    print(f"  > Exécution {scenario_name}...")
    start_time = time.time()
    best_solution = system.run(n_iterations=iterations, verbose=False)
    duration = time.time() - start_time
    
    stats = system.get_statistics()
    best_fitness = stats['global_best_fitness']
    
    print(f"    Terminé en {duration:.2f}s. Makespan: {best_fitness}")
    return best_fitness

def main():
    print("================================================================")
    print("   GÉNÉRATEUR DE DONNÉES POUR TABLEAUX COMPARATIFS (PDF)")
    print("================================================================")
    
    # Paramètres globaux
    ITERATIONS = 50  # Réduit pour la démo, augmenter à 200+ pour résultats réels
    N_RUNS = 1       # Nombre d'exécutions pour faire une moyenne (augmenter pour robustesse)
    
    # --- TABLEAU 1 (Slide 25) : SANS COLLABORATION ---
    print("\n\n--- TABLEAU 1 : Comparaison SANS Collaboration ---")
    print(f"{'Jour':<5} | {'AG':<8} | {'Tabou':<8} | {'RS':<8} | {'SMA_NoLearn':<12} | {'SMA_Learn':<12}")
    print("-" * 65)

    # Simulation de "Jours" (Différents seeds aléatoires pour simuler différentes arrivées/conditions)
    for day in range(1, 4): # Simuler 3 jours
        env = create_default_environment()
        
        # 1. Métaheuristiques seules (Simulées via un SMA à 1 agent)
        # AG
        mas_ag = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=False)
        mas_ag.add_agent('genetic', 'AG_Solo', population_size=20)
        res_ag = run_benchmark_scenario("AG Solo", mas_ag, ITERATIONS)
        
        # Tabou
        mas_tabu = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=False)
        mas_tabu.add_agent('tabu', 'Tabu_Solo', tabu_tenure=10)
        res_tabu = run_benchmark_scenario("Tabu Solo", mas_tabu, ITERATIONS)
        
        # RS (Recuit Simulé)
        mas_rs = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=False)
        mas_rs.add_agent('sa', 'RS_Solo', initial_temp=100)
        res_rs = run_benchmark_scenario("RS Solo", mas_rs, ITERATIONS)
        
        # 2. SMA sans apprentissage (Multi-Agents classiques, pas de Q-Learning)
        mas_nolearn = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=False)
        mas_nolearn.add_agent('genetic', 'AG')
        mas_nolearn.add_agent('tabu', 'Tabu')
        mas_nolearn.add_agent('sa', 'RS')
        res_sma_nl = run_benchmark_scenario("SMA NoLearn", mas_nolearn, ITERATIONS)
        
        # 3. SMA avec apprentissage (Q-Learning activé)
        mas_learn = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=True)
        mas_learn.add_agent('genetic', 'AG')
        mas_learn.add_agent('tabu', 'Tabu')
        mas_learn.add_agent('sa', 'RS')
        res_sma_l = run_benchmark_scenario("SMA Learn", mas_learn, ITERATIONS)
        
        print(f"J{day:<4} | {res_ag:<8.1f} | {res_tabu:<8.1f} | {res_rs:<8.1f} | {res_sma_nl:<12.1f} | {res_sma_l:<12.1f}")


    # --- TABLEAU 2 (Slide 26) : AVEC COLLABORATION (AMIS vs ENNEMIS) ---
    print("\n\n--- TABLEAU 2 : Comparaison AVEC Collaboration ---")
    print("Scénarios complexes (Combinaisons d'agents)")
    
    scenarios = [
        ("AG+Tabou", ['genetic', 'tabu']),
        ("AG+RS", ['genetic', 'sa']),
        ("Tabou+RS", ['tabu', 'sa']),
        ("Trio Complet", ['genetic', 'tabu', 'sa'])
    ]
    
    print(f"{'Configuration':<15} | {'Amis (No QL)':<12} | {'Ennemis (No QL)':<15} | {'Amis (With QL)':<12} | {'Ennemis (With QL)':<15}")
    print("-" * 80)
    
    env = create_default_environment()
    
    for name, agent_types in scenarios:
        results = []
        
        # On teste les 4 variantes pour chaque configuration d'agents
        for mode in [CollaborationMode.FRIENDS, CollaborationMode.ENEMIES]:
            for use_ql in [False, True]:
                mas = MultiAgentSystem(env, mode=mode, use_qlearning=use_ql)
                
                # Ajout des agents demandés
                for idx, atype in enumerate(agent_types):
                    if atype == 'genetic':
                        mas.add_agent('genetic', f'AG_{idx}', population_size=15)
                    elif atype == 'tabu':
                        mas.add_agent('tabu', f'Tabu_{idx}', tabu_tenure=10)
                    elif atype == 'sa':
                        mas.add_agent('sa', f'RS_{idx}', initial_temp=100)
                
                res = run_benchmark_scenario(f"{name} {mode} QL={use_ql}", mas, ITERATIONS)
                results.append(res)
        
        # Affichage de la ligne
        print(f"{name:<15} | {results[0]:<12.1f} | {results[1]:<15.1f} | {results[2]:<12.1f} | {results[3]:<15.1f}")

if __name__ == "__main__":
    main()
