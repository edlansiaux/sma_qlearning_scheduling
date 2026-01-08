"""
main.py - Point d'entrée principal remasterisé.
"""

import sys
import os
import random
import numpy as np

# Imports core
# Assurez-vous que core.environment, core.agents, etc. sont bien accessibles
from core.environment import create_default_environment
from core.agents import MultiAgentSystem, CollaborationMode
from visualization import plot_gantt

def print_header():
    print("\n" + "="*70)
    print("   SYSTÈME MULTI-AGENTS - ORDONNANCEMENT SANTÉ (STRICT)")
    print("="*70)
    print("Configuration: Données WhatsApp (Contraintes strictes)")
    print("Voisinages actifs: C (Insertion), E (Swap Intra-Ressource)")

def run_single_demo():
    """Exécute une démonstration visuelle unique."""
    print("\n--- DÉMONSTRATION UNIQUE (Mode Amis) ---")
    
    env = create_default_environment()
    print(f"Environnement chargé: {env.num_patients} patients, {len(env.skills)} compétences.")
    
    # Création du système
    mas = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=True)
    mas.add_agent('genetic', 'AG_Demo', population_size=20)
    mas.add_agent('tabu', 'Tabu_Demo', tabu_tenure=10)
    mas.add_agent('sa', 'RS_Demo', initial_temp=100)
    
    print("Lancement de l'optimisation (100 itérations)...")
    best_sol = mas.run(n_iterations=100, verbose=True)
    
    if best_sol:
        print(f"\nMeilleur Makespan trouvé: {best_sol.fitness}")
        
        # Génération du Gantt
        cmax, task_times, _ = env.evaluate(best_sol.sequences, return_schedule=True)
        try:
            # On utilise la nouvelle fonction plot_gantt de visualization.py
            output_file = "gantt_resultat_demo.png"
            plot_gantt(task_times, env.skills, env.num_patients, 
                      title=f"Planning Optimisé (Cmax={cmax})", 
                      save_path=output_file)
            print(f"Gantt sauvegardé sous '{output_file}'")
        except Exception as e:
            print(f"Impossible de générer le graphique: {e}")
    else:
        print("Aucune solution trouvée.")

def main():
    print_header()
    
    print("\nQue voulez-vous faire ?")
    print("1. Lancer une démonstration rapide (avec Gantt)")
    print("2. Lancer le Benchmark complet (Génération des tableaux et courbes)")
    
    choice = input("\nVotre choix (1/2) [1]: ").strip()
    
    if choice == "2":
        # Import dynamique corrigé pour appeler le script dans core/
        try:
            from core import benchmark_tables
            benchmark_tables.main()
        except ImportError as e:
            print(f"Erreur critique: {e}")
            print("Vérifiez que le fichier 'core/benchmark_tables.py' existe et que les dépendances sont installées.")
        except Exception as e:
            print(f"Erreur lors de l'exécution du benchmark: {e}")
    else:
        run_single_demo()

if __name__ == "__main__":
    main()
