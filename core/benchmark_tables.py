"""
benchmark_tables.py - Script pour générer les données des tableaux et les VISUALISATIONS.
"""
import sys
import os
import time
import shutil

# Assurer que le dossier courant est dans le path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import create_default_environment
from core.agents import MultiAgentSystem, CollaborationMode
from visualization import plot_gantt, plot_convergence

# Dossier de sortie pour les images
OUTPUT_DIR = "benchmark_results"

def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier '{OUTPUT_DIR}' créé.")

def run_benchmark_scenario(scenario_name, system, iterations=50, save_plots=True):
    """Exécute un scénario, retourne le makespan et sauvegarde les graphiques."""
    print(f"  > Exécution {scenario_name}...")
    
    start_time = time.time()
    best_solution = system.run(n_iterations=iterations, verbose=False)
    duration = time.time() - start_time
    
    stats = system.get_statistics()
    best_fitness = stats['global_best_fitness']
    
    print(f"    Terminé en {duration:.2f}s. Makespan: {best_fitness}")
    
    # --- GÉNÉRATION DES VISUALISATIONS ---
    if save_plots:
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "Plus")
        
        # 1. Courbes de convergence
        histories = {aid: agent.fitness_history for aid, agent in system.agents.items()}
        plot_convergence(
            histories, 
            title=f"Convergence - {scenario_name}",
            save_path=os.path.join(OUTPUT_DIR, f"{safe_name}_convergence.png")
        )
        
        # 2. Gantt du meilleur résultat
        if best_solution:
            _, task_times, _ = system.env.evaluate(best_solution.sequences, return_schedule=True)
            plot_gantt(
                task_times, system.env.skills, system.env.num_patients,
                title=f"Planning {scenario_name} (Cmax={best_fitness})",
                save_path=os.path.join(OUTPUT_DIR, f"{safe_name}_gantt.png")
            )
            
    return best_fitness

def main():
    print("================================================================")
    print("   GÉNÉRATEUR DE BENCHMARK ET VISUALISATIONS")
    print("================================================================")
    setup_output_dir()
    
    ITERATIONS = 50 
    
    # --- TABLEAU 1 : SANS COLLABORATION ---
    print("\n\n--- TABLEAU 1 : Comparaison SANS Collaboration ---")
    print(f"{'Jour':<5} | {'AG':<8} | {'Tabou':<8} | {'RS':<8} | {'SMA_NoLearn':<12} | {'SMA_Learn':<12}")
    print("-" * 65)

    # Pour l'exemple, on ne fait qu'un seul "Jour" pour éviter de générer trop d'images
    # Remplacez range(1, 2) par range(1, 4) pour plus de résultats
    for day in range(1, 2): 
        env = create_default_environment()
        
        # AG
        mas_ag = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=False)
        mas_ag.add_agent('genetic', 'AG_Solo', population_size=20)
        res_ag = run_benchmark_scenario(f"J{day}_AG_Solo", mas_ag, ITERATIONS)
        
        # Tabou
        mas_tabu = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=False)
        mas_tabu.add_agent('tabu', 'Tabu_Solo', tabu_tenure=10)
        res_tabu = run_benchmark_scenario(f"J{day}_Tabou_Solo", mas_tabu, ITERATIONS)
        
        # RS
        mas_rs = MultiAgentSystem(env, mode=CollaborationMode.ENEMIES, use_qlearning=False)
        mas_rs.add_agent('sa', 'RS_Solo', initial_temp=100)
        res_rs = run_benchmark_scenario(f"J{day}_RS_Solo", mas_rs, ITERATIONS)
        
        # SMA NoLearn
        mas_nl = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=False)
        mas_nl.add_agent('genetic', 'AG')
        mas_nl.add_agent('tabu', 'Tabu')
        mas_nl.add_agent('sa', 'RS')
        res_nl = run_benchmark_scenario(f"J{day}_SMA_NoLearn", mas_nl, ITERATIONS)
        
        # SMA Learn
        mas_l = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=True)
        mas_l.add_agent('genetic', 'AG')
        mas_l.add_agent('tabu', 'Tabu')
        mas_l.add_agent('sa', 'RS')
        res_l = run_benchmark_scenario(f"J{day}_SMA_Learn", mas_l, ITERATIONS)
        
        print(f"J{day:<4} | {res_ag:<8.1f} | {res_tabu:<8.1f} | {res_rs:<8.1f} | {res_nl:<12.1f} | {res_l:<12.1f}")

    print(f"\n✅ Terminé. Les graphiques ont été sauvegardés dans le dossier '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
