"""
visualization.py - Visualisation Gantt et Statistiques (Enrichi).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import random
import numpy as np

def patient_colors(num_patients):
    """Génère une couleur unique par patient."""
    cmap = plt.get_cmap('tab20')
    return {i: cmap(i % 20) for i in range(1, num_patients + 5)}

def plot_gantt(task_times, skills, num_patients, title="Gantt", save_path=None):
    """
    Trace le Gantt.
    task_times: dict (i, j, s) -> (start, end, duration)
    """
    if not task_times:
        print("Aucune tâche à afficher.")
        return

    horizon = max(t[1] for t in task_times.values())
    colors = patient_colors(num_patients)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_height = 0.8
    y_mapping = {s: i for i, s in enumerate(sorted(skills))}
    
    ax.set_ylim(-0.5, len(skills) - 0.5)
    ax.set_xlim(0, horizon + 1)
    
    for (pid, op_stage, skill), (start, end, dur) in task_times.items():
        y = y_mapping[skill]
        rect = Rectangle((start, y - bar_height/2), dur, bar_height,
                         facecolor=colors[pid], edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        cx = start + dur/2
        cy = y
        ax.text(cx, cy, f"P{pid}\nO{op_stage}", ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(list(y_mapping.values()))
    ax.set_yticklabels([f"Skill {s}" for s in sorted(skills)])
    
    legend_elements = [Patch(facecolor=colors[i], label=f'Patient {i}') 
                       for i in range(1, num_patients+1)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel("Temps")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Fermer pour libérer la mémoire
    else:
        plt.show()

def plot_convergence(fitness_history_dict, title="Convergence", save_path=None):
    """
    Trace les courbes de convergence des agents.
    fitness_history_dict: dict {agent_id: [fitness_iter1, fitness_iter2, ...]}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for agent_id, history in fitness_history_dict.items():
        ax.plot(history, label=agent_id, linewidth=1.5, alpha=0.8)
        
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Makespan (Cmax)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_gantt_comparison(initial_times, final_times, skills, num_patients, 
                          init_cmax, final_cmax, title="Comparaison", save_path=None):
    """Affiche deux Gantt l'un sous l'autre pour comparer."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    colors = patient_colors(num_patients)
    bar_height = 0.8
    y_mapping = {s: i for i, s in enumerate(sorted(skills))}
    yticks = list(y_mapping.values())
    yticklabels = [f"Skill {s}" for s in sorted(skills)]

    # --- Plot 1: Initial ---
    horizon1 = max(t[1] for t in initial_times.values()) if initial_times else 0
    ax1.set_ylim(-0.5, len(skills) - 0.5)
    ax1.set_xlim(0, horizon1 + 1)
    
    for (pid, op_stage, skill), (start, end, dur) in initial_times.items():
        y = y_mapping[skill]
        rect = Rectangle((start, y - bar_height/2), dur, bar_height,
                         facecolor=colors[pid], edgecolor='black', alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(start + dur/2, y, f"P{pid}", ha='center', va='center', 
                 color='white', fontsize=7, fontweight='bold')
    
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_title(f"Avant Optimisation (Cmax = {init_cmax})")
    ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

    # --- Plot 2: Final ---
    horizon2 = max(t[1] for t in final_times.values()) if final_times else 0
    ax2.set_ylim(-0.5, len(skills) - 0.5)
    ax2.set_xlim(0, horizon2 + 1)

    for (pid, op_stage, skill), (start, end, dur) in final_times.items():
        y = y_mapping[skill]
        rect = Rectangle((start, y - bar_height/2), dur, bar_height,
                         facecolor=colors[pid], edgecolor='black', alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(start + dur/2, y, f"P{pid}", ha='center', va='center', 
                 color='white', fontsize=7, fontweight='bold')

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.set_title(f"Après Optimisation (Cmax = {final_cmax})")
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Temps")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# Fonctions placeholder pour compatibilité avec le notebook complet
# Si vous souhaitez implémenter la logique réelle, il faudrait accéder aux structures internes
def plot_multi_agent_convergence(*args, **kwargs): pass
def plot_q_table(*args, **kwargs): pass
def plot_neighborhood_usage(*args, **kwargs): pass
def plot_diversity_matrix(*args, **kwargs): pass
def plot_agent_contributions(*args, **kwargs): pass
