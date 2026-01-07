"""
visualization.py - Visualisation Gantt robuste.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import random

def patient_colors(num_patients):
    """Génère une couleur unique par patient."""
    # Palette fixe pour cohérence visuelle
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

    # Calcul de l'horizon
    horizon = max(t[1] for t in task_times.values())
    colors = patient_colors(num_patients)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Hauteur d'une barre
    bar_height = 0.8
    # Espace vertical par skill
    y_mapping = {s: i for i, s in enumerate(sorted(skills))}
    
    # Fond
    ax.set_ylim(-0.5, len(skills) - 0.5)
    ax.set_xlim(0, horizon + 1)
    
    # Tracer les tâches
    for (pid, op_stage, skill), (start, end, dur) in task_times.items():
        y = y_mapping[skill]
        # Rectangle(xy, width, height)
        rect = Rectangle((start, y - bar_height/2), dur, bar_height,
                         facecolor=colors[pid], edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        
        # Texte (P1-O1)
        cx = start + dur/2
        cy = y
        ax.text(cx, cy, f"P{pid}\nO{op_stage}", ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')

    # Labels Y
    ax.set_yticks(list(y_mapping.values()))
    ax.set_yticklabels([f"Skill {s}" for s in sorted(skills)])
    
    # Légende Patients
    legend_elements = [Patch(facecolor=colors[i], label=f'Patient {i}') 
                       for i in range(1, num_patients+1)]
    # On met la légende à l'extérieur
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel("Temps")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
