"""
Visualization Module - Visualisations pour le système multi-agents
Inclut: Diagrammes de Gantt, courbes de convergence, tables Q, diversité EMP
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


def patient_colors(num_patients: int) -> Dict[int, tuple]:
    """Génère une palette de couleurs pour les patients."""
    cmap = plt.colormaps.get_cmap("tab20")
    n = min(20, num_patients)
    return {i+1: cmap(i / max(n-1, 1)) for i in range(num_patients)}


def build_gantt_data(task_times: Dict, skills: List[int]) -> Tuple[Dict, int]:
    """
    Formate les données pour le diagramme de Gantt.
    
    Args:
        task_times: Dict (i, j, s) -> (start, end, duration)
        skills: Liste des compétences
    
    Returns:
        (données par skill, horizon temporel)
    """
    by_skill = {s: [] for s in skills}
    horizon = 0
    
    for (i, j, s), (start, finish, p) in task_times.items():
        by_skill[s].append({
            "start": start, "end": finish, "dur": p,
            "patient": i, "op": j
        })
        horizon = max(horizon, finish)
    
    for s in skills:
        by_skill[s].sort(key=lambda x: (x["start"], x["patient"], x["op"]))
    
    return by_skill, horizon


def plot_gantt(task_times: Dict, skills: List[int], num_patients: int,
               title: str = "Diagramme de Gantt", figsize: Tuple = None,
               annotate: bool = True, save_path: str = None, dpi: int = 150):
    """
    Génère un diagramme de Gantt pour le planning.
    
    Args:
        task_times: Temps des tâches {(i,j,s): (start, end, dur)}
        skills: Liste des compétences
        num_patients: Nombre de patients
        title: Titre du graphique
        figsize: Taille de la figure
        annotate: Annoter les tâches
        save_path: Chemin pour sauvegarder
        dpi: Résolution
    """
    by_skill, horizon = build_gantt_data(task_times, skills)
    colors = patient_colors(num_patients)
    
    if horizon == 0:
        print("Makespan est 0. Impossible de tracer le Gantt.")
        return None
    
    if figsize is None:
        figsize = (max(12, horizon * 0.4), 1.5 * len(skills) + 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    lane_height = 0.8
    y_gap = 0.6
    y_positions = {s: (len(skills)-idx-1)*(lane_height + y_gap) 
                   for idx, s in enumerate(skills)}
    ymin = -0.5
    ymax = max(y_positions.values()) + lane_height + 0.5
    
    # Dessiner les rectangles
    for s in skills:
        y = y_positions[s]
        ax.add_patch(Rectangle((0, y - 0.1), horizon, lane_height + 0.2,
                               facecolor=(0.95, 0.95, 0.95), edgecolor="none"))
        
        for it in by_skill[s]:
            start = it["start"]
            dur = it["dur"]
            i = it["patient"]
            j = it["op"]
            
            rect = Rectangle((start, y), dur, lane_height,
                             facecolor=colors[i], edgecolor="black", linewidth=0.7)
            ax.add_patch(rect)
            
            if annotate:
                label = f"P{i}-O{j}"
                ax.text(start + dur/2, y + lane_height/2, label,
                        ha="center", va="center", fontsize=8, fontweight='bold')
    
    # Configuration des axes
    ax.set_xlim(0, math.ceil(horizon))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Temps (unités)", fontsize=11)
    ax.set_ylabel("Compétences (Skills)", fontsize=11)
    ax.set_yticks([y_positions[s] + lane_height/2 for s in skills])
    ax.set_yticklabels([f"Skill {s}" for s in skills])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    
    # Légende des patients
    legend_handles = [Patch(facecolor=colors[i], edgecolor="black", 
                           label=f"Patient {i}") 
                     for i in range(1, num_patients + 1)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), 
              loc="upper left", borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Gantt sauvegardé: {save_path}")
    
    return fig


def plot_gantt_comparison(task_times_before: Dict, task_times_after: Dict,
                          skills: List[int], num_patients: int,
                          cmax_before: float, cmax_after: float,
                          title: str = "Comparaison Avant/Après Optimisation",
                          figsize: Tuple = (16, 10), save_path: str = None):
    """
    Compare deux diagrammes de Gantt côte à côte.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = patient_colors(num_patients)
    
    def plot_single(ax, task_times, subtitle, makespan):
        by_skill, horizon = build_gantt_data(task_times, skills)
        
        if horizon == 0:
            horizon = 1
        
        lane_height = 0.8
        y_gap = 0.6
        y_positions = {s: (len(skills)-idx-1)*(lane_height + y_gap) 
                       for idx, s in enumerate(skills)}
        
        for s in skills:
            y = y_positions[s]
            ax.add_patch(Rectangle((0, y - 0.1), horizon, lane_height + 0.2,
                                   facecolor=(0.95, 0.95, 0.95), edgecolor="none"))
            
            for it in by_skill[s]:
                start = it["start"]
                dur = it["dur"]
                i = it["patient"]
                j = it["op"]
                
                rect = Rectangle((start, y), dur, lane_height,
                                 facecolor=colors[i], edgecolor="black", linewidth=0.7)
                ax.add_patch(rect)
                ax.text(start + dur/2, y + lane_height/2, f"P{i}",
                        ha="center", va="center", fontsize=7)
        
        ax.set_xlim(0, max(math.ceil(horizon), 1))
        ax.set_ylim(-0.5, max(y_positions.values()) + lane_height + 0.5)
        ax.set_xlabel("Temps")
        ax.set_yticks([y_positions[s] + lane_height/2 for s in skills])
        ax.set_yticklabels([f"S{s}" for s in skills])
        ax.set_title(f"{subtitle}\nCmax = {makespan}", fontsize=11, fontweight='bold')
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    
    plot_single(ax1, task_times_before, "Avant Optimisation", cmax_before)
    plot_single(ax2, task_times_after, "Après Optimisation", cmax_after)
    
    # Légende commune
    legend_handles = [Patch(facecolor=colors[i], edgecolor="black", 
                           label=f"P{i}") 
                     for i in range(1, num_patients + 1)]
    fig.legend(handles=legend_handles, loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(10, num_patients), frameon=True)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_convergence(fitness_history: List[float], 
                     title: str = "Courbe de Convergence",
                     figsize: Tuple = (10, 6), save_path: str = None):
    """
    Trace la courbe de convergence de la fitness.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fitness_history, 'b-', linewidth=1.5, label='Fitness')
    
    # Marquer le meilleur point
    best_idx = np.argmin(fitness_history)
    best_val = fitness_history[best_idx]
    ax.scatter([best_idx], [best_val], color='red', s=100, zorder=5, 
               label=f'Meilleur: {best_val} (it. {best_idx})')
    
    ax.set_xlabel("Itération", fontsize=11)
    ax.set_ylabel("Makespan (Cmax)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_multi_agent_convergence(agent_histories: Dict[str, List[float]],
                                 global_history: List[float] = None,
                                 title: str = "Convergence Multi-Agents",
                                 figsize: Tuple = (12, 6), save_path: str = None):
    """
    Trace les courbes de convergence de plusieurs agents.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_histories)))
    
    for (agent_id, history), color in zip(agent_histories.items(), colors):
        ax.plot(history, '-', color=color, linewidth=1, alpha=0.7, label=agent_id)
    
    if global_history:
        ax.plot(global_history, 'k-', linewidth=2.5, label='Global Best')
    
    ax.set_xlabel("Itération", fontsize=11)
    ax.set_ylabel("Makespan (Cmax)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_q_table(q_table: Dict[str, Dict[str, float]], 
                 title: str = "Table Q (Q-Learning)",
                 figsize: Tuple = (8, 6), save_path: str = None):
    """
    Visualise la table Q du Q-Learning.
    """
    states = list(q_table.keys())
    n = len(states)
    
    # Créer la matrice
    matrix = np.zeros((n, n))
    for i, s in enumerate(states):
        for j, a in enumerate(states):
            matrix[i, j] = q_table[s][a]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_xlabel("Action (Prochain voisinage)", fontsize=11)
    ax.set_ylabel("État (Voisinage actuel)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Annoter les cellules
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha="center", va="center", fontsize=9,
                          color="white" if abs(matrix[i, j]) > 0.5 else "black")
    
    plt.colorbar(im, ax=ax, label="Valeur Q")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_neighborhood_usage(stats: Dict[str, Dict], 
                           title: str = "Utilisation des Voisinages",
                           figsize: Tuple = (10, 5), save_path: str = None):
    """
    Visualise l'utilisation des différents voisinages.
    """
    neighborhoods = list(stats.keys())
    calls = [stats[n]['calls'] for n in neighborhoods]
    improvements = [stats[n]['improvements'] for n in neighborhoods]
    
    x = np.arange(len(neighborhoods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, calls, width, label='Appels', color='steelblue')
    bars2 = ax.bar(x + width/2, improvements, width, label='Améliorations', color='forestgreen')
    
    ax.set_xlabel("Voisinage", fontsize=11)
    ax.set_ylabel("Nombre", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Voisinage {n}" for n in neighborhoods])
    ax.legend()
    
    # Annoter les barres
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_diversity_matrix(distance_matrix: List[List[int]],
                          title: str = "Matrice de Diversité (EMP)",
                          figsize: Tuple = (8, 6), save_path: str = None):
    """
    Visualise la matrice des distances entre solutions de l'EMP.
    """
    matrix = np.array(distance_matrix)
    n = len(matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"S{i+1}" for i in range(n)])
    ax.set_yticklabels([f"S{i+1}" for i in range(n)])
    ax.set_xlabel("Solution", fontsize=11)
    ax.set_ylabel("Solution", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Annoter les cellules
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", fontsize=8)
    
    plt.colorbar(im, ax=ax, label="Distance")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_agent_contributions(contributions: Dict[str, int],
                            title: str = "Contributions des Agents",
                            figsize: Tuple = (8, 5), save_path: str = None):
    """
    Visualise les contributions de chaque agent à l'amélioration globale.
    """
    agents = list(contributions.keys())
    values = list(contributions.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
    bars = ax.bar(agents, values, color=colors, edgecolor='black')
    
    ax.set_xlabel("Agent", fontsize=11)
    ax.set_ylabel("Nombre d'améliorations", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Annoter les barres
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_exploration_exploitation(epsilon_history: List[float],
                                  title: str = "Balance Exploration/Exploitation",
                                  figsize: Tuple = (10, 5), save_path: str = None):
    """
    Visualise l'évolution du taux d'exploration (epsilon) au cours du temps.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(len(epsilon_history))
    exploration = epsilon_history
    exploitation = [1 - e for e in epsilon_history]
    
    ax.fill_between(iterations, exploration, alpha=0.3, color='blue', label='Exploration')
    ax.fill_between(iterations, [0]*len(iterations), exploitation, alpha=0.3, color='green', label='Exploitation')
    ax.plot(iterations, epsilon_history, 'b-', linewidth=2)
    
    ax.set_xlabel("Itération", fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig
