"""
visualization.py - Visualisations Gantt, Convergence, Q-Table, Diversité.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np


def patient_colors(num_patients):
    cmap = plt.get_cmap("tab20")
    return {i: cmap(i % 20) for i in range(1, num_patients + 5)}


def plot_gantt(task_times, skills, num_patients, title="Gantt", save_path=None):
    if not task_times:
        print("Aucune tâche à afficher.")
        return

    horizon = max(t[1] for t in task_times.values())
    colors = patient_colors(num_patients)

    fig, ax = plt.subplots(figsize=(14, max(4, len(skills) * 0.9)))
    bar_height = 0.8
    y_mapping = {s: i for i, s in enumerate(sorted(skills))}

    ax.set_ylim(-0.5, len(skills) - 0.5)
    ax.set_xlim(0, horizon + 1)

    for (pid, op_stage, skill), (start, end, dur) in task_times.items():
        y = y_mapping.get(skill, 0)
        rect = Rectangle(
            (start, y - bar_height / 2), dur, bar_height,
            facecolor=colors[pid], edgecolor="black", alpha=0.8,
        )
        ax.add_patch(rect)
        if dur >= 1:
            ax.text(
                start + dur / 2, y, f"P{pid}\nO{op_stage}",
                ha="center", va="center", color="white", fontsize=7, fontweight="bold",
            )

    ax.set_yticks(list(y_mapping.values()))
    ax.set_yticklabels([f"Skill {s}" for s in sorted(skills)])

    legend_elements = [
        Patch(facecolor=colors[i], label=f"Patient {i}")
        for i in range(1, num_patients + 1)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_xlabel("Temps")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_convergence(fitness_history_dict, title="Convergence", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_id, history in fitness_history_dict.items():
        ax.plot(history, label=agent_id, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Itérations")
    ax.set_ylabel("Makespan (Cmax)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_gantt_comparison(
    initial_times, final_times, skills, num_patients,
    init_cmax, final_cmax, title="Comparaison", save_path=None,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    colors = patient_colors(num_patients)
    bar_height = 0.8
    y_mapping = {s: i for i, s in enumerate(sorted(skills))}
    yticks = list(y_mapping.values())
    yticklabels = [f"Skill {s}" for s in sorted(skills)]

    for ax, times, cmax_label, subtitle in [
        (ax1, initial_times, init_cmax, "Avant Optimisation"),
        (ax2, final_times, final_cmax, "Après Optimisation"),
    ]:
        h = max(t[1] for t in times.values()) if times else 0
        ax.set_ylim(-0.5, len(skills) - 0.5)
        ax.set_xlim(0, h + 1)
        for (pid, op_stage, skill), (start, end, dur) in times.items():
            y = y_mapping.get(skill, 0)
            rect = Rectangle(
                (start, y - bar_height / 2), dur, bar_height,
                facecolor=colors[pid], edgecolor="black", alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                start + dur / 2, y, f"P{pid}",
                ha="center", va="center", color="white", fontsize=7, fontweight="bold",
            )
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title(f"{subtitle} (Cmax = {cmax_label})")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    ax2.set_xlabel("Temps")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ── Visualisations avancées ──────────────────────────────────────────────────

def plot_q_table(agent, title="Q-Table", save_path=None):
    """Heatmap de la Q-Table d'un agent."""
    if not hasattr(agent, "q_selector") or agent.q_selector is None:
        print(f"L'agent {agent.agent_id} n'utilise pas le Q-Learning.")
        return

    q_dict = agent.q_selector.q_agent.get_q_table_formatted()
    states = sorted(q_dict.keys())
    actions = list(q_dict[states[0]].keys()) if states else []
    if not states or not actions:
        return

    matrix = np.array([[q_dict[s][a] for a in actions] for s in states])

    fig, ax = plt.subplots(figsize=(max(6, len(actions) * 1.5), max(4, len(states) * 0.8)))
    cax = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    fig.colorbar(cax, label="Q-Value")

    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels(actions, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=11)

    for i in range(len(states)):
        for j in range(len(actions)):
            val = matrix[i, j]
            max_abs = np.max(np.abs(matrix)) if np.max(np.abs(matrix)) > 0 else 1
            tc = "white" if abs(val) > max_abs / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=tc, fontsize=10, fontweight="bold")

    ax.set_title(f"{title} – {getattr(agent, 'agent_id', '')}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actions (Voisinages)")
    ax.set_ylabel("États")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_neighborhood_usage(agent, title="Utilisation des voisinages", save_path=None):
    """Diagramme en barres de l'utilisation de chaque voisinage."""
    if not hasattr(agent, "q_selector") or agent.q_selector is None:
        return
    stats = agent.q_selector.get_statistics()
    if not stats:
        return

    names = sorted(stats.keys())
    calls = [stats[n]["calls"] for n in names]
    improvements = [stats[n]["improvements"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, calls, width, label="Appels", color="steelblue")
    ax.bar(x + width / 2, improvements, width, label="Améliorations", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight="bold")
    ax.set_ylabel("Nombre")
    ax.set_title(f"{title} – {getattr(agent, 'agent_id', '')}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_diversity_matrix(emp, title="Matrice des Distances (EMP)", save_path=None):
    """Heatmap des distances entre les solutions de l'EMP."""
    solutions = emp.solutions
    n = len(solutions)
    if n < 2:
        print(f"Pas assez de solutions ({n}).")
        return

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = emp.calculate_distance(solutions[i].sequences, solutions[j].sequences)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.45)))
    cax = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    fig.colorbar(cax, label="Distance de Hamming")

    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            max_val = np.max(matrix) if np.max(matrix) > 0 else 1
            c = "white" if val < max_val / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=c, fontsize=8, fontweight="bold")

    ax.set_title(f"{title} – {n} solutions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Index Solution")
    ax.set_ylabel("Index Solution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_agent_contributions(mas, title="Contributions des agents", save_path=None):
    """Histogramme des meilleures fitness par agent."""
    agents = mas.agents
    names = list(agents.keys())
    best_fits = [agents[a].best_fitness for a in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, best_fits, color="cornflowerblue", edgecolor="black")

    for bar, val in zip(bars, best_fits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Meilleur Makespan")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_multi_agent_convergence(mas, title="Convergence Multi-Agents", save_path=None):
    """Courbes de convergence de tous les agents d'un MAS."""
    histories = {aid: agent.fitness_history for aid, agent in mas.agents.items()}
    plot_convergence(histories, title=title, save_path=save_path)
