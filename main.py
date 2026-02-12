"""
main.py - Point d'entrée principal avec options CLI.
"""

import sys
import os
import argparse
import random
import numpy as np

from core.environment import (
    create_default_environment,
    generate_random_data,
    SchedulingEnvironment,
    DEFAULT_SKILLS,
    DEFAULT_MAX_OPS,
)
from core.data_generator import (
    get_reference_data,
    generate_parametric_data,
    generate_balanced_data,
    generate_realistic_healthcare_data,
    print_data_summary,
)
from core.agents import MultiAgentSystem, CollaborationMode
from visualization import plot_gantt, plot_convergence


def print_header():
    print("\n" + "=" * 70)
    print("   SYSTÈME MULTI-AGENTS – ORDONNANCEMENT PATIENTS")
    print("   5 voisinages (A–E) · Q-Learning · Amis / Ennemis")
    print("=" * 70)


def run_optimize(env, args):
    """Optimisation simple avec affichage Gantt."""
    mode = (
        CollaborationMode.FRIENDS
        if args.collaboration == "FRIENDS"
        else CollaborationMode.ENEMIES
    )
    use_ql = not args.no_learning

    mas = MultiAgentSystem(env, mode=mode, use_qlearning=use_ql)
    mas.add_agent("genetic", "AG", population_size=15)
    mas.add_agent("tabu", "Tabu", tabu_tenure=10)
    mas.add_agent("sa", "RS", initial_temp=100)

    print(f"Mode : {mode.upper()} | Q-Learning : {'ON' if use_ql else 'OFF'}")
    print(f"Itérations : {args.iterations}")
    print("Lancement de l'optimisation…")

    best_sol = mas.run(n_iterations=args.iterations, verbose=not args.quiet)

    if best_sol:
        print(f"\nMeilleur Makespan : {best_sol.fitness}")
        cmax, task_times, _ = env.evaluate(best_sol.sequences, return_schedule=True)
        output_file = "gantt_resultat.png"
        plot_gantt(
            task_times, env.skills, env.num_patients,
            title=f"Planning Optimisé (Cmax={cmax})",
            save_path=output_file,
        )
        print(f"Gantt sauvegardé : {output_file}")
    else:
        print("Aucune solution trouvée.")


def run_benchmark():
    """Lance le benchmark complet (Tableaux 1 & 2)."""
    from core.benchmark_tables import main as bench_main
    bench_main()


def main():
    parser = argparse.ArgumentParser(description="SMA Ordonnancement Patients")

    # Données
    parser.add_argument("--use-reference", action="store_true", help="Données de référence (10 patients, 6 skills)")
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--skills", type=int, default=6)
    parser.add_argument("--max-operations", type=int, default=5)
    parser.add_argument("--generator", choices=["parametric", "balanced", "realistic"], default="parametric")
    parser.add_argument("--seed", type=int, default=None)

    # Optimisation
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--collaboration", choices=["FRIENDS", "ENEMIES"], default="FRIENDS")
    parser.add_argument("--no-learning", action="store_true")

    # Exécution
    parser.add_argument("--mode", choices=["optimize", "benchmark", "both"], default="optimize")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    print_header()

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Construction de l'environnement
    if args.use_reference:
        env = create_default_environment()
        print(f"Données de référence : {env.num_patients} patients, {len(env.skills)} compétences")
    else:
        skills = list(range(1, args.skills + 1))
        if args.generator == "balanced":
            data, skills = generate_balanced_data(args.patients, args.skills, args.max_operations, seed=args.seed)
        elif args.generator == "realistic":
            data, skills = generate_realistic_healthcare_data(args.patients, args.skills, seed=args.seed)
        else:
            data, skills = generate_parametric_data(args.patients, args.skills, args.max_operations, seed=args.seed)

        env = SchedulingEnvironment(data, skills, args.patients, args.max_operations)
        print(f"Données générées ({args.generator}) : {args.patients} patients, {len(skills)} compétences")
        print_data_summary(data, skills)

    # Exécution
    if args.mode in ("optimize", "both"):
        run_optimize(env, args)

    if args.mode in ("benchmark", "both"):
        run_benchmark()


if __name__ == "__main__":
    main()
