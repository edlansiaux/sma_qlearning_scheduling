"""
main.py - Point d'entrée pour l'exécution du projet SMA Santé avec génération paramétrable.
"""
import argparse
import sys
from core.data_generator import (
    generate_parametric_data,
    generate_balanced_data,
    generate_realistic_healthcare_data,
    get_reference_data,
    print_data_summary
)
from core.environment import SchedulingEnvironment
from core.agents import MultiAgentSystem


def run_optimization(
    data, 
    skills, 
    num_patients, 
    iterations=50, 
    mode='FRIENDS',
    use_learning=True,
    verbose=True
):
    """
    Exécute l'optimisation avec les données fournies.
    
    Args:
        data: Dictionnaire des données patients
        skills: Liste des compétences disponibles
        num_patients: Nombre de patients
        iterations: Nombre d'itérations d'optimisation
        mode: Mode de collaboration ('FRIENDS' ou 'ENEMIES')
        use_learning: Utiliser le Q-Learning ou non
        verbose: Afficher les détails
    
    Returns:
        float: Meilleur makespan trouvé
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMISATION EN COURS")
        print(f"{'='*70}")
        print(f"Mode de collaboration: {mode}")
        print(f"Q-Learning activé: {use_learning}")
        print(f"Nombre d'itérations: {iterations}")
        print(f"{'='*70}\n")
    
    # Création de l'environnement
    env = SchedulingEnvironment(data, skills, num_patients)
    
    # Configuration des agents
    agents_config = [
        {'id': 'AG_1', 'type': 'AG', 'learning': use_learning},
        {'id': 'Tabu_1', 'type': 'Tabu', 'learning': use_learning},
        {'id': 'RS_1', 'type': 'RS', 'learning': use_learning}
    ]
    
    # Création et exécution du système multi-agents
    mas = MultiAgentSystem(env, agents_config, mode=mode)
    best_makespan = mas.run(iterations=iterations)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RÉSULTATS")
        print(f"{'='*70}")
        print(f"Meilleur makespan trouvé: {best_makespan} slots")
        print(f"Équivalent en temps: {best_makespan * 5} minutes ({best_makespan * 5 / 60:.2f} heures)")
        print(f"{'='*70}\n")
    
    return best_makespan


def run_benchmark_comparison(data, skills, num_patients, iterations=15, verbose=True):
    """
    Exécute une comparaison benchmark complète.
    
    Args:
        data: Dictionnaire des données patients
        skills: Liste des compétences disponibles
        num_patients: Nombre de patients
        iterations: Nombre d'itérations pour chaque test
        verbose: Afficher les détails
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARK COMPARATIF")
        print(f"{'='*70}")
        print_data_summary(data, skills)
    
    env = SchedulingEnvironment(data, skills, num_patients)
    
    results = {}
    
    # 1. Agents individuels sans apprentissage
    if verbose:
        print("\n1. Test des agents individuels (sans apprentissage)...")
    
    for agent_type in ['AG', 'Tabu', 'RS']:
        mas = MultiAgentSystem(env, [
            {'id': agent_type, 'type': agent_type, 'learning': False}
        ], mode='ENEMIES')
        results[f'{agent_type}_solo'] = mas.run(iterations)
        if verbose:
            print(f"   {agent_type} seul: {results[f'{agent_type}_solo']} slots")
    
    # 2. SMA sans apprentissage (mode FRIENDS)
    if verbose:
        print("\n2. Test du SMA sans apprentissage (mode FRIENDS)...")
    
    mas = MultiAgentSystem(env, [
        {'id': '1', 'type': 'AG', 'learning': False},
        {'id': '2', 'type': 'Tabu', 'learning': False},
        {'id': '3', 'type': 'RS', 'learning': False}
    ], mode='FRIENDS')
    results['SMA_no_learn'] = mas.run(iterations)
    if verbose:
        print(f"   SMA sans apprentissage: {results['SMA_no_learn']} slots")
    
    # 3. SMA avec apprentissage (mode FRIENDS)
    if verbose:
        print("\n3. Test du SMA avec apprentissage (mode FRIENDS)...")
    
    mas = MultiAgentSystem(env, [
        {'id': '1', 'type': 'AG', 'learning': True},
        {'id': '2', 'type': 'Tabu', 'learning': True},
        {'id': '3', 'type': 'RS', 'learning': True}
    ], mode='FRIENDS')
    results['SMA_with_learn'] = mas.run(iterations)
    if verbose:
        print(f"   SMA avec apprentissage: {results['SMA_with_learn']} slots")
    
    # 4. SMA avec apprentissage (mode ENEMIES)
    if verbose:
        print("\n4. Test du SMA avec apprentissage (mode ENEMIES)...")
    
    mas = MultiAgentSystem(env, [
        {'id': '1', 'type': 'AG', 'learning': True},
        {'id': '2', 'type': 'Tabu', 'learning': True},
        {'id': '3', 'type': 'RS', 'learning': True}
    ], mode='ENEMIES')
    results['SMA_with_learn_enemies'] = mas.run(iterations)
    if verbose:
        print(f"   SMA avec apprentissage (enemies): {results['SMA_with_learn_enemies']} slots")
    
    # Affichage du résumé
    if verbose:
        print(f"\n{'='*70}")
        print(f"RÉSUMÉ COMPARATIF")
        print(f"{'='*70}")
        print(f"{'Configuration':<40} {'Makespan':>10} {'Amélioration':>15}")
        print(f"{'-'*70}")
        baseline = results['AG_solo']
        for key, value in results.items():
            improvement = ((baseline - value) / baseline * 100) if baseline > 0 else 0
            print(f"{key:<40} {value:>10} {improvement:>14.1f}%")
        print(f"{'='*70}\n")
    
    return results


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Projet SMA Santé - Ordonnancement Collaboratif avec Données Paramétrables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Utiliser les données de référence (image)
  python main.py --use-reference
  
  # Générer 15 patients avec 8 skills
  python main.py --patients 15 --skills 8 --mode optimize
  
  # Benchmark avec données équilibrées
  python main.py --patients 20 --skills 6 --generator balanced --mode benchmark
  
  # Optimisation réaliste (contexte médical)
  python main.py --patients 12 --skills 5 --generator realistic --iterations 100
        """
    )
    
    # Paramètres de génération de données
    data_group = parser.add_argument_group('Génération de données')
    data_group.add_argument(
        '--use-reference', 
        action='store_true',
        help="Utiliser les données de référence de l'image (10 patients, 6 skills)"
    )
    data_group.add_argument(
        '--patients', 
        type=int, 
        default=10,
        help="Nombre de patients à générer (défaut: 10)"
    )
    data_group.add_argument(
        '--skills', 
        type=int, 
        default=6,
        help="Nombre de compétences/ressources (défaut: 6)"
    )
    data_group.add_argument(
        '--max-operations', 
        type=int, 
        default=5,
        help="Nombre maximum d'opérations par patient (défaut: 5)"
    )
    data_group.add_argument(
        '--generator',
        type=str,
        choices=['parametric', 'balanced', 'realistic'],
        default='parametric',
        help="Type de générateur de données (défaut: parametric)"
    )
    data_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Seed pour la génération aléatoire (pour reproductibilité)"
    )
    
    # Paramètres d'optimisation
    optim_group = parser.add_argument_group('Optimisation')
    optim_group.add_argument(
        '--iterations', 
        type=int, 
        default=50,
        help="Nombre d'itérations pour l'optimisation (défaut: 50)"
    )
    optim_group.add_argument(
        '--collaboration',
        type=str,
        choices=['FRIENDS', 'ENEMIES'],
        default='FRIENDS',
        help="Mode de collaboration entre agents (défaut: FRIENDS)"
    )
    optim_group.add_argument(
        '--no-learning',
        action='store_true',
        help="Désactiver le Q-Learning"
    )
    
    # Mode d'exécution
    parser.add_argument(
        '--mode',
        type=str,
        choices=['optimize', 'benchmark', 'both'],
        default='optimize',
        help="Mode d'exécution (défaut: optimize)"
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Mode silencieux (moins de détails affichés)"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale."""
    args = parse_arguments()
    
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*70)
        print(" " * 15 + "PROJET SMA SANTÉ")
        print(" " * 10 + "Ordonnancement Collaboratif & Q-Learning")
        print("="*70)
    
    # Génération ou chargement des données
    if args.use_reference:
        if verbose:
            print("\nUtilisation des données de référence (image)...")
        data, skills, num_patients = get_reference_data()
    else:
        if verbose:
            print(f"\nGénération des données ({args.generator})...")
        
        if args.generator == 'parametric':
            data, skills = generate_parametric_data(
                num_patients=args.patients,
                num_skills=args.skills,
                max_operations=args.max_operations,
                seed=args.seed
            )
        elif args.generator == 'balanced':
            data, skills = generate_balanced_data(
                num_patients=args.patients,
                num_skills=args.skills,
                max_operations=args.max_operations,
                seed=args.seed
            )
        else:  # realistic
            data, skills = generate_realistic_healthcare_data(
                num_patients=args.patients,
                num_skills=args.skills,
                seed=args.seed
            )
        
        num_patients = args.patients
    
    if verbose:
        print_data_summary(data, skills)
    
    # Exécution selon le mode choisi
    if args.mode in ['optimize', 'both']:
        run_optimization(
            data=data,
            skills=skills,
            num_patients=num_patients,
            iterations=args.iterations,
            mode=args.collaboration,
            use_learning=not args.no_learning,
            verbose=verbose
        )
    
    if args.mode in ['benchmark', 'both']:
        run_benchmark_comparison(
            data=data,
            skills=skills,
            num_patients=num_patients,
            iterations=min(args.iterations, 20),  # Limite pour le benchmark
            verbose=verbose
        )
    
    if verbose:
        print("\n" + "="*70)
        print(" " * 20 + "TERMINÉ")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
