"""
main.py - Point d'entrée pour l'exécution du projet SMA Santé.
Supporte les arguments CLI pour tester la scalabilité.
"""
import argparse
import sys
from core import benchmark_tables

def parse_arguments():
    parser = argparse.ArgumentParser(description="Projet SMA Santé - Ordonnancement Collaboratif")
    
    parser.add_argument('--patients', type=int, default=20, 
                        help="Nombre de patients à générer (défaut: 20)")
    
    parser.add_argument('--iterations', type=int, default=15, 
                        help="Nombre d'itérations pour la simulation (défaut: 15)")
    
    parser.add_argument('--mode', type=str, choices=['benchmark', 'demo'], default='benchmark',
                        help="Mode d'exécution (défaut: benchmark)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print("================================================================")
    print("   Projet SMA Santé : Optimisation Collaborative & Q-Learning   ")
    print("================================================================")
    print(f"Configuration : {args.patients} Patients | {args.iterations} Itérations")
    print("----------------------------------------------------------------")

    # Appel avec les paramètres dynamiques
    if args.mode == 'benchmark':
        benchmark_tables.run_benchmark(num_patients=args.patients, iterations=args.iterations)
    else:
        print("Mode démo non implémenté dans cette version simplifiée.")
