"""
core/benchmark_tables.py - Génération des tableaux comparatifs (Slides 25 & 26).
"""
from core.environment import SchedulingEnvironment, generate_random_data
from core.agents import MultiAgentSystem
import random

def run_benchmark():
    print("Génération des données et exécution des benchmarks...")
    # Setup environnement test
    skills = [1, 2, 3, 4]
    # Scénario moyen avec 20 patients
    data = generate_random_data(num_patients=20, max_ops=4, skills=skills)
    env = SchedulingEnvironment(data, skills, 20)
    ITERATIONS = 15

    # --- TABLEAU 1: Comparaison SANS collaboration (Slide 25) ---
    print("\n--- Tableau de comparaison sans collaboration (Slide 25) [cite: 1197] ---")
    print('"Jour","Nombre de patients",,"Métaheuristique",,,"Métaheuristique Agents Métaheuristique_Agents_Apprentissage"')
    print(',,"AG","Tabou","RS","Agent_AG Agent_Tabou Agent_RS","Agent_AG Agent_Tabou Agent_RS"')

    # 1. Classique (Simulé par Agent Solo en mode ENNEMIS sans Learning, isolés)
    ag_solo = MultiAgentSystem(env, [{'id':'AG','type':'AG','learning':False}], mode='ENEMIES').run(ITERATIONS)
    tabu_solo = MultiAgentSystem(env, [{'id':'Tabu','type':'Tabu','learning':False}], mode='ENEMIES').run(ITERATIONS)
    rs_solo = MultiAgentSystem(env, [{'id':'RS','type':'RS','learning':False}], mode='ENEMIES').run(ITERATIONS)

    # 2. Agents Sans Apprentissage (SMA No Learn - Mode Amis pour coopération de base)
    sma_no_learn = MultiAgentSystem(env, [
        {'id':'1','type':'AG','learning':False},
        {'id':'2','type':'Tabu','learning':False},
        {'id':'3','type':'RS','learning':False}
    ], mode='FRIENDS').run(ITERATIONS)

    # 3. Agents Avec Apprentissage (SMA Learn - Le coeur du sujet)
    sma_learn = MultiAgentSystem(env, [
        {'id':'1','type':'AG','learning':True},
        {'id':'2','type':'Tabu','learning':True},
        {'id':'3','type':'RS','learning':True}
    ], mode='FRIENDS').run(ITERATIONS)

    # Affichage ligne format CSV
    print(f'"J1","20",,"{ag_solo}","{tabu_solo}","{rs_solo}","{sma_no_learn}","{sma_learn}"')


    # --- TABLEAU 2: Comparaison AVEC collaboration (Slide 26) ---
    print("\n\n--- Tableau de comparaison avec collaboration (Slide 26) [cite: 1200] ---")
    print('"Jour","Nb Patients",,,"SMA sans apprentissage",,,,,,"SMA avec apprentissage"')
    print(',,,"Amis",,,,,,"Ennemis"')
    print(',,"AG_Tabou","AG_RS","Tabou_RS",,"AG_Tabou","AG_RS","Tabou_RS"')
    
    # Configuration des paires d'agents pour tester les interactions
    pairs_config = [
        # Amis Sans Apprentissage
        ([{'id':'1','type':'AG','learning':False},{'id':'2','type':'Tabu','learning':False}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':False},{'id':'3','type':'RS','learning':False}], 'FRIENDS'),
        ([{'id':'2','type':'Tabu','learning':False},{'id':'3','type':'RS','learning':False}], 'FRIENDS'),
        
        # Ennemis Avec Apprentissage (Note: Le titre slide 26 est complexe, 
        # ici on illustre la colonne "SMA avec apprentissage - Ennemis")
        ([{'id':'1','type':'AG','learning':True},{'id':'2','type':'Tabu','learning':True}], 'ENEMIES'),
        ([{'id':'1','type':'AG','learning':True},{'id':'3','type':'RS','learning':True}], 'ENEMIES'),
        ([{'id':'2','type':'Tabu','learning':True},{'id':'3','type':'RS','learning':True}], 'ENEMIES'),
    ]
    
    results = []
    for conf, mode in pairs_config:
        res = MultiAgentSystem(env, conf, mode=mode).run(ITERATIONS)
        results.append(res)
        
    print(f'"J1","20",,"{results[0]}","{results[1]}","{results[2]}",,"{results[3]}","{results[4]}","{results[5]}"')

if __name__ == "__main__":
    run_benchmark()
