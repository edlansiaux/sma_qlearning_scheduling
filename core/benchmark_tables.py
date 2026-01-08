"""
core/benchmark_tables.py - Génération des tableaux comparatifs (Slides 25 & 26).
"""
from core.environment import SchedulingEnvironment, generate_random_data
from core.agents import MultiAgentSystem

def run_benchmark(num_patients=20, iterations=15):
    """
    Exécute le benchmark avec des paramètres configurables.
    """
    print(f"Génération des données pour {num_patients} patients...")
    
    # Setup environnement
    skills = [1, 2, 3, 4]
    # Génération dynamique basée sur le paramètre num_patients
    # On ajuste max_ops si nécessaire, mais on garde 4 pour l'instant
    data = generate_random_data(num_patients=num_patients, max_ops=4, skills=skills)
    env = SchedulingEnvironment(data, skills, num_patients)

    # --- TABLEAU 1: Comparaison SANS collaboration (Slide 25) ---
    print("\n--- Tableau de comparaison sans collaboration (Slide 25) ---")
    print(f'"Jour","Nombre de patients",,"Métaheuristique",,,"Métaheuristique Agents Métaheuristique_Agents_Apprentissage"')
    print(',,"AG","Tabou","RS","Agent_AG Agent_Tabou Agent_RS","Agent_AG Agent_Tabou Agent_RS"')

    # 1. Classique (Simulé par Agent Solo en mode ENNEMIS sans Learning)
    ag_solo = MultiAgentSystem(env, [{'id':'AG','type':'AG','learning':False}], mode='ENEMIES').run(iterations)
    tabu_solo = MultiAgentSystem(env, [{'id':'Tabu','type':'Tabu','learning':False}], mode='ENEMIES').run(iterations)
    rs_solo = MultiAgentSystem(env, [{'id':'RS','type':'RS','learning':False}], mode='ENEMIES').run(iterations)

    # 2. Agents Sans Apprentissage (SMA No Learn)
    sma_no_learn = MultiAgentSystem(env, [
        {'id':'1','type':'AG','learning':False},
        {'id':'2','type':'Tabu','learning':False},
        {'id':'3','type':'RS','learning':False}
    ], mode='FRIENDS').run(iterations)

    # 3. Agents Avec Apprentissage (SMA Learn)
    sma_learn = MultiAgentSystem(env, [
        {'id':'1','type':'AG','learning':True},
        {'id':'2','type':'Tabu','learning':True},
        {'id':'3','type':'RS','learning':True}
    ], mode='FRIENDS').run(iterations)

    print(f'"J1","{num_patients}",,"{ag_solo}","{tabu_solo}","{rs_solo}","{sma_no_learn}","{sma_learn}"')


    # --- TABLEAU 2: Comparaison AVEC collaboration (Slide 26) ---
    print("\n--- Tableau Slide 26 ---")
    print('"Jour","Nb patients","SMA sans apprentissage (Amis)",,,,"SMA avec apprentissage (Amis)",,,,"SMA sans apprentissage (Ennemis)",,,,,,,,"SMA avec apprentissage (Ennemis)"')
    print(',,"AG_Tabou","AG_RS","Tabou_RS","AG_Tabou_RS","AG_Tabou","AG_RS","Tabou_RS","AG_Tabou","AG_RS","Tabou_RS","AG_Tabou_RS", "AG_Tabou","AG_RS","Tabou_RS","AG_Tabou_RS"')
    
    pairs = [
        ([{'id':'1','type':'AG','learning':False},{'id':'2','type':'Tabu','learning':False}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':False},{'id':'3','type':'RS','learning':False}], 'FRIENDS'),
        ([{'id':'2','type':'Tabu','learning':False},{'id':'3','type':'RS','learning':False}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':False},{'id':'2','type':'Tabu','learning':False},{'id':'3','type':'RS','learning':False}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':True},{'id':'2','type':'Tabu','learning':True}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':True},{'id':'3','type':'RS','learning':True}], 'FRIENDS'),
        ([{'id':'2','type':'Tabu','learning':True},{'id':'3','type':'RS','learning':True}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':True},{'id':'2','type':'Tabu','learning':True},{'id':'3','type':'RS','learning':True}], 'FRIENDS'),
        ([{'id':'1','type':'AG','learning':False},{'id':'2','type':'Tabu','learning':False}], 'ENEMIES'),
        ([{'id':'1','type':'AG','learning':False},{'id':'3','type':'RS','learning':False}], 'ENEMIES'),
        ([{'id':'2','type':'Tabu','learning':False},{'id':'3','type':'RS','learning':False}], 'ENEMIES'),
        ([{'id':'1','type':'AG','learning':False},{'id':'2','type':'Tabu','learning':False},{'id':'3','type':'RS','learning':False}], 'ENEMIES'),
        ([{'id':'1','type':'AG','learning':True},{'id':'2','type':'Tabu','learning':True}], 'ENEMIES'),
        ([{'id':'1','type':'AG','learning':True},{'id':'3','type':'RS','learning':True}], 'ENEMIES'),
        ([{'id':'2','type':'Tabu','learning':True},{'id':'3','type':'RS','learning':True}], 'ENEMIES'),
        ([{'id':'1','type':'AG','learning':True},{'id':'2','type':'Tabu','learning':True},{'id':'3','type':'RS','learning':True}], 'ENEMIES')
    ]

    
    res = []
    for conf, mode in pairs:
        res.append(MultiAgentSystem(env, conf, mode=mode).run(iterations))
        
    print(f'"J1","{num_patients}","{res[0]}","{res[1]}","{res[2]}",,"{res[3]}","{res[4]}","{res[5]}","{res[6]}","{res[7]}","{res[8]}","{res[9]}","{res[10]}","{res[11]}","{res[12]}","{res[13]}","{res[14]}","{res[15]}"')
        
    print(f'"J1","{num_patients}",,"{results[0]}","{results[1]}","{results[2]}",,"{results[3]}","{results[4]}","{results[5]}"')

if __name__ == "__main__":
    # Fallback si exécuté directement sans main.py
    run_benchmark(20, 15)
