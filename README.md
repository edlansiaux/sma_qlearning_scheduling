# Multi-Agent System with Q-Learning for Patient Scheduling

Ce projet implémente un système multi-agents collaboratif pour l'optimisation de l'ordonnancement de patients en milieu hospitalier. Il intègre l'apprentissage par renforcement (Q-Learning) pour l'auto-adaptation des agents.

## 🎯 Objectif

**Minimiser le makespan** : Réduire le temps total nécessaire pour traiter tous les patients en optimisant l'allocation des ressources (compétences/personnel médical) et le séquencement des opérations.

## 📋 Fonctionnalités Principales

### Génération de Données Paramétrable

Le projet propose plusieurs méthodes de génération de données pour tester différents scénarios :

- **Données de référence** : Basées sur l'image fournie (10 patients, 6 skills)
- **Générateur paramétrique** : Personnalisation complète (nombre de patients, skills, opérations, durées, etc.)
- **Générateur équilibré** : Équilibre automatique de la charge entre ressources
- **Générateur réaliste** : Parcours de soins réalistes (consultation → examens → traitement)

### Métaheuristiques Hybrides

- **Algorithme Génétique (AG)** : Avec opérateurs de croisement ordonnés et mutation adaptative
- **Recherche Tabu** : Avec liste tabu et critère d'aspiration
- **Recuit Simulé (SA)** : Avec refroidissement exponentiel

### Modes de Collaboration

- **Mode Amis (FRIENDS)** : Les agents partagent des solutions complètes via la Mémoire Partagée (SMP)
- **Mode Ennemis (ENEMIES)** : Les agents ne partagent que les valeurs de fitness (compétition)

### Auto-Adaptation via Q-Learning

- Processus de Décision Markovien (MDP)
- Q-Table pour la sélection des voisinages
- Équilibre exploration/exploitation (ε-greedy)

### Mémoire Partagée (SMP)

- Contrôle de diversité basé sur la distance entre solutions
- Insertion conditionnelle selon un seuil de diversité
- Remplacement de la pire solution si nécessaire

### 5 Fonctions de Voisinage

- **A** : Réassignation à un autre personnel médical
- **B** : Réassignation de tâches successives
- **C** : Insertion dans le même planning (déplacement temporel)
- **D** : Échange entre différents personnels
- **E** : Échange au sein du même personnel

## 🏗️ Architecture du Projet

```
sma_qlearning_scheduling/
├── core/
│   ├── __init__.py              # Exports du package
│   ├── environment.py           # Environnement d'ordonnancement
│   ├── data_generator.py        # 🆕 Générateurs de données paramétrables
│   ├── neighborhoods.py         # 5 fonctions de voisinage
│   ├── qlearning.py             # Q-Learning et MDP
│   ├── shared_memory.py         # SMP avec contrôle de diversité
│   └── agents.py                # Agents et système multi-agents
├── visualization.py             # Visualisations (Gantt, convergence, etc.)
├── main.py                      # 🆕 Script principal avec options CLI
├── notebook_demo.ipynb          # Notebook Jupyter interactif
└── README.md                    # Ce fichier
```

## 🚀 Installation et Utilisation

### Prérequis

```bash
pip install numpy matplotlib
```

### Utilisation Rapide

#### 1. Utiliser les données de référence (image)

```bash
python main.py --use-reference
```

#### 2. Générer des données personnalisées

```bash
# 15 patients avec 8 skills
python main.py --patients 15 --skills 8

# 20 patients avec 6 skills, générateur équilibré
python main.py --patients 20 --skills 6 --generator balanced

# 12 patients avec 5 skills, contexte médical réaliste
python main.py --patients 12 --skills 5 --generator realistic
```

#### 3. Exécuter un benchmark comparatif

```bash
python main.py --patients 20 --skills 6 --mode benchmark
```

#### 4. Mode complet (optimisation + benchmark)

```bash
python main.py --patients 15 --skills 7 --mode both --iterations 100
```

### Options de Ligne de Commande Complètes

#### Génération de données

| Option | Description | Défaut |
|--------|-------------|--------|
| `--use-reference` | Utiliser les données de l'image (10 patients, 6 skills) | False |
| `--patients N` | Nombre de patients | 10 |
| `--skills N` | Nombre de compétences/ressources | 6 |
| `--max-operations N` | Nombre max d'opérations par patient | 5 |
| `--generator TYPE` | Type de générateur (`parametric`, `balanced`, `realistic`) | `parametric` |
| `--seed N` | Seed pour reproductibilité | None |

#### Optimisation

| Option | Description | Défaut |
|--------|-------------|--------|
| `--iterations N` | Nombre d'itérations | 50 |
| `--collaboration MODE` | Mode de collaboration (`FRIENDS`, `ENEMIES`) | `FRIENDS` |
| `--no-learning` | Désactiver le Q-Learning | False |

#### Exécution

| Option | Description | Défaut |
|--------|-------------|--------|
| `--mode MODE` | Mode d'exécution (`optimize`, `benchmark`, `both`) | `optimize` |
| `--quiet` | Mode silencieux | False |

### Exemples d'Utilisation

```bash
# Exemple 1 : Test rapide avec données de référence
python main.py --use-reference --iterations 30

# Exemple 2 : Scénario hospitalier réaliste avec 25 patients
python main.py --patients 25 --skills 8 --generator realistic --iterations 100

# Exemple 3 : Benchmark complet avec données équilibrées
python main.py --patients 20 --skills 6 --generator balanced --mode benchmark

# Exemple 4 : Test de scalabilité
python main.py --patients 50 --skills 10 --iterations 150 --quiet

# Exemple 5 : Comparaison des modes de collaboration
python main.py --patients 15 --skills 7 --collaboration FRIENDS --mode benchmark
python main.py --patients 15 --skills 7 --collaboration ENEMIES --mode benchmark

# Exemple 6 : Reproductibilité avec seed
python main.py --patients 20 --skills 6 --seed 42 --mode both
```

### Utilisation Programmatique

```python
from core import (
    generate_parametric_data,
    generate_balanced_data,
    get_reference_data,
    print_data_summary,
    SchedulingEnvironment,
    MultiAgentSystem
)

# Méthode 1 : Utiliser les données de référence
data, skills, num_patients = get_reference_data()

# Méthode 2 : Générer des données personnalisées
data, skills = generate_parametric_data(
    num_patients=15,
    num_skills=8,
    max_operations=5,
    operation_probability=0.75,
    min_duration=15,
    max_duration=60,
    seed=42  # Pour reproductibilité
)
num_patients = 15

# Méthode 3 : Données équilibrées
data, skills = generate_balanced_data(
    num_patients=20,
    num_skills=6,
    max_operations=5,
    seed=42
)

# Afficher le résumé des données
print_data_summary(data, skills)

# Créer l'environnement
env = SchedulingEnvironment(data, skills, num_patients)

# Créer le système multi-agents
mas = MultiAgentSystem(
    env, 
    agents_config=[
        {'id': 'AG_1', 'type': 'AG', 'learning': True},
        {'id': 'Tabu_1', 'type': 'Tabu', 'learning': True},
        {'id': 'SA_1', 'type': 'RS', 'learning': True}
    ],
    mode='FRIENDS'  # ou 'ENEMIES'
)

# Exécuter l'optimisation
best_makespan = mas.run(iterations=100)
print(f"Meilleur makespan: {best_makespan} slots ({best_makespan * 5} minutes)")
```

### Utiliser le Notebook

```bash
jupyter notebook notebook_demo.ipynb
```

## 📊 Paramètres Principaux

### Q-Learning

| Paramètre | Description | Valeur par Défaut |
|-----------|-------------|-------------------|
| α (alpha) | Taux d'apprentissage | 0.1 |
| γ (gamma) | Facteur d'actualisation | 0.9 |
| ε (epsilon) | Taux d'exploration initial | 0.1 |

### SMP (Mémoire Partagée)

| Paramètre | Description | Valeur par Défaut |
|-----------|-------------|-------------------|
| max_size | Taille maximale | 20 |
| R (min_distance) | Distance minimale entre solutions | 2 |
| DT (diversity_threshold) | Seuil de diversité | 0.5 |

### Générateurs de Données

#### Générateur Paramétrique

```python
generate_parametric_data(
    num_patients=10,           # Nombre de patients
    num_skills=6,              # Nombre de compétences
    max_operations=5,          # Opérations max par patient
    operation_probability=0.7, # Probabilité qu'une opération existe
    min_duration=10,           # Durée min (minutes)
    max_duration=60,           # Durée max (minutes)
    max_tasks_per_operation=3, # Tâches max par opération
    seed=None                  # Seed pour reproductibilité
)
```

#### Générateur Équilibré

Génère des données où chaque skill est utilisée de manière équilibrée pour tester l'équilibrage de charge.

```python
generate_balanced_data(
    num_patients=10,
    num_skills=6,
    max_operations=5,
    seed=None
)
```

#### Générateur Réaliste

Génère des parcours de soins réalistes avec des séquences logiques (consultation → examens → traitement).

```python
generate_realistic_healthcare_data(
    num_patients=10,
    num_skills=6,
    seed=None
)
```

## 📈 Visualisations Disponibles

- **Diagramme de Gantt** : Planning des tâches par compétence
- **Courbe de Convergence** : Évolution du makespan
- **Q-Table** : Valeurs apprises par Q-Learning
- **Matrice de Diversité** : Distances entre solutions dans la SMP
- **Contributions des Agents** : Améliorations par agent

## 🎓 Données de Référence (Image)

Les données de référence correspondent à l'exemple de la table de compétences fournie :

- **10 patients** (Patient 1 à Patient 10)
- **6 compétences** (Skill 1 à Skill 6)
- **5 opérations maximum** par patient
- Répartition variable des tâches selon le patient

Ces données peuvent être utilisées comme benchmark de référence avec `--use-reference`.

## 🔬 Basé Sur

Ce projet est basé sur la présentation :
> "Optimisation Collaborative: Agents Auto-Adaptatifs, Apprentissage par Renforcement"

### Références Conceptuelles

- Processus de Décision Markovien (Bellman)
- Q-Learning (Watkins & Dayan, 1992)
- Systèmes Multi-Agents pour l'optimisation (Jin & Liu 2002, Milano & Roli 2004)
- Métaheuristiques hybrides (Fernandes et al. 2009)

## 📄 Licence

Ce projet est publié sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 👥 Auteurs & Remerciements

- [Mohammed Berrajaa](https://github.com/medberrajaa)
- [Guillaume Gauguet](https://github.com/GAUGUET)
- [Hugo Kazzi](https://github.com/hugokazzi63)
- [Abdallah Lafendi](https://github.com/imadlaf2503)
- [Edouard Lansiaux](https://github.com/edlansiaux)
- [Aurélien Loison](https://github.com/lsnaurelien)

Merci à tous les contributeurs qui aident à améliorer ce projet !

## 🚦 Tests de Scalabilité

Le projet permet de tester facilement la scalabilité de l'approche :

```bash
# Test avec 10 patients (petit problème)
python main.py --patients 10 --skills 5 --mode benchmark

# Test avec 30 patients (problème moyen)
python main.py --patients 30 --skills 8 --mode benchmark

# Test avec 50 patients (grand problème)
python main.py --patients 50 --skills 10 --mode benchmark --iterations 200
```

## 💡 Conseils d'Utilisation

1. **Pour débuter** : Utilisez `--use-reference` pour tester rapidement avec les données de l'image
2. **Pour comparer** : Utilisez `--mode benchmark` pour comparer les différentes approches
3. **Pour la production** : Utilisez `--generator realistic` pour des scénarios réalistes
4. **Pour la recherche** : Utilisez `--seed` pour garantir la reproductibilité des expériences
5. **Pour la scalabilité** : Augmentez progressivement `--patients` et `--skills` pour tester les limites

## 🐛 Dépannage

- **Problème** : Le makespan ne diminue pas
  - **Solution** : Augmentez le nombre d'itérations avec `--iterations`
  
- **Problème** : Résultats trop lents
  - **Solution** : Utilisez `--quiet` pour désactiver l'affichage verbose
  
- **Problème** : Besoin de reproduire des résultats
  - **Solution** : Utilisez `--seed` avec une valeur fixe

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub ou contacter les auteurs.
