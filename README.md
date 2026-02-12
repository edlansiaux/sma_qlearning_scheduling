# Ordonnancement de Patients : Multi-Agents, (Amis/Ennemis) et Q-Learning

Système multi-agents (SMA) pour l'optimisation d'ordonnancement hospitalier, intégrant l'apprentissage par renforcement (Q-Learning) pour l'auto-adaptation des agents.

## 🎯 Objectif

**Minimiser le Makespan (Cmax)** : réduire le temps total de prise en charge de tous les patients en optimisant l'allocation des ressources (compétences médicales) et le séquencement des opérations.

---

## 📁 Architecture du projet

```
sma_qlearning_scheduling/
│
├── core/                          # Moteur d'optimisation
│   ├── __init__.py                # Exports centralisés
│   ├── environment.py             # Environnement d'ordonnancement + évaluation Cmax
│   ├── data_generator.py          # Générateurs de données (paramétrique, équilibré, réaliste)
│   ├── neighborhoods.py           # 5 fonctions de voisinage (A–E)
│   ├── qlearning.py               # Q-Learning + MDP + sélection adaptative
│   ├── shared_memory.py           # Espace Mémoire Partagé (EMP) + ElitePool
│   ├── agents.py                  # Agents (AG, Tabou, RS) + Système Multi-Agents
│   └── benchmark_tables.py        # Génération des Tableaux 1 & 2 du diaporama
│
├── visualization.py               # Gantt, convergence, Q-Table, diversité
├── main.py                        # Point d'entrée CLI
├── notebook_demo.ipynb            # Notebook Jupyter de démonstration complète
├── requirements.txt               # Dépendances Python
├── benchmark_results/             # Dossier de sortie (graphiques générés)
└── README.md
```

---

## 🔬 Méthodes implémentées

### 3 Métaheuristiques

| Agent | Stratégie | Paramètres clés |
|-------|-----------|-----------------|
| **AG** (Algorithme Génétique) | Crossover OX + mutation par voisinage | `population_size`, `mutation_rate` |
| **Tabou** (Recherche Tabou) | Liste tabou + critère d'aspiration | `tabu_tenure`, `candidate_limit` |
| **RS** (Recuit Simulé) | Refroidissement exponentiel | `initial_temp`, `cooling_rate` |

### 5 Fonctions de voisinage (Algorithme 3)

| Voisinage | Description |
|-----------|-------------|
| **A** | Réaffectation de position (retrait + réinsertion aléatoire) |
| **B** | Déplacement groupé (bloc de tâches du même patient) |
| **C** | Décalage local (shift ∈ [−6, 6] positions) |
| **D** | Échange inter-files (entre deux stages de la même compétence) |
| **E** | Swap intra-file (échange de deux tâches dans la même file) |

### Q-Learning (Algorithme 2)

Sélection adaptative du voisinage via un processus de décision de Markov :
- **États** : {A, B, C, D, E}
- **Actions** : choix du voisinage
- **Récompense** : amélioration du makespan
- **Paramètres** : α=0.1, γ=0.9, ε=0.9 (décroissance ε-greedy)

### 2 Modes de collaboration (Algorithme 4)

| Mode | Comportement |
|------|-------------|
| **AMIS (FRIENDS)** | Partage de solutions complètes via l'EMP, adoption probabiliste (p=0.2) |
| **ENNEMIS (ENEMIES)** | Partage de fitness uniquement, adoption forcée si meilleure |

---

## 🚀 Installation et utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

### Ligne de commande

```bash
# Démonstration rapide avec données de référence (10 patients)
python main.py --use-reference

# 20 patients, générateur paramétrique, mode AMIS + Q-Learning
python main.py --patients 20 --skills 6 --iterations 100

# Générateur réaliste (parcours de soins)
python main.py --patients 25 --skills 8 --generator realistic

# Benchmark complet (Tableaux 1 & 2)
python main.py --mode benchmark

# Mode ENNEMIS sans Q-Learning
python main.py --patients 15 --collaboration ENEMIES --no-learning

# Reproductibilité
python main.py --patients 20 --seed 42 --mode both
```

### Options CLI complètes

| Option | Description | Défaut |
|--------|-------------|--------|
| `--use-reference` | Données de référence (10 patients, 6 skills) | `False` |
| `--patients N` | Nombre de patients | `10` |
| `--skills N` | Nombre de compétences | `6` |
| `--max-operations N` | Opérations max par patient | `5` |
| `--generator TYPE` | `parametric` / `balanced` / `realistic` | `parametric` |
| `--seed N` | Graine aléatoire | `None` |
| `--iterations N` | Itérations d'optimisation | `50` |
| `--collaboration MODE` | `FRIENDS` / `ENEMIES` | `FRIENDS` |
| `--no-learning` | Désactiver le Q-Learning | `False` |
| `--mode MODE` | `optimize` / `benchmark` / `both` | `optimize` |
| `--quiet` | Mode silencieux | `False` |

### Notebook Jupyter

```bash
jupyter notebook notebook_demo.ipynb
```

Le notebook reproduit intégralement les tableaux du diaporama et génère toutes les visualisations.

### Utilisation programmatique

```python
from core import *

# Données de référence
env = create_default_environment()

# OU données personnalisées
data, skills = generate_parametric_data(num_patients=20, num_skills=8, seed=42)
env = SchedulingEnvironment(data, skills, 20, 5)

# Système Multi-Agents
mas = MultiAgentSystem(env, mode=CollaborationMode.FRIENDS, use_qlearning=True)
mas.add_agent('genetic', 'AG', population_size=15)
mas.add_agent('tabu', 'Tabu', tabu_tenure=10)
mas.add_agent('sa', 'RS', initial_temp=100)

best = mas.run(n_iterations=100)
print(f"Meilleur Cmax : {best.fitness}")
```

---

## 📊 Tableaux de résultats (diaporama)

### Tableau 1 — Sans collaboration

Comparaison des métaheuristiques seules (sans/avec Q-Learning) sur 4 jours avec un nombre de patients variable.

| Jour | Patients | AG | Tabou | RS | Agent_AG | Agent_Tabou | Agent_RS |
|------|----------|----:|------:|----:|---------:|------------:|---------:|

### Tableau 2 — Avec collaboration

Pour chaque combinaison d'agents (AG_Tabou, AG_RS, Tabou_RS, AG_Tabou_RS), comparaison Amis vs Ennemis, sans et avec apprentissage.

---

## 📈 Visualisations disponibles

- **Diagramme de Gantt** : planning par compétence avec code couleur patient
- **Courbes de convergence** : évolution du Cmax par agent
- **Heatmap Q-Table** : valeurs Q apprises (5×5 pour les voisinages A–E)
- **Utilisation des voisinages** : histogramme appels / améliorations
- **Matrice de diversité** : distances de Hamming entre solutions de l'EMP
- **Contributions des agents** : meilleur fitness par agent

---

## 🔗 Références

- Q-Learning (Watkins & Dayan, 1992)
- Systèmes Multi-Agents pour l'optimisation (Jin & Liu, 2002 ; Milano & Roli, 2004)
- Métaheuristiques hybrides (Fernandes et al., 2009)

## 👥 Auteurs

- Mohammed Berrajaa
- Guillaume Gauguet
- Hugo Kazzi
- Abdallah Lafendi
- Edouard Lansiaux
- Aurélien Loison

## 📄 Licence

MIT License
