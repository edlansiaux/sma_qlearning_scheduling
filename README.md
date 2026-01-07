# Multi-Agent System with Q-Learning for Patient Scheduling

This project implements a collaborative multi-agent system for optimizing patient scheduling in a healthcare environment. It integrates reinforcement learning (Q-Learning) for agent self-adaptation.

## 📋 Main Features

### Hybrid Metaheuristics
- **Genetic Algorithm (GA)**: With ordered crossover operators and adaptive mutation
- **Tabu Search**: With tabu list and aspiration criterion
- **Simulated Annealing (SA)**: With exponential cooling

### Collaboration Modes
- **Friends Mode**: Agents share complete solutions via the Shared Memory Pool (SMP)
- **Enemies Mode**: Agents only share fitness values (competition)

### Self-Adaptation via Q-Learning
- Markov Decision Process (MDP)
- Q-Table for neighborhood selection
- Exploration/exploitation balance (ε-greedy)

### Shared Memory Pool (SMP)
- Diversity control based on distance between solutions
- Conditional insertion according to diversity threshold
- Replacement of worst solution when necessary

### 5 Neighborhood Functions
- **A**: Task reassignment to another staff member (MID)
- **B**: Successive tasks reassignment
- **C**: Insertion within the same schedule (MIS)
- **D**: Swap between different staff members (SDMS)
- **E**: Swap within the same schedule (SSMS)

## 🏗️ Project Architecture

```
sma_qlearning_scheduling/
├── core/
│   ├── __init__.py          # Package exports
│   ├── environment.py       # Scheduling environment
│   ├── neighborhoods.py     # 5 neighborhood functions
│   ├── qlearning.py         # Q-Learning and MDP
│   ├── shared_memory.py     # SMP with diversity control
│   └── agents.py            # Agents and multi-agent system
├── visualization.py         # Visualizations (Gantt, convergence, etc.)
├── main.py                  # Main demonstration script
├── notebook_demo.ipynb      # Interactive Jupyter notebook
└── README.md                # This file
```

## 🚀 Installation and Usage

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Main Script
```bash
python main.py
```

### Using the Notebook
```bash
jupyter notebook notebook_demo.ipynb
```

### Programmatic Usage

```python
from core import (
    create_default_environment,
    MultiAgentSystem,
    CollaborationMode
)

# Create the environment
env = create_default_environment()

# Create the multi-agent system
mas = MultiAgentSystem(
    env, 
    mode=CollaborationMode.FRIENDS,  # or ENEMIES
    use_qlearning=True
)

# Add agents
mas.add_agent('genetic', 'GA_1', population_size=15)
mas.add_agent('tabu', 'Tabu_1', tabu_tenure=10)
mas.add_agent('sa', 'SA_1', initial_temp=100)

# Run optimization
best_solution = mas.run(n_iterations=200, verbose=True)

# Get statistics
stats = mas.get_statistics()
print(f"Best Makespan: {stats['global_best_fitness']}")
```

## 📊 Main Parameters

### Q-Learning
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| α (alpha) | Learning rate | 0.15 |
| γ (gamma) | Discount factor | 0.9 |
| ε (epsilon) | Initial exploration rate | 0.3-0.5 |
| ε_decay | Epsilon decay | 0.995 |

### SMP (Shared Memory Pool)
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| max_size | Maximum size | 20-30 |
| R (min_distance) | Minimum distance between solutions | 5 |
| DT (diversity_threshold) | Diversity threshold | 0.3-0.5 |

### Agents
| Agent | Key Parameters |
|-------|----------------|
| Genetic | population_size, mutation_rate |
| Tabu | tabu_tenure, candidate_limit |
| Simulated Annealing | initial_temp, cooling_rate |

## 📈 Available Visualizations

- **Gantt Chart**: Task schedule by skill
- **Convergence Curve**: Makespan evolution
- **Q-Table**: Learned Q-Learning values
- **Diversity Matrix**: Distances between SMP solutions
- **Agent Contributions**: Improvements by agent

## 🔬 Based On

This project is based on the presentation:
> "Collaborative Optimization: Self-Adaptive Agents, Reinforcement Learning"

### Conceptual References
- Markov Decision Process (Bellman)
- Q-Learning (Watkins & Dayan, 1992)
- Multi-Agent Systems for optimization (Jin & Liu 2002, Milano & Roli 2004)
- Hybrid metaheuristics (Fernandes et al. 2009)

Please follow good practices: clean code, tests, documentation, and consistent architecture.

## License

This project is released under the MIT License. See the [LICENSE](#license) file for details.

## Authors & Acknowledgments

- [Mohammed Berrajaa](https://github.com/medberrajaa) (main author)
- [Guillaume Gauguet](https://github.com/GAUGUET) (main author)
- [Hugo Kazzi]() (main author)
- [Abdallah Lafendi]() (main author)
- [Edouard Lansiaux](https://github/edlansiaux) (main author)
- [Aurélien Loison](https://github.com/lsnaurelien) (main author)

Contributors via pull requests, reviews, or ideas

Special thanks to all contributors who help improve this project!
