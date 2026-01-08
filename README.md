# Multi-Agent System with Q-Learning for Patient Scheduling

This project implements a collaborative multi-agent system for optimizing patient scheduling in healthcare environments. It integrates reinforcement learning (Q-Learning) for agent self-adaptation.

> 📦 **This project is part of the [Red Thread Project](https://github.com/edlansiaux/red-thread-project)** - A comprehensive research initiative on optimization algorithms and multi-agent systems.

## 🎯 Objective

**Minimize makespan**: Reduce the total time required to process all patients by optimizing resource allocation (medical skills/staff) and operation sequencing.

## 📋 Main Features

### Configurable Data Generation

The project offers several data generation methods to test different scenarios:

- **Reference data**: Based on the provided image (10 patients, 6 skills)
- **Parametric generator**: Full customization (number of patients, skills, operations, durations, etc.)
- **Balanced generator**: Automatic load balancing between resources
- **Realistic generator**: Realistic care pathways (consultation → examinations → treatment)

### Hybrid Metaheuristics

- **Genetic Algorithm (GA)**: With ordered crossover operators and adaptive mutation
- **Tabu Search**: With tabu list and aspiration criterion
- **Simulated Annealing (SA)**: With exponential cooling

### Collaboration Modes

- **Friends Mode (FRIENDS)**: Agents share complete solutions via Shared Memory Pool (SMP)
- **Enemies Mode (ENEMIES)**: Agents only share fitness values (competition)

### Self-Adaptation via Q-Learning

- Markov Decision Process (MDP)
- Q-Table for neighborhood selection
- Exploration/exploitation balance (ε-greedy)

### Shared Memory Pool (SMP)

- Diversity control based on distance between solutions
- Conditional insertion according to diversity threshold
- Worst solution replacement when necessary

### 5 Neighborhood Functions

- **A**: Reassignment to another medical staff
- **B**: Successive task reassignment
- **C**: Insertion within the same schedule (temporal shift)
- **D**: Exchange between different staff members
- **E**: Exchange within the same staff member

## 🏗️ Project Architecture

```
sma_qlearning_scheduling/
├── core/
│   ├── __init__.py              # Package exports
│   ├── environment.py           # Scheduling environment
│   ├── data_generator.py        # 🆕 Configurable data generators
│   ├── neighborhoods.py         # 5 neighborhood functions
│   ├── qlearning.py             # Q-Learning and MDP
│   ├── shared_memory.py         # SMP with diversity control
│   └── agents.py                # Agents and multi-agent system
├── visualization.py             # Visualizations (Gantt, convergence, etc.)
├── main.py                      # 🆕 Main script with CLI options
├── notebook_demo.ipynb          # Interactive Jupyter notebook
└── README.md                    # This file
```

## 🚀 Installation and Usage

### Prerequisites

```bash
pip install numpy matplotlib
```

### Quick Start

#### 1. Use reference data (image)

```bash
python main.py --use-reference
```

#### 2. Generate custom data

```bash
# 15 patients with 8 skills
python main.py --patients 15 --skills 8

# 20 patients with 6 skills, balanced generator
python main.py --patients 20 --skills 6 --generator balanced

# 12 patients with 5 skills, realistic medical context
python main.py --patients 12 --skills 5 --generator realistic
```

#### 3. Run a comparative benchmark

```bash
python main.py --patients 20 --skills 6 --mode benchmark
```

#### 4. Full mode (optimization + benchmark)

```bash
python main.py --patients 15 --skills 7 --mode both --iterations 100
```

### Complete Command Line Options

#### Data Generation

| Option | Description | Default |
|--------|-------------|---------|
| `--use-reference` | Use image data (10 patients, 6 skills) | False |
| `--patients N` | Number of patients | 10 |
| `--skills N` | Number of skills/resources | 6 |
| `--max-operations N` | Maximum operations per patient | 5 |
| `--generator TYPE` | Generator type (`parametric`, `balanced`, `realistic`) | `parametric` |
| `--seed N` | Seed for reproducibility | None |

#### Optimization

| Option | Description | Default |
|--------|-------------|---------|
| `--iterations N` | Number of iterations | 50 |
| `--collaboration MODE` | Collaboration mode (`FRIENDS`, `ENEMIES`) | `FRIENDS` |
| `--no-learning` | Disable Q-Learning | False |

#### Execution

| Option | Description | Default |
|--------|-------------|---------|
| `--mode MODE` | Execution mode (`optimize`, `benchmark`, `both`) | `optimize` |
| `--quiet` | Silent mode | False |

### Usage Examples

```bash
# Example 1: Quick test with reference data
python main.py --use-reference --iterations 30

# Example 2: Realistic hospital scenario with 25 patients
python main.py --patients 25 --skills 8 --generator realistic --iterations 100

# Example 3: Complete benchmark with balanced data
python main.py --patients 20 --skills 6 --generator balanced --mode benchmark

# Example 4: Scalability test
python main.py --patients 50 --skills 10 --iterations 150 --quiet

# Example 5: Collaboration mode comparison
python main.py --patients 15 --skills 7 --collaboration FRIENDS --mode benchmark
python main.py --patients 15 --skills 7 --collaboration ENEMIES --mode benchmark

# Example 6: Reproducibility with seed
python main.py --patients 20 --skills 6 --seed 42 --mode both
```

### Programmatic Usage

```python
from core import (
    generate_parametric_data,
    generate_balanced_data,
    get_reference_data,
    print_data_summary,
    SchedulingEnvironment,
    MultiAgentSystem
)

# Method 1: Use reference data
data, skills, num_patients = get_reference_data()

# Method 2: Generate custom data
data, skills = generate_parametric_data(
    num_patients=15,
    num_skills=8,
    max_operations=5,
    operation_probability=0.75,
    min_duration=15,
    max_duration=60,
    seed=42  # For reproducibility
)
num_patients = 15

# Method 3: Balanced data
data, skills = generate_balanced_data(
    num_patients=20,
    num_skills=6,
    max_operations=5,
    seed=42
)

# Display data summary
print_data_summary(data, skills)

# Create the environment
env = SchedulingEnvironment(data, skills, num_patients)

# Create the multi-agent system
mas = MultiAgentSystem(
    env, 
    agents_config=[
        {'id': 'GA_1', 'type': 'GA', 'learning': True},
        {'id': 'Tabu_1', 'type': 'Tabu', 'learning': True},
        {'id': 'SA_1', 'type': 'SA', 'learning': True}
    ],
    mode='FRIENDS'  # or 'ENEMIES'
)

# Run the optimization
best_makespan = mas.run(iterations=100)
print(f"Best makespan: {best_makespan} slots ({best_makespan * 5} minutes)")
```

### Using the Notebook

```bash
jupyter notebook notebook_demo.ipynb
```

## 📊 Main Parameters

### Q-Learning

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| α (alpha) | Learning rate | 0.1 |
| γ (gamma) | Discount factor | 0.9 |
| ε (epsilon) | Initial exploration rate | 0.1 |

### SMP (Shared Memory Pool)

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| max_size | Maximum size | 20 |
| R (min_distance) | Minimum distance between solutions | 2 |
| DT (diversity_threshold) | Diversity threshold | 0.5 |

### Data Generators

#### Parametric Generator

```python
generate_parametric_data(
    num_patients=10,           # Number of patients
    num_skills=6,              # Number of skills
    max_operations=5,          # Max operations per patient
    operation_probability=0.7, # Probability that an operation exists
    min_duration=10,           # Min duration (minutes)
    max_duration=60,           # Max duration (minutes)
    max_tasks_per_operation=3, # Max tasks per operation
    seed=None                  # Seed for reproducibility
)
```

#### Balanced Generator

Generates data where each skill is used in a balanced manner to test load balancing.

```python
generate_balanced_data(
    num_patients=10,
    num_skills=6,
    max_operations=5,
    seed=None
)
```

#### Realistic Generator

Generates realistic care pathways with logical sequences (consultation → examinations → treatment).

```python
generate_realistic_healthcare_data(
    num_patients=10,
    num_skills=6,
    seed=None
)
```

## 📈 Available Visualizations

- **Gantt Chart**: Task scheduling by skill
- **Convergence Curve**: Makespan evolution
- **Q-Table**: Values learned by Q-Learning
- **Diversity Matrix**: Distances between solutions in the SMP
- **Agent Contributions**: Improvements by agent

## 🎓 Reference Data (Image)

The reference data corresponds to the provided skill table example:

- **10 patients** (Patient 1 to Patient 10)
- **6 skills** (Skill 1 to Skill 6)
- **5 operations maximum** per patient
- Variable task distribution depending on the patient

This data can be used as a reference benchmark with `--use-reference`.

## 🔬 Based On

This project is based on the presentation:
> "Collaborative Optimization: Self-Adaptive Agents, Reinforcement Learning"

### Conceptual References

- Markov Decision Process (Bellman)
- Q-Learning (Watkins & Dayan, 1992)
- Multi-Agent Systems for optimization (Jin & Liu 2002, Milano & Roli 2004)
- Hybrid metaheuristics (Fernandes et al. 2009)

## 📄 License

This project is released under the MIT License. See the LICENSE file for more details.

## 🔗 Related Projects

This repository is derived from the main project:
- **[Red Thread Project](https://github.com/edlansiaux/red-thread-project)** - The original repository containing the complete research work and additional resources.

## 👥 Authors & Acknowledgments

- [Mohammed Berrajaa](https://github.com/medberrajaa)
- [Guillaume Gauguet](https://github.com/GAUGUET)
- [Hugo Kazzi](https://github.com/hugokazzi63)
- [Abdallah Lafendi](https://github.com/imadlaf2503)
- [Edouard Lansiaux](https://github.com/edlansiaux)
- [Aurélien Loison](https://github.com/lsnaurelien)

Thanks to all contributors who help improve this project!

## 🚦 Scalability Tests

The project allows easy testing of the approach's scalability:

```bash
# Test with 10 patients (small problem)
python main.py --patients 10 --skills 5 --mode benchmark

# Test with 30 patients (medium problem)
python main.py --patients 30 --skills 8 --mode benchmark

# Test with 50 patients (large problem)
python main.py --patients 50 --skills 10 --mode benchmark --iterations 200
```

## 💡 Usage Tips

1. **To get started**: Use `--use-reference` to quickly test with the image data
2. **To compare**: Use `--mode benchmark` to compare different approaches
3. **For production**: Use `--generator realistic` for realistic scenarios
4. **For research**: Use `--seed` to ensure experiment reproducibility
5. **For scalability**: Gradually increase `--patients` and `--skills` to test limits

## 🐛 Troubleshooting

- **Problem**: Makespan doesn't decrease
  - **Solution**: Increase the number of iterations with `--iterations`
  
- **Problem**: Results too slow
  - **Solution**: Use `--quiet` to disable verbose output
  
- **Problem**: Need to reproduce results
  - **Solution**: Use `--seed` with a fixed value

## 📞 Support

For any questions or issues, please open an issue on GitHub or contact the authors.
