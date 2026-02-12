"""
Core package for the Multi-Agent Scheduling System with Q-Learning
"""

from core.environment import (
    SchedulingEnvironment,
    Task,
    create_default_environment,
    generate_random_data,
    DEFAULT_DATA,
    DEFAULT_SKILLS,
    DEFAULT_NUM_PATIENTS,
    DEFAULT_MAX_OPS,
)

from core.data_generator import (
    get_reference_data,
    generate_parametric_data,
    generate_balanced_data,
    generate_realistic_healthcare_data,
    print_data_summary,
)

from core.neighborhoods import (
    NeighborhoodFunction,
    NeighborhoodA,
    NeighborhoodB,
    NeighborhoodC,
    NeighborhoodD,
    NeighborhoodE,
    NeighborhoodManager,
)

from core.qlearning import (
    QLearningAgent,
    AdaptiveNeighborhoodSelector,
    MarkovDecisionProcess,
)

from core.shared_memory import (
    Solution,
    SharedMemoryPool,
    ElitePool,
)

from core.agents import (
    BaseAgent,
    GeneticAgent,
    TabuAgent,
    SimulatedAnnealingAgent,
    CollaborationMode,
    MultiAgentSystem,
)

__all__ = [
    # Environment
    "SchedulingEnvironment",
    "Task",
    "create_default_environment",
    "generate_random_data",
    "DEFAULT_DATA",
    "DEFAULT_SKILLS",
    "DEFAULT_NUM_PATIENTS",
    "DEFAULT_MAX_OPS",
    # Data Generator
    "get_reference_data",
    "generate_parametric_data",
    "generate_balanced_data",
    "generate_realistic_healthcare_data",
    "print_data_summary",
    # Neighborhoods
    "NeighborhoodFunction",
    "NeighborhoodA",
    "NeighborhoodB",
    "NeighborhoodC",
    "NeighborhoodD",
    "NeighborhoodE",
    "NeighborhoodManager",
    # Q-Learning
    "QLearningAgent",
    "AdaptiveNeighborhoodSelector",
    "MarkovDecisionProcess",
    # Shared Memory
    "Solution",
    "SharedMemoryPool",
    "ElitePool",
    # Agents
    "BaseAgent",
    "GeneticAgent",
    "TabuAgent",
    "SimulatedAnnealingAgent",
    "CollaborationMode",
    "MultiAgentSystem",
]
