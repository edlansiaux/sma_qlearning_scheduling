"""
Core package for the Multi-Agent Scheduling System with Q-Learning
"""

from core.environment import (
    SchedulingEnvironment,
    Task,
    create_default_environment,
    DEFAULT_DATA,
    DEFAULT_SKILLS,
    DEFAULT_NUM_PATIENTS,
    DEFAULT_MAX_OPS
)

from core.neighborhoods import (
    NeighborhoodFunction,
    NeighborhoodA,
    NeighborhoodB,
    NeighborhoodC,
    NeighborhoodD,
    NeighborhoodE,
    NeighborhoodManager
)

# Import corrigé : MarkovDecisionProcess est maintenant disponible
from core.qlearning import (
    QLearningAgent,
    AdaptiveNeighborhoodSelector,
    MarkovDecisionProcess
)

# Import corrigé : ElitePool est maintenant disponible
from core.shared_memory import (
    Solution,
    SharedMemoryPool,
    ElitePool
)

from core.agents import (
    BaseAgent,
    GeneticAgent,
    TabuAgent,
    SimulatedAnnealingAgent,
    CollaborationMode,
    MultiAgentSystem
)

__all__ = [
    # Environment
    'SchedulingEnvironment',
    'Task',
    'create_default_environment',
    'DEFAULT_DATA',
    'DEFAULT_SKILLS',
    'DEFAULT_NUM_PATIENTS',
    'DEFAULT_MAX_OPS',
    
    # Neighborhoods
    'NeighborhoodFunction',
    'NeighborhoodA',
    'NeighborhoodB',
    'NeighborhoodC',
    'NeighborhoodD',
    'NeighborhoodE',
    'NeighborhoodManager',
    
    # Q-Learning
    'QLearningAgent',
    'AdaptiveNeighborhoodSelector',
    'MarkovDecisionProcess',
    
    # Shared Memory
    'Solution',
    'SharedMemoryPool',
    'ElitePool',
    
    # Agents
    'BaseAgent',
    'GeneticAgent',
    'TabuAgent',
    'SimulatedAnnealingAgent',
    'CollaborationMode',
    'MultiAgentSystem',
]
