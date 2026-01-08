"""
Core package for the Multi-Agent Scheduling System with Q-Learning
"""

from core.environment import (
    SchedulingEnvironment,
    Solution,
    Task
)

# Intégration du nouveau module de génération de données
from core.data_generator import (
    generate_parametric_data,
    generate_balanced_data,
    generate_realistic_healthcare_data,
    get_reference_data,
    print_data_summary
)

from core.neighborhoods import NeighborhoodManager

from core.qlearning import QLearningModel

from core.shared_memory import SharedMemory

from core.agents import (
    MetaheuristicAgent,
    MultiAgentSystem
)

__all__ = [
    # Environment
    'SchedulingEnvironment',
    'Solution',
    'Task',
    
    # Data Generation
    'generate_parametric_data',
    'generate_balanced_data',
    'generate_realistic_healthcare_data',
    'get_reference_data',
    'print_data_summary',
    
    # Neighborhoods
    'NeighborhoodManager',
    
    # Q-Learning
    'QLearningModel',
    
    # Shared Memory
    'SharedMemory',
    
    # Agents
    'MetaheuristicAgent',
    'MultiAgentSystem'
]
