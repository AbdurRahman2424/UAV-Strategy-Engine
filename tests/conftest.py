import pytest
from config import EnvironmentConfig,UAVConfig
from src.environment import Environment
from src.pathfinding import AStarPlanner
from src.fitness import FitnessEvaluator


@pytest.fixture
def setup():
    env = Environment(EnvironmentConfig(size=20,seed=91,target_count=10))
    spec = UAVConfig(count=3,waypoints=3,max_range=45)
    planner = AStarPlanner(env)
    return env,spec,planner,FitnessEvaluator(env,spec,planner)


@pytest.fixture
def open_setup():
    """A transparent analytic scenario for independent physical-model assertions."""
    env = Environment(EnvironmentConfig(size=12,seed=4,target_count=5))
    env.traversable[:] = True
    env.obstacles[:] = env.no_fly[:] = False
    env.hazard[:] = .5
    env.priority[:] = 1
    env.target_probability[:] = 1/(env.size**2)
    env.rebuild_navigation()
    spec = UAVConfig(count=1,waypoints=2,max_range=30)
    planner = AStarPlanner(env)
    return env,spec,planner,FitnessEvaluator(env,spec,planner)
