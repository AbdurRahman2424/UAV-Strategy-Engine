from dataclasses import replace
import numpy as np
import pytest
from config import EnvironmentConfig,GAConfig
from src.environment import Environment
from src.fitness import FitnessEvaluator
from src.genetic_algorithm import GeneticAlgorithm
from src.pathfinding import AStarPlanner
from src.random_search import RandomSearch


def test_ga_execution_elitism_and_budget(setup):
    env,spec,planner,evaluator = setup
    config = GAConfig(population=10,generations=4,elites=2,seed=7)
    result = GeneticAlgorithm(env,spec,evaluator,config).run()
    assert result.evaluation.feasible
    assert result.evaluations == evaluator.evaluations == 10+4*8
    assert result.valid_solutions == result.evaluations
    best = [h['best_fitness'] for h in result.history]
    assert np.all(np.diff(best)>=0)
    assert all(h['average_fitness']<=h['best_fitness']+1e-12 for h in result.history)


def test_reproducibility_and_random_search(setup):
    env,spec,_,_ = setup
    config = GAConfig(population=8,generations=3,seed=6)
    def ga_run():
        evaluator = FitnessEvaluator(env,spec,AStarPlanner(env))
        return GeneticAlgorithm(env,spec,evaluator,config).run()
    a,b = ga_run(),ga_run()
    assert a.best.to_dict() == b.best.to_dict()
    assert a.history == b.history
    baseline_evaluator = FitnessEvaluator(env,spec,AStarPlanner(env))
    random = RandomSearch(env,spec,baseline_evaluator,seed=10006).run(a.evaluations)
    assert random.evaluation.feasible
    assert random.evaluations == baseline_evaluator.evaluations == a.evaluations
    assert random.valid_solutions == random.evaluations
    other = RandomSearch(env,spec,FitnessEvaluator(env,spec,AStarPlanner(env)),seed=10006).run(a.evaluations)
    assert random.best.to_dict() == other.best.to_dict()
    assert np.all(np.diff([h['best_fitness'] for h in random.history])>=0)


def test_partial_generation_equal_budget(setup):
    env,spec,_,evaluator = setup
    c = GAConfig(population=10,generations=100,seed=8)
    result = GeneticAlgorithm(env,spec,evaluator,c).run(evaluation_budget=23)
    assert result.evaluations == evaluator.evaluations == 23
    assert result.evaluation.feasible
    assert result.history[-1]['evaluations']==23
    assert all(h['population_size']==c.population for h in result.history)


def test_environment_reproducibility():
    a,b = Environment(EnvironmentConfig(seed=2026)),Environment(EnvironmentConfig(seed=2026))
    for field in ('traversable','priority','hazard','targets','target_probability'):
        assert np.array_equal(getattr(a,field),getattr(b,field))


def test_early_stopping(setup):
    env,spec,_,evaluator = setup
    c = GAConfig(population=6,generations=20,mutation=0,crossover=0,immigrant_fraction=0,elites=2,patience=2)
    result = GeneticAlgorithm(env,spec,evaluator,c).run()
    assert result.history[-1]['generation'] == 2


@pytest.mark.parametrize('kwargs',[{'population':2},{'mutation':1.2},{'elites':60},{'tournament':70}])
def test_configuration_errors(kwargs):
    with pytest.raises(ValueError):
        replace(GAConfig(),**kwargs)
