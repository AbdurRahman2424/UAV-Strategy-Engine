import numpy as np
import pytest
from src.chromosome import Chromosome


def test_random_valid_initialization_and_bounds(setup):
    env,spec,planner,evaluator = setup
    for seed in range(6):
        c = Chromosome.random(env,spec,planner,np.random.default_rng(seed))
        assert not c.validate(env,spec)
        assert evaluator.evaluate(c).feasible
        assert all(0<=x<env.size and 0<=y<env.size for p in c.plans for x,y in [p.start,*p.waypoints])


def test_deep_copy(setup):
    env,spec,planner,_ = setup
    a = Chromosome.random(env,spec,planner,np.random.default_rng(3))
    b = a.copy()
    b.plans[0].speeds[0] = 2.9
    assert a.plans[0].speeds[0] != b.plans[0].speeds[0]


@pytest.mark.parametrize('method',['segment','geometric'])
def test_crossover_correctness(setup,method):
    env,spec,planner,evaluator = setup
    rng = np.random.default_rng(4)
    a,b = [Chromosome.random(env,spec,planner,rng) for _ in range(2)]
    parents = a.to_dict(),b.to_dict()
    for _ in range(8):
        child = a.crossover(b,env,spec,planner,rng,method)
        assert not child.validate(env,spec)
        assert evaluator.evaluate(child).feasible
    assert (a.to_dict(),b.to_dict()) == parents


def test_mutation_correctness(setup):
    env,spec,planner,evaluator = setup
    rng = np.random.default_rng(5)
    parent = Chromosome.random(env,spec,planner,rng)
    saved = parent.to_dict()
    child = parent.mutate(env,spec,planner,rng,1.0)
    assert child.to_dict() != saved
    assert parent.to_dict() == saved
    assert not child.validate(env,spec)
    assert evaluator.evaluate(child).feasible


def test_structural_corruption_rejected(setup):
    env,spec,planner,evaluator = setup
    c = Chromosome.random(env,spec,planner,np.random.default_rng(2))
    c.plans[0].speeds.pop()
    assert not evaluator.evaluate(c).feasible
    with pytest.raises(ValueError,match='gene-array'):
        c.repair(env,spec,planner)
