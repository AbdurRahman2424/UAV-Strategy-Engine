from dataclasses import replace
import numpy as np
import pytest
from src.chromosome import Chromosome
from src.drone import UAVPlan
from src.fitness import FitnessEvaluator
from src.pathfinding import AStarPlanner


def test_astar_avoids_no_fly_and_obstacles(setup):
    env,spec,planner,evaluator = setup
    rng = np.random.default_rng(19)
    for _ in range(8):
        result = evaluator.evaluate(Chromosome.random(env,spec,planner,rng))
        assert result.feasible
        for path in result.paths:
            assert all(env.traversable[y,x] and not env.no_fly[y,x] for x,y in path)
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))


def test_range_and_battery_repair(setup):
    env,spec,planner,_ = setup
    tight = replace(spec,max_range=8,battery_capacity=2.5)
    evaluator = FitnessEvaluator(env,tight,planner)
    c = Chromosome.random(env,spec,planner,np.random.default_rng(2))
    assert not evaluator.evaluate(c).feasible
    c.repair(env,tight,planner)
    result = evaluator.evaluate(c)
    assert result.feasible
    for p in result.uav_metrics:
        assert p['distance'] <= tight.max_range+1e-9
        assert p['energy'] <= tight.battery_capacity+1e-9


def test_invalid_deployment_and_speed_rejected_then_repaired(setup):
    env,spec,planner,evaluator = setup
    c = Chromosome.random(env,spec,planner,np.random.default_rng(8))
    c.plans[0].start = (-5,999)
    c.plans[0].speeds[0] = float('nan')
    c.plans[0].sensor_power[0] = 99
    assert not evaluator.evaluate(c).feasible
    c.repair(env,spec,planner)
    assert evaluator.evaluate(c).feasible


def test_unreachable_route_is_never_valid(open_setup):
    env,spec,_,_ = open_setup
    env.traversable[:,6] = False
    env.no_fly[:,6] = True
    env.rebuild_navigation()
    planner = AStarPlanner(env)
    assert planner.route((2,2),(9,2)) is None
    c = Chromosome([UAVPlan(0,(2,2),[(9,2),(9,8)],[2.,2.],[1,1],[1.,1.])])
    evaluator = FitnessEvaluator(env,spec,planner)
    bad = evaluator.evaluate(c)
    assert not bad.feasible and bad.fitness < -1
    assert 'unreachable' in bad.violations[0]
    c.repair(env,spec,planner)
    good = evaluator.evaluate(c)
    assert good.feasible
    assert all(x<6 for path in good.paths for x,y in path)


def test_astar_hazard_detour(open_setup):
    env,spec,_,_ = open_setup
    env.hazard[:] = 0
    env.hazard[5,2:10] = 1
    safe = AStarPlanner(env,hazard_weight=5).route((1,5),(10,5))
    direct = AStarPlanner(env,hazard_weight=0).route((1,5),(10,5))
    assert len(safe) > len(direct)
    assert sum(env.hazard[y,x] for x,y in safe) < sum(env.hazard[y,x] for x,y in direct)


def test_blocked_environment_rejected(open_setup):
    env = open_setup[0]
    env.traversable[:] = False
    with pytest.raises(ValueError,match='no traversable'):
        env.rebuild_navigation()
