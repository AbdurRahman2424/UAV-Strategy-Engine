from dataclasses import replace
import numpy as np
import pytest
from src.chromosome import Chromosome
from src.drone import UAVPlan,energy_per_cell
from src.fitness import FitnessEvaluator


def mission(sensors=(1,1),power=(1.,1.),speeds=(1.5,1.5),waypoints=((7,3),(7,7))):
    return Chromosome([UAVPlan(0,(3,3),list(waypoints),list(speeds),list(sensors),list(power))])


def test_fitness_matches_formula(open_setup):
    env,spec,_,evaluator = open_setup
    value = evaluator.evaluate(mission())
    assert value.feasible
    assert value.fitness == pytest.approx(.35*value.coverage+.25*value.detection
                                         -.15*value.energy/spec.battery_capacity
                                         -.10*value.mission_time/(spec.max_range/spec.min_speed)
                                         -.15*value.risk/(spec.max_range/spec.min_speed))
    assert value.energy == pytest.approx(8*energy_per_cell(1.5,1,1,spec))
    assert value.risk == pytest.approx(8*.5/1.5)
    assert all(0<=v<=1 for v in [value.coverage,value.detection,value.normalized_energy,
                                value.normalized_time,value.normalized_risk])


def test_sensor_activation_changes_coverage_detection_energy(open_setup):
    evaluator = open_setup[3]
    on = evaluator.evaluate(mission())
    off = evaluator.evaluate(mission(sensors=(0,0)))
    assert on.coverage > off.coverage == 0
    assert on.detection > off.detection == 0
    assert on.energy > off.energy


def test_sensor_power_and_speed_influence_physics(open_setup):
    evaluator = open_setup[3]
    low = evaluator.evaluate(mission(power=(.5,.5)))
    high = evaluator.evaluate(mission(power=(1.5,1.5)))
    assert high.detection > low.detection
    assert high.coverage > low.coverage
    assert high.energy > low.energy
    slow = evaluator.evaluate(mission(speeds=(1.,1.)))
    fast = evaluator.evaluate(mission(speeds=(3.,3.)))
    assert fast.mission_time < slow.mission_time
    assert fast.energy > slow.energy
    assert fast.detection < slow.detection


def test_repeated_observations_do_not_create_unique_coverage(open_setup):
    evaluator = open_setup[3]
    once = mission(waypoints=((7,3),(7,3)))
    # Duplicate the same path with a second UAV to isolate repeated sensing.
    env,spec,planner,_ = open_setup
    double_spec = replace(spec,count=2)
    double = once.copy()
    extra = once.plans[0]
    double.plans.append(type(extra)(1,extra.start,list(extra.waypoints),list(extra.speeds),
                                   list(extra.sensors),list(extra.sensor_power)))
    a = evaluator.evaluate(once)
    b = FitnessEvaluator(env,double_spec,planner).evaluate(double)
    assert b.coverage == pytest.approx(a.coverage)
    assert b.detection > a.detection
    assert b.mission_time == pytest.approx(a.mission_time)
    assert b.energy == pytest.approx(2*a.energy)


def test_priority_weighted_coverage(open_setup):
    env,spec,planner,evaluator = open_setup
    c = mission()
    value = evaluator.evaluate(c)
    covered = value.observation_counts>0
    env.priority[covered] = 5
    altered = FitnessEvaluator(env,spec,planner).evaluate(c)
    expected = env.priority[covered].sum()/env.priority[env.traversable].sum()
    assert altered.coverage == pytest.approx(expected)
    assert altered.coverage > value.coverage


def test_stationary_routes_do_not_sense_for_free(open_setup):
    value = open_setup[3].evaluate(mission(waypoints=((3,3),(3,3))))
    assert value.feasible
    assert value.coverage == value.detection == value.energy == value.mission_time == 0
