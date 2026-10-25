"""Spatial perturbations and heterogeneous gene mutations."""
import numpy as np


def mutate(chromosome, env, spec, planner, rng, probability=.15):
    child = chromosome.copy()
    for p in child.plans:
        if rng.random() < probability:
            p.start = tuple(np.rint(rng.normal(p.start,3)).astype(int))
        for j in range(spec.waypoints):
            if rng.random() < probability:
                p.waypoints[j] = tuple(np.rint(rng.normal(p.waypoints[j],4)).astype(int))
            if rng.random() < probability*.15:
                p.waypoints[j] = tuple(map(int,env.valid_cells[rng.integers(len(env.valid_cells))]))
            if rng.random() < probability:
                p.speeds[j] += float(rng.normal(0,.35))
            if rng.random() < probability:
                p.sensors[j] = 1-p.sensors[j]
            if rng.random() < probability:
                p.sensor_power[j] = float(rng.choice(spec.power_levels))
    return child.repair(env,spec,planner)
