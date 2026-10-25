"""Linked segment swapping and arithmetic/geometric waypoint recombination."""
import numpy as np


def crossover(a,b,env,spec,planner,rng,method='segment'):
    child = a.copy()
    for target, other in zip(child.plans,b.plans):
        if method == 'segment':
            cut = int(rng.integers(spec.waypoints))
            for name in ('waypoints','speeds','sensors','sensor_power'):
                getattr(target,name)[cut:] = list(getattr(other,name)[cut:])
            if rng.random() < .5:
                target.start = other.start
        elif method == 'geometric':
            alpha = float(rng.uniform(.2,.8))
            target.start = tuple(np.rint(alpha*np.array(target.start)+(1-alpha)*np.array(other.start)).astype(int))
            for j in range(spec.waypoints):
                target.waypoints[j] = tuple(np.rint(alpha*np.array(target.waypoints[j])+
                                                    (1-alpha)*np.array(other.waypoints[j])).astype(int))
                target.speeds[j] = alpha*target.speeds[j]+(1-alpha)*other.speeds[j]
                if rng.random() < .5:
                    target.sensors[j] = other.sensors[j]
                    target.sensor_power[j] = other.sensor_power[j]
        else:
            raise ValueError(f'Unknown crossover method: {method}')
    return child.repair(env,spec,planner)
