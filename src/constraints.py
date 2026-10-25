"""Hard feasibility repair. Never substitute straight lines for missing A* paths."""
import numpy as np
from .drone import energy_per_cell


def repair_chromosome(chromosome, env, spec, planner):
    if len(chromosome.plans) != spec.count:
        raise ValueError('Cannot repair a chromosome with the wrong fleet size.')
    for uid, plan in enumerate(chromosome.plans):
        plan.uid = uid
        k = spec.waypoints
        if any(len(a) != k for a in (plan.waypoints,plan.speeds,plan.sensors,plan.sensor_power)):
            raise ValueError('Cannot repair inconsistent gene-array lengths.')
        plan.start = env.nearest_valid(plan.start)
        component = env.components[plan.start[1],plan.start[0]]
        previous = plan.start
        distance = energy = 0.
        for j in range(k):
            v = float(np.nan_to_num(plan.speeds[j], nan=spec.min_speed,
                                    posinf=spec.max_speed, neginf=spec.min_speed))
            plan.speeds[j] = v = float(np.clip(v,spec.min_speed,spec.max_speed))
            plan.sensors[j] = int(bool(plan.sensors[j]))
            raw_power = float(np.nan_to_num(plan.sensor_power[j], nan=1.0))
            plan.sensor_power[j] = power = min(spec.power_levels, key=lambda p:abs(p-raw_power))
            goal = env.nearest_valid(plan.waypoints[j], component)
            path = planner.route(previous,goal)
            if path is None:
                raise RuntimeError('Component-aware repair unexpectedly found no route.')
            unit_energy = energy_per_cell(v,plan.sensors[j],power,spec)
            allowance = min(len(path)-1, int(np.floor(spec.max_range-distance+1e-9)),
                            int(np.floor((spec.battery_capacity-energy)/unit_energy+1e-9)))
            allowance = max(0,allowance)
            # A new hazard-aware A* route to a truncated point can differ from a prefix.
            # Replan and shorten until the actual encoded segment meets both bounds.
            while True:
                goal = path[allowance]
                actual = planner.route(previous,goal)
                steps = len(actual)-1
                if distance+steps <= spec.max_range+1e-9 and energy+steps*unit_energy <= spec.battery_capacity+1e-9:
                    break
                allowance -= 1
            plan.waypoints[j] = goal
            distance += steps
            energy += steps*unit_energy
            previous = goal
