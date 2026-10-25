"""Shared normalized fitness and independent physical feasibility checks."""
from dataclasses import dataclass, field
import numpy as np
from config import FitnessWeights
from .drone import energy_per_cell


@dataclass
class Evaluation:
    fitness: float
    coverage: float = 0.
    detection: float = 0.
    energy: float = 0.
    mission_time: float = 0.
    risk: float = 0.
    normalized_energy: float = 0.
    normalized_time: float = 0.
    normalized_risk: float = 0.
    expected_detections: float = 0.
    feasible: bool = True
    violations: list = field(default_factory=list)
    uav_metrics: list = field(default_factory=list)
    paths: list = field(default_factory=list)
    observation_counts: object = None
    detection_map: object = None

    def summary(self):
        return {key:getattr(self,key) for key in ('fitness','coverage','detection','energy',
                'mission_time','risk','normalized_energy','normalized_time','normalized_risk',
                'expected_detections','feasible','violations','uav_metrics')}


class FitnessEvaluator:
    """Units: energy in model units, time in seconds, risk in hazard-seconds.

    Sensor observations occur at each *arrival* cell, with exposure time 1/speed.
    Zero-length segments cannot collect free observations. Deployment is external
    to the modeled one-way mission; there is no implicit return-to-base leg.
    """

    def __init__(self, environment, spec, planner, weights=FitnessWeights()):
        self.env, self.spec, self.planner, self.weights = environment,spec,planner,weights
        self.evaluations = 0
        self.footprints = {}
        n = environment.size
        for power in spec.power_levels:
            radius = spec.sensor_radius*np.sqrt(power)
            offsets = [(dx,dy) for dy in range(-int(radius),int(radius)+1)
                       for dx in range(-int(radius),int(radius)+1) if dx*dx+dy*dy <= radius*radius]
            y,x = np.mgrid[:n,:n]
            ids = np.full((n*n,len(offsets)), n*n, dtype=np.int32)
            rates = np.zeros(ids.shape)
            for k,(dx,dy) in enumerate(offsets):
                tx,ty = x.ravel()+dx,y.ravel()+dy
                inside = (tx>=0)&(ty>=0)&(tx<n)&(ty<n)
                source = np.flatnonzero(inside)
                target = ty[inside]*n+tx[inside]
                allowed = environment.traversable.ravel()[target]
                source,target = source[allowed],target[allowed]
                ids[source,k] = target
                rates[source,k] = (.65*power*np.exp(-(dx*dx+dy*dy)/(2*radius*radius))
                                   *(1-.4*environment.hazard.ravel()[target]))
            self.footprints[power] = ids,rates

    def evaluate(self, chromosome):
        self.evaluations += 1
        errors = chromosome.validate(self.env,self.spec)
        if errors:
            return Evaluation(-1.-len(errors),feasible=False,violations=errors)
        spec,n = self.spec,self.env.size
        all_ids,all_rates = [],[]
        result = Evaluation(0.)
        for plan in chromosome.plans:
            previous = plan.start
            full_path = [previous]
            distance = energy = elapsed = risk = 0.
            for goal,v,on,power in zip(plan.waypoints,plan.speeds,plan.sensors,plan.sensor_power):
                route = self.planner.route(previous,goal)
                if route is None:
                    result.violations.append(f'UAV {plan.uid}: unreachable segment')
                    break
                # A* returns adjacent, traversable cells; recheck route geometry as a guard.
                if any(not self.env.valid(p) for p in route) or any(
                        abs(a[0]-b[0])+abs(a[1]-b[1]) != 1 for a,b in zip(route,route[1:])):
                    result.violations.append(f'UAV {plan.uid}: disconnected or blocked path')
                    break
                steps = len(route)-1
                distance += steps
                energy += steps*energy_per_cell(v,on,power,spec)
                elapsed += steps/v
                if steps:
                    cells = np.array([y*n+x for x,y in route[1:]],dtype=int)
                    risk += float(self.env.hazard.ravel()[cells].sum()/v)
                    if on:
                        ids,rates = self.footprints[power]
                        all_ids.append(ids[cells].ravel())
                        all_rates.append((rates[cells]/v).ravel())
                full_path.extend(route[1:])
                previous = goal
            if distance > spec.max_range+1e-8:
                result.violations.append(f'UAV {plan.uid}: maximum range exceeded')
            if energy > spec.battery_capacity+1e-8:
                result.violations.append(f'UAV {plan.uid}: battery capacity exceeded')
            result.uav_metrics.append(dict(uid=plan.uid,distance=float(distance),energy=float(energy),
                                           time=float(elapsed),risk=float(risk)))
            result.paths.append(full_path)
            result.energy += energy
            result.mission_time = max(result.mission_time,elapsed)
            result.risk += risk
        if result.violations:
            result.feasible = False
            result.fitness = -1.-len(result.violations)
            return result
        if all_ids:
            ids,rates = np.concatenate(all_ids),np.concatenate(all_rates)
            # Padded entries have zero rate and point to a sentinel outside the map.
            exposure = np.bincount(ids,weights=rates,minlength=n*n+1)[:n*n]
            counts = np.bincount(ids[rates>0],minlength=n*n+1)[:n*n]
        else:
            exposure,counts = np.zeros(n*n),np.zeros(n*n,dtype=int)
        detection = -np.expm1(-exposure)
        priorities = self.env.priority.ravel()*self.env.traversable.ravel()
        result.coverage = float(priorities[counts>0].sum()/priorities.sum())
        result.detection = float(np.dot(self.env.target_probability.ravel(),detection))
        result.expected_detections = float(detection[self.env.targets.ravel()].sum())
        result.normalized_energy = result.energy/(spec.count*spec.battery_capacity)
        max_time = spec.max_range/spec.min_speed
        result.normalized_time = result.mission_time/max_time
        result.normalized_risk = result.risk/(spec.count*max_time)
        result.observation_counts = counts.reshape(n,n)
        result.detection_map = detection.reshape(n,n)
        w = self.weights
        result.fitness = (w.coverage*result.coverage+w.detection*result.detection
                          -w.energy*result.normalized_energy-w.time*result.normalized_time
                          -w.risk*result.normalized_risk)
        return result
