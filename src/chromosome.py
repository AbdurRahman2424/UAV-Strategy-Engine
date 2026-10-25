"""Hybrid fleet chromosome: integer geometry, real speeds, binary sensing, discrete power."""
from copy import deepcopy
from dataclasses import dataclass
from .drone import UAVPlan


@dataclass
class Chromosome:
    plans: list

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return {'uavs': [p.to_dict() for p in self.plans]}

    @classmethod
    def random(cls, environment, spec, planner, rng):
        plans = []
        for uid in range(spec.count):
            start = tuple(map(int, environment.valid_cells[rng.integers(len(environment.valid_cells))]))
            point = start
            waypoints = []
            for _ in range(spec.waypoints):
                component = environment.components[start[1],start[0]]
                point = environment.nearest_valid(rng.normal(point, environment.size*.23), component)
                waypoints.append(point)
            plans.append(UAVPlan(uid, start, waypoints,
                                 rng.uniform(spec.min_speed,spec.max_speed,spec.waypoints).tolist(),
                                 (rng.random(spec.waypoints)<.85).astype(int).tolist(),
                                 rng.choice(spec.power_levels,spec.waypoints).tolist()))
        chromosome = cls(plans)
        chromosome.repair(environment, spec, planner)
        return chromosome

    def validate(self, environment, spec):
        """Structural validation only; evaluator also checks expanded paths/resources."""
        errors = []
        if len(self.plans) != spec.count or sorted(p.uid for p in self.plans) != list(range(spec.count)):
            errors.append('Fleet IDs/count do not match configuration')
        for p in self.plans:
            if any(len(x) != spec.waypoints for x in (p.waypoints,p.speeds,p.sensors,p.sensor_power)):
                errors.append(f'UAV {p.uid}: inconsistent segment arrays')
                continue
            if not all(environment.valid(x) for x in [p.start,*p.waypoints]):
                errors.append(f'UAV {p.uid}: invalid deployment/waypoint')
            if not all(spec.min_speed <= v <= spec.max_speed for v in p.speeds):
                errors.append(f'UAV {p.uid}: speed limits')
            if not all(x in (0,1) for x in p.sensors):
                errors.append(f'UAV {p.uid}: invalid sensor state')
            if not all(x in spec.power_levels for x in p.sensor_power):
                errors.append(f'UAV {p.uid}: invalid power level')
        return errors

    def repair(self, environment, spec, planner):
        from .constraints import repair_chromosome
        repair_chromosome(self, environment, spec, planner)
        return self

    def mutate(self, environment, spec, planner, rng, probability=.15):
        from .mutation import mutate
        return mutate(self, environment, spec, planner, rng, probability)

    def crossover(self, other, environment, spec, planner, rng, method='segment'):
        from .crossover import crossover
        return crossover(self, other, environment, spec, planner, rng, method)

    def evaluate(self, evaluator):
        return evaluator.evaluate(self)
