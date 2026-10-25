"""Typed, serializable configuration. Distances are grid cells; time is seconds."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvironmentConfig:
    size: int = 50
    seed: int = 2026
    target_count: int = 28

    def __post_init__(self):
        if self.size < 12 or self.target_count < 1:
            raise ValueError('Use a grid of at least 12 cells and at least one target.')


@dataclass(frozen=True)
class UAVConfig:
    count: int = 3
    waypoints: int = 4
    battery_capacity: float = 65.0
    max_range: float = 90.0
    min_speed: float = 0.8
    max_speed: float = 3.0
    sensor_radius: float = 2.6
    sensor_consumption: float = 0.22
    power_levels: tuple = (0.5, 1.0, 1.5)

    def __post_init__(self):
        if self.count < 1 or self.waypoints < 1:
            raise ValueError('UAV count and waypoint count must be positive.')
        if min(self.battery_capacity, self.max_range, self.min_speed,
               self.sensor_radius, self.sensor_consumption) <= 0:
            raise ValueError('Physical specifications must be positive.')
        if self.max_speed < self.min_speed or not self.power_levels or min(self.power_levels) <= 0:
            raise ValueError('Invalid speed limits or sensor power levels.')


@dataclass(frozen=True)
class FitnessWeights:
    coverage: float = 0.35
    detection: float = 0.25
    energy: float = 0.15
    time: float = 0.10
    risk: float = 0.15

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if min(values) < 0 or abs(sum(values) - 1) > 1e-9:
            raise ValueError('Nonnegative weights must sum to one.')


@dataclass(frozen=True)
class GAConfig:
    population: int = 60
    generations: int = 100
    crossover: float = 0.85
    mutation: float = 0.15
    tournament: int = 3
    elites: int = 2
    immigrant_fraction: float = 0.05
    seed: int = 42
    patience: int = 0

    def __post_init__(self):
        if self.population < 4 or self.generations < 0:
            raise ValueError('Population >= 4 and generations >= 0 required.')
        if not 0 <= self.elites < self.population or not 1 <= self.tournament <= self.population:
            raise ValueError('Invalid elitism or tournament size.')
        if any(not 0 <= x <= 1 for x in (self.crossover, self.mutation, self.immigrant_fraction)):
            raise ValueError('Probabilities must lie in [0, 1].')
        if self.patience < 0:
            raise ValueError('Patience cannot be negative.')
