"""One UAV's heterogeneous genes and physical limits."""
from dataclasses import dataclass
from config import UAVConfig


@dataclass
class UAVPlan:
    uid: int
    start: tuple
    waypoints: list
    speeds: list
    sensors: list
    sensor_power: list

    def to_dict(self):
        return dict(uid=self.uid, start=list(self.start), waypoints=[list(p) for p in self.waypoints],
                    speeds=list(self.speeds), sensors=list(self.sensors), sensor_power=list(self.sensor_power))


def energy_per_cell(speed, sensor_on, power, spec: UAVConfig):
    """Propulsion rises quadratically with speed; sensing draws power per second."""
    return .26 + .045*speed**2 + spec.sensor_consumption*sensor_on*power/speed
