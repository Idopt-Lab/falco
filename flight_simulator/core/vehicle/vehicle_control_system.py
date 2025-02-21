from abc import ABC, abstractmethod
from typing import List

import csdl_alpha as csdl


class VehicleControlSystem(ABC):

    def __init__(self, pitch_control: list,
                 roll_control: list,
                 yaw_control: list,
                 throttle_control: list):
        self.pitch_control = pitch_control
        self.roll_control = roll_control
        self.yaw_control = yaw_control
        self.throttle_control = throttle_control
        pass

    @abstractmethod
    def control_order(self) -> List[str]:
        raise NotImplementedError


class ControlSurface:
    def __init__(self, name, lb: float, ub: float, component=None):
        self._lb = lb
        self._ub = ub
        if component is None:
            self.deflection = csdl.Variable(name=name+'_deflection', shape=(1, ), value=0.)
        else:
            assert isinstance(component.parameters.actuate_angle, csdl.Variable)
            self.deflection = component.parameters.actuate_angle

    @property
    def lower_bound(self):
        return self._lb

    @property
    def upper_bound(self):
        return self._ub


class PropulsiveControl:
    def __init__(self, name, lb: float=0., ub: float=1.):
        self._lb = lb
        self._ub = ub
        self.throttle = csdl.Variable(name=name+'_throttle', shape=(1, ), value=0.)

    @property
    def lower_bound(self):
        return self._lb

    @property
    def upper_bound(self):
        return self._ub





