from abc import ABC, abstractmethod
from typing import List
import csdl_alpha as csdl   


class VehicleControlSystem(ABC):

    def __init__(self, pitch: list, roll: list, yaw: list, throttle: list):
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.throttle = throttle    
        pass

    @abstractmethod
    def control_order(self) -> List[str]:
        raise NotImplementedError
    
class ControlSurface:
    def __init__(self, name, min_value: float, max_value: float):
        self.name = name
        self._min_value = min_value
        self._max_value = max_value
        self.deflection = csdl.Variable(name=name + "_deflection", shape=(1, ), value=0.)
    
    @property
    def min_value(self):
        return self._min_value
    
    @property
    def max_value(self):
        return self._max_value
    
class PropulsiveControl:
    def __init__(self, name, min_value: float=0., max_value: float=1.):
        self.name = name
        self._min_value = min_value
        self._max_value = max_value
        self.throttle = csdl.Variable(name=name + "_throttle", shape=(1, ), value=0.)
    
    @property
    def min_value(self):
        return self._min_value
    
    @property
    def max_value(self):
        return self._max_value
