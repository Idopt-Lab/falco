from abc import ABC, abstractmethod
from typing import List

from sympy import Union

import csdl_alpha as csdl


class ControlSurface:
    def __init__(self, name, lb: float, ub: float, component=None):
        self._lb = lb
        self._ub = ub
        if component is None: # Massless control surface case with no deflection pre-defined
            self.deflection = csdl.Variable(name=name+'_deflection', shape=(1, ), value=0.)
        elif component is not None and component.parameters.actuate_angle is None:
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
    def __init__(self, name, lb: float=0., ub: float=1., throttle:float=0.):
        self._lb = lb
        self._ub = ub
        self.throttle = csdl.Variable(name=name+'_throttle', shape=(1, ), value=throttle)

    @property
    def lower_bound(self):
        return self._lb

    @property
    def upper_bound(self):
        return self._ub
    
class PropulsiveControlRPM:
    def __init__(self, name, lb: float=0., ub: float=1., rpm:float=0.):
        self._lb = lb
        self._ub = ub
        self.rpm = csdl.Variable(name=name+'_rpm', shape=(1, ), value=rpm)

    @property
    def lower_bound(self):
        return self._lb

    @property
    def upper_bound(self):
        return self._ub


class VehicleControlSystem(ABC):

    def __init__(self, pitch_control: list,
                 roll_control: list,
                 yaw_control: list,
                 rpm_control: list):
        self.pitch_control_vector = self._create_pitch_control_vector(pitch_control=pitch_control)
        self.roll_control = roll_control
        self.yaw_control = yaw_control
        self.rpm_control = rpm_control
        pass

    def _create_pitch_control_vector(self, pitch_control):

        if len(pitch_control) == 1:
            if isinstance(pitch_control[0], ControlSurface):
                return pitch_control[0].deflection
            elif isinstance(pitch_control[0], PropulsiveControl):
                return pitch_control[0].throttle
            else:
                raise TypeError(f"Control {pitch_control[0]} is not of type ControlSurface or PropulsiveControl")
        else:
            pc_obj_list = list()
            for idx, pc in enumerate(pitch_control):
                if isinstance(pc, ControlSurface):
                    pc_obj_list.append(pc.deflection)
                elif isinstance(pc, PropulsiveControl):
                    pc_obj_list.append(pc.throttle)
                else:
                    raise TypeError(f"Control {pc} is not of type ControlSurface or PropulsiveControl")
            return csdl.concatenate(tuple(pc_obj_list), axis=0)

    @abstractmethod
    def control_order(self) -> List[str]:
        raise NotImplementedError




