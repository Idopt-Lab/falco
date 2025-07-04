from abc import ABC, abstractmethod
from typing import List

from sympy import Union

import csdl_alpha as csdl


class ControlSurface:
    """Represents a control surface with a deflection angle and bounds.

    Parameters
    ----------
    name : str
        Name of the control surface.
    lb : float
        Lower bound for deflection.
    ub : float
        Upper bound for deflection.
    component : object, optional
        Associated component; if provided, may supply actuate_angle.

    Attributes
    ----------
    deflection : csdl.Variable
        The deflection variable for the control surface.
    """
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
    """Represents a propulsive control with a throttle value and bounds.

    Parameters
    ----------
    name : str
        Name of the propulsive control.
    lb : float, optional
        Lower bound for throttle (default 0.).
    ub : float, optional
        Upper bound for throttle (default 1.).
    throttle : float, optional
        Initial throttle value (default 0.).

    Attributes
    ----------
    throttle : csdl.Variable
        The throttle variable for the propulsive control.
    """
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

    """Represents a propulsive control with an RPM value and bounds.

    Parameters
    ----------
    name : str
        Name of the propulsive control.
    lb : float, optional
        Lower bound for RPM (default 0.).
    ub : float, optional
        Upper bound for RPM (default 1.).
    rpm : float, optional
        Initial RPM value (default 0.).

    Attributes
    ----------
    rpm : csdl.Variable
        The RPM variable for the propulsive control.
    """

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

    """Abstract base class for vehicle control in flight simulation.

    Handles pitch, roll, yaw, and throttle inputs. Subclasses must define the control order.

    Parameters
    ----------
    pitch_control : list
        List of pitch control surfaces or propulsive controls.
    roll_control : list
        List of roll control surfaces.
    yaw_control : list
        List of yaw control surfaces.
    throttle_control : list
        List of throttle controls.

    Attributes
    ----------
    pitch_control_vector : csdl.Variable
        Concatenated vector of pitch control variables.
    pitch_control : list
        List of pitch control objects.
    roll_control : list
        List of roll control objects.
    yaw_control : list
        List of yaw control objects.
    throttle_control : list
        List of throttle control objects.
    """
    def __init__(self, pitch_control: list,
                 roll_control: list,
                 yaw_control: list,
                 throttle_control: list):
        self.pitch_control_vector = self._create_pitch_control_vector(pitch_control=pitch_control)
        self.pitch_control = pitch_control
        self.roll_control = roll_control
        self.yaw_control = yaw_control
        self.throttle_control = throttle_control
        pass
        #TODO: add self.u as a part of this generic class

    def _create_pitch_control_vector(self, pitch_control):
        """Create a concatenated vector of pitch control variables.

        Parameters
        ----------
        pitch_control : list
            List of pitch control surfaces or propulsive controls.

        Returns
        -------
        csdl.Variable
            Concatenated pitch control variable(s).

        Raises
        ------
        TypeError
            If an element is not a ControlSurface or PropulsiveControl.
        """

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
        """Return the order of control variables as a list of strings.

        Returns
        -------
        List[str]
            List of control variable names in order.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        raise NotImplementedError




