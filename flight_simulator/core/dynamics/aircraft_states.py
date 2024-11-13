from abc import ABC, abstractmethod
import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_

class RigidBodyStates(ABC):
    def __init__(self, state_axis):
        self.state_axis = state_axis

    @abstractmethod
    def return_state_vector(self):
        raise NotImplementedError

    @abstractmethod
    def update_state_from_vector(self, x_input):
        raise NotImplementedError

    @abstractmethod
    def update(self, t_input, x_input):
        raise NotImplementedError


class MassSpringDamperState(RigidBodyStates):
    def __init__(self, state_axis, x=Union[ureg.Quantity, csdl.Variable], x_dot=Union[ureg.Quantity, csdl.Variable]):
        super().__init__(state_axis=state_axis)

        self.x = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.x_dot = csdl.Variable(shape=(1,), value=np.array([0, ]))

        self.x = x
        self.x_dot = x_dot

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x_value):
        if x_value is None:
            self._x = None
        elif isinstance(x_value, ureg.Quantity):
            vector_x = x_value.to_base_units()
            self._x.set_value(vector_x.magnitude)
            self._x.add_tag(str(vector_x.units))
        elif isinstance(x_value, csdl.Variable):
            self._translation = x_value
        else:
            raise IOError

    @property
    def x_dot(self):
        return self._x_dot

    @x_dot.setter
    def x_dot(self, x_dot_value):
        if x_dot_value is None:
            self._x_dot = None
        elif isinstance(x_dot_value, ureg.Quantity):
            vector_x = x_dot_value.to_base_units()
            self._x_dot.set_value(vector_x.magnitude)
            self._x_dot.add_tag(str(vector_x.units))
        elif isinstance(x_dot_value, csdl.Variable):
            self._translation = x_dot_value
        else:
            raise IOError

    def update(self, t, input_vector):
        # Potential t-based property updates
        self.update_state_from_vector(input_vector)

    def update_state_from_vector(self, input_vector):
        self.x = Q_(input_vector.vector.value[0], 'm')
        self.x_dot = Q_(input_vector.vector.value[1], 'm/s')

    def return_state_vector(self):
        return np.array([self.x.value, self.x_dot.value], dtype=float)

@dataclass
class AircaftStates(csdl.VariableGroup):
    u : Union[float, int, csdl.Variable, np.ndarray] = 0
    v : Union[float, int, csdl.Variable, np.ndarray] = 0
    w : Union[float, int, csdl.Variable, np.ndarray] = 0
    p : Union[float, int, csdl.Variable, np.ndarray] = 0
    q : Union[float, int, csdl.Variable, np.ndarray] = 0
    r : Union[float, int, csdl.Variable, np.ndarray] = 0
    phi : Union[float, int, csdl.Variable, np.ndarray] = 0
    theta : Union[float, int, csdl.Variable, np.ndarray] = 0
    psi : Union[float, int, csdl.Variable, np.ndarray] = 0
    x : Union[float, int, csdl.Variable, np.ndarray] = 0
    y : Union[float, int, csdl.Variable, np.ndarray] = 0
    z : Union[float, int, csdl.Variable, np.ndarray] = 0