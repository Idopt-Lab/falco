from abc import ABC, abstractmethod

import NRLMSIS2
import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo


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
class AircraftStates:

    @dataclass
    class States6dof(csdl.VariableGroup):
        u: csdl.Variable
        v: csdl.Variable
        w: csdl.Variable
        p: csdl.Variable
        q: csdl.Variable
        r: csdl.Variable
        phi: csdl.Variable
        theta: csdl.Variable
        psi: csdl.Variable
        x: csdl.Variable
        y: csdl.Variable
        z: csdl.Variable

        def define_checks(self):
            self.add_check('u', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('v', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('w', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('p', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('q', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('r', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('phi', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('theta', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('psi', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('x', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('y', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('z', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

        def _check_parameters(self, name, value):
            if self._metadata[name]['type'] is not None:
                if type(value) not in self._metadata[name]['type']:
                    raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")

            if self._metadata[name]['variablize']:
                if isinstance(value, ureg.Quantity):
                    value_si = value.to_base_units()
                    value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                    value.add_tag(tag=str(value_si.units))

            if self._metadata[name]['shape'] is not None:
                if value.shape != self._metadata[name]['shape']:
                    raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
            return value

    @dataclass
    class StatesInertialFrameWindVelocityVector(csdl.VariableGroup):
        Vwx: csdl.Variable
        Vwy: csdl.Variable
        Vwz: csdl.Variable

        def define_checks(self):
            self.add_check('Vwx', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Vwy', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Vwz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

        def _check_parameters(self, name, value):
            if self._metadata[name]['type'] is not None:
                if type(value) not in self._metadata[name]['type']:
                    raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")

            if self._metadata[name]['variablize']:
                if isinstance(value, ureg.Quantity):
                    value_si = value.to_base_units()
                    value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                    value.add_tag(tag=str(value_si.units))

            if self._metadata[name]['shape'] is not None:
                if value.shape != self._metadata[name]['shape']:
                    raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
            return value

        
    def __init__(self,
                 axis: Union[Axis, AxisLsdoGeo],
                 u: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 v: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 w: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 p: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),
                 q: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),
                 r: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),
                 Vwx: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 Vwy: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 Vwz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 ):
        
        
        self.axis = axis
        self.atm = NRLMSIS2.Atmosphere()

        self.atmospheric_states = self.atm.evaluate(-self.axis.translation_from_origin.z)


        self.states = self.States6dof(
            u=u, v=v, w=w,
            p=p, q=q, r=r,
            x=self.axis.translation_from_origin.x, 
            y=self.axis.translation_from_origin.y, 
            z=self.axis.translation_from_origin.z,
            phi=axis.euler_angles.phi, 
            theta=axis.euler_angles.theta, 
            psi=axis.euler_angles.psi
        )

        self.states_inertial_frame_wind = self.StatesInertialFrameWindVelocityVector(Vwx=Vwx, Vwy=Vwy, Vwz=Vwz)

        self.body_frame_velocity_vector = csdl.concatenate((self.states.u, self.states.v, self.states.w), axis=0)
        self.inertial_frame_wind_velocity_vector = csdl.concatenate((self.states_inertial_frame_wind.Vwx,
                                                                     self.states_inertial_frame_wind.Vwy,
                                                                     self.states_inertial_frame_wind.Vwz), axis=0)
        self.angular_rates_vector = csdl.concatenate((self.states.p, self.states.q, self.states.r), axis=0)
        self.position_vector = self.axis.translation_from_origin_vector
        self.euler_angles_vector = self.axis.euler_angles_vector

        self.states_vector = csdl.concatenate(
            (self.body_frame_velocity_vector, self.angular_rates_vector, self.euler_angles_vector, self.position_vector),
            axis=0
        )

        self.linear_acceleration = csdl.VariableGroup()
        self.linear_acceleration.uDot = csdl.Variable(value=0, shape=(1,), name='uDot', tags=['m/s^2'])
        self.linear_acceleration.vDot = csdl.Variable(value=0, shape=(1,), name='vDot', tags=['m/s^2'])
        self.linear_acceleration.wDot = csdl.Variable(value=0, shape=(1,), name='wDot', tags=['m/s^2'])
        self.linear_acceleration_vector = csdl.concatenate(
            (self.linear_acceleration.uDot, self.linear_acceleration.vDot, self.linear_acceleration.wDot),
            axis=0)

        self.angular_acceleration = csdl.VariableGroup()
        self.angular_acceleration.pDot = csdl.Variable(value=0, shape=(1,), name='pDot', tags=['rad/s^2'])
        self.angular_acceleration.qDot = csdl.Variable(value=0, shape=(1,), name='qDot', tags=['rad/s^2'])
        self.angular_acceleration.rDot = csdl.Variable(value=0, shape=(1,), name='rDot', tags=['rad/s^2'])
        self.angular_acceleration_vector = csdl.concatenate(
            (self.angular_acceleration.pDot, self.angular_acceleration.qDot, self.angular_acceleration.rDot),
            axis=0)

        self.statesdot_vector = csdl.concatenate(
            (self.linear_acceleration_vector, self.angular_acceleration_vector, self.angular_rates_vector, self.body_frame_velocity_vector),
            axis=0
        )

        self.VTAS = csdl.norm(self.body_frame_velocity_vector, ord=2)
        self.VTAS.add_name(name='VTAS')
        self.VTAS.add_tag(tag='m/s')

        self.alpha = csdl.arctan(self.states.w / self.states.u)  # Todo: Change to atan2
        self.alpha.add_name(name='alpha')
        self.alpha.add_tag(tag='rad')

        self.beta = csdl.arcsin(self.states.v / self.VTAS)
        self.beta.add_name(name='beta')
        self.beta.add_tag(tag='rad')

        self.gamma = self.axis.euler_angles.theta - self.alpha
        self.gamma.add_name(name='gamma')
        self.gamma.add_tag(tag='rad')

        self.sigma = self.beta - self.axis.euler_angles.psi  # Aircraft Heading
        self.sigma.add_name(name='sigma')
        self.sigma.add_tag(tag='rad')

        self.Mach = self.VTAS / self.atmospheric_states.speed_of_sound
        self.Mach.add_name(name='Mach')
        self.Mach.add_tag(tag='')

        # Ground Speed
        R_B_to_I = np.eye(3)  # Todo: Implement rotation matrix
        self.inertial_velocity_vector = csdl.matvec(R_B_to_I, self.body_frame_velocity_vector) + self.inertial_frame_wind_velocity_vector

        self.course_angle = csdl.arctan(self.inertial_velocity_vector[1]/self.inertial_velocity_vector[0])
    
    def _assemble_state_vector(self):
        """
        Assemble the state vector.
        
        Returns:
            The concatenated state vector.
        """
        return csdl.concatenate((
                self.state_vector.u,
                self.state_vector.v,
                self.state_vector.w,
                self.state_vector.p,
                self.state_vector.q,
                self.state_vector.r,
                self.state_vector.phi,
                self.state_vector.theta,
                self.state_vector.psi,
                self.state_vector.x,
                self.state_vector.y,
                self.state_vector.z
            ), axis=0)