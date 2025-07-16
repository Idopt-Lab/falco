from abc import ABC, abstractmethod

import NRLMSIS2
import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from falco import ureg, Q_
from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.dynamics.axis_lsdogeo import AxisLsdoGeo


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
    """Represents the state of a mass-spring-damper system.

    Attributes
    ----------
    x : csdl.Variable or ureg.Quantity
        Displacement.
    x_dot : csdl.Variable or ureg.Quantity
        Velocity.
    """
    def __init__(self, state_axis, x=Union[ureg.Quantity, csdl.Variable], x_dot=Union[ureg.Quantity, csdl.Variable]):
        """Initialize the mass-spring-damper state.

        Parameters
        ----------
        state_axis : Axis
            The axis in which the state is defined.
        x : ureg.Quantity or csdl.Variable, optional
            Initial displacement.
        x_dot : ureg.Quantity or csdl.Variable, optional
            Initial velocity.
        """
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
        """Update the state at a given time with a given input vector.

        Parameters
        ----------
        t : float
            Time.
        input_vector : object
            Input vector containing state information.
        """
        # Potential t-based property updates
        self.update_state_from_vector(input_vector)

    def update_state_from_vector(self, input_vector):
        """Update the state from a given input vector.

        Parameters
        ----------
        input_vector : object
            Input vector containing state information.
        """
        self.x = Q_(input_vector.vector.value[0], 'm')
        self.x_dot = Q_(input_vector.vector.value[1], 'm/s')

    def return_state_vector(self):
        """Return the state vector as a NumPy array.

        Returns
        -------
        np.ndarray
            Array containing displacement and velocity.
        """
        return np.array([self.x.value, self.x_dot.value], dtype=float)


@dataclass
class AircraftStates:
    """Represents the full state of an aircraft, including 6-DOF and wind states.

    Attributes
    ----------
    axis : Axis or AxisLsdoGeo
        The axis in which the states are defined.
    states : AircraftStates.States6dof
        The 6-DOF state variables.
    states_inertial_frame_wind : AircraftStates.StatesInertialFrameWindVelocityVector
        Wind velocity components in the inertial frame.
    atmospheric_states : object
        Atmospheric state variables from NRLMSIS2.
    body_frame_velocity_vector : csdl.Variable
        Body-frame velocity vector [u, v, w].
    inertial_frame_wind_velocity_vector : csdl.Variable
        Inertial-frame wind velocity vector [Vwx, Vwy, Vwz].
    angular_rates_vector : csdl.Variable
        Angular rates vector [p, q, r].
    position_vector : csdl.Variable
        Position vector in the inertial frame.
    euler_angles_vector : csdl.Variable
        Euler angles vector [phi, theta, psi].
    states_vector : csdl.Variable
        Concatenated state vector.
    linear_acceleration : csdl.VariableGroup
        Linear acceleration variables.
    angular_acceleration : csdl.VariableGroup
        Angular acceleration variables.
    VTAS : csdl.Variable
        True airspeed.
    alpha : csdl.Variable
        Angle of attack.
    beta : csdl.Variable
        Sideslip angle.
    windAxis : Axis
        Wind axis object.
    alpha_dot : csdl.Variable
        Time derivative of angle of attack.
    gamma : csdl.Variable
        Flight path angle.
    sigma : csdl.Variable
        Heading angle.
    Mach : csdl.Variable
        Mach number.
    inertial_velocity_vector : csdl.Variable
        Velocity vector in the inertial frame.
    course_angle : csdl.Variable
        Course angle.
    """
    @dataclass
    class States6dof(csdl.VariableGroup):
        """6-DOF state variables for an aircraft.

        Attributes
        ----------
        u, v, w : csdl.Variable
            Body-frame velocity components.
        p, q, r : csdl.Variable
            Body-frame angular rates.
        phi, theta, psi : csdl.Variable
            Euler angles.
        x, y, z : csdl.Variable
            Position in the inertial frame.
        """
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
        """Wind velocity components in the inertial frame.

        Attributes
        ----------
        Vwx, Vwy, Vwz : csdl.Variable
            Wind velocity components in the inertial frame.
        """
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
                 u: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 v: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 w: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 p: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 q: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 r: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 Vwx: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'), # WRT To Inertial Frame
                 Vwy: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'), # WRT To Inertial Frame
                 Vwz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'), # WRT To Inertial Frame
                 ):
        
        """Initialize the aircraft states.

        Parameters
        ----------
        axis : Axis or AxisLsdoGeo
            The axis in which the states are defined.
        u, v, w : ureg.Quantity or csdl.Variable, optional
            Body-frame velocity components.
        p, q, r : ureg.Quantity or csdl.Variable, optional
            Body-frame angular rates.
        Vwx, Vwy, Vwz : ureg.Quantity or csdl.Variable, optional
            Wind velocity components in the inertial frame.
        """
        self.axis = axis
        self.atm = NRLMSIS2.Atmosphere()

        self.atmospheric_states = self.atm.evaluate(-self.axis.translation_from_origin.z)


        self.states = self.States6dof(
            u=u, v=v, w=w,
            p=p, q=q, r=r,
            x=self.axis.translation_from_origin.x,    # WRT To Inertial Frame
            y=self.axis.translation_from_origin.y,    # WRT To Inertial Frame
            z=self.axis.translation_from_origin.z,    # WRT To Inertial Frame
            phi=axis.euler_angles.phi,                # WRT To Inertial Frame
            theta=axis.euler_angles.theta,            # WRT To Inertial Frame
            psi=axis.euler_angles.psi                 # WRT To Inertial Frame
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

        self.windAxis = Axis(
            name='Wind Axis',
            x=Q_(0, 'ft'),
            y=Q_(0, 'ft'),
            z=Q_(0, 'ft'),
            phi=Q_(0, 'deg'),
            theta=self.alpha,
            psi=-self.beta,
            sequence=np.array([3, 2, 1]),
            reference=self.axis,
            origin=ValidOrigins.Inertial.value
        )

        self.alpha_dot = (csdl.cos(self.alpha) * self.linear_acceleration.wDot - csdl.sin(self.alpha) * self.linear_acceleration.uDot) / (self.VTAS * csdl.cos(self.beta))

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
        """Assemble the state vector.

        Returns
        -------
        csdl.Variable
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