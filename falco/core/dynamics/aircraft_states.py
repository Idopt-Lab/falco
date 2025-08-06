from abc import ABC, abstractmethod

import pymsis
import jax
import jax.numpy as jnp
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
    x : jnp.ndarray or ureg.Quantity
        Displacement.
    x_dot : jnp.ndarray or ureg.Quantity
        Velocity.
    """
    def __init__(self, state_axis, x=Union[ureg.Quantity, jnp.ndarray], x_dot=Union[ureg.Quantity, jnp.ndarray]):
        """Initialize the mass-spring-damper state.

        Parameters
        ----------
        state_axis : Axis
            The axis in which the state is defined.
        x : ureg.Quantity or jnp.ndarray, optional
            Initial displacement.
        x_dot : ureg.Quantity or jnp.ndarray, optional
            Initial velocity.
        """
        super().__init__(state_axis=state_axis)

        self._x = jnp.array([0.0])
        self._x_dot = jnp.array([0.0])

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
            self._x = jnp.array([vector_x.magnitude])
        elif isinstance(x_value, jnp.ndarray):
            self._x = x_value
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
            self._x_dot = jnp.array([vector_x.magnitude])
        elif isinstance(x_dot_value, jnp.ndarray):
            self._x_dot = x_dot_value
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
    body_frame_velocity_vector : jnp.ndarray
        Body-frame velocity vector [u, v, w].
    inertial_frame_wind_velocity_vector : jnp.ndarray
        Inertial-frame wind velocity vector [Vwx, Vwy, Vwz].
    angular_rates_vector : jnp.ndarray
        Angular rates vector [p, q, r].
    position_vector : jnp.ndarray
        Position vector in the inertial frame.
    euler_angles_vector : jnp.ndarray
        Euler angles vector [phi, theta, psi].
    states_vector : jnp.ndarray
        Concatenated state vector.
    linear_acceleration : object
        Linear acceleration variables.
    angular_acceleration : object
        Angular acceleration variables.
    VTAS : jnp.ndarray
        True airspeed.
    alpha : jnp.ndarray
        Angle of attack.
    beta : jnp.ndarray
        Sideslip angle.
    windAxis : Axis
        Wind axis object.
    alpha_dot : jnp.ndarray
        Time derivative of angle of attack.
    gamma : jnp.ndarray
        Flight path angle.
    sigma : jnp.ndarray
        Heading angle.
    Mach : jnp.ndarray
        Mach number.
    inertial_velocity_vector : jnp.ndarray
        Velocity vector in the inertial frame.
    course_angle : jnp.ndarray
        Course angle.
    """
    @dataclass
    class States6dof:
        """6-DOF state variables for an aircraft.

        Attributes
        ----------
        u, v, w : jnp.ndarray
            Body-frame velocity components.
        p, q, r : jnp.ndarray
            Body-frame angular rates.
        phi, theta, psi : jnp.ndarray
            Euler angles.
        x, y, z : jnp.ndarray
            Position in the inertial frame.
        """
        u: jnp.ndarray
        v: jnp.ndarray
        w: jnp.ndarray
        p: jnp.ndarray
        q: jnp.ndarray
        r: jnp.ndarray
        phi: jnp.ndarray
        theta: jnp.ndarray
        psi: jnp.ndarray
        x: jnp.ndarray
        y: jnp.ndarray
        z: jnp.ndarray

    @dataclass
    class StatesInertialFrameWindVelocityVector:
        """Wind velocity components in the inertial frame.

        Attributes
        ----------
        Vwx, Vwy, Vwz : jnp.ndarray
            Wind velocity components in the inertial frame.
        """
        Vwx: jnp.ndarray
        Vwy: jnp.ndarray
        Vwz: jnp.ndarray

    @staticmethod
    @jax.jit
    def _concatenate_vectors(*vectors):
        """JIT-compiled concatenation of vectors."""
        return jnp.concatenate(vectors, axis=0)

    @staticmethod
    @jax.jit
    def _compute_norm(vector):
        """JIT-compiled computation of vector norm."""
        return jnp.linalg.norm(vector, ord=2)

    @staticmethod
    @jax.jit
    def _compute_arctan(y, x):
        """JIT-compiled computation of arctangent."""
        return jnp.arctan2(y, x)

    @staticmethod
    @jax.jit
    def _compute_arcsin(x):
        """JIT-compiled computation of arcsine."""
        return jnp.arcsin(x)

    @staticmethod
    @jax.jit
    def _compute_cos(x):
        """JIT-compiled computation of cosine."""
        return jnp.cos(x)

    @staticmethod
    @jax.jit
    def _compute_sin(x):
        """JIT-compiled computation of sine."""
        return jnp.sin(x)

    @staticmethod
    @jax.jit
    def _compute_matvec(matrix, vector):
        """JIT-compiled matrix-vector multiplication."""
        return jnp.matmul(matrix, vector)

    @staticmethod
    @jax.jit
    def _compute_alpha_dot(alpha, beta, w_dot, u_dot, VTAS):
        """JIT-compiled computation of alpha_dot."""
        return (jnp.cos(alpha) * w_dot - jnp.sin(alpha) * u_dot) / (VTAS * jnp.cos(beta))

    @staticmethod
    @jax.jit
    def _compute_rotation_matrix(phi, theta, psi):
        """JIT-compiled computation of rotation matrix from body to inertial frame."""
        # Rotation matrix from body to inertial frame
        # R = Rz(psi) * Ry(theta) * Rx(phi)
        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_theta, s_theta = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        
        Rx = jnp.array([[1, 0, 0],
                        [0, c_phi, -s_phi],
                        [0, s_phi, c_phi]])
        
        Ry = jnp.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
        
        Rz = jnp.array([[c_psi, -s_psi, 0],
                        [s_psi, c_psi, 0],
                        [0, 0, 1]])
        
        return jnp.matmul(jnp.matmul(Rz, Ry), Rx)
        
    def __init__(self,
                 axis: Union[Axis, AxisLsdoGeo],
                 u: Union[ureg.Quantity, jnp.ndarray]=Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 v: Union[ureg.Quantity, jnp.ndarray]=Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 w: Union[ureg.Quantity, jnp.ndarray]=Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 p: Union[ureg.Quantity, jnp.ndarray]=Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 q: Union[ureg.Quantity, jnp.ndarray]=Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 r: Union[ureg.Quantity, jnp.ndarray]=Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 Vwx: Union[ureg.Quantity, jnp.ndarray] = Q_(0, 'm/s'), # WRT To Inertial Frame
                 Vwy: Union[ureg.Quantity, jnp.ndarray] = Q_(0, 'm/s'), # WRT To Inertial Frame
                 Vwz: Union[ureg.Quantity, jnp.ndarray] = Q_(0, 'm/s'), # WRT To Inertial Frame
                 ):
        
        """Initialize the aircraft states.

        Parameters
        ----------
        axis : Axis or AxisLsdoGeo
            The axis in which the states are defined.
        u, v, w : ureg.Quantity or jnp.ndarray, optional
            Body-frame velocity components.
        p, q, r : ureg.Quantity or jnp.ndarray, optional
            Body-frame angular rates.
        Vwx, Vwy, Vwz : ureg.Quantity or jnp.ndarray, optional
            Wind velocity components in the inertial frame.
        """
        self.axis = axis
        
        # Convert altitude to numpy for pymsis compatibility
        # Handle both ureg.Quantity and JAX array cases
        z_value = self.axis.translation_from_origin.z
        if hasattr(z_value, 'to_base_units'):
            # It's a ureg.Quantity
            altitude_np = float(z_value.to_base_units().magnitude)
        else:
            # It's a JAX or numpy array, possibly with shape (1,)
            if hasattr(z_value, 'shape') and z_value.shape == (1,):
                altitude_np = float(z_value[0])
            else:
                altitude_np = float(z_value)
        
        # Use pymsis.calculate() instead of Atmosphere class
        # Default parameters for atmospheric calculation
        import numpy as np
        from datetime import datetime
        
        # Convert altitude from meters to kilometers for pymsis
        altitude_km = altitude_np / 1000.0
        
        # Use current date and default location
        date = np.datetime64(datetime.now().strftime("%Y-%m-%dT%H:%M"))
        lon = 0.0  # Default longitude
        lat = 0.0  # Default latitude
        
        # Calculate atmospheric properties using pymsis with default parameters
        # PYMSIS will use standard default values for f107, f107a, and ap if not specified
        atmospheric_data = pymsis.calculate(date, lon, lat, np.array([altitude_km]))
        
        # Extract relevant atmospheric properties
        # atmospheric_data has shape (1, 11) for single point
        # Index 0 is total mass density (kg/m³)
        # Index 10 is temperature (K)
        self.atmospheric_states = type('AtmosphericStates', (), {
            'density': atmospheric_data[0, 0],  # Total mass density (kg/m³)
            'temperature': atmospheric_data[0, 10],  # Temperature (K)
            'speed_of_sound': np.sqrt(1.4 * 287.0 * atmospheric_data[0, 10])  # Speed of sound (m/s)
        })()

        # Convert inputs to JAX arrays
        def _convert_to_jax(value):
            if isinstance(value, ureg.Quantity):
                return jnp.array([value.to_base_units().magnitude])
            elif isinstance(value, jnp.ndarray):
                return value
            else:
                return jnp.array([float(value)])

        u_jax = _convert_to_jax(u)
        v_jax = _convert_to_jax(v)
        w_jax = _convert_to_jax(w)
        p_jax = _convert_to_jax(p)
        q_jax = _convert_to_jax(q)
        r_jax = _convert_to_jax(r)
        Vwx_jax = _convert_to_jax(Vwx)
        Vwy_jax = _convert_to_jax(Vwy)
        Vwz_jax = _convert_to_jax(Vwz)

        # Position and orientation from axis
        # Handle both ureg.Quantity and JAX array cases
        def _extract_value(prop):
            if hasattr(prop, 'to_base_units'):
                # It's a ureg.Quantity
                return jnp.array([prop.to_base_units().magnitude])
            else:
                # It's already a JAX array or numpy array
                if hasattr(prop, 'shape') and prop.shape == (1,):
                    return jnp.array([float(prop[0])])
                else:
                    return jnp.array([float(prop)])

        x_jax = _extract_value(self.axis.translation_from_origin.x)
        y_jax = _extract_value(self.axis.translation_from_origin.y)
        z_jax = _extract_value(self.axis.translation_from_origin.z)
        phi_jax = _extract_value(axis.euler_angles.phi)
        theta_jax = _extract_value(axis.euler_angles.theta)
        psi_jax = _extract_value(axis.euler_angles.psi)

        self.states = self.States6dof(
            u=u_jax, v=v_jax, w=w_jax,
            p=p_jax, q=q_jax, r=r_jax,
            x=x_jax, y=y_jax, z=z_jax,
            phi=phi_jax, theta=theta_jax, psi=psi_jax
        )

        self.states_inertial_frame_wind = self.StatesInertialFrameWindVelocityVector(
            Vwx=Vwx_jax, Vwy=Vwy_jax, Vwz=Vwz_jax
        )

        # JIT-compiled vector computations
        self.body_frame_velocity_vector = self._concatenate_vectors(
            self.states.u, self.states.v, self.states.w
        )
        self.inertial_frame_wind_velocity_vector = self._concatenate_vectors(
            self.states_inertial_frame_wind.Vwx,
            self.states_inertial_frame_wind.Vwy,
            self.states_inertial_frame_wind.Vwz
        )
        self.angular_rates_vector = self._concatenate_vectors(
            self.states.p, self.states.q, self.states.r
        )
        self.position_vector = self._concatenate_vectors(
            self.states.x, self.states.y, self.states.z
        )
        self.euler_angles_vector = self._concatenate_vectors(
            self.states.phi, self.states.theta, self.states.psi
        )

        self.states_vector = self._concatenate_vectors(
            self.body_frame_velocity_vector, 
            self.angular_rates_vector, 
            self.euler_angles_vector, 
            self.position_vector
        )

        # Initialize acceleration vectors
        self.linear_acceleration = type('LinearAcceleration', (), {
            'uDot': jnp.array([0.0]),
            'vDot': jnp.array([0.0]),
            'wDot': jnp.array([0.0])
        })()
        self.linear_acceleration_vector = self._concatenate_vectors(
            self.linear_acceleration.uDot, 
            self.linear_acceleration.vDot, 
            self.linear_acceleration.wDot
        )

        self.angular_acceleration = type('AngularAcceleration', (), {
            'pDot': jnp.array([0.0]),
            'qDot': jnp.array([0.0]),
            'rDot': jnp.array([0.0])
        })()
        self.angular_acceleration_vector = self._concatenate_vectors(
            self.angular_acceleration.pDot, 
            self.angular_acceleration.qDot, 
            self.angular_acceleration.rDot
        )

        self.statesdot_vector = self._concatenate_vectors(
            self.linear_acceleration_vector, 
            self.angular_acceleration_vector, 
            self.angular_rates_vector, 
            self.body_frame_velocity_vector
        )

        # JIT-compiled aerodynamic computations
        self.VTAS = self._compute_norm(self.body_frame_velocity_vector)

        self.alpha = self._compute_arctan(self.states.w, self.states.u)

        self.beta = self._compute_arcsin(self.states.v / self.VTAS)

        self.windAxis = Axis(
            name='Wind Axis',
            x=Q_(0, 'ft'),
            y=Q_(0, 'ft'),
            z=Q_(0, 'ft'),
            phi=Q_(0, 'deg'),
            theta=Q_(self.alpha[0], 'rad'),
            psi=Q_(-self.beta[0], 'rad'),
            sequence=np.array([3, 2, 1]),
            reference=self.axis,
            origin=ValidOrigins.Inertial.value
        )

        self.alpha_dot = self._compute_alpha_dot(
            self.alpha, self.beta, 
            self.linear_acceleration.wDot, 
            self.linear_acceleration.uDot, 
            self.VTAS
        )

        self.gamma = self.states.theta - self.alpha

        self.sigma = self.beta - self.states.psi

        self.Mach = self.VTAS / self.atmospheric_states.speed_of_sound

        # Ground Speed - compute rotation matrix and transform
        R_B_to_I = self._compute_rotation_matrix(
            self.states.phi[0], self.states.theta[0], self.states.psi[0]
        )
        self.inertial_velocity_vector = (
            self._compute_matvec(R_B_to_I, self.body_frame_velocity_vector) + 
            self.inertial_frame_wind_velocity_vector
        )

        self.course_angle = self._compute_arctan(
            self.inertial_velocity_vector[1], 
            self.inertial_velocity_vector[0]
        )
    
    def _assemble_state_vector(self):
        """Assemble the state vector.

        Returns
        -------
        jnp.ndarray
            The concatenated state vector.
        """
        return self._concatenate_vectors(
            self.states.u,
            self.states.v,
            self.states.w,
            self.states.p,
            self.states.q,
            self.states.r,
            self.states.phi,
            self.states.theta,
            self.states.psi,
            self.states.x,
            self.states.y,
            self.states.z
        )


if __name__ == "__main__":
    # Example usage of the AircraftStates class
    # Create an axis for the aircraft

    inertial_axis = Axis(
    name='Inertial Axis',
    origin=ValidOrigins.Inertial.value)


    axis = Axis(
        name='Aircraft Axis',
        x=Q_(0, 'ft'),
        y=Q_(0, 'ft'),
        z=Q_(1000, 'ft'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    # Create an instance of AircraftStates
    aircraft_states = AircraftStates(
        axis=axis,
        u=Q_(100, 'm/s'),
        v=Q_(0, 'm/s'),
        w=Q_(0, 'm/s'),
        p=Q_(0, 'rad/s'),
        q=Q_(0, 'rad/s'),
        r=Q_(0, 'rad/s'),
        Vwx=Q_(10, 'm/s'),
        Vwy=Q_(0, 'm/s'),
        Vwz=Q_(0, 'm/s')
    )

    # Print some properties
    print(f"Position: {aircraft_states.position_vector}")
    print(f"Velocity: {aircraft_states.body_frame_velocity_vector}")
    print(f"Wind: {aircraft_states.inertial_frame_wind_velocity_vector}")
    print(f"Alpha: {aircraft_states.alpha}")
    print(f"Beta: {aircraft_states.beta}")
    print(f"VTAS: {aircraft_states.VTAS}")
    print(f"Mach: {aircraft_states.Mach}")
    print(f"Course Angle: {aircraft_states.course_angle}")