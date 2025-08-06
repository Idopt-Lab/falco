from abc import ABC, abstractmethod
import pymsis
import jax
import jax.numpy as jnp
from typing import Union, Any
from dataclasses import dataclass
import numpy as np
from falco import ureg, Q_
from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.dynamics.axis_lsdogeo import AxisLsdoGeo

import jax.numpy as jnp
import jax


def process_input_value(value):
    """Convert input value to JAX array format with consistent shape.
    
    This function standardizes input values to JAX arrays with shape (1,) to ensure
    compatibility with JAX operations. It handles various input types including
    Pint quantities, scalars, and arrays.
    
    Parameters
    ----------
    value : ureg.Quantity, int, float, np.number, list, tuple, np.ndarray, or jnp.ndarray
        Input value to be converted to JAX array format.
        
    Returns
    -------
    jnp.ndarray
        JAX array with shape (1,) containing the converted value in SI base units
        if the input was a Pint quantity, otherwise the original value.
        
    """
    if hasattr(value, 'to_base_units'):  # It's a ureg.Quantity
        value_si = value.to_base_units()
        return jnp.array([value_si.magnitude])
    elif isinstance(value, (int, float, np.number)):
        return jnp.array([value])
    elif isinstance(value, (list, tuple, np.ndarray)):
        return jnp.array(value)
    else:
        return value


@jax.jit
def build_state_vector(u, v, w, p, q, r, phi, theta, psi, x, y, z):
    """Build the complete 12-DOF state vector for aircraft dynamics.
    
    Constructs a comprehensive state vector containing all primary aircraft state
    variables including linear velocities, angular rates, Euler angles, and position
    components. This vector follows the standard aircraft dynamics convention.
    
    Parameters
    ----------
    u, v, w : jax.Array
        Body-frame linear velocity components [m/s].
        - u: Forward velocity (positive forward)
        - v: Side velocity (positive right)  
        - w: Vertical velocity (positive down)
    p, q, r : jax.Array
        Body-frame angular velocity components [rad/s].
        - p: Roll rate about x-axis
        - q: Pitch rate about y-axis
        - r: Yaw rate about z-axis
    phi, theta, psi : jax.Array
        Euler angles [rad] following aerospace convention.
        - phi: Roll angle (rotation about x-axis)
        - theta: Pitch angle (rotation about y-axis)
        - psi: Yaw angle (rotation about z-axis)
    x, y, z : jax.Array
        Position components in inertial frame [m].
        - x: North position
        - y: East position
        - z: Down position (positive down)
        
    Returns
    -------
    jax.Array
        12-element state vector [u, v, w, p, q, r, phi, theta, psi, x, y, z]
        

    """
    return jnp.stack([u, v, w, p, q, r, phi, theta, psi, x, y, z])

@jax.jit
def build_angular_rates_vector(p, q, r):
    """Build angular rates vector for rotational dynamics.
    
    Constructs a 3-element vector containing the aircraft's angular velocity
    components in the body-fixed coordinate system.
    
    Parameters
    ----------
    p : jax.Array
        Roll rate about body x-axis [rad/s]. Positive p corresponds to 
        right wing down rotation.
    q : jax.Array
        Pitch rate about body y-axis [rad/s]. Positive q corresponds to 
        nose up rotation.
    r : jax.Array
        Yaw rate about body z-axis [rad/s]. Positive r corresponds to 
        nose right rotation.
        
    Returns
    -------
    jax.Array
        3-element angular rates vector [p, q, r] in body frame [rad/s]
        

    """
    return jnp.stack([p, q, r])

@jax.jit
def build_position_vector(x, y, z):
    """Build position vector in inertial reference frame.
    
    Constructs a 3-element position vector representing the aircraft's location
    in the inertial (Earth-fixed) coordinate system.
    
    Parameters
    ----------
    x : jax.Array
        North position component [m]. Positive x points true north.
    y : jax.Array
        East position component [m]. Positive y points true east.
    z : jax.Array
        Down position component [m]. Positive z points toward Earth center
        (NED convention).
        
    Returns
    -------
    jax.Array
        3-element position vector [x, y, z] in inertial frame [m]
        
    Notes
    -----
    Uses North-East-Down (NED) coordinate system convention:
    - x: North (positive northward)
    - y: East (positive eastward)  
    - z: Down (positive downward)
    
    """
    return jnp.stack([x, y, z])

@jax.jit
def build_euler_angles_vector(phi, theta, psi):
    """Build Euler angles vector for aircraft orientation.
    
    Constructs a 3-element vector containing the aircraft's orientation
    angles following the aerospace Z-Y-X (yaw-pitch-roll) Euler sequence.
    
    Parameters
    ----------
    phi : jax.Array
        Roll angle [rad]. Rotation about body x-axis.
        - Positive phi: right wing down (clockwise when viewed from behind)
        - Range: typically [-π, π]
    theta : jax.Array
        Pitch angle [rad]. Rotation about body y-axis.
        - Positive theta: nose up
        - Range: typically [-π/2, π/2] to avoid gimbal lock
    psi : jax.Array
        Yaw angle [rad]. Rotation about body z-axis.
        - Positive psi: nose right (clockwise when viewed from above)
        - Range: typically [-π, π] or [0, 2π]
        
    Returns
    -------
    jax.Array
        3-element Euler angles vector [phi, theta, psi] [rad]
        
    Notes
    -----
    Euler angle sequence (Z-Y-X):
    1. Yaw (ψ) about inertial z-axis
    2. Pitch (θ) about intermediate y-axis  
    3. Roll (φ) about final x-axis (body x-axis)
    
    """
    return jnp.stack([phi, theta, psi])

@jax.jit
def calculate_VTAS(u, v, w):
    """Calculate true airspeed from body-frame velocity components.
    
    Computes the magnitude of the velocity vector in the body-fixed frame,
    which represents the true airspeed (VTAS) of the aircraft relative to
    the surrounding air mass.
    
    Parameters
    ----------
    u : jax.Array
        Forward velocity component in body frame [m/s]
    v : jax.Array  
        Side velocity component in body frame [m/s]
    w : jax.Array
        Vertical velocity component in body frame [m/s]
        
    Returns
    -------
    jax.Array
        True airspeed magnitude [m/s]
        
    Notes
    -----
    True airspeed is fundamental for:
    - Aerodynamic force and moment calculations
    - Flight envelope monitoring
    - Performance analysis
    - Air data system validation
    
    VTAS = √(u² + v² + w²)
    
    This differs from groundspeed, which is velocity relative to the ground
    and includes wind effects.
    
    """
    return jnp.linalg.norm(jnp.stack([u, v, w]))

@jax.jit
def calculate_alpha(u, w):
    """Calculate angle of attack from body-frame velocity components.
    
    Computes the angle of attack (α), which is the angle between the aircraft's
    longitudinal axis and the relative wind vector projected onto the aircraft's
    x-z (vertical) plane.
    
    Parameters
    ----------
    u : jax.Array
        Forward velocity component in body frame [m/s]
    w : jax.Array
        Vertical velocity component in body frame [m/s] 
        (positive down in NED convention)
        
    Returns
    -------
    jax.Array
        Angle of attack [rad]
        - Positive α: nose up relative to relative wind
        - Negative α: nose down relative to relative wind
        - Range: typically [-π, π]
        
    Notes
    -----
    Angle of attack is critical for:
    - Lift and drag calculations
    - Stall prediction and prevention
    - Flight control system design
    - Performance optimization
    
    α = arctan2(w, u)

    
    """
    return jnp.arctan2(w, u)

@jax.jit
def calculate_beta(u, v, w):
    """Calculate sideslip angle from body-frame velocity components.
    
    Computes the sideslip angle (β), which is the angle between the aircraft's
    longitudinal axis and the relative wind vector projected onto the aircraft's
    x-y (horizontal) plane.
    
    Parameters
    ----------
    u : jax.Array
        Forward velocity component in body frame [m/s]
    v : jax.Array
        Side velocity component in body frame [m/s]
        (positive right in body frame)
    w : jax.Array
        Vertical velocity component in body frame [m/s]
        (used to compute total horizontal velocity)
        
    Returns
    -------
    jax.Array
        Sideslip angle [rad]
        - Positive β: relative wind from right side of aircraft
        - Negative β: relative wind from left side of aircraft  
        - Range: typically [-π/2, π/2]
        
    
    β = arctan2(v, √(u² + w²))
    
    The denominator √(u² + w²) represents the velocity magnitude in the
    longitudinal-vertical plane, ensuring proper scaling.
    
    Flight considerations:
    - β = 0: coordinated flight (ball centered)
    - β ≠ 0: uncoordinated flight, creates side forces
    - Large β can lead to adverse handling characteristics
    
    """
    return jnp.arctan2(v, jnp.sqrt(u**2 + w**2))

@jax.jit
def calculate_gamma(u, v):
    """Calculate flight path angle from velocity components.
    
    Computes the flight path angle (γ), which is the angle between the
    velocity vector and the horizontal plane. This represents the aircraft's
    climb or descent angle relative to the horizontal.
    
    Parameters
    ----------
    u : jax.Array
        Forward velocity component [m/s]
    v : jax.Array
        Vertical velocity component [m/s]
        (Note: This may be w in some conventions)
        
    Returns
    -------
    jax.Array
        Flight path angle [rad]
        - Positive γ: climbing flight
        - Negative γ: descending flight
        - Zero γ: level flight
        
    Notes
    -----
    
    γ = arctan2(v, u)
    
    This differs from pitch angle (θ), as flight path angle is defined
    relative to the velocity vector, while pitch angle is relative to
    the aircraft's longitudinal axis.
    
    Relationship: γ = θ - α (approximately for small angles)
    
    """
    return jnp.arctan2(v, u)

@jax.jit
def build_wind_velocity_vector(Vwx, Vwy, Vwz):
    """Build wind velocity vector in inertial reference frame.
    
    Constructs a 3-element vector representing atmospheric wind velocity
    components in the inertial (Earth-fixed) coordinate system.
    
    Parameters
    ----------
    Vwx : jax.Array
        North wind velocity component [m/s]. Positive indicates wind 
        blowing toward the north.
    Vwy : jax.Array
        East wind velocity component [m/s]. Positive indicates wind
        blowing toward the east.
    Vwz : jax.Array
        Vertical wind velocity component [m/s]. Positive indicates
        upward wind (updraft) in NED convention.
        
    Returns
    -------
    jax.Array
        3-element wind velocity vector [Vwx, Vwy, Vwz] in inertial frame [m/s]
        
    Notes
    -----
    Wind effects are crucial for:
    - Ground speed calculations
    - Navigation accuracy
    - Fuel consumption estimates
    - Flight planning and optimization
    - Turbulence and gust modeling
    
    The wind vector is used to transform between:
    - Airspeed (velocity relative to air mass)
    - Groundspeed (velocity relative to Earth)
    
    Relationship: V_ground = V_air + V_wind
    
    """
    return jnp.stack([Vwx, Vwy, Vwz])

@jax.jit
def calculate_alpha_dot(alpha, wDot, uDot, VTAS, beta):
    """Calculate time derivative of angle of attack.
    
    Computes the rate of change of angle of attack (α̇) using the kinematic
    relationship derived from the definition of angle of attack and the
    aircraft's velocity dynamics.
    
    Parameters
    ----------
    alpha : jax.Array
        Current angle of attack [rad]
    wDot : jax.Array
        Time derivative of vertical velocity component [m/s²]
    uDot : jax.Array
        Time derivative of forward velocity component [m/s²]
    VTAS : jax.Array
        True airspeed magnitude [m/s]
    beta : jax.Array
        Sideslip angle [rad]
        
    Returns
    -------
    jax.Array
        Time derivative of angle of attack [rad/s]
        
    Notes
    -----
    The angle of attack rate is derived from:
    α̇ = (cos(α) * ẇ - sin(α) * u̇) / (VTAS * cos(β))
    
    The denominator VTAS * cos(β) represents the velocity component
    in the longitudinal-vertical plane, ensuring proper normalization.
    
    Physical interpretation:
    - Positive α̇: increasing angle of attack (nose rising relative to airflow)
    - Negative α̇: decreasing angle of attack (nose lowering relative to airflow)
    

    """
    return (jnp.cos(alpha) * wDot - jnp.sin(alpha) * uDot) / (VTAS * jnp.cos(beta))


@jax.jit
def build_statesdot_vector(linear_acceleration_vector, angular_acceleration_vector, 
                          angular_rates_vector, body_frame_velocity_vector):
    """Build complete state derivative vector for numerical integration.
    
    Constructs the time derivative of the full aircraft state vector by
    concatenating acceleration and rate vectors. This vector is used in
    numerical integration schemes to advance the aircraft state in time.
    
    Parameters
    ----------
    linear_acceleration_vector : jax.Array
        3-element vector of linear acceleration components [u̇, v̇, ẇ] [m/s²]
        in body-fixed frame
    angular_acceleration_vector : jax.Array
        3-element vector of angular acceleration components [ṗ, q̇, ṙ] [rad/s²]
        in body-fixed frame
    angular_rates_vector : jax.Array
        3-element vector of current angular rates [p, q, r] [rad/s]
        used for Euler angle rate computation
    body_frame_velocity_vector : jax.Array
        3-element vector of velocity components [u, v, w] [m/s]
        in body-fixed frame
        
    Returns
    -------
    jax.Array
        12-element state derivative vector [u̇, v̇, ẇ, ṗ, q̇, ṙ, φ̇, θ̇, ψ̇, ẋ, ẏ, ż]
        
    Notes
    -----
    The state derivative vector ordering follows:
    [linear_accelerations, angular_accelerations, euler_rate_equations, position_rates]
    
    """
    return jnp.concatenate([
        linear_acceleration_vector, 
        angular_acceleration_vector, 
        angular_rates_vector, 
        body_frame_velocity_vector
    ])

@jax.jit
def compute_rotation_matrix(phi, theta, psi):
    """Compute rotation matrix from body to inertial frame."""
    # Rotation matrices for each axis
    Rx = jnp.array([[1, 0, 0],
                    [0, jnp.cos(phi), -jnp.sin(phi)],
                    [0, jnp.sin(phi), jnp.cos(phi)]])
    
    Ry = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                    [0, 1, 0],
                    [-jnp.sin(theta), 0, jnp.cos(theta)]])
    
    Rz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                    [jnp.sin(psi), jnp.cos(psi), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix (ZYX sequence)
    return Rz @ Ry @ Rx

@jax.jit
def aircraft_states(phi, theta, psi, x, y, z, u, v, w, p, q, r):
    """Create aircraft states as a dictionary of JAX arrays.
    
    Args:
        phi, theta, psi: Euler angles
        x, y, z: Position components
        u, v, w: Velocity components in body frame
        p, q, r: Angular velocity components in body frame  
        
    Returns:
        Dictionary containing all state variables and derived quantities
    """
    state_vector = build_state_vector(u, v, w, p, q, r, phi, theta, psi, x, y, z)
    angular_rates_vector = build_angular_rates_vector(p, q, r)
    position_vector = build_position_vector(x, y, z)
    euler_angles_vector = build_euler_angles_vector(phi, theta, psi)
    VTAS = calculate_VTAS(u, v, w)
    alpha = calculate_alpha(u, w)
    beta = calculate_beta(u, v, w)
    gamma = calculate_gamma(u, v)
    
    return {
        # Primary state variables
        'u': u, 'v': v, 'w': w,
        'p': p, 'q': q, 'r': r,
        'phi': phi, 'theta': theta, 'psi': psi,
        'x': x, 'y': y, 'z': z,
        
        # Derived state vectors
        'state_vector': state_vector,
        'angular_rates_vector': angular_rates_vector,
        'position_vector': position_vector,
        'euler_angles_vector': euler_angles_vector,
        
        # Flight dynamics quantities
        'VTAS': VTAS,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }




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


        
    def __init__(self,
                 axis: Union[Axis, AxisLsdoGeo],
                 u: Any = Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 v: Any = Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 w: Any = Q_(0, 'm/s'),     # WRT To Body-Fixed Frame
                 p: Any = Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 q: Any = Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 r: Any = Q_(0, 'rad/s'),   # WRT To Body-Fixed Frame
                 Vwx: Any = Q_(0, 'm/s'), # WRT To Inertial Frame
                 Vwy: Any = Q_(0, 'm/s'), # WRT To Inertial Frame
                 Vwz: Any = Q_(0, 'm/s'), # WRT To Inertial Frame
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
        
        # Process input values to JAX arrays
        u_jax = process_input_value(u)
        v_jax = process_input_value(v)
        w_jax = process_input_value(w)
        p_jax = process_input_value(p)
        q_jax = process_input_value(q)
        r_jax = process_input_value(r)
        Vwx_jax = process_input_value(Vwx)
        Vwy_jax = process_input_value(Vwy)
        Vwz_jax = process_input_value(Vwz)
        
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


        # Use the JAX functional approach to create aircraft states
        self.aircraft_states_dict = aircraft_states(
            self.axis.euler_angles.phi[0],
            self.axis.euler_angles.theta[0], 
            self.axis.euler_angles.psi[0],
            self.axis.translation_from_origin.x[0],
            self.axis.translation_from_origin.y[0],
            self.axis.translation_from_origin.z[0],
            u_jax[0], v_jax[0], w_jax[0], p_jax[0], q_jax[0], r_jax[0]
        )


        self.states = self.States6dof(
            u=u_jax, v=v_jax, w=w_jax,
            p=p_jax, q=q_jax, r=r_jax,
            x=axis.translation_from_origin.x, y=axis.translation_from_origin.y, z=axis.translation_from_origin.z,
            phi=axis.euler_angles.phi, theta=axis.euler_angles.theta, psi=axis.euler_angles.psi
        )

        self.states_inertial_frame_wind = self.StatesInertialFrameWindVelocityVector(
            Vwx=Vwx_jax, Vwy=Vwy_jax, Vwz=Vwz_jax
        )

        # Use JAX functional approach results for vector computations
        self.body_frame_velocity_vector = jnp.stack([u_jax[0], v_jax[0], w_jax[0]])
        self.inertial_frame_wind_velocity_vector = build_wind_velocity_vector(
            Vwx_jax[0], Vwy_jax[0], Vwz_jax[0]
        )
        self.angular_rates_vector = self.aircraft_states_dict['angular_rates_vector']
        self.position_vector = self.aircraft_states_dict['position_vector']
        self.euler_angles_vector = self.aircraft_states_dict['euler_angles_vector']

        self.states_vector = self.aircraft_states_dict['state_vector']

        # Initialize acceleration vectors
        self.linear_acceleration = type('LinearAcceleration', (), {
            'uDot': jnp.array([0.0]),
            'vDot': jnp.array([0.0]),
            'wDot': jnp.array([0.0])
        })()
        self.linear_acceleration_vector = jnp.stack([
            self.linear_acceleration.uDot[0], 
            self.linear_acceleration.vDot[0], 
            self.linear_acceleration.wDot[0]
        ])

        self.angular_acceleration = type('AngularAcceleration', (), {
            'pDot': jnp.array([0.0]),
            'qDot': jnp.array([0.0]),
            'rDot': jnp.array([0.0])
        })()
        self.angular_acceleration_vector = jnp.stack([
            self.angular_acceleration.pDot[0], 
            self.angular_acceleration.qDot[0], 
            self.angular_acceleration.rDot[0]
        ])

        self.statesdot_vector = build_statesdot_vector(
            self.linear_acceleration_vector, 
            self.angular_acceleration_vector, 
            self.angular_rates_vector, 
            self.body_frame_velocity_vector
        )

        # Aerodynamic computations using JAX functions
        self.VTAS = calculate_VTAS(u_jax[0], v_jax[0], w_jax[0])

        self.alpha = calculate_alpha(u_jax[0], w_jax[0])

        self.beta = calculate_beta(u_jax[0], v_jax[0], w_jax[0])

        self.windAxis = Axis(
            name='Wind Axis',
            x=Q_(0, 'ft'),
            y=Q_(0, 'ft'),
            z=Q_(0, 'ft'),
            phi=Q_(0, 'deg'),
            theta=Q_(float(self.alpha), 'rad'),
            psi=Q_(float(-self.beta), 'rad'),
            sequence=np.array([3, 2, 1]),
            reference=self.axis,
            origin=ValidOrigins.Inertial.value
        )


        self.alpha_dot = calculate_alpha_dot(self.alpha, self.linear_acceleration.wDot, self.linear_acceleration.uDot, self.VTAS, self.beta)

        self.gamma = calculate_gamma(u_jax[0], w_jax[0])

        self.sigma = self.beta - self.axis.euler_angles.psi

        self.Mach = self.VTAS / self.atmospheric_states.speed_of_sound

        # Ground Speed - compute rotation matrix and transform using JAX functions
        R_B_to_I = compute_rotation_matrix(
            self.windAxis.euler_angles.phi[0], 
            self.windAxis.euler_angles.theta[0], 
            self.windAxis.euler_angles.psi[0]
        )
        self.inertial_velocity_vector = (
            R_B_to_I @ self.body_frame_velocity_vector + 
            self.inertial_frame_wind_velocity_vector
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
    
    # Demonstrate the JIT-compiled function for performance-critical applications
    print(f"Assembled State Vector: {aircraft_states.states_vector}")
