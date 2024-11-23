from abc import ABC, abstractmethod

from . import ForcesMoments
from . import Q_
from . import Type
from . import np
from .Axis import Axis, build_rotation_matrix
from .Environment import StandardAtmosphereMixin, ConstantWindMixin


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
    def __init__(self, state_axis, x=Q_(1, 'm'), x_dot=Q_(0, 'm/s')):
        super().__init__(state_axis=state_axis)
        self._x = x
        self._x_dot = x_dot

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x_value):
        self._x = x_value.to_base_units()

    @property
    def x_dot(self):
        return self._x_dot

    @x_dot.setter
    def x_dot(self, x_dot_value):
        self._x_dot = x_dot_value.to_base_units()

    def update(self, t, input_vector):
        # Potential t-based property updates
        self.update_state_from_vector(input_vector)

    def update_state_from_vector(self, input_vector):
        self.x = Q_(input_vector[0], 'm')
        self.x_dot = Q_(input_vector[1], 'm/s')

    def return_state_vector(self):
        return np.array([self.x.magnitude, self.x_dot.magnitude], dtype=float)


class VehicleState(RigidBodyStates, StandardAtmosphereMixin):
    def __init__(self, Mach: float,
                 xref: Type[Q_], yref: Type[Q_], altitude: Type[Q_],
                 state_axis: Axis,
                 u=Q_(1, 'm/s'), v=Q_(0, 'm/s'), w=Q_(0, 'm/s'),
                 p=Q_(0, 'rad/s'), q=Q_(0, 'rad/s'), r=Q_(0, 'rad/s'),
                 Phi=Q_(0, 'deg'), Theta=Q_(0, 'deg'), Psi=Q_(0, 'deg'),
                 u_dot=Q_(0, 'm/(s*s)'), v_dot=Q_(0, 'm/(s*s)'), w_dot=Q_(0, 'm/(s*s)'),
                 p_dot=Q_(0, 'rad/(s*s)'), q_dot=Q_(0, 'rad/(s*s)'), r_dot=Q_(0, 'rad/(s*s)')):
        """
        Vehicle state will be defined at a minimum with Mach number, altitude
        and a reference point.
        Vehicle state will contain body-fixed states as independent properties.
        Wind-fixed states will be treated as dependent parameters.
        :param Mach:
        :param altitude:
        """
        super().__init__(state_axis)
        self._u, self._v, self._w = u.to_base_units(), v.to_base_units(), w.to_base_units()
        self._p, self._q, self._r = p.to_base_units(), q.to_base_units(), r.to_base_units()
        self._Phi = Phi.to_base_units()
        self._Theta = Theta.to_base_units()
        self._Psi = Psi.to_base_units()
        self._X = xref.to_base_units()
        self._Y = yref.to_base_units()
        self._Z = altitude.to_base_units()
        self._u_dot = u_dot.to_base_units()
        self._v_dot = v_dot.to_base_units()
        self._w_dot = w_dot.to_base_units()
        self._p_dot = p_dot.to_base_units()
        self._q_dot = q_dot.to_base_units()
        self._r_dot = r_dot.to_base_units()

        self.Mach = Mach

        self._Phi_W = Q_(0, 'rad')

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u_val):
        self._u = u_val.to_base_units()

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v_val):
        self._v = v_val.to_base_units()

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w_val):
        self._w = w_val.to_base_units()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p_value):
        self._p = p_value.to_base_units()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q_value):
        self._q = q_value.to_base_units()

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r_value):
        self._r = r_value.to_base_units()

    @property
    def u_dot(self):
        return self._u_dot

    @u_dot.setter
    def u_dot(self, u_dot_value):
        self._u_dot = u_dot_value.to_base_units()

    @property
    def v_dot(self):
        return self._v_dot

    @v_dot.setter
    def v_dot(self, v_dot_value):
        self._v_dot = v_dot_value.to_base_units()

    @property
    def w_dot(self):
        return self._w_dot

    @w_dot.setter
    def w_dot(self, w_dot_value):
        self._w_dot = w_dot_value.to_base_units()

    @property
    def p_dot(self):
        return self._p_dot

    @p_dot.setter
    def p_dot(self, p_dot_value):
        self._p_dot = p_dot_value.to_base_units()

    @property
    def q_dot(self):
        return self._q_dot

    @q_dot.setter
    def q_dot(self, q_dot_value):
        self._q_dot = q_dot_value.to_base_units()

    @property
    def r_dot(self):
        return self._r

    @r_dot.setter
    def r_dot(self, r_dot_value):
        self._r_dot = r_dot_value.to_base_units()

    @property
    def Phi(self):
        return self._Phi

    @Phi.setter
    def Phi(self, phi_value):
        self._Phi = phi_value.to_base_units()
        self.update_state_axis()

    @property
    def Theta(self):
        return self._Theta

    @Theta.setter
    def Theta(self, theta_value):
        self._Theta = theta_value.to_base_units()
        self.update_state_axis()

    @property
    def Psi(self):
        return self._Psi

    @Psi.setter
    def Psi(self, psi_value):
        self._Psi = psi_value.to_base_units()
        self.update_state_axis()

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, xref_value):
        self._X = xref_value.to_base_units()
        self.update_state_axis()

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, yref_value):
        self._Y = yref_value.to_base_units()
        self.update_state_axis()

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, altitude_value):
        self._Z = altitude_value.to_base_units()
        self.update_state_axis()

    @property
    def VTAS(self):
        return np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)

    @VTAS.setter
    def VTAS(self, V_value):
        V_value = V_value.to_base_units()
        VTAS = self.VTAS
        self.u = self.u * V_value / VTAS
        self.v = self.v * V_value / VTAS
        self.w = self.w * V_value / VTAS

    @property
    def alpha(self):
        return np.arctan2(self.w, self.u)

    @alpha.setter
    def alpha(self, alpha_value):
        alpha_value = alpha_value.to_base_units()
        V_const = self.VTAS
        beta_const = self.beta
        self.u = V_const * np.cos(beta_const) * np.cos(alpha_value)
        self.v = V_const * np.sin(beta_const)
        self.w = V_const * np.cos(beta_const) * np.sin(alpha_value)

    @property
    def beta(self):
        return np.arcsin(self.v / self.VTAS)

    @beta.setter
    def beta(self, beta_value):
        beta_value = beta_value.to_base_units()
        V_const = self.VTAS
        alpha_const = self.alpha
        self.u = V_const * np.cos(beta_value) * np.cos(alpha_const)
        self.v = V_const * np.sin(beta_value)
        self.w = V_const * np.cos(beta_value) * np.sin(alpha_const)

    @property
    def Phi_W(self):
        return self._Phi_W

    @Phi_W.setter
    def Phi_W(self, Phi_W_value):
        self._Phi_W = Phi_W_value.to_base_units()

    @property
    def gamma(self):
        return self.Theta - self.alpha

    @gamma.setter
    def gamma(self, gamma_value):
        # EH - Theta and [uvw] treated as independent, so direct setting of gamma will imply a change in theta at constant speed (i.e angle of attack)
        self.Theta = self.alpha + gamma_value.to_base_units()

    @property
    def Psi_W(self):
        return self.beta - self.Psi

    @Psi_W.setter
    def Psi_W(self, Psi_W_value):
        # EH - Psi and [uvw] is treated as independent, so direct setting of Psi_W will imply a change Psi at constant speed (i.e. sideslip angle)
        self.Psi = self.beta - Psi_W_value.to_base_units()

    @property
    def Mach(self):
        return self.VTAS / self.atmosphere_properties['a']

    @Mach.setter
    def Mach(self, mach_value):
        self.VTAS = mach_value * self.atmosphere_properties['a']

    def get_Position_BodyFixed(self):
        return np.array([self.X.magnitude, self.Y.magnitude, self.Z.magnitude])

    def update_position(self, x_new, y_new, z_new):
        self.X = x_new
        self.Y = y_new
        self.Z = z_new
        pass

    def get_Attitude_BodyFixed(self):
        return np.array([self.Phi.magnitude, self.Theta.magnitude, self.Psi.magnitude])

    def update_attitude_bodyfixed(self, Phi_new, Theta_new, Psi_new):
        self.Phi = Phi_new
        self.Theta = Theta_new
        self.Psi = Psi_new

    def get_Attitude_Wind(self):
        return np.array([self.Phi_W.magnitude, self.gamma.magnitude, self.Psi_W.magnitude])

    def update_attitude_wind(self, Phi_W_new, gamma_new, Psi_W_new):
        self.Phi_W = Phi_W_new
        self.gamma = gamma_new
        self.Psi_W = Psi_W_new

    def get_AngRates(self):
        return np.array([self.p.magnitude, self.q.magnitude, self.r.magnitude])

    def update_angular_rates(self, p_new, q_new, r_new):
        self.p = p_new
        self.q = q_new
        self.r = r_new

    def update_state_axis(self):
        translations = [self.X.magnitude, self.Y.magnitude, self.Z.magnitude]
        angles = [self.Phi.magnitude, self.Theta.magnitude, self.Psi.magnitude]
        self.state_axis.translation = Q_(np.array(translations), 'm')
        self.state_axis.angles = Q_(np.array(angles), 'rad')

    def return_structural_bodyfixed_axis(self):
        FDbodyFixedAxis = self.return_fd_bodyfixed_axis()  # needs to call FD Body Fixed Axis to ensure correct orientation
        bodyFixedAxis = Axis(name='Structural Body Fixed Axis',
                             translation=Q_(np.asarray([0, 0, 0]), 'm'),
                             angles=Q_(np.asarray([0, -180, 0]), 'deg'),
                             sequence=np.array([3, 2, 1]),
                             reference=FDbodyFixedAxis)
        return bodyFixedAxis

    def return_fd_bodyfixed_axis(self):
        self.update_state_axis()
        return self.state_axis

    def return_wind_axis(self):
        FDbodyFixedAxis = self.return_fd_bodyfixed_axis()

        windAxis = Axis(name='Wind Axis',
                        translation=Q_(np.asarray([0, 0, 0]), 'm'),
                        angles=Q_(np.asarray([0,
                                              self.alpha.magnitude,
                                              -self.beta.magnitude]), 'rad'),
                        sequence=np.array([3, 2, 1]),
                        reference=FDbodyFixedAxis
                        )
        return windAxis

    def transfer_to_axis(self, name, translation, angles, origin):
        # Transform vehicle state to identical state expressed in new axis system. Applies rigid-body transformations
        # only - if effect assumes that new axis system is attached to a point on the rigid body
        FDbodyFixedAxis = self.return_fd_bodyfixed_axis()
        new_axis = Axis(name=name,
                        translation=translation,
                        angles=angles,
                        sequence=np.array([3, 2, 1]),
                        origin=origin,
                        reference=FDbodyFixedAxis)
        new_axis_rotation = build_rotation_matrix(angles=new_axis.angles, seq=new_axis.sequence)
        new_state_obj = VehicleState(Mach=self.Mach, xref=self.X + translation[0],
                                     yref=self.Y + translation[1],
                                     altitude=self.Z + translation[2],
                                     state_axis=new_axis)

        radius = translation.magnitude
        velocity = np.array([self.u.magnitude, self.v.magnitude, self.w.magnitude])
        velocity_dot = np.array([self.u_dot.magnitude, self.v_dot.magnitude, self.w_dot.magnitude])
        omega = np.array([self.p.magnitude, self.q.magnitude, self.r.magnitude])
        omega_dot = np.array([self.p_dot.magnitude, self.q_dot.magnitude, self.r_dot.magnitude])

        velocity_new = velocity + np.cross(omega, radius)
        velocity_dot_new = velocity_dot + np.cross(omega_dot, radius) + np.cross(omega, np.cross(omega, radius))
        # All points on a rigid body have the same angular velocity and angular acceleration
        omega_new = omega
        omega_dot_new = omega_dot

        velocity_new = np.dot(new_axis_rotation, velocity_new)
        velocity_dot_new = np.dot(new_axis_rotation, velocity_dot_new)  # Assumes no relative rotation between axes
        omega_new = np.dot(new_axis_rotation, omega_new)
        omega_dot_new = np.dot(new_axis_rotation, omega_dot_new)

        new_state_obj.u = Q_(velocity_new[0], 'm/s')
        new_state_obj.v = Q_(velocity_new[1], 'm/s')
        new_state_obj.w = Q_(velocity_new[2], 'm/s')
        new_state_obj.p = Q_(omega_new[0], 'rad/s')
        new_state_obj.q = Q_(omega_new[1], 'rad/s')
        new_state_obj.r = Q_(omega_new[2], 'rad/s')
        new_state_obj.Phi = new_axis.angles[0]
        new_state_obj.Theta = new_axis.angles[1]
        new_state_obj.Psi = new_axis.angles[2]
        new_state_obj.u_dot = Q_(velocity_dot_new[0], 'm/s**2')
        new_state_obj.v_dot = Q_(velocity_dot_new[1], 'm/s**2')
        new_state_obj.w_dot = Q_(velocity_dot_new[2], 'm/s**2')
        new_state_obj.p_dot = Q_(omega_dot_new[0], 'rad/s**2')
        new_state_obj.q_dot = Q_(omega_dot_new[1], 'rad/s**2')
        new_state_obj.r_dot = Q_(omega_dot_new[2], 'rad/s**2')

        return new_state_obj

    def get_attitude_air(self):
        return np.asarray([self.alpha.magnitude, self.beta.magnitude])

    def set_attitude_air(self, value):
        self.alpha, self.beta = value[0], value[1]

    attitude_air = property(get_attitude_air, set_attitude_air)

    def get_velocity_bodyfixed(self):
        return np.array([self.u.magnitude, self.v.magnitude, self.w.magnitude])

    def set_velocity_bodyfixed(self, value):
        self.u, self.v, self.w = value[0], value[1], value[2]

    velocity_bodyfixed = property(get_velocity_bodyfixed, set_velocity_bodyfixed)

    @property
    def atmosphere_properties(self):
        atm = self.standard_atm()
        VTAS = self.VTAS
        qBar = 0.5 * atm['rho'] * VTAS ** 2
        return {'rho': atm['rho'], 'a': atm['a'], 'p': atm['p'], 'T': atm['T'], 'g': atm['g'], 'qBar': qBar}

    #    def set_atmosphere_properties(self, value):
    #        self.rho, self.a, self.p, self.T, self.g, \
    #        self.qBar = value[0], value[1], value[2], value[3], value[4], value[5]

    #    atmosphere_properties = property(get_atmosphere_properties, set_atmosphere_properties)

    def return_gravity_vector(self):
        g = ForcesMoments.Vector(x=-self.atmosphere_properties['g'] * np.sin(self.Theta.magnitude),
                                 y=self.atmosphere_properties['g'] * np.cos(self.Theta.magnitude) * np.sin(
                                     self.Phi.magnitude),
                                 z=self.atmosphere_properties['g'] * np.cos(self.Theta.magnitude) * np.cos(
                                     self.Phi.magnitude),
                                 axis=self.return_fd_bodyfixed_axis())
        return g

    def __str__(self):
        print_string = """Summary of Vehicle State
        States represent rigid body states in the %s 
        Vehicle Position: %.2f %s, %.2f %s, %.2f %s
        Vehicle Velocity: %.2f %s, %.2f %s, %.2f %s
        Vehicle Angular Velocity: %.2f %s, %.2f %s, %.2f %s
        Vehicle Attitude: %.2f %s, %.2f %s, %.2f %s
        Vehicle Mach Number: %.2f
        Vehicle Aerodynamic Angles: Angle of Attack: %.2f %s; Sideslip: %.2f %s""" % (self.state_axis.name,
                                                                                      self.X.magnitude, self.X.units,
                                                                                      self.Y.magnitude, self.Y.units,
                                                                                      self.Z.magnitude, self.Z.units,
                                                                                      self.u.magnitude, self.u.units,
                                                                                      self.v.magnitude, self.v.units,
                                                                                      self.w.magnitude, self.w.units,
                                                                                      self.p.magnitude, self.p.units,
                                                                                      self.q.magnitude, self.q.units,
                                                                                      self.r.magnitude, self.r.units,
                                                                                      self.Phi.magnitude,
                                                                                      self.Phi.units,
                                                                                      self.Theta.magnitude,
                                                                                      self.Theta.units,
                                                                                      self.Psi.magnitude,
                                                                                      self.Psi.units,
                                                                                      self.Mach,
                                                                                      self.alpha.magnitude,
                                                                                      self.alpha.units,
                                                                                      self.beta.magnitude,
                                                                                      self.beta.units)
        return print_string

    def update(self, t, input_vector):
        # Potential t-based property updates
        self.update_state_from_vector(input_vector)

    def update_state_from_vector(self, x_input):
        self.u = Q_(x_input[0], 'm/s')
        self.v = Q_(x_input[1], 'm/s')
        self.w = Q_(x_input[2], 'm/s')
        self.p = Q_(x_input[3], 'rad/s')
        self.q = Q_(x_input[4], 'rad/s')
        self.r = Q_(x_input[5], 'rad/s')
        self.Phi = Q_(x_input[6], 'rad')
        self.Theta = Q_(x_input[7], 'rad')
        self.Psi = Q_(x_input[8], 'rad')
        self.X = Q_(x_input[9], 'm')
        self.Y = Q_(x_input[10], 'm')
        self.Z = Q_(x_input[11], 'm')

    def return_state_vector(self):
        return np.array(
            [self.u.magnitude, self.v.magnitude, self.w.magnitude,
             self.p.magnitude, self.q.magnitude, self.r.magnitude,
             self.Phi.magnitude, self.Theta.magnitude, self.Psi.magnitude,
             self.X.magnitude, self.Y.magnitude, self.Z.magnitude], dtype=float)


class VehicleStateConstantWind(VehicleState, ConstantWindMixin):
    def __init__(self, Mach: float,
                 xref: Type[Q_], yref: Type[Q_], altitude: Type[Q_],
                 state_axis: Axis,
                 wind_direction=0, wind_speed=Q_(1, 'm/s'),
                 u_g=Q_(1, 'm/s'), v_g=Q_(0, 'm/s'), w_g=Q_(0, 'm/s'),
                 p=Q_(0, 'rad/s'), q=Q_(0, 'rad/s'), r=Q_(0, 'rad/s'),
                 Phi=Q_(0, 'deg'), Theta=Q_(0, 'deg'), Psi=Q_(0, 'deg'),
                 u_dot=Q_(0, 'm/(s*s)'), v_dot=Q_(0, 'm/(s*s)'), w_dot=Q_(0, 'm/(s*s)'),
                 p_dot=Q_(0, 'rad/(s*s)'), q_dot=Q_(0, 'rad/(s*s)'), r_dot=Q_(0, 'rad/(s*s)')):
        """
        Vehicle state will be defined at a minimum with Mach number, altitude
        and a reference point.
        Vehicle state will contain body-fixed states as independent properties.
        Wind-fixed states will be treated as dependent parameters.
        :param Mach:
        :param altitude:
        """
        self.wind_speed_ned = self.wind_speed(wind_direction, wind_speed)
        self._u_g, self._v_g, self._w_g = u_g.to_base_units(), v_g.to_base_units(), w_g.to_base_units()
        self._wind_u, self._wind_v, self._wind_w = Q_(0, 'm/s'), Q_(0, 'm/s'), Q_(0, 'm/s')
        super().__init__(Mach, xref, yref, altitude, state_axis=state_axis,
                         p=p, q=q, r=r, Phi=Phi, Theta=Theta, Psi=Psi,
                         u_dot=u_dot, v_dot=v_dot, w_dot=w_dot,
                         p_dot=p_dot, q_dot=q_dot, r_dot=r_dot)

        self.Mach = Mach

    @property
    def u_g(self):
        return self._u_g

    @u_g.setter
    def u_g(self, u_g_val):
        self._u_g = u_g_val.to_base_units()

    @property
    def v_g(self):
        return self._v_g

    @v_g.setter
    def v_g(self, v_g_val):
        self._v_g = v_g_val.to_base_units()

    @property
    def w_g(self):
        return self._w_g

    @w_g.setter
    def w_g(self, w_g_val):
        self._w_g = w_g_val.to_base_units()

    @property
    def wind_u(self):
        self._wind_u = self.wind_speed_ned[0] * np.cos(self.Theta) * np.cos(self.Psi) \
                       + self.wind_speed_ned[1] * np.cos(self.Theta) * np.sin(self.Psi) \
                       - self.wind_speed_ned[2] * np.sin(self.Theta)
        return self._wind_u

    @property
    def wind_v(self):
        self._wind_v = self.wind_speed_ned[0] * (np.sin(self.Phi) * np.sin(self.Theta) * np.cos(self.Psi)
                                                 + np.cos(self.Phi) * np.sin(self.Psi)) + self.wind_speed_ned[1] * (
                               np.sin(self.Phi) * np.sin(self.Theta) * np.sin(self.Psi)
                               + np.cos(self.Phi) * np.cos(self.Psi)) \
                       + self.wind_speed_ned[2] * np.sin(self.Phi) * np.cos(self.Theta)
        return self._wind_v

    @property
    def wind_w(self):
        self._wind_w = self.wind_speed_ned[0] * (
                np.cos(self.Phi) * np.sin(self.Theta) * np.cos(self.Psi) + np.sin(self.Phi) * np.sin(self.Psi)) + \
                       self.wind_speed_ned[1] * (
                               np.cos(self.Phi) * np.sin(self.Theta) * np.sin(self.Psi) - np.sin(self.Phi) * np.cos(
                           self.Psi)) \
                       + self.wind_speed_ned[2] * np.cos(self.Phi) * np.cos(self.Theta)
        return self._wind_w

    @property
    def u(self):
        return self.u_g - self.wind_u

    @u.setter
    def u(self, u_val):
        self.u_g = u_val.to_base_units() + self.wind_u

    @property
    def v(self):
        return self.v_g - self.wind_v

    @v.setter
    def v(self, v_val):
        self.v_g = v_val.to_base_units() + self.wind_v

    @property
    def w(self):
        return self.w_g - self.wind_w

    @w.setter
    def w(self, w_val):
        self.w_g = w_val.to_base_units() + self.wind_w

    def __str__(self):
        print_string = """Summary of Vehicle State
        States represent rigid body states in the %s 
        Vehicle Position: %.2f %s, %.2f %s, %.2f %s
        Vehicle Airspeed: %.2f %s, %.2f %s, %.2f %s
        Vehicle Ground Speed: %.2f %s, %.2f %s, %.2f %s
        Wind Speed: %.2f %s, %.2f %s, %.2f %s
        Vehicle Angular Velocity: %.2f %s, %.2f %s, %.2f %s
        Vehicle Attitude: %.2f %s, %.2f %s, %.2f %s
        Vehicle Mach Number: %.2f
        Vehicle Aerodynamic Angles: Angle of Attack: %.2f %s; Sideslip: %.2f %s""" % (self.state_axis.name,
                                                                                      self.X.magnitude, self.X.units,
                                                                                      self.Y.magnitude, self.Y.units,
                                                                                      self.Z.magnitude, self.Z.units,
                                                                                      self.u.magnitude, self.u.units,
                                                                                      self.v.magnitude, self.v.units,
                                                                                      self.w.magnitude, self.w.units,
                                                                                      self.u_g.magnitude,
                                                                                      self.u_g.units,
                                                                                      self.v_g.magnitude,
                                                                                      self.v_g.units,
                                                                                      self.w_g.magnitude,
                                                                                      self.w_g.units,
                                                                                      self.wind_u.magnitude,
                                                                                      self.wind_u.units,
                                                                                      self.wind_v.magnitude,
                                                                                      self.wind_v.units,
                                                                                      self.wind_w.magnitude,
                                                                                      self.w_g.units,
                                                                                      self.p.magnitude, self.p.units,
                                                                                      self.q.magnitude, self.q.units,
                                                                                      self.r.magnitude, self.r.units,
                                                                                      self.Phi.magnitude,
                                                                                      self.Phi.units,
                                                                                      self.Theta.magnitude,
                                                                                      self.Theta.units,
                                                                                      self.Psi.magnitude,
                                                                                      self.Psi.units,
                                                                                      self.Mach,
                                                                                      self.alpha.magnitude,
                                                                                      self.alpha.units,
                                                                                      self.beta.magnitude,
                                                                                      self.beta.units)
        return print_string