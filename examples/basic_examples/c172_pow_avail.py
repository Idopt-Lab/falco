import numpy as np
from scipy.interpolate import Akima1DInterpolator, RectBivariateSpline

from falco import Q_, ureg
from falco.core.dynamics.aircraft_states import AircraftStates
from falco.core.dynamics.vector import Vector
from falco.core.loads.forces_moments import ForcesMoments
from falco.core.loads.loads import Loads
from falco.core.vehicle.conditions import aircraft_conditions
from falco.core.vehicle.conditions.aircraft_conditions import CruiseCondition, ClimbCondition
from falco.core.vehicle.controls.vehicle_control_system import (
    VehicleControlSystem, ControlSurface, PropulsiveControl)
from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.loads.mass_properties import MassProperties, MassMI
from falco.core.vehicle.components.component import Component
from falco.core.vehicle.components.aircraft import Aircraft
from typing import Union
from typing import List
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem, SLSQP, IPOPT
import time


# Every CSDl code starts with a recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# region Axis

# region General Parameters
h = 8000
Throttle = 0.94863
PropRadius = 0.94902815
Range = 10000
M_Aircraft = 1043.2616
delta_elevator = 0
flight_path_angle = 0
Speed = 50
Pitch = 0
# endregion

# region Inertial Axis
# I am picking the inertial axis location as the OpenVSP (0,0,0)
inertial_axis = Axis(
    name='Inertial Axis',
    origin=ValidOrigins.Inertial.value
)
# endregion

# region Aircraft FD Axis

fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    x=Q_(0, 'ft'),
    y=Q_(0, 'ft'),
    z=Q_(-h, 'ft'),  # z is positive down in FD axis
    phi=Q_(0, 'deg'),
    theta=Q_(0, 'deg'),
    psi=Q_(0, 'deg'),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
# endregion

# region Aircraft Component

aircraft_component = Aircraft()

# endregion

# region Fuselage Component
fuselage_component = Component(name='Fuselage')
aircraft_component.add_subcomponent(fuselage_component)

# region Engine Component
engine_axis = Axis(
    name='Engine Axis',
    x=Q_(0, 'ft'),
    y=Q_(0, 'ft'),
    z=Q_(0, 'ft'),  # z is positive down in FD axis
    phi=Q_(0, 'deg'),
    theta=Q_(0, 'deg'),
    psi=Q_(0, 'deg'),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)

engine_component = Component(name='Engine')
c172_engine_mass_properties = MassProperties(mass=Q_(0, 'kg'),
                                             inertia=MassMI(axis=engine_axis,
                                                            Ixx=Q_(0, 'kg*(m*m)'),
                                                            Iyy=Q_(0, 'kg*(m*m)'),
                                                            Izz=Q_(0, 'kg*(m*m)'),
                                                            Ixy=Q_(0, 'kg*(m*m)'),
                                                            Ixz=Q_(0, 'kg*(m*m)'),
                                                            Iyz=Q_(0, 'kg*(m*m)')),
                                             cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'),
                                                       axis=engine_axis))
engine_component.mass_properties = c172_engine_mass_properties

radius_c172 = csdl.Variable(name='prop_radius', shape=(1,), value=0.94)
engine_component.parameters.radius = radius_c172
fuselage_component.add_subcomponent(engine_component)
# endregion
# endregion

# region Wing Component

wing_axis = Axis(
    name='Wing Axis',
    x=Q_(0, 'ft'),
    y=Q_(0, 'ft'),
    z=Q_(0, 'ft'),  # z is positive down in FD axis
    phi=Q_(0, 'deg'),
    theta=Q_(0, 'deg'),
    psi=Q_(0, 'deg'),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
c172_wing_mass_properties = MassProperties(mass=Q_(0, 'kg'),
                                           inertia=MassMI(axis=wing_axis,
                                                          Ixx=Q_(0, 'kg*(m*m)'),
                                                          Iyy=Q_(0, 'kg*(m*m)'),
                                                          Izz=Q_(0, 'kg*(m*m)'),
                                                          Ixy=Q_(0, 'kg*(m*m)'),
                                                          Ixz=Q_(0, 'kg*(m*m)'),
                                                          Iyz=Q_(0, 'kg*(m*m)')),
                                           cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'),
                                                     axis=wing_axis))

wing_component = Component(name='Wing')
wing_component.mass_properties = c172_wing_mass_properties
aircraft_component.add_subcomponent(wing_component)
# endregion

# region Control Surface Components
rudder_component = Component(name='Rudder')
rudder_component.parameters.actuate_angle = csdl.Variable(name="Rudder Actuate Angle", shape=(1,), value=np.deg2rad(0))
# aircraft_component.add_subcomponent(rudder_component)

elevator_component = Component(name='Elevator')
elevator_component.parameters.actuate_angle = csdl.Variable(name="Elevator Actuate Angle", shape=(1,), value=np.deg2rad(delta_elevator))
# aircraft_component.add_subcomponent(elevator_component)

right_aileron_component = Component(name='Right Aileron')
right_aileron_component.parameters.actuate_angle = csdl.Variable(name="Right Aileron Actuate Angle", shape=(1,), value=np.deg2rad(0))
wing_component.add_subcomponent(right_aileron_component)
# endregion

# region Aircraft Controls
class C172Control(VehicleControlSystem):

    def __init__(self, elevator_component, aileron_right_component, aileron_left_component, rudder_component, symmetrical: bool = True):
        self.symmetrical = symmetrical

        self.elevator = ControlSurface('elevator_left', lb=-26, ub=28, component=elevator_component)
        if not symmetrical:
            self.aileron_left = ControlSurface('aileron_left', lb=-15, ub=20, component=aileron_right_component)
            self.aileron_right = ControlSurface('aileron_right', lb=-15, ub=20, component=aileron_left_component)
        else:
            self.aileron = ControlSurface('aileron', lb=-15, ub=20, component=aileron_right_component)
        self.rudder = ControlSurface('rudder', lb=-16, ub=16, component=rudder_component)

        self.engine = PropulsiveControl(name='engine', throttle=Throttle)

        if symmetrical:
            self.u = csdl.concatenate((self.aileron.deflection,
                                      -self.aileron.deflection,
                                      self.elevator.deflection,
                                      self.rudder.deflection,
                                      self.engine.throttle), axis=0)
        else:
            self.u = csdl.concatenate((self.aileron_left.deflection,
                                      self.aileron_right.deflection,
                                      self.elevator.deflection,
                                      self.rudder.deflection,
                                      self.engine.throttle), axis=0)

        if symmetrical:
            super().__init__(pitch_control=[self.elevator],
                             roll_control=[self.aileron],
                             yaw_control=[self.rudder],
                             throttle_control=[self.engine])
        else:
            super().__init__(pitch_control=[self.elevator],
                             roll_control=[self.aileron_left, self.aileron_right],
                             yaw_control=[self.rudder],
                             throttle_control=[self.engine])

    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']

    @property
    def lower_bounds(self):
        lb_elevator = self.elevator.lower_bound
        lb_rudder = self.rudder.lower_bound
        lb_thr = self.engine.lower_bound

        if self.symmetrical:
            lb_aileron_left = self.aileron.lower_bound
            lb_aileron_right = self.aileron.lower_bound
            return np.array([lb_aileron_left, lb_aileron_right, lb_elevator, lb_rudder, lb_thr])
        else:
            lb_aileron_left = self.aileron_left.lower_bound
            lb_aileron_right = self.aileron_right.lower_bound
            return np.array([lb_aileron_left, lb_aileron_right, lb_elevator, lb_rudder, lb_thr])

    @property
    def upper_bounds(self):
        ub_elevator = self.elevator.upper_bound
        ub_rudder = self.rudder.upper_bound
        ub_thr = self.engine.upper_bound

        if self.symmetrical:
            ub_aileron_left = self.aileron.upper_bound
            ub_aileron_right = self.aileron.upper_bound
            return np.array([ub_aileron_left, ub_aileron_right, ub_elevator, ub_rudder, ub_thr])
        else:
            ub_aileron_left = self.aileron_left.upper_bound
            ub_aileron_right = self.aileron_right.upper_bound
            return np.array([ub_aileron_left, ub_aileron_right, ub_elevator, ub_rudder, ub_thr])


c172_controls = C172Control(symmetrical=True, elevator_component=elevator_component, aileron_right_component=right_aileron_component,
                            aileron_left_component=None, rudder_component=rudder_component)
pass
# endregion


# region Aerodynamic Model
class AeroCurve(csdl.CustomExplicitOperation):
    def __init__(self):
        super().__init__()

        # Curves for aerodynamic constants
        # Controls & state input
        self.alpha_data = np.array([-7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 17, 18, 19.5])  # degree
        self.delta_aile_data = np.array([-15, -10, -5, -2.5, 0, 5, 10, 15, 20])  # degree
        self.delta_elev_data = np.array([-26, -20, -10, -5, 0, 7.5, 15, 22.5, 28])  # degree
        # Ouptuts
        # CD
        """Initialize all drag coefficient splines"""
        CD_data = np.array([0.044, 0.034, 0.03, 0.03, 0.036, 0.048, 0.067, 0.093, 0.15, 0.169, 0.177, 0.184])
        self.CD = Akima1DInterpolator(self.alpha_data, CD_data, method="akima")
        self.CD_derivative = self.CD.derivative()
        # CD_elevator_influence
        CD_delta_elev_data = np.array(
            [[0.0135, 0.0119, 0.0102, 0.00846, 0.0067, 0.0049, 0.00309, 0.00117, -0.0033, -0.00541, -0.00656, -0.00838],
             [0.0121, 0.0106, 0.00902, 0.00738, 0.00574, 0.00406, 0.00238, 0.00059, -0.00358, -0.00555, -0.00661,
              -0.00831],
             [0.00651, 0.00552, 0.00447, 0.00338, 0.00229, 0.00117, 0.0000517, -0.00114, -0.00391, -0.00522, -0.00593,
              -0.00706],
             [0.00249, 0.002, 0.00147, 0.000931, 0.000384, -0.000174, -0.000735, -0.00133, -0.00272, -0.00337, -0.00373,
              -0.00429],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-0.00089, -0.00015, 0.00064, 0.00146, 0.00228, 0.00311, 0.00395, 0.00485, 0.00693, 0.00791, 0.00844,
              0.00929],
             [0.00121, 0.00261, 0.00411, 0.00566, 0.00721, 0.00879, 0.0104, 0.0121, 0.016, 0.0179, 0.0189, 0.0205],
             [0.00174, 0.00323, 0.00483, 0.00648, 0.00814, 0.00983, 0.0115, 0.0133, 0.0175, 0.0195, 0.0206, 0.0223],
             [0.00273, 0.00438, 0.00614, 0.00796, 0.0098, 0.0117, 0.0135, 0.0155, 0.0202, 0.0224, 0.0236, 0.0255]])
        self.CD_delta_elev = RectBivariateSpline(self.delta_elev_data, self.alpha_data, CD_delta_elev_data)

        """Initialize all lift coefficient splines"""
        # CL
        CL_data = np.array([-0.571, -0.321, -0.083, 0.148, 0.392, 0.65, 0.918, 1.195, 1.659, 1.789, 1.84, 1.889])
        self.CL = Akima1DInterpolator(self.alpha_data, CL_data)
        self.CL_derivative = self.CL.derivative()

        # CL_dot (alphadot)
        CL_alphadot_data = np.array(
            [2.434, 2.362, 2.253, 2.209, 2.178, 2.149, 2.069, 1.855, 1.185, 0.8333, 0.6394, 0.4971])
        self.CL_dot = Akima1DInterpolator(self.alpha_data, CL_alphadot_data)
        self.CL_dot_derivative = self.CL_dot.derivative()

        # CL_q
        CL_q_data = np.array([7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282])
        self.CL_q = Akima1DInterpolator(self.alpha_data, CL_q_data)
        self.CL_q_derivative = self.CL_q.derivative()

        # CL_delta_elev
        CL_delta_elev_data = np.array([-0.132, -0.123, -0.082, -0.041, 0, 0.061, 0.116, 0.124, 0.137])
        self.CL_delta_elev = Akima1DInterpolator(self.delta_elev_data, CL_delta_elev_data)
        self.CL_delta_elev_derivative = self.CL_delta_elev.derivative()

        """Initialize all moment coefficient splines"""
        # Cm
        CM_data = np.array(
            [0.0597, 0.0498, 0.0314, 0.0075, -0.0248, -0.068, -0.1227, -0.1927, -0.3779, -0.4605, -0.5043, -0.5496])
        self.CM = Akima1DInterpolator(self.alpha_data, CM_data)
        self.CM_derivative = self.CM.derivative()

        # Cm_q
        CM_q_data = np.array(
            [-6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232])
        self.CM_q = Akima1DInterpolator(self.alpha_data, CM_q_data)
        self.CM_q_derivative = self.CM_q.derivative()

        # Cm_dot (alphadot)
        CM_alphadot_data = np.array(
            [-6.64, -6.441, -6.146, -6.025, -5.942, -5.861, -5.644, -5.059, -3.233, -2.273, -1.744, -1.356])
        self.CM_dot = Akima1DInterpolator(self.alpha_data, CM_alphadot_data)
        self.CM_dot_derivative = self.CM_dot.derivative()

        # Cm_delta_elev
        CM_delta_elev_data = np.array([0.3302, 0.3065, 0.2014, 0.1007, -0.0002, -0.1511, -0.2863, -0.3109, -0.345])
        self.CM_delta_elev = Akima1DInterpolator(self.delta_elev_data, CM_delta_elev_data)
        self.CM_delta_elev_derivative = self.CM_delta_elev.derivative()

        """Initialize all side force coefficient splines"""
        # CY_beta
        CY_beta_data = np.array(
            [-0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268])
        self.CY_beta = Akima1DInterpolator(self.alpha_data, CY_beta_data)
        self.CY_beta_derivative = self.CY_beta.derivative()

        # CY_p
        CY_p_data = np.array(
            [-0.032, -0.0372, -0.0418, -0.0463, -0.051, -0.0563, -0.0617, -0.068, -0.0783, -0.0812, -0.0824, -0.083])
        self.CY_p = Akima1DInterpolator(self.alpha_data, CY_p_data)
        self.CY_p_derivative = self.CY_p.derivative()

        # CY_r
        CY_r_data = np.array(
            [0.2018, 0.2054, 0.2087, 0.2115, 0.2139, 0.2159, 0.2175, 0.2187, 0.2198, 0.2198, 0.2196, 0.2194])
        self.CY_r = Akima1DInterpolator(self.alpha_data, CY_r_data)
        self.CY_r_derivative = self.CY_r.derivative()

        # CY_delta_rud
        CY_delta_rud_data = (-1) * np.array(
            [0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561])
        self.CY_delta_rud = Akima1DInterpolator(self.alpha_data, CY_delta_rud_data)
        self.CY_delta_rud_derivative = self.CY_delta_rud.derivative()

        """Initialize all roll coefficient splines"""
        # Cl_beta
        CL_beta_data = np.array(
            [-0.178, -0.186, -0.1943, -0.202, -0.2103, -0.219, -0.2283, -0.2376, -0.2516, -0.255, -0.256, -0.257])
        self.CL_beta = Akima1DInterpolator(self.alpha_data, CL_beta_data)
        self.CL_beta_derivative = self.CL_beta.derivative()

        # Cl_p
        CL_p_data = np.array(
            [-0.4968, -0.4678, -0.4489, -0.4595, 0.487, -0.5085, -0.5231, -0.4916, -0.301, -0.203, -0.1498, -0.0671])
        self.CL_p = Akima1DInterpolator(self.alpha_data, CL_p_data)
        self.CL_p_derivative = self.CL_p.derivative()

        # Cl_r
        CL_r_data = np.array(
            [-0.09675, -0.05245, -0.01087, 0.02986, 0.07342, 0.1193, 0.1667, 0.2152, 0.2909, 0.3086, 0.3146, 0.3197])
        self.CL_r = Akima1DInterpolator(self.alpha_data, CL_r_data)
        self.CL_r_derivative = self.CL_r.derivative()

        # Cl_delta_rud
        CL_delta_rud_data = (-1) * np.array(
            [0.091, 0.082, 0.072, 0.063, 0.053, 0.0432, 0.0333, 0.0233, 0.0033, -0.005, -0.009, -0.015])
        self.CL_delta_rud = Akima1DInterpolator(self.alpha_data, CL_delta_rud_data)
        self.CL_delta_rud_derivative = self.CL_delta_rud.derivative()

        # Cl_delta_aile
        CL_delta_aile_data = np.array(
            [-0.078052, -0.059926, -0.036422, -0.018211, 0, 0.018211, 0.036422, 0.059926, 0.078052])
        self.CL_delta_aile = Akima1DInterpolator(self.delta_aile_data, CL_delta_aile_data)
        self.CL_delta_aile_derivative = self.CL_delta_aile.derivative()

        """Initialize all yaw coefficient splines"""
        # Cn_beta
        CN_beta_data = np.array(
            [0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126])
        self.CN_beta = Akima1DInterpolator(self.alpha_data, CN_beta_data)
        self.CN_beta_derivative = self.CN_beta.derivative()

        # Cn_p
        CN_p_data = np.array(
            [0.03, 0.016, 0.00262, -0.0108, -0.0245, -0.0385, -0.0528, -0.0708, -0.113, -0.1284, -0.1356, -0.1422])
        self.CN_p = Akima1DInterpolator(self.alpha_data, CN_p_data)
        self.CN_p_derivative = self.CN_p.derivative()

        # Cn_r
        CN_r_data = np.array(
            [-0.028, -0.027, -0.027, -0.0275, -0.0293, -0.0325, -0.037, -0.043, -0.05484, -0.058, -0.0592, -0.06015])
        self.CN_r = Akima1DInterpolator(self.alpha_data, CN_r_data)
        self.CN_r_derivative = self.CN_r.derivative()

        # Cn_delta_rud
        CN_delta_rud_data = (-1) * np.array(
            [-0.211, -0.215, -0.218, -0.22, -0.224, -0.226, -0.228, -0.229, -0.23, -0.23, -0.23, -0.23])
        self.CN_delta_rud = Akima1DInterpolator(self.alpha_data, CN_delta_rud_data)
        self.CN_delta_rud_derivative = self.CN_delta_rud.derivative()

        CN_delta_aile_data = np.array([[-0.004321, -0.002238, -0.0002783, 0.001645, 0.003699, 0.005861, 0.008099,
                                        0.01038, 0.01397, 0.01483, 0.01512, 0.01539],
                                       [-0.003318, -0.001718, -0.0002137, 0.001263, 0.00284, 0.0045, 0.006218, 0.00797,
                                        0.01072, 0.01138, 0.01161, 0.01181],
                                       [-0.002016, -0.001044, -0.000123, 0.0007675, 0.00173, 0.002735, 0.0038, 0.004844,
                                        0.00652, 0.00692, 0.00706, 0.0072],
                                       [-0.00101, -0.000522, -0.0000649, 0.000384, 0.000863, 0.00137, 0.0019, 0.00242,
                                        0.00326, 0.00346, 0.00353, 0.0036],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.00101, 0.00052, 0.000065, -0.000384, -0.00086, -0.0014, -0.002, -0.002422,
                                        -0.00326, -0.00346, -0.00353, -0.0036],
                                       [0.00202, 0.001044, 0.00013, -0.0008, -0.00173, -0.002735, -0.0038, -0.004844,
                                        -0.00652, -0.00692, -0.00706, -0.0072],
                                       [0.00332, 0.00172, 0.000214, -0.001263, -0.00284, -0.0045, -0.00622, -0.008,
                                        -0.01072, -0.01138, -0.01161, -0.01181],
                                       [0.004321, 0.00224, 0.00028, -0.001645, -0.0037, -0.00586, -0.0081, -0.0104,
                                        -0.014, -0.01483, -0.01512, -0.0154]])
        self.CN_delta_aile = RectBivariateSpline(self.delta_aile_data,
                                                 self.alpha_data,
                                                 CN_delta_aile_data)

    def evaluate(self, alpha: csdl.Variable, delta_aileron: csdl.Variable, delta_elev: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('alpha', alpha)
        self.declare_input('delta_aileron', delta_aileron)
        self.declare_input('delta_elev', delta_elev)

        # declare output variables
        CD = self.create_output('CD', alpha.shape)
        CD_delta_elev = self.create_output('CD_delta_elev', alpha.shape)

        CL = self.create_output('CL', alpha.shape)
        CL_dot = self.create_output('CL_dot', alpha.shape)
        CL_q = self.create_output('CL_q', alpha.shape)
        CL_delta_elev = self.create_output('CL_delta_elev', alpha.shape)

        CM = self.create_output('CM', alpha.shape)
        CM_q = self.create_output('CM_q', alpha.shape)
        CM_dot = self.create_output('CM_dot', alpha.shape)
        CM_delta_elev = self.create_output('CM_delta_elev', alpha.shape)

        CY_beta = self.create_output('CY_beta', alpha.shape)
        CY_p = self.create_output('CY_p', alpha.shape)
        CY_r = self.create_output('CY_r', alpha.shape)
        CY_delta_rud = self.create_output('CY_delta_rud', alpha.shape)

        CL_beta = self.create_output('CL_beta', alpha.shape)
        CL_p = self.create_output('CL_p', alpha.shape)
        CL_r = self.create_output('CL_r', alpha.shape)
        CL_delta_rud = self.create_output('CL_delta_rud', alpha.shape)
        CL_delta_aile = self.create_output('CL_delta_aile', alpha.shape)

        CN_beta = self.create_output('CN_beta', alpha.shape)
        CN_p = self.create_output('CN_p', alpha.shape)
        CN_r = self.create_output('CN_r', alpha.shape)
        CN_delta_rud = self.create_output('CN_delta_rud', alpha.shape)
        CN_delta_aile = self.create_output('CN_delta_aile', alpha.shape)


        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.CD = CD
        outputs.CD_delta_elev = CD_delta_elev
        outputs.CL = CL
        outputs.CL_dot = CL_dot
        outputs.CL_q = CL_q
        outputs.CL_delta_elev = CL_delta_elev
        outputs.CM = CM
        outputs.CM_q = CM_q
        outputs.CM_dot = CM_dot
        outputs.CM_delta_elev = CM_delta_elev
        outputs.CY_beta = CY_beta
        outputs.CY_p = CY_p
        outputs.CY_r = CY_r
        outputs.CY_delta_rud = CY_delta_rud
        outputs.CL_beta = CL_beta
        outputs.CL_p = CL_p
        outputs.CL_r = CL_r
        outputs.CL_delta_rud = CL_delta_rud
        outputs.CL_delta_aile = CL_delta_aile
        outputs.CN_beta = CN_beta
        outputs.CN_p = CN_p
        outputs.CN_r = CN_r
        outputs.CN_delta_rud = CN_delta_rud
        outputs.CN_delta_aile = CN_delta_aile

        return outputs

    def compute(self, input_vals, output_vals):
        alpha = input_vals['alpha']
        delta_aileron = input_vals['delta_aileron']
        delta_elev = input_vals['delta_elev']

        output_vals['CD'] = self.CD(alpha)
        output_vals['CD_delta_elev'] = self.CD_delta_elev(delta_elev, alpha)
        output_vals['CL'] = self.CL(alpha)
        output_vals['CL_dot'] = self.CL_dot(alpha)
        output_vals['CL_q'] = self.CL_q(alpha)
        output_vals['CL_delta_elev'] = self.CL_delta_elev(delta_elev)
        output_vals['CM'] = self.CM(alpha)
        output_vals['CM_q'] = self.CM_q(alpha)
        output_vals['CM_dot'] = self.CM_dot(alpha)
        output_vals['CM_delta_elev'] = self.CM_delta_elev(delta_elev)
        output_vals['CY_beta'] = self.CY_beta(alpha)
        output_vals['CY_p'] = self.CY_p(alpha)
        output_vals['CY_r'] = self.CY_r(alpha)
        output_vals['CY_delta_rud'] = self.CY_delta_rud(alpha)
        output_vals['CL_beta'] = self.CL_beta(alpha)
        output_vals['CL_p'] = self.CL_p(alpha)
        output_vals['CL_r'] = self.CL_r(alpha)
        output_vals['CL_delta_rud'] = self.CL_delta_rud(alpha)
        output_vals['CL_delta_aile'] = self.CL_delta_aile(delta_aileron)
        output_vals['CN_beta'] = self.CN_beta(alpha)
        output_vals['CN_p'] = self.CN_p(alpha)
        output_vals['CN_r'] = self.CN_r(alpha)
        output_vals['CN_delta_rud'] = self.CN_delta_rud(alpha)
        output_vals['CN_delta_aile'] = self.CN_delta_aile(delta_aileron, alpha)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        alpha = input_vals['alpha']
        delta_aileron = input_vals['delta_aileron']
        delta_elev = input_vals['delta_elev']

        derivatives['CD', 'alpha'] = np.diag(self.CD_derivative(alpha))
        derivatives['CD_delta_elev', 'delta_elev'] = np.diag(self.CD_delta_elev(delta_elev, alpha, dx=1, dy=0))
        derivatives['CD_delta_elev', 'alpha'] = np.diag(self.CD_delta_elev(delta_elev, alpha, dx=0, dy=1))
        derivatives['CL', 'alpha'] = np.diag(self.CL_derivative(alpha))
        derivatives['CL_dot', 'alpha'] = np.diag(self.CL_dot_derivative(alpha))
        derivatives['CL_q', 'alpha'] = np.diag(self.CL_q_derivative(alpha))
        derivatives['CL_delta_elev', 'delta_elev'] = np.diag(self.CL_delta_elev_derivative(delta_elev))
        derivatives['CM', 'alpha'] = np.diag(self.CM_derivative(alpha))
        derivatives['CM_q', 'alpha'] = np.diag(self.CM_q_derivative(alpha))
        derivatives['CM_dot', 'alpha'] = np.diag(self.CM_dot_derivative(alpha))
        derivatives['CM_delta_elev', 'delta_elev'] = np.diag(self.CM_delta_elev_derivative(delta_elev))
        derivatives['CY_beta', 'alpha'] = np.diag(self.CY_beta_derivative(alpha))
        derivatives['CY_p', 'alpha'] = np.diag(self.CY_p_derivative(alpha))
        derivatives['CY_r', 'alpha'] = np.diag(self.CY_r_derivative(alpha))
        derivatives['CY_delta_rud', 'alpha'] = np.diag(self.CY_delta_rud_derivative(alpha))
        derivatives['CL_beta', 'alpha'] = np.diag(self.CL_beta_derivative(alpha))
        derivatives['CL_p', 'alpha'] = np.diag(self.CL_p_derivative(alpha))
        derivatives['CL_r', 'alpha'] = np.diag(self.CL_r_derivative(alpha))
        derivatives['CL_delta_rud', 'alpha'] = np.diag(self.CL_delta_rud_derivative(alpha))
        derivatives['CL_delta_aile', 'delta_aileron'] = np.diag(self.CL_delta_aile_derivative(delta_aileron))
        derivatives['CN_beta', 'alpha'] = np.diag(self.CN_beta_derivative(alpha))
        derivatives['CN_p', 'alpha'] = np.diag(self.CN_p_derivative(alpha))
        derivatives['CN_r', 'alpha'] = np.diag(self.CN_r_derivative(alpha))
        derivatives['CN_delta_rud', 'alpha'] = np.diag(self.CN_delta_rud_derivative(alpha))
        derivatives['CN_delta_aile', 'delta_aileron'] = np.diag(self.CN_delta_aile(delta_aileron, alpha, dx=1, dy=0))
        derivatives['CN_delta_aile', 'delta_elev'] = np.diag(self.CN_delta_aile(delta_aileron, alpha, dx=0, dy=1))

class C172Aerodynamics(Loads):

    def __init__(self, S:Union[ureg.Quantity, csdl.Variable], c:Union[ureg.Quantity, csdl.Variable],
                 b:Union[ureg.Quantity, csdl.Variable], aero_curves:AeroCurve):

        if S is None:
            self.S = csdl.Variable(name='S', shape=(1,), value=16.2)
        elif isinstance(S, ureg.Quantity):
            self.S = csdl.Variable(name='S', shape=(1,), value=S.to_base_units())
        else:
            self.S = S

        if c is None:
            self.c = csdl.Variable(name='c', shape=(1,), value=1.49352)
        elif isinstance(c, ureg.Quantity):
            self.c = csdl.Variable(name='c', shape=(1,), value=c.to_base_units())
        else:
            self.c = c

        if b is None:
            self.b = csdl.Variable(name='b', shape=(1,), value=10.91184)
        elif isinstance(c, ureg.Quantity):
            self.b = csdl.Variable(name='b', shape=(1,), value=b.to_base_units())
        else:
            self.b = b

        self.c172_aero_curves = aero_curves

        self.L = csdl.Variable(name='L', shape=(1,), value=0.0)
        self.D = csdl.Variable(name='D', shape=(1,), value=0.0)

    def get_FM_localAxis(self, states, controls, axis):
        rad2deg = 180.0 / np.pi

        # Geometric Design Variables
        S = self.S
        c = self.c
        b = self.b
        # State Variables (angles for the tables are ALL in degrees)
        velocity = states.VTAS
        p = states.states.p
        q = states.states.q
        r = states.states.r
        density = states.atmospheric_states.density
        alpha_eff = states.alpha * rad2deg + axis.euler_angles_vector[1] * rad2deg
        beta = states.beta * rad2deg
        alpha_dot = states.alpha_dot  # Keeping it in rad/s
        # Controls
        left_aileron = controls.u[0] * rad2deg
        elevator = controls.u[2] * rad2deg
        rudder = controls.u[3] * rad2deg

        curve_outputs = self.c172_aero_curves.evaluate(alpha=alpha_eff,delta_aileron=left_aileron, delta_elev=elevator)

        CD = curve_outputs.CD
        CD_delta_elev = curve_outputs.CD_delta_elev
        CL = curve_outputs.CL
        CL_dot = curve_outputs.CL_dot
        CL_q = curve_outputs.CL_q
        CL_delta_elev = curve_outputs.CL_delta_elev
        CM = curve_outputs.CM
        CM_q = curve_outputs.CM_q
        CM_dot = curve_outputs.CM_dot
        CM_delta_elev = curve_outputs.CM_delta_elev
        CY_beta = curve_outputs.CY_beta
        CY_p = curve_outputs.CY_p
        CY_r = curve_outputs.CY_r
        CY_delta_rud = curve_outputs.CY_delta_rud
        CL_beta = curve_outputs.CL_beta
        CL_p = curve_outputs.CL_p
        CL_r = curve_outputs.CL_r
        CL_delta_rud = curve_outputs.CL_delta_rud
        CL_delta_aile = curve_outputs.CL_delta_aile
        CN_beta = curve_outputs.CN_beta
        CN_p = curve_outputs.CN_p
        CN_r = curve_outputs.CN_r
        CN_delta_rud = curve_outputs.CN_delta_rud
        CN_delta_aile = curve_outputs.CN_delta_aile

        CL_total = (
                CL + CL_delta_elev +
                c / (2 * velocity) * (CL_q * q + CL_dot * alpha_dot)
        )
        CD_total = CD + CD_delta_elev

        CM_total = (
                CM + CM_delta_elev +
                c / (2 * velocity) * (
                            2 * CM_q * q + CM_dot * alpha_dot)
        )

        CY_total = (
                CY_beta * beta +
                CY_delta_rud * rudder +
                b / (2 * velocity) * (CY_p * p + CY_r * r)
        )
        Cl_total = (
                0.1 * CL_beta * beta +
                CL_delta_aile +
                0.075 * CL_delta_rud * rudder +
                b / (2 * velocity) * (CL_p * p + CL_r * r)
        )
        Cn_total = (
                CN_beta * beta +
                CN_delta_aile +
                0.075 * CN_delta_rud * rudder +
                b / (2 * velocity) * (CN_p * p + CN_r * r)
        )

        # Compute Forces from coefficients:
        qBar = 0.5 * density * velocity ** 2

        L = qBar * S * CL_total
        D = qBar * S * CD_total
        Y = qBar * S * CY_total
        l = qBar * S * b * Cl_total
        m = qBar * S * c * CM_total
        n = qBar * S * b * Cn_total

        wind_axis = states.windAxis

        self.L = L
        self.D = D


        force_vector = Vector(vector=csdl.concatenate((-D,
                                                       Y,
                                                       -L),
                                                      axis=0), axis=wind_axis)

        moment_vector = Vector(vector=csdl.concatenate((l,
                                                       m,
                                                       n),
                                                      axis=0), axis=wind_axis)
        loads_waxis = ForcesMoments(force=force_vector, moment=moment_vector)

        return loads_waxis

c172_aero_curves = AeroCurve()

c172_aerodynamics = C172Aerodynamics(S=None, b=None, c=None, aero_curves=c172_aero_curves)
# loads = c172_aerodynamics.get_FM_localAxis(states=state, controls=c172_controls)
wing_component.load_solvers.append(c172_aerodynamics)
# wing_component.compute_total_loads(fd_state=state, controls=c172_controls)

pass
# endregion

# region Propulsion Model

# Propeller Data
class PropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        # Obtained with JavaProp
        J_data = np.array(
            [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.76, 0.77, 0.78,
             0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94])
        Ct_data = np.array(
            [0.102122, 0.11097, 0.107621, 0.105191, 0.102446, 0.09947, 0.096775, 0.094706, 0.092341, 0.088912, 0.083878,
             0.076336, 0.066669, 0.056342, 0.045688, 0.034716, 0.032492, 0.030253, 0.028001, 0.025735, 0.023453,
             0.021159, 0.018852, 0.016529, 0.014194, 0.011843, 0.009479, 0.0071, 0.004686, 0.002278, -0.0002, -0.002638,
             -0.005145, -0.007641, -0.010188])
        self.ct = Akima1DInterpolator(J_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)

    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, advance_ratio: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('advance_ratio', advance_ratio)

        # declare output variables
        ct = self.create_output('ct', advance_ratio.shape)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.ct = ct

        return outputs

    def compute(self, input_vals, output_vals):
        advance_ratio = input_vals['advance_ratio']
        output_vals['ct'] = self.ct(advance_ratio)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        advance_ratio = input_vals['advance_ratio']
        derivatives['ct', 'advance_ratio'] = np.diag(self.ct_derivative(advance_ratio))


c172_prop_curve = PropCurve()
# adv_rt = csdl.Variable(shape=(1,), value=0.1)
# prop_data_outputs = c172_prop_curve.evaluate(advance_ratio=adv_rt)
# print(prop_data_outputs.ct.value)

class C172Propulsion(Loads):

    def __init__(self, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:PropCurve):
        self.c172_prop_curve = prop_curve

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=PropRadius)
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius

        self.c172_thrust = csdl.Variable(name='Thrust', shape=(1,), value=0.0)

    def get_FM_localAxis(self, states, controls, axis):
        throttle = controls.u[4]
        density = states.atmospheric_states.density
        velocity = states.VTAS

        # Compute RPM
        rpm = 1000 + (2800 - 1000) * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s

        # Compute advance ratio
        J = (np.pi * velocity) / (omega_RAD * self.radius)  # non-dimensional

        # Compute Ct
        ct = self.c172_prop_curve.evaluate(advance_ratio=J).ct

        # Compute Thrust
        T =  (2 / np.pi) ** 2 * density * (omega_RAD * self.radius) ** 2 * ct  # N

        self.c172_thrust = T

        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads

    # def get_torque_power(self, states, controls):
    #     throttle = controls.u[4]
    #     density = states.atmospheric_states.density
    #     velocity = states.VTAS
    #
    #     # Compute RPM
    #     rpm = 1000 + (2800 - 1000) * throttle
    #     omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s
    #
    #     # Compute advance ratio
    #     J = (np.pi * velocity) / (omega_RAD * self.radius)  # non-dimensional


c172_propulsion = C172Propulsion( radius=radius_c172, prop_curve=c172_prop_curve)
# loads = c172_propulsion.get_FM_localAxis(states=state, controls=c172_controls, axis=engine_axis)

engine_component.load_solvers.append(c172_propulsion)
# engine_component.compute_total_loads(fd_state=state, controls=c172_controls)
# endregion


# region Climb Condition

cruise_cond = aircraft_conditions.RateofClimb(fd_axis=fd_axis, controls=c172_controls, altitude=Q_(h, 'ft'),
                                              range=Q_(Range, 'km'), speed=Q_(Speed, 'm/s'),
                                              pitch_angle=Q_(Pitch, 'deg'),
                               flight_path_angle=Q_(flight_path_angle, 'radians'))

# Create a Mass Properties object with given values
c172_mi = MassMI(axis=cruise_cond.ac_states.axis,
                 Ixx=Q_(1285.3154166, 'kg*(m*m)'),
                 Iyy=Q_(1824.9309607, 'kg*(m*m)'),
                 Izz=Q_(2666.89390765, 'kg*(m*m)'),
                 Ixy=Q_(0, 'kg*(m*m)'),
                 Ixz=Q_(0, 'kg*(m*m)'),
                 Iyz=Q_(0, 'kg*(m*m)'))
c172_mass_properties = MassProperties(mass=Q_(M_Aircraft, 'kg'),
                                      inertia=c172_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))
aircraft_component.mass_properties = c172_mass_properties
# endregion

# endregion

tf, tm = aircraft_component.compute_total_loads(fd_state=cruise_cond.ac_states,
                                                controls=cruise_cond.controls)

print(tf.value)
print(tm.value)
# FM = csdl.concatenate((tf, tm), axis=0)

Lift_scaling = 1 / (M_Aircraft * 9.81)
Drag_scaling = Lift_scaling * 10
Moment_scaling = Lift_scaling / 10

res1 = tf[0] * Drag_scaling
res1.name = 'Fx Force'
res1.set_as_constraint(equals=0.0)  # setting a small value to ensure the thrust is close to zero

res2 = tf[2] * Lift_scaling
res2.name = 'Fz Force'
res2.set_as_constraint(equals=0.0)

res4 = tm[1] * Moment_scaling
res4.name = 'My Moment'
res4.set_as_constraint(equals=0.0)

cruise_r, cruise_x = cruise_cond.evaluate_eom(component=aircraft_component, forces=tf, moments=tm)

h_dot = cruise_r[11]
h_dot_scaling = 1e-1  # scaling factor for the rate of climb, should be in m/s
# h_dot is the rate of climb in m/s, we want to maximize this, so we will minimize its negative value
h_dot_residual = h_dot * h_dot_scaling
h_dot_residual.name = 'Rate of Climb Residual'
h_dot_residual.set_as_objective()

c172_controls.engine.throttle.set_as_design_variable(lower=0.5, upper=1.0)
c172_controls.elevator.deflection.set_as_design_variable(lower=c172_controls.elevator.lower_bound*np.pi/180,
                                                     upper=c172_controls.elevator.upper_bound*np.pi/180, scaler=100)

cruise_cond.parameters.pitch_angle.set_as_design_variable(lower=(-1)*np.pi/180,
                                                     upper=30*np.pi/180, scaler=100)
cruise_cond.parameters.flight_path_angle.set_as_design_variable(lower=np.deg2rad(-20), upper=np.deg2rad(20), scaler=1e2)


sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        gpu=False,
        additional_inputs=[cruise_cond.parameters.speed],
        additional_outputs=[cruise_cond.ac_states.VTAS, c172_propulsion.c172_thrust],
        derivatives_kwargs= {
            "concatenate_ofs" : True
        })

speeds = np.linspace(33.528, 69, 20)  # in m/s,

prop_J = np.asarray([0.17809162500971004, 0.2378395628066537, 0.29759253932829605,0.35952854332568435,0.4199856816239811,0.4833501640734731,0.5459550587746242,0.6136805574509094,0.6769744977546185,0.7822095227697872,0.8741423145801799,0.942549300771975,1.0300354180356954,1.078540701348489])
prop_nu = np.asarray([0.24373927958833608, 0.3150943396226414, 0.38782161234991414, 0.4550600343053173, 0.5195540308747855, 0.5758147512864493, 0.625214408233276, 0.6691252144082332, 0.7061749571183533, 0.765180102915952, 0.8015437392795883, 0.8310463121783875, 0.8564322469982846, 0.8660377358490565])

prop_eff_curve = Akima1DInterpolator(prop_J, prop_nu, method="akima")

Drags = []
Lifts = []
LD_ratios = []
vtas_list = []
eta_list = []
Jval_list = []
RoD_list = []
torque_list = []
P_excess_list = []
Thrust_power_list = []
throttles = []
elevators = []
pitches = []

fpa = []
climb_rate = []
for a_speed_i in range(speeds.shape[0]):
    sim[cruise_cond.parameters.speed] = speeds[a_speed_i]
    recorder.execute()



    t1 = time.time()
    prob = CSDLAlphaProblem(problem_name='trim_power_avail_max_opt', simulator=sim)
    optimizer = IPOPT(problem=prob)
    optimizer.solve()
    optimizer.print_results()

    t2 = time.time()
    print('Time to solve', t2 - t1)
    recorder.execute()

    RoD = h_dot
    RoD_list.append(RoD.value[0])  # Rate of Descent in m/s
    Trust = c172_propulsion.c172_thrust.value
    Power = c172_propulsion.c172_thrust.value * speeds[a_speed_i]
    Throttle = c172_controls.engine.throttle.value

    # Compute RPM
    rpm = 1000 + (2800 - 1000) * Throttle
    omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s

    # Compute advance ratio
    J = (np.pi * speeds[a_speed_i]) / (omega_RAD * PropRadius)  # non-dimensional

    prop_eff = prop_eff_curve(J[0]) # Find if possible a propeller curve

    net_power = Power[0] / prop_eff
    P_excess_list.append(net_power)
    Thrust_power_list.append(Power[0])
    fpa.append(cruise_cond.parameters.flight_path_angle.value[0]*180/np.pi)
    climb_rate.append(-h_dot.value[0])

    baseline_Drag = wing_component.load_solvers[0].D.value
    baseline_Lift = wing_component.load_solvers[0].L.value
    Drags.append(baseline_Drag)
    Lifts.append(baseline_Lift)
    dv_save_dict = {}
    constraints_save_dict = {}
    obj_save_dict = {}

    # Print Cruise Trim Results:
    print("=====Aircraft States=====")
    print("Throttle")
    print(c172_controls.engine.throttle.value)
    throttles.append(c172_controls.engine.throttle.value)
    print("Elevator Deflection (deg)")
    print(c172_controls.elevator.deflection.value * 180 / np.pi)
    print(c172_controls.elevator.deflection.value * 180 / np.pi)
    elevators.append(c172_controls.elevator.deflection.value * 180 / np.pi)
    print("Pitch Angle (deg)")
    print(cruise_cond.parameters.pitch_angle.value * 180 / np.pi)
    pitches.append(cruise_cond.parameters.pitch_angle.value * 180 / np.pi)

print(np.asarray(speeds))
print(fpa)
print(climb_rate)
print(P_excess_list)
print(Thrust_power_list)
#
import matplotlib.pyplot as plt
plt.figure()
plt.plot(speeds, np.asarray(fpa), marker='s', linestyle='--', label='Flight Path Angle (deg)')
plt.plot(speeds, np.asarray(climb_rate), marker='o', linestyle='-', label='Climb Rate (m/s)')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Degrees / Climb Rate')
plt.title('Attitude / Climb Rate vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(speeds, P_excess_list, marker='s', linestyle='--', label='Shaft Power')
plt.plot(speeds, Thrust_power_list, marker='s', linestyle='--', label='Power Available')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Power (W)')
plt.title('Power vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()




