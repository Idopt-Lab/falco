import numpy as np
from scipy.interpolate import Akima1DInterpolator

from flight_simulator import Q_, ureg
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.dynamics.vector import Vector
from flight_simulator.core.loads.forces_moments import ForcesMoments
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.vehicle.controls.vehicle_control_system import (
    VehicleControlSystem, ControlSurface, PropulsiveControl)
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.components.aircraft import Aircraft
from typing import Union
from typing import List
import csdl_alpha as csdl


# Every CSDl code starts with a recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# region Axis

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
    z=Q_(-12000, 'ft'),  # z is positive down in FD axis
    phi=Q_(0, 'deg'),
    theta=Q_(4, 'deg'),
    psi=Q_(0, 'deg'),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
# endregion

# region Aircraft Wind Axis

wind_axis = Axis(
        name='Wind Axis',
        x=Q_(0, 'ft'),
        y=Q_(0, 'ft'),
        z=Q_(0, 'ft'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=fd_axis,
        origin=ValidOrigins.Inertial.value
    )
# endregion

# region Aircraft Component

# Create a Mass Properties object with given values
c172_mi = MassMI(axis=fd_axis,
                 Ixx=Q_(1285.3154166, 'kg*(m*m)'),
                 Iyy=Q_(1824.9309607, 'kg*(m*m)'),
                 Izz=Q_(2666.89390765, 'kg*(m*m)'),
                 Ixy=Q_(0, 'kg*(m*m)'),
                 Ixz=Q_(0, 'kg*(m*m)'),
                 Iyz=Q_(0, 'kg*(m*m)'))
c172_mass_properties = MassProperties(mass=Q_(1043.2616, 'kg'),
                                      inertia=c172_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))

aircraft_component = Aircraft()
aircraft_component.quantities.mass_properties = c172_mass_properties
# endregion

# region Fuselage Component
fuselage_component = Component(name='Fuselage')
aircraft_component.add_subcomponent(fuselage_component)

# region Engine Component
engine_component = Component(name='Engine')
radius_c172 = csdl.Variable(name='prop_radius', shape=(1,), value=0.94)
engine_component.parameters.radius = radius_c172
fuselage_component.add_subcomponent(engine_component)
# endregion
# endregion

# region Wing Component
wing_component = Component(name='Wing')
aircraft_component.add_subcomponent(wing_component)
# endregion

# region Control Surface Components
rudder_component = Component(name='Rudder')
rudder_component.parameters.actuate_angle = csdl.Variable(name="Rudder Actuate Angle", shape=(1,), value=np.deg2rad(0))
# aircraft_component.add_subcomponent(rudder_component)

elevator_component = Component(name='Elevator')
elevator_component.parameters.actuate_angle = csdl.Variable(name="Elevator Actuate Angle", shape=(1,), value=np.deg2rad(0))
# aircraft_component.add_subcomponent(elevator_component)

right_aileron_component = Component(name='Right Aileron')
right_aileron_component.parameters.actuate_angle = csdl.Variable(name="Right Aileron Actuate Angle", shape=(1,), value=np.deg2rad(15))
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

        self.engine = PropulsiveControl(name='engine', throttle=1.0)

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
            self.radius = csdl.Variable(name='radius', shape=(1,), value=0.94)
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius

    def get_FM_localAxis(self, states, controls):
        throttle = controls.u[4]
        density = states.atmospheric_states.density
        velocity = states.VTAS
        axis = states.axis

        # Compute RPM
        rpm = 1000 + (2800 - 1000) * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s

        # Compute advance ratio
        J = (np.pi * velocity) / (omega_RAD * self.radius)  # non-dimensional

        # Compute Ct
        ct = self.c172_prop_curve.evaluate(advance_ratio=J).ct

        # Compute Thrust
        T =  (2 / np.pi) ** 2 * density * (omega_RAD * self.radius) ** 2 * ct  # N

        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads


c172_propulsion = C172Propulsion( radius=radius_c172, prop_curve=c172_prop_curve)
# state = AircraftStates(axis=fd_axis, u=Q_(125, 'mph'))
# loads = c172_propulsion.get_FM_localAxis(states=state, controls=c172_controls)

engine_component.quantities.load_solvers.append(c172_propulsion)
# endregion

state = AircraftStates(axis=fd_axis, u=Q_(125, 'mph'))
tf, tm = aircraft_component.compute_total_loads(fd_state=state, controls=c172_controls)
pass
# Create Aerodynamic Model





