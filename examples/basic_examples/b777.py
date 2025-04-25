import numpy as np
from scipy.interpolate import Akima1DInterpolator

from flight_simulator import Q_, ureg
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.dynamics.vector import Vector
from flight_simulator.core.loads.forces_moments import ForcesMoments
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.vehicle.conditions.aircraft_conditions import CruiseCondition
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel, AircraftAerodynamics
from flight_simulator.core.vehicle.controls.vehicle_control_system import (
    VehicleControlSystem, ControlSurface, PropulsiveControl)
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.components.aircraft import Aircraft
from typing import Union
from typing import List
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP, COBYLA
import time
import scipy.io as sio
import NRLMSIS2

# Every CSDl code starts with a recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# Load Data files
thrust_data = sio.loadmat('./DELPHI-master/AircraftData/b777/TminTmax.mat')
dragpolar_data = sio.loadmat('./DELPHI-master/AircraftData/b777/DragPolar.mat')
aerocoeffs_data = sio.loadmat('./DELPHI-master/AircraftData/b777/AeroDerivatives.mat')


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
b777_mi = MassMI(axis=fd_axis,
                 Ixx=Q_(18663676.936422713, 'kg*(m*m)'),
                 Iyy=Q_(47812509.66393921, 'kg*(m*m)'),
                 Izz=Q_(63555637.7196545, 'kg*(m*m)'),
                 Ixy=Q_(0, 'kg*(m*m)'),
                 Ixz=Q_(0, 'kg*(m*m)'),
                 Iyz=Q_(0, 'kg*(m*m)'))
b777_mass_properties = MassProperties(mass=Q_(650209.35*0.4535924, 'kg'),
                                      inertia=b777_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))

aircraft_component = Aircraft()
aircraft_component.mass_properties = b777_mass_properties
# endregion

# region Fuselage Component
fuselage_component = Component(name='Fuselage')
aircraft_component.add_subcomponent(fuselage_component)

# region Engine Component
left_engine_component = Component(name='Left Engine')
fuselage_component.add_subcomponent(left_engine_component)
right_engine_component = Component(name='Right Engine')
fuselage_component.add_subcomponent(right_engine_component)
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

left_aileron_component = Component(name='Left Aileron')
left_aileron_component.parameters.actuate_angle = csdl.Variable(name="Left Aileron Actuate Angle", shape=(1,), value=np.deg2rad(0))
right_aileron_component = Component(name='Right Aileron')
right_aileron_component.parameters.actuate_angle = csdl.Variable(name="Right Aileron Actuate Angle", shape=(1,), value=np.deg2rad(0))
wing_component.add_subcomponent(right_aileron_component)
# endregion

# region Aircraft Controls
class B777Control(VehicleControlSystem):

    def __init__(self, elevator_component, aileron_right_component, aileron_left_component, rudder_component):

        self.elevator = ControlSurface('elevator_left', lb=-30, ub=30, component=elevator_component)
        self.aileron_left = ControlSurface('aileron_left', lb=-15, ub=20, component=aileron_right_component)
        self.aileron_right = ControlSurface('aileron_right', lb=-15, ub=20, component=aileron_left_component)
        self.engine_left = PropulsiveControl(name='engine_left', throttle=0.45)
        self.engine_right = PropulsiveControl(name='engine_right', throttle=0.52)
        self.rudder = ControlSurface('rudder', lb=-16, ub=16, component=rudder_component)
        self.u = csdl.concatenate((self.aileron_left.deflection,
                                    self.aileron_right.deflection,
                                    self.elevator.deflection,
                                    self.rudder.deflection,
                                    self.engine_left.throttle,
                                    self.engine_right.throttle), axis=0)

        super().__init__(pitch_control=[self.elevator],
                            roll_control=[self.aileron_left, self.aileron_right],
                            yaw_control=[self.rudder],
                            throttle_control=[self.engine_left,self.engine_right])

    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']

    @property
    def lower_bounds(self):
        lb_elevator = self.elevator.lower_bound
        lb_rudder = self.rudder.lower_bound
        lb_thr_left = self.engine_left.lower_bound
        lb_thr_right = self.engine_right.lower_bound
        lb_aileron_left = self.aileron_left.lower_bound
        lb_aileron_right = self.aileron_right.lower_bound
        return np.array([lb_aileron_left, lb_aileron_right, lb_elevator, lb_rudder, lb_thr_left, lb_thr_right])

    @property
    def upper_bounds(self):
        ub_elevator = self.elevator.upper_bound
        ub_rudder = self.rudder.upper_bound
        ub_thr_left = self.engine_left.upper_bound
        ub_thr_right = self.engine_right.upper_bound
        ub_aileron_left = self.aileron_left.upper_bound
        ub_aileron_right = self.aileron_right.upper_bound
        return np.array([ub_aileron_left, ub_aileron_right, ub_elevator, ub_rudder, ub_thr_left, ub_thr_right])


B777_controls = B777Control(elevator_component=elevator_component, aileron_right_component=right_aileron_component,
                            aileron_left_component=left_aileron_component, rudder_component=rudder_component)
pass
# endregion

# region Propulsion Model

# Engine Data
class B777Propulsion(Loads):

    def __init__(self, Tmax: csdl.Variable, Tmin: csdl.Variable, is_left_engine:bool = True):
        if not isinstance(Tmax, csdl.Variable) or not isinstance(Tmin, csdl.Variable):
            raise TypeError("Tmax and Tmin must be instances of csdl.Variable")
        
        self.Tmax = Tmax
        self.Tmin = Tmin

        self.is_left_engine = is_left_engine

    def get_FM_refPoint(self, x_bar, u_bar):
        if self.is_left_engine:
            throttle = u_bar.engine_left.throttle
        else:
            throttle = u_bar.engine_right.throttle


        atm_SL = NRLMSIS2.Atmosphere()
        atmospheric_states_SL = atm_SL.evaluate(csdl.Variable(value=0.0, shape=(1,))) 
        density_SL = atmospheric_states_SL.density
        temperature_SL = atmospheric_states_SL.temperature
        R_specific = 287.058  # J/(kg*K)
        pressure_SL = density_SL * R_specific * temperature_SL

        density = x_bar.atmospheric_states.density
        temperature = x_bar.atmospheric_states.temperature
        pressure = density * R_specific * temperature
        velocity = x_bar.VTAS
        axis = x_bar.axis
        mach = velocity / x_bar.atmospheric_states.speed_of_sound

        # Compute Thrust
        theta = temperature / temperature_SL
        delta = pressure / pressure_SL
        gamma = csdl.Variable(value=1.4, shape=(1,))
        theta0 = theta*(1+(gamma-1)/2*mach**2)
        delta0 = delta*(1+(gamma-1)/2*mach**2)**(gamma/(gamma-1))
        TR= csdl.Variable(value=1.0, shape=(1,))
        TSFCsl = csdl.Variable(value=0.3, shape=(1,))
        if theta0.value <= TR.value:
            Tmin = self.Tmin * delta0 * (1-0.3*mach**1)
            Tmax = self.Tmax * delta0 * (1-0.3*mach**1)
        else: 
            Tmin = self.Tmin * delta0 * (1-0.3*mach**1 - 1.7*(theta0-TR)/theta0)
            Tmax = self.Tmax * delta0 * (1-0.3*mach**1 - 1.7*(theta0-TR)/theta0)

        T = Tmin + throttle * (Tmax - Tmin)
        TSFC = TSFCsl*1*(1+0.35*(mach))*(T/self.Tmax)**0.5


        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads

Tmin = csdl.Variable(shape=(1,), value = thrust_data['TL_N'][0][0])
Tmax = csdl.Variable(shape=(1,), value = thrust_data['TU_N'][0][0])
B777_left_engine = B777Propulsion(Tmax=Tmax, Tmin=Tmin, is_left_engine=True)
B777_right_engine = B777Propulsion(Tmax=Tmax, Tmin=Tmin, is_left_engine=False)

left_engine_component.load_solvers.append(B777_left_engine)
right_engine_component.load_solvers.append(B777_right_engine)

b = csdl.Variable(name='wing_span', shape=(1,), value=64.8)
S = csdl.Variable(name='wing_area', shape=(1,), value=436.8)
incidence = csdl.Variable(name='incidence', shape=(1,), value=2*np.pi/180)
CD0 = csdl.Variable(name='CD0', shape=(1,), value=0.002)
e = csdl.Variable(name='e', shape=(1,), value=0.8)
AR = b**2/S
lift_model = LiftModel(
            AR=AR,
            e=e,
            CD0=CD0,
            S=S,
            incidence=incidence,
        )

wing_component.mass_properties.mass = Q_(152.88*10, 'kg')
wing_component.mass_properties.cg_vector = Vector(vector=Q_(0, 'm'), axis=fd_axis)
wing_aero = AircraftAerodynamics(component=wing_component, lift_model=lift_model)
wing_component.load_solvers.append(wing_aero)


### THRUST AVAILABLE AND POWER AVAILABLE CALCULATION ###
# throttles = np.array([1])
# machs = np.arange(0.1, 0.9, 0.05)
# vtas_list = []
# Tavailable = []
# Pavailable = []
# Trequired = []
# Prequired = []
# vtas_csdl_list = []
# Tavailable_csdl = []
# Pavailable_csdl = []
# Trequired_csdl = []
# Prequired_csdl = []
# for i, throttle in enumerate(throttles):
#     B777_controls.engine_left.throttle = throttle
#     B777_controls.engine_right.throttle = throttle

#     for j, mach in enumerate(machs):
#         cruise_cond = CruiseCondition(fd_axis=fd_axis, controls=B777_controls,
#                                 altitude=Q_(12000, 'ft'), mach_number=Q_(mach, 'dimensionless'),
#                                 range=Q_(10000, 'm'), pitch_angle=Q_(0, 'deg'))
        
#         ThrustAvailable = (B777_left_engine.get_FM_refPoint(cruise_cond.ac_states, B777_controls).F.vector + B777_right_engine.get_FM_refPoint(cruise_cond.ac_states, B777_controls).F.vector)
#         ThrustRequired = -wing_aero.get_FM_refPoint(cruise_cond.ac_states, B777_controls).F.vector
#         PowerAvailable = cruise_cond.ac_states.VTAS * ThrustAvailable
#         PowerRequired = cruise_cond.ac_states.VTAS * ThrustRequired
#         Tavailable.append(ThrustAvailable.value[0])
#         Pavailable.append(PowerAvailable.value[0])
#         Trequired.append(ThrustRequired.value[0])
#         Prequired.append(PowerRequired.value[0])
#         vtas_list.append(cruise_cond.ac_states.VTAS.value[0])
#         Tavailable_csdl.append(ThrustAvailable[0])
#         Pavailable_csdl.append(PowerAvailable[0])
#         Trequired_csdl.append(ThrustRequired[0])
#         Prequired_csdl.append(PowerRequired[0])
#         vtas_csdl_list.append(cruise_cond.ac_states.VTAS[0])

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(vtas_list, Tavailable, marker='o', linestyle='-', label='Available Thrust')
# plt.plot(vtas_list, Trequired, marker='s', linestyle='--', label='Required Thrust')
# plt.xlabel('True Airspeed (VTAS) [ft/s]')
# plt.ylabel('Thrust (lbf)')
# plt.title('Thrust vs. VTAS')
# plt.grid(True)
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(vtas_list, Pavailable, marker='o', linestyle='-', label='Available Power')
# plt.plot(vtas_list, Prequired, marker='s', linestyle='--', label='Required Power')
# plt.xlabel('True Airspeed (VTAS) [ft/s]')
# plt.ylabel('Power')
# plt.title('Power vs. VTAS')
# plt.grid(True)
# plt.legend()
# plt.show()

# region Create Conditions
pass
# region Cruise Condition
mach=0.6
alt=35000
cruise_cond = CruiseCondition(fd_axis=fd_axis, controls=B777_controls,
                              altitude=Q_(alt, 'ft'), mach_number=Q_(mach, 'dimensionless'),
                              range=Q_(10000, 'm'), pitch_angle=Q_(2, 'deg'))
# endregion

# endregion

tf, tm = aircraft_component.compute_total_loads(fd_state=cruise_cond.ac_states,
                                                controls=cruise_cond.controls)
throttle=csdl.Variable(name='throttle', shape=(1,), value=0.5)
B777_controls.engine_left.throttle = throttle
B777_controls.engine_right.throttle = throttle
throttle.set_as_design_variable(lower=0.4, upper=1)
# B777_controls.engine_right.throttle.set_as_design_variable(lower=0.4,
#                                                      upper=1)
# B777_controls.engine_left.throttle.set_as_design_variable(lower=0.4,
#                                                      upper=1)
cruise_cond.parameters.mach_number.set_as_design_variable(lower=0.1, upper=0.9)
cruise_cond.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(5), upper=np.deg2rad(5))
cruise_cond.parameters.altitude.set_as_design_variable(lower=0, upper=50000)

throttle_diff = B777_controls.engine_right.throttle - B777_controls.engine_left.throttle
throttle_diff.set_as_constraint(lower=-1e-6, upper=1e-6)
throttle_diff.name = 'Throttle (R-L) Difference'
ThrustAvailable = (B777_left_engine.get_FM_refPoint(cruise_cond.ac_states, B777_controls).F.vector + B777_right_engine.get_FM_refPoint(cruise_cond.ac_states, B777_controls).F.vector)[0]
ThrustRequired = -wing_aero.get_FM_refPoint(cruise_cond.ac_states, B777_controls).F.vector[0]
PowerAvailable = cruise_cond.ac_states.VTAS * ThrustAvailable
PowerRequired = cruise_cond.ac_states.VTAS * ThrustRequired
ThrustAvailable.name = 'Thrust Available'
ThrustRequired.name = 'Thrust Required'

residual1 = csdl.absolute(ThrustAvailable-ThrustRequired)
residual1.name = 'TA=TR'
residual1.set_as_constraint(lower=-1e-6, upper=1e-6)
# residual.set_as_objective()

residual2 = tf[2]
residual2.name = 'Fz=0'
residual2.set_as_constraint(lower=-1e-6, upper=1e-6)

residual3 = tf[0]
residual3.name = 'Fx=0'
residual3.set_as_constraint(lower=-1e-6, upper=1e-6)

residual4 = tf[1]
residual4.name = 'Fy=0'
residual4.set_as_constraint(lower=-1e-6, upper=1e-6)

residual5 = tm[0]
residual5.name = 'Mx=0'
residual5.set_as_constraint(lower=-1e-6, upper=1e-6)

residual6 = tm[1]
residual6.name = 'My=0'
residual6.set_as_constraint(lower=-1e-6, upper=1e-6)

residual7 = tm[2]
residual7.name = 'Mz=0'
residual7.set_as_constraint(lower=-1e-6, upper=1e-6)

# r, xtest = cruise_cond.eval_linear_stability(component=aircraft_component)
J = cruise_cond.evaluate_trim_res(component=aircraft_component)
J.name = 'J: Trim Scalar'
J.set_as_objective()



pass
sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        gpu=False,
        derivatives_kwargs= {
            "concatenate_ofs" : True
        })


sim.check_optimization_derivatives()


t1 = time.time()
prob = CSDLAlphaProblem(problem_name='thrust_drag_equal_opt', simulator=sim)
optimizer = IPOPT(problem=prob)
optimizer.solve()
optimizer.print_results()
t2 = time.time()
print('Time to solve', t2-t1)
recorder.execute()
dv_save_dict = {}
constraints_save_dict = {}
obj_save_dict = {}

dv_dict = recorder.design_variables
constraint_dict = recorder.constraints
obj_dict = recorder.objectives

for dv in dv_dict.keys():
    dv_save_dict[dv.name] = dv.value
    print("Design Variable", dv.name, dv.value)

for c in constraint_dict.keys():
    constraints_save_dict[c.name] = c.value
    print("Constraint", c.name, c.value)

for obj in obj_dict.keys():
    obj_save_dict[obj.name] = obj.value
    print("Objective", obj.name, obj.value)



pass






