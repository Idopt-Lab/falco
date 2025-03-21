import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt
import lsdo_geo as lg
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER, ureg, Q_
from flight_simulator.core.vehicle.component import Component
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from typing import Union, List
from dataclasses import dataclass
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.euler_rotations import build_rotation_matrix
from flight_simulator.core.vehicle.aircraft_control_system import AircraftControlSystem
from flight_simulator.core.vehicle.models.propulsion.propulsion_model import HLPropCurve, CruisePropCurve, AircraftPropulsion
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel, AircraftAerodynamics
from flight_simulator.core.vehicle.models.mass_properties.mass_prop_model import GravityLoads
from flight_simulator.core.vehicle.models.weights.weights_model import StructuralWeights
from flight_simulator.core.vehicle.components.wing import Wing as WingComp
from flight_simulator.core.vehicle.components.fuselage import Fuselage as FuseComp
from flight_simulator.core.vehicle.components.aircraft import Aircraft as AircraftComp
from flight_simulator.core.vehicle.components.rotor import Rotor as RotorComp
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

lfs.num_workers = 1

debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=debug)

recorder.start()


in2m=0.0254
ft2m = 0.3048


geometry = import_geometry(
    file_name="x57.stp",
    file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57',
    refit=False,
    scale=in2m,
    rotate_to_body_fixed_frame=True
)


wing = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4'], name='wing')
aileronR = geometry.declare_component(function_search_names=['Rt_Aileron'], name='aileronR')
aileronL = geometry.declare_component(function_search_names=['Lt_Aileron'], name='aileronL')
flapL = geometry.declare_component(function_search_names=['Flap, 1'], name='left_flap')
flapR = geometry.declare_component(function_search_names=['Flap, 0'], name='right_flap')
h_tail = geometry.declare_component(function_search_names=['HorzStab'], name='h_tail')
trimTab = geometry.declare_component(function_search_names=['TrimTab'], name='trimTab')
htALL = geometry.declare_component(function_search_names=['HorzStab', 'TrimTab'], name='CompleteHT')
vertTail = geometry.declare_component(function_search_names=['VertTail'], name='vertTail')
rudder = geometry.declare_component(function_search_names=['Rudder'], name='rudder')
vtALL = geometry.declare_component(function_search_names=['VertTail', 'Rudder'], name='CompleteVT')
fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
gear_pod = geometry.declare_component(function_search_names=['GearPod'], name='gear_pod')

pylon1 = geometry.declare_component(function_search_names=['Pylon_12, 1'], name='pylon1')
pylon2 = geometry.declare_component(function_search_names=['Pylon_11, 1'], name='pylon2')
pylon3 = geometry.declare_component(function_search_names=['Pylon_10, 1'], name='pylon3')
pylon4 = geometry.declare_component(function_search_names=['Pylon_09, 1'], name='pylon4')
pylon5 = geometry.declare_component(function_search_names=['Pylon_08, 1'], name='pylon5')
pylon6 = geometry.declare_component(function_search_names=['Pylon_07, 1'], name='pylon6')
pylon7 = geometry.declare_component(function_search_names=['Pylon_07, 0'], name='pylon7')
pylon8 = geometry.declare_component(function_search_names=['Pylon_08, 0'], name='pylon8')
pylon9 = geometry.declare_component(function_search_names=['Pylon_09, 0'], name='pylon9')
pylon10 = geometry.declare_component(function_search_names=['Pylon_10, 0'], name='pylon10')
pylon11 = geometry.declare_component(function_search_names=['Pylon_11, 0'], name='pylon11')
pylon12 = geometry.declare_component(function_search_names=['Pylon_12, 0'], name='pylon12')

nacelle7 = geometry.declare_component(function_search_names=['HLNacelle_7_Tail'], name='nacelle7')
nacelle8 = geometry.declare_component(function_search_names=['HLNacelle_8_Tail'], name='nacelle8')
nacelle9 = geometry.declare_component(function_search_names=['HLNacelle_9_Tail'], name='nacelle9')
nacelle10 = geometry.declare_component(function_search_names=['HLNacelle_10_Tail'], name='nacelle10')
nacelle11 = geometry.declare_component(function_search_names=['HLNacelle_11_Tail'], name='nacelle11')
nacelle12 = geometry.declare_component(function_search_names=['HLNacelle_12_Tail'], name='nacelle12')

spinner1 = geometry.declare_component(function_search_names=['HL_Spinner12, 1'], name='spinner1')
spinner2 = geometry.declare_component(function_search_names=['HL_Spinner11, 1'], name='spinner2')
spinner3 = geometry.declare_component(function_search_names=['HL_Spinner10, 1'], name='spinner3')
spinner4 = geometry.declare_component(function_search_names=['HL_Spinner9, 1'], name='spinner4')
spinner5 = geometry.declare_component(function_search_names=['HL_Spinner8, 1'], name='spinner5')
spinner6 = geometry.declare_component(function_search_names=['HL_Spinner7, 1'], name='spinner6')
spinner7 = geometry.declare_component(function_search_names=['HL_Spinner7, 0'], name='spinner7')
spinner8 = geometry.declare_component(function_search_names=['HL_Spinner8, 0'], name='spinner8')
spinner9 = geometry.declare_component(function_search_names=['HL_Spinner9, 0'], name='spinner9')
spinner10 = geometry.declare_component(function_search_names=['HL_Spinner10, 0'], name='spinner10')
spinner11 = geometry.declare_component(function_search_names=['HL_Spinner11, 0'], name='spinner11')
spinner12 = geometry.declare_component(function_search_names=['HL_Spinner12, 0'], name='spinner12')

prop = geometry.declare_component(function_search_names=['HL-Prop'], name='prop')
motor = geometry.declare_component(function_search_names=['HL_Motor'], name='motor')
motor_interface = geometry.declare_component(function_search_names=['HL_Motor_Controller_Interface'], name='motor_interface')

cruise_spinner1 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 0'], name='cruise_spinner1')
cruise_spinner2 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 1'], name='cruise_spinner2')

wingALL = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4','Rt_Aileron','Lt_Aileron','Flap, 0','Flap, 1',
                                                            'Pylon_12, 1','Pylon_11, 1','Pylon_10, 1','Pylon_09, 1','Pylon_08, 1','Pylon_07, 1',
                                                            'Pylon_12, 0','Pylon_11, 0','Pylon_10, 0','Pylon_09, 0','Pylon_08, 0','Pylon_07, 0',
                                                            'HL_Spinner12, 1','HL_Spinner11, 1','HL_Spinner10, 1','HL_Spinner9, 1','HL_Spinner8, 1','HL_Spinner7, 1',
                                                            'HL_Spinner12, 0','HL_Spinner11, 0','HL_Spinner10, 0','HL_Spinner9, 0','HL_Spinner8, 0','HL_Spinner7, 0',
                                                            'CruiseNacelle-Spinner, 0','CruiseNacelle-Spinner, 1', 'Flap_Cover_7','Flap_Cover_9','Flap_Cover_11'], name='CompleteWing')

                                                     


cruise_motor =  geometry.declare_component(function_search_names=['CruiseNacelle-Motor'], name='cruise_motor')
cruise_nacelle =  geometry.declare_component(function_search_names=['CruiseNacelle-Tail'], name='cruise_nacelle')
cruise_prop = geometry.declare_component(function_search_names=['Cruise-Prop'], name='cruise_prop')

M1_components = [pylon7, nacelle7, spinner1, prop, motor, motor_interface]
M2_components = [pylon8, nacelle8, spinner2, prop, motor, motor_interface]
M3_components = [pylon9, nacelle9, spinner3, prop, motor, motor_interface]
M4_components = [pylon10, nacelle10, spinner4, prop, motor, motor_interface]
M5_components = [pylon11, nacelle11, spinner5, prop, motor, motor_interface]
M6_components = [pylon12, nacelle12, spinner6, prop, motor, motor_interface]
M7_components = [pylon7, nacelle7, spinner7, prop, motor, motor_interface]
M8_components = [pylon8, nacelle8, spinner8, prop, motor, motor_interface]
M9_components = [pylon9, nacelle9, spinner9, prop, motor, motor_interface]
M10_components = [pylon10, nacelle10, spinner10, prop, motor, motor_interface]
M11_components = [pylon11, nacelle11, spinner11, prop, motor, motor_interface]
M12_components = [pylon12, nacelle12, spinner12, prop, motor, motor_interface]
CM1_components = [cruise_nacelle, cruise_spinner1, cruise_prop, cruise_motor]
CM2_components = [cruise_nacelle, cruise_spinner2, cruise_prop, cruise_motor]
total_HL_motor_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components]
total_prop_sys_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components, CM1_components, CM2_components]
# geometry.plot()


# Wing Region Info
wing_le_left_guess = np.array([-12.356, -16, -5.5])*ft2m
wing_le_left_parametric = wing.project(wing_le_left_guess, plot=False)
wing_le_left = geometry.evaluate(wing_le_left_parametric)

wing_le_right_guess = np.array([-12.356, 16, -5.5])*ft2m
wing_le_right_parametric = wing.project(wing_le_right_guess, plot=False)
wing_le_right = geometry.evaluate(wing_le_right_parametric)

wing_le_center_guess = np.array([-12.356, 0., -5.5])*ft2m
wing_le_center_parametric = wing.project(wing_le_center_guess, plot=False)
wing_le_center = geometry.evaluate(wing_le_center_parametric)

wing_te_left_guess = np.array([-14.25, -16, -5.5])*ft2m
wing_te_left_parametric = wing.project(wing_te_left_guess, plot=False)
wing_te_left = geometry.evaluate(wing_te_left_parametric)

wing_te_right_guess = np.array([-14.25, 16, -5.5])*ft2m
wing_te_right_parametric = wing.project(wing_te_right_guess, plot=False)
wing_te_right = geometry.evaluate(wing_te_right_parametric)

wing_te_center_guess = np.array([-14.25, 0., -5.5])*ft2m
wing_te_center_parametric = wing.project(wing_te_center_guess, plot=False)
wing_te_center = geometry.evaluate(wing_te_center_parametric)

wing_qc_center_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False)
wing_qc_tip_right_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), 16., -5.5])*ft2m, plot=False)
wing_qc_tip_left_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), -16., -5.5])*ft2m, plot=False)


wing_parametric_geometry = [
    wing_le_left_parametric,
    wing_le_right_parametric,
    wing_le_center_parametric,
    wing_te_left_parametric,
    wing_te_right_parametric,
    wing_te_center_parametric,
    wing_qc_center_parametric,
    wing_qc_tip_right_parametric,
    wing_qc_tip_left_parametric
]

wingspan = np.linalg.norm(wing_le_left.value - wing_le_right.value)

## ADD CONTROL SURFACE INFO PROJECTIONS HERE

left_aileron_le_left_guess = np.array([-13.85, -15.5, -7.5])*ft2m
left_aileron_le_left_parametric = aileronL.project(left_aileron_le_left_guess, plot=False)
left_aileron_le_left = geometry.evaluate(left_aileron_le_left_parametric)

left_aileron_le_right_guess = np.array([-13.85, -11.4, -7.5])*ft2m
left_aileron_le_right_parametric = aileronL.project(left_aileron_le_right_guess, plot=False)
left_aileron_le_right = geometry.evaluate(left_aileron_le_right_parametric)

left_aileron_le_center_guess = np.array([-13.85, -13.4, -7.5])*ft2m
left_aileron_le_center_parametric = aileronL.project(left_aileron_le_center_guess, plot=False)
left_aileron_le_center = geometry.evaluate(left_aileron_le_center_parametric)

left_aileron_le_center_on_wing_te_guess = left_aileron_le_center_guess
left_aileron_le_center_on_wing_te_parametric = wing.project(left_aileron_le_center_on_wing_te_guess, plot=False)

left_aileron_te_center_parametric = aileronL.project(np.array([-14.25, -13.4, -7.5])*ft2m, plot=False)
left_aileron_te_center = geometry.evaluate(left_aileron_te_center_parametric)

left_aileron_te_left_parametric = aileronL.project(np.array([-14.25, -15.5, -7.5])*ft2m, plot=False)
left_aileron_te_left = geometry.evaluate(left_aileron_te_left_parametric)

left_aileron_te_right_parametric = aileronL.project(np.array([-14.25, -11.4, -7.5])*ft2m, plot=False)
left_aileron_te_right = geometry.evaluate(left_aileron_te_right_parametric)

left_aileron_qc_center_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -13.4, -7.5])*ft2m, plot=False)
left_aileron_qc = geometry.evaluate(left_aileron_qc_center_parametric)
left_aileron_qc_tip_right_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.4, -7.5])*ft2m, plot=False)
left_aileron_qc_tip_left_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -15.15, -7.5])*ft2m, plot=False)

left_aileron_span = np.linalg.norm(left_aileron_le_left.value - left_aileron_le_right.value)
left_aileron_chord = np.linalg.norm(left_aileron_le_center[0].value - left_aileron_te_center[0].value)

left_aileron_parametric_geometry = [
    left_aileron_le_left_parametric,
    left_aileron_le_right_parametric,
    left_aileron_le_center_parametric,
    left_aileron_te_left_parametric,
    left_aileron_te_right_parametric,
    left_aileron_te_center_parametric,
    left_aileron_qc_center_parametric,
    left_aileron_qc_tip_right_parametric,
    left_aileron_qc_tip_left_parametric
]

right_aileron_le_left_guess = np.array([-13.85, 11.4, -7.5])*ft2m
right_aileron_le_left_parametric = aileronR.project(right_aileron_le_left_guess, plot=False)
right_aileron_le_left = geometry.evaluate(right_aileron_le_left_parametric)

right_aileron_le_right_guess = np.array([-13.85, 15.5, -7.5])*ft2m
right_aileron_le_right_parametric = aileronR.project(right_aileron_le_right_guess, plot=False)
right_aileron_le_right = geometry.evaluate(right_aileron_le_right_parametric)

right_aileron_le_center_guess = np.array([-13.85, 13.4, -7.5])*ft2m
right_aileron_le_center_parametric = aileronR.project(right_aileron_le_center_guess, plot=False)
right_aileron_le_center = geometry.evaluate(right_aileron_le_center_parametric)

right_aileron_le_center_on_wing_te_guess = right_aileron_le_center_guess
right_aileron_le_center_on_wing_te_parametric = wing.project(right_aileron_le_center_on_wing_te_guess, plot=False)

right_aileron_te_center_parametric = aileronR.project(np.array([-14.25, 13.4, -7.5])*ft2m, plot=False)
right_aileron_te_center = geometry.evaluate(right_aileron_te_center_parametric)

right_aileron_te_left_parametric = aileronR.project(np.array([-14.25, 11.4, -7.5])*ft2m, plot=False)
right_aileron_te_left = geometry.evaluate(right_aileron_te_left_parametric)

right_aileron_te_right_parametric = aileronR.project(np.array([-14.25, 15.5, -7.5])*ft2m, plot=False)
right_aileron_te_right = geometry.evaluate(right_aileron_te_right_parametric)

right_aileron_qc_center_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 13.12, -7.5])*ft2m, plot=False)
right_aileron_qc = geometry.evaluate(right_aileron_qc_center_parametric)
right_aileron_qc_tip_right_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 11.4, -7.5])*ft2m, plot=False)
right_aileron_qc_tip_left_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 15.15, -7.5])*ft2m, plot=False)
                                                         
right_aileron_parametric_geometry = [
    right_aileron_le_left_parametric,
    right_aileron_le_right_parametric,
    right_aileron_le_center_parametric,
    right_aileron_te_left_parametric,
    right_aileron_te_right_parametric,
    right_aileron_te_center_parametric,
    right_aileron_qc_center_parametric,
    right_aileron_qc_tip_right_parametric,
    right_aileron_qc_tip_left_parametric
]


right_aileron_span = np.linalg.norm(right_aileron_le_left.value - right_aileron_le_right.value)
right_aileron_chord = np.linalg.norm(right_aileron_le_center[0].value - right_aileron_te_center[0].value)

left_flap_le_left_guess = np.array([-13.85, -11.5, -7.5])*ft2m
left_flap_le_left_parametric = flapL.project(left_flap_le_left_guess, plot=False)
left_flap_le_left = geometry.evaluate(left_flap_le_left_parametric)

left_flap_le_right_guess = np.array([-13.85, -0.6, -7.5])*ft2m
left_flap_le_right_parametric = flapL.project(left_flap_le_right_guess, plot=False)
left_flap_le_right = geometry.evaluate(left_flap_le_right_parametric)

left_flap_le_center_guess = np.array([-13.85, -6.05, -7.5])*ft2m
left_flap_le_center_parametric = flapL.project(left_flap_le_center_guess, plot=False)
left_flap_le_center = geometry.evaluate(left_flap_le_center_parametric)

left_flap_le_center_on_wing_te_guess = left_flap_le_center_guess
left_flap_le_center_on_wing_te_parametric = wing.project(left_flap_le_center_on_wing_te_guess, plot=False)

left_flap_te_center_parametric = flapL.project(np.array([-14.25, -6.05, -5.5])*ft2m, plot=False)
left_flap_te_center = geometry.evaluate(left_flap_te_center_parametric)

left_flap_te_left_parametric = flapL.project(np.array([-14.25, -11.5, -5.5])*ft2m, plot=False)
left_flap_te_left = geometry.evaluate(left_flap_te_left_parametric)

left_flap_te_right_parametric = flapL.project(np.array([-14.25, -0.6, -5.5])*ft2m, plot=False)
left_flap_te_right = geometry.evaluate(left_flap_te_right_parametric)

left_flap_qc_center_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -6.56, -5.5])*ft2m, plot=False)
left_flap_qc = geometry.evaluate(left_flap_qc_center_parametric)
left_flap_qc_tip_right_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -0.6, -5.5])*ft2m, plot=False)
left_flap_qc_tip_left_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.48, -5.5])*ft2m, plot=False)
                                                  
left_flap_parametric_geometry = [
    left_flap_le_left_parametric,
    left_flap_le_right_parametric,
    left_flap_le_center_parametric,
    left_flap_te_left_parametric,
    left_flap_te_right_parametric,
    left_flap_te_center_parametric,
    left_flap_qc_center_parametric,
    left_flap_qc_tip_right_parametric,
    left_flap_qc_tip_left_parametric
]

left_flap_span = np.linalg.norm(left_flap_le_left.value - left_flap_le_right.value)
left_flap_chord = np.linalg.norm(left_flap_le_center[0].value - left_flap_te_center[0].value)


right_flap_le_left_guess = np.array([-13.85, 0.6, -7.5])*ft2m
right_flap_le_left_parametric = flapR.project(right_flap_le_left_guess, plot=False)
right_flap_le_left = geometry.evaluate(right_flap_le_left_parametric)

right_flap_le_right_guess = np.array([-13.85, 11.5, -7.5])*ft2m
right_flap_le_right_parametric = flapR.project(right_flap_le_right_guess, plot=False)
right_flap_le_right = geometry.evaluate(right_flap_le_right_parametric)

right_flap_le_center_guess = np.array([-13.85, 6.05, -7.5])*ft2m
right_flap_le_center_parametric = flapR.project(right_flap_le_center_guess, plot=False)
right_flap_le_center = geometry.evaluate(right_flap_le_center_parametric)

right_flap_le_center_on_wing_te_guess = right_flap_le_center_guess
right_flap_le_center_on_wing_te_parametric = wing.project(right_flap_le_center_on_wing_te_guess, plot=False)

right_flap_te_center_parametric = flapR.project(np.array([-14.25, 6.05, -7.5])*ft2m, plot=False)
right_flap_te_center = geometry.evaluate(right_flap_te_center_parametric)

right_flap_te_left_parametric = flapR.project(np.array([-14.25, 0.6, -7.5])*ft2m, plot=False)
right_flap_te_left = geometry.evaluate(right_flap_te_left_parametric)

right_flap_te_right_parametric = flapR.project(np.array([-14.25, 11.5, -7.5])*ft2m, plot=False)
right_flap_te_right = geometry.evaluate(right_flap_te_right_parametric)

right_flap_qc_center_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 6.56, -7.5])*ft2m, plot=False)
right_flap_qc = geometry.evaluate(right_flap_qc_center_parametric)
right_flap_qc_tip_right_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 11.48, -7.5])*ft2m, plot=False)
right_flap_qc_tip_left_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 1.5, -7.5])*ft2m, plot=False)

right_flap_parametric_geometry = [
    right_flap_le_left_parametric,
    right_flap_le_right_parametric,
    right_flap_le_center_parametric,
    right_flap_te_left_parametric,
    right_flap_te_right_parametric,
    right_flap_te_center_parametric,
    right_flap_qc_center_parametric,
    right_flap_qc_tip_right_parametric,
    right_flap_qc_tip_left_parametric
]

right_flap_span = np.linalg.norm(right_flap_le_left.value - right_flap_le_right.value)
right_flap_chord = np.linalg.norm(right_flap_le_center[0].value - right_flap_te_center[0].value)

# HT Region Info
ht_le_left_parametric = h_tail.project(np.array([-26.5, -5.25, -5.5])*ft2m, plot=False)
ht_le_left = geometry.evaluate(ht_le_left_parametric)

ht_le_center_parametric = h_tail.project(np.array([-27, 0., -5.5])*ft2m, plot=False)
ht_le_center = geometry.evaluate(ht_le_center_parametric)

ht_le_right_parametric = h_tail.project(np.array([-26.5, 5.25, -5.5])*ft2m, plot=False)
ht_le_right = geometry.evaluate(ht_le_right_parametric)

ht_te_left_parametric = h_tail.project(np.array([-30, -5.25, -5.5])*ft2m, plot=False)
ht_te_left = geometry.evaluate(ht_te_left_parametric)

ht_te_center_guess = np.array([-30, 0., -5.5])*ft2m
ht_te_center_parametric = h_tail.project(ht_te_center_guess, plot=False)
ht_te_center = geometry.evaluate(h_tail.project(ht_te_center_guess, plot=False))

ht_te_right_parametric = h_tail.project(np.array([-30, 5.25, -5.5])*ft2m, plot=False)
ht_te_right = geometry.evaluate(ht_te_right_parametric)

ht_qc_center_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False)
ht_qc = geometry.evaluate(ht_qc_center_parametric)
ht_qc_tip_right_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), 5.25, -5.5])*ft2m, plot=False)
ht_qc_tip_left_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), -5.25, -5.5])*ft2m, plot=False)
                                            
ht_qc = geometry.evaluate(h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))

ht_parametric_geometry = [
    ht_le_left_parametric,
    ht_le_right_parametric,
    ht_le_center_parametric,
    ht_te_left_parametric,
    ht_te_right_parametric,
    ht_te_center_parametric,
    ht_qc_center_parametric,
    ht_qc_tip_right_parametric,
    ht_qc_tip_left_parametric
]
ht_span = np.linalg.norm(ht_le_left.value - ht_le_right.value)
ht_chord = np.linalg.norm(ht_le_center[0].value - ht_te_center[0].value)

trimTab_le_left_parametric = trimTab.project(np.array([-29.4, -3.3, -5.5])*ft2m, plot=False)
trimTab_le_right_parametric = trimTab.project(np.array([-29.4, 3.3, -5.5])*ft2m, plot=False)
trimTab_le_left = geometry.evaluate(trimTab_le_left_parametric)
trimTab_le_center_parametric = trimTab.project(np.array([-29.4, 0, -5.5])*ft2m, plot=False)
trimTab_le_center = geometry.evaluate(trimTab_le_center_parametric)
trimTab_le_right = geometry.evaluate(trimTab_le_right_parametric)

trimTab_te_left_parametric = trimTab.project(np.array([-30, -3.3, -5.5])*ft2m, plot=False)
trimTab_te_center_parametric = trimTab.project(np.array([-30, 0, -5.5])*ft2m, plot=False)
trimTab_te_right_parametric = trimTab.project(np.array([-30, 3.3, -5.5])*ft2m, plot=False)

trimTab_te_center = geometry.evaluate(trimTab.project(np.array([-30, 0, -5.5])*ft2m, plot=False))

trimTab_qc_center_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), 0, -5.5])*ft2m, plot=False)
trimTab_qc_tip_right_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), 3.3, -5.5])*ft2m, plot=False)
trimTab_qc_tip_left_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), -3.3, -5.5])*ft2m, plot=False)                                                  


trimTab_parametric_geometry = [
    trimTab_le_left_parametric,
    trimTab_le_right_parametric,
    trimTab_le_center_parametric,
    trimTab_te_left_parametric,
    trimTab_te_right_parametric,
    trimTab_te_center_parametric,
    trimTab_qc_center_parametric,
    trimTab_qc_tip_right_parametric,
    trimTab_qc_tip_left_parametric
]


trim_tab_span = np.linalg.norm(trimTab_le_left.value - trimTab_le_right.value)

trimTab_chord = np.linalg.norm(trimTab_le_center[0].value - trimTab_te_center[0].value)

# VT Region Info
vt_le_base_parametric = vertTail.project(np.array([-23, 0, -5.5])*ft2m, plot=False)
vt_le_base = geometry.evaluate(vt_le_base_parametric)
vt_le_mid_parametric = vertTail.project(np.array([-26, 0., -8])*ft2m, plot=False)
vt_le_mid = geometry.evaluate(vt_le_mid_parametric)
vt_le_tip_parametric = vertTail.project(np.array([-28.7, 0, -11])*ft2m, plot=False)
vt_le_tip = geometry.evaluate(vt_le_tip_parametric)

vt_te_base_parametric = vertTail.project(np.array([-27.75, 0, -5.5])*ft2m, plot=False)
vt_te_base = geometry.evaluate(vt_te_base_parametric)
vt_te_mid_parametric = vertTail.project(np.array([-28.7, 0., -8])*ft2m, plot=False)
vt_te_mid= geometry.evaluate(vt_te_mid_parametric)
vt_te_tip_parametric = vertTail.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False)
vt_te_tip = geometry.evaluate(vt_te_tip_parametric)
vt_qc = geometry.evaluate(vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m, plot=False))
vt_qc_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)
vt_qc_base_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)
vt_qc_tip_parametric = vertTail.project(np.array([-27.33 + (0.25*(-28.7+23)), 0., -10.85])*ft2m,plot=False)


vt_span = np.linalg.norm(vt_te_base.value - vt_te_tip.value)

vt_chord = np.linalg.norm(vt_le_mid.value - vt_te_mid.value)

rudder_le_base_parametric = rudder.project(np.array([-23, 0, -5.5])*ft2m, plot=False)
rudder_le_base = geometry.evaluate(rudder_le_base_parametric)
rudder_le_mid_parametric = rudder.project(np.array([-28.7, 0., -8.5])*ft2m, plot=False)
rudder_le_mid = geometry.evaluate(rudder_le_mid_parametric)
rudder_le_tip = geometry.evaluate(rudder.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False))

rudder_te_base_parametric = rudder.project(np.array([-29.5, 0, -5.5])*ft2m, plot=False)
rudder_te_mid_parametric = rudder.project(np.array([-30., 0., -8.5])*ft2m, plot=False)
rudder_te_tip_parametric = rudder.project(np.array([-30.4, 0, -11.])*ft2m, plot=False)


rudder_span = np.linalg.norm(rudder_le_base.value - rudder_le_tip.value)
rudder_chord = np.linalg.norm(rudder_le_mid.value - rudder_le_tip.value)

vt_parametric_geometry = [
    vt_le_base_parametric,
    vt_le_tip_parametric,
    vt_le_base_parametric,
    rudder_te_base_parametric,
    rudder_te_tip_parametric,
    rudder_te_mid_parametric,
    vt_qc_parametric,
    vt_qc_base_parametric,
    vt_qc_tip_parametric
]

# Fuselage Region Info
fuselage_wing_le_center_parametric = fuselage.project(np.array([-12.356, 0., -5.5])*ft2m, plot=False)
fuselage_wing_qc_center_parametric = fuselage.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False)
fuselage_wing_qc = geometry.evaluate(fuselage_wing_qc_center_parametric)
fuselage_wing_te_center_parametric = fuselage.project(np.array([-14.25, 0., -5.5])*ft2m, plot=False)
fuselage_wing_te_center = geometry.evaluate(fuselage_wing_te_center_parametric)
fuselage_tail_qc = geometry.evaluate(fuselage.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))
fuselage_tail_te_center_parametric = fuselage.project(np.array([-30, 0., -5.5])*ft2m, plot=False)
fuselage_tail_te_center = geometry.evaluate(fuselage_tail_te_center_parametric)


# Propeller Region Info
M12_disk_pt =  np.array([-12.5, 14, -7.355])*ft2m
M1_disk_pt = np.array([-12.5, -14, -7.355])*ft2m
M11_disk_pt =  np.array([-12.35, 12, -7.355])*ft2m
M2_disk_pt = np.array([-12.35, -12, -7.355])*ft2m
M10_disk_pt = np.array([-12.2, 10, -7.659])*ft2m
M3_disk_pt = np.array([-12.2, -10, -7.659])*ft2m
M9_disk_pt = np.array([-12, 8, -7.659])*ft2m
M4_disk_pt = np.array([-12, -8, -7.659])*ft2m
M8_disk_pt = np.array([-11.8, 6, -7.659])*ft2m
M5_disk_pt = np.array([-11.8, -6, -7.659])*ft2m
M7_disk_pt = np.array([-11.6, 4, -7.659])*ft2m
M6_disk_pt = np.array([-11.6, -4, -7.659])*ft2m

M1_disk_on_wing = spinner1.project(M1_disk_pt, plot=False)
M1_disk = geometry.evaluate(M1_disk_on_wing)

M2_disk_on_wing = spinner2.project(M2_disk_pt, plot=False)
M2_disk = geometry.evaluate(M2_disk_on_wing)

M3_disk_on_wing = spinner3.project(M3_disk_pt, plot=False)
M3_disk = geometry.evaluate(M3_disk_on_wing)

M4_disk_on_wing = spinner4.project(M4_disk_pt, plot=False)
M4_disk = geometry.evaluate(M4_disk_on_wing)

M5_disk_on_wing = spinner5.project(M5_disk_pt, plot=False)
M5_disk = geometry.evaluate(M5_disk_on_wing)

M6_disk_on_wing = spinner6.project(M6_disk_pt, plot=False)
M6_disk = geometry.evaluate(M6_disk_on_wing)

M7_disk_on_wing = spinner7.project(M7_disk_pt, plot=False)
M7_disk = geometry.evaluate(M7_disk_on_wing)

M8_disk_on_wing = spinner8.project(M8_disk_pt, plot=False)
M8_disk = geometry.evaluate(M8_disk_on_wing)

M9_disk_on_wing = spinner9.project(M9_disk_pt, plot=False)
M9_disk = geometry.evaluate(M9_disk_on_wing)

M10_disk_on_wing = spinner10.project(M10_disk_pt, plot=False)
M10_disk = geometry.evaluate(M10_disk_on_wing)

M11_disk_on_wing = spinner11.project(M11_disk_pt, plot=False)
M11_disk = geometry.evaluate(M11_disk_on_wing)

M12_disk_on_wing = spinner12.project(M12_disk_pt, plot=False)
M12_disk = geometry.evaluate(M12_disk_on_wing)

MotorDisks = [M1_disk, M2_disk, M3_disk, M4_disk, M5_disk, M6_disk, M7_disk, M8_disk, M9_disk, M10_disk, M11_disk, M12_disk]


M1_pylon_pt = np.array([-12.5, -14, -7.355])*ft2m 
M2_pylon_pt = np.array([-12.35, -12, -7.355])*ft2m
M3_pylon_pt = np.array([-12.2, -10, -7.659])*ft2m
M4_pylon_pt = np.array([-12, -8, -7.659])*ft2m
M5_pylon_pt = np.array([-11.8, -6, -7.659])*ft2m
M6_pylon_pt = np.array([-11.6, -4, -7.659])*ft2m
M7_pylon_pt = np.array([-11.6, 4, -7.659])*ft2m
M8_pylon_pt = np.array([-11.8, 6, -7.659])*ft2m
M9_pylon_pt = np.array([-12, 8, -7.659])*ft2m
M10_pylon_pt = np.array([-12.2, 10, -7.659])*ft2m
M11_pylon_pt = np.array([-12.35, 12, -7.355])*ft2m
M12_pylon_pt = np.array([-12.5, 14, -7.355])*ft2m

M1_pylon_on_wing = pylon1.project(M1_pylon_pt, plot=False)
M2_pylon_on_wing = pylon2.project(M2_pylon_pt, plot=False)
M3_pylon_on_wing = pylon3.project(M3_pylon_pt, plot=False)
M4_pylon_on_wing = pylon4.project(M4_pylon_pt, plot=False)
M5_pylon_on_wing = pylon5.project(M5_pylon_pt, plot=False)
M6_pylon_on_wing = pylon6.project(M6_pylon_pt, plot=False)
M7_pylon_on_wing = pylon7.project(M7_pylon_pt, plot=False)
M8_pylon_on_wing = pylon8.project(M8_pylon_pt, plot=False)
M9_pylon_on_wing = pylon9.project(M9_pylon_pt, plot=False)
M10_pylon_on_wing = pylon10.project(M10_pylon_pt, plot=False)
M11_pylon_on_wing = pylon11.project(M11_pylon_pt, plot=False)
M12_pylon_on_wing = pylon12.project(M12_pylon_pt, plot=False)




fuselage_nose_guess = np.array([-1.75, 0, -4])*ft2m
fuselage_rear_guess = np.array([-29.5, 0, -5.5])*ft2m
fuselage_nose_pts_parametric = fuselage.project(fuselage_nose_guess, grid_search_density_parameter=20, plot=False)
fuselage_nose = geometry.evaluate(fuselage_nose_pts_parametric)
fuselage_rear_pts_parametric = fuselage.project(fuselage_rear_guess, plot=False)
fuselage_rear = geometry.evaluate(fuselage_rear_pts_parametric)

# For Cruise Motor 1 Hub Region
cruise_motor1_tip_guess = np.array([-13, -15.83, -5.5])*ft2m
cruise_motor1_tip_parametric = cruise_spinner1.project(cruise_motor1_tip_guess, plot=False)
cruise_motor1_tip = geometry.evaluate(cruise_motor1_tip_parametric)

cruise_motor1_base_guess = cruise_motor1_tip + np.array([-1.67, 0, 0])*ft2m
cruise_motor1_base_parametric = cruise_spinner1.project(cruise_motor1_base_guess, plot=False)
cruise_motor1_base= geometry.evaluate(cruise_motor1_base_parametric)

# For Cruise Motor 2 Hub Region
cruise_motor2_tip_guess = np.array([-13, 15.83, -5.5])*ft2m
cruise_motor2_tip_parametric = cruise_spinner2.project(cruise_motor2_tip_guess, plot=False)
cruise_motor2_tip = geometry.evaluate(cruise_motor1_tip_parametric)

cruise_motor2_base_guess = cruise_motor2_tip + np.array([-1.67, 0, 0])*ft2m
cruise_motor2_base_parametric = cruise_spinner2.project(cruise_motor2_base_guess, plot=False)
cruise_motor2_base= geometry.evaluate(cruise_motor2_base_parametric)

wing_on_cruise_motor1_parametric = cruise_spinner2.project(wing_le_left_guess, plot=False)
wing_on_cruise_motor2_parametric = cruise_spinner1.project(wing_le_right_guess, plot=False)





## AXIS/AXISLSDOGEO CREATION


def axes_create():

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    # OpenVSP Model Axis
    openvsp_axis = Axis(
        name='OpenVSP Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        origin=ValidOrigins.OpenVSP.value

    )
    
    wing_axis = AxisLsdoGeo(
        name='Wing Axis',
        geometry=wing,
        parametric_coords=wing_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    left_flap_axis = AxisLsdoGeo(
        name='Left Flap Axis',
        geometry=flapL,
        parametric_coords=left_flap_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_flap_axis = AxisLsdoGeo(
        name='Right Flap Axis',
        geometry=flapR,
        parametric_coords=right_flap_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    left_aileron_axis = AxisLsdoGeo(
        name='Left Aileron Axis',
        geometry=aileronL,
        parametric_coords=left_aileron_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_aileron_axis = AxisLsdoGeo(
        name='Right Aileron Axis',
        geometry=aileronR,
        parametric_coords=right_aileron_le_left_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Tail Region Axis

    ht_tail_axis = AxisLsdoGeo(
        name='Horizontal Tail Axis',
        geometry=h_tail,
        parametric_coords=ht_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    trimTab_axis = AxisLsdoGeo(
        name='Trim Tab Axis',
        geometry=trimTab,
        parametric_coords=trimTab_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=ht_tail_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    vt_tail_axis = AxisLsdoGeo(
        name='Vertical Tail Axis',
        geometry=rudder,
        parametric_coords=rudder_le_mid_parametric,
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Distributed Propulsion Motors Axes

    M1_axis = AxisLsdoGeo(
        name= 'Motor 1 Axis',
        geometry=spinner1,
        parametric_coords=M1_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    M2_axis = AxisLsdoGeo(
        name= 'Motor 2 Axis',
        geometry=spinner2,
        parametric_coords=M2_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    M3_axis = AxisLsdoGeo(
        name= 'Motor 3 Axis',
        geometry=spinner3,
        parametric_coords=M3_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M4_axis = AxisLsdoGeo(
        name= 'Motor 4 Axis',
        geometry=spinner4,
        parametric_coords=M4_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M5_axis = AxisLsdoGeo(
        name= 'Motor 5 Axis',
        geometry=spinner5,
        parametric_coords=M5_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M6_axis = AxisLsdoGeo(
        name= 'Motor 6 Axis',
        geometry=spinner6,
        parametric_coords=M6_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M7_axis = AxisLsdoGeo(
        name= 'Motor 7 Axis',
        geometry=spinner7,
        parametric_coords=M7_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M8_axis = AxisLsdoGeo(
        name= 'Motor 8 Axis',
        geometry=spinner8,
        parametric_coords=M8_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M9_axis = AxisLsdoGeo(
        name= 'Motor 9 Axis',
        geometry=spinner9,
        parametric_coords=M9_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M10_axis = AxisLsdoGeo(
        name= 'Motor 10 Axis',
        geometry=spinner10,
        parametric_coords=M10_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M11_axis = AxisLsdoGeo(
        name= 'Motor 11 Axis',
        geometry=spinner11,
        parametric_coords=M11_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M12_axis = AxisLsdoGeo(
        name= 'Motor 12 Axis',
        geometry=spinner12,
        parametric_coords=M12_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    HL_motor_axes = [M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis]
    # Cruise Motor Region


    cruise_motor1_axis = AxisLsdoGeo(
        name= 'Cruise Motor 1 Axis',
        geometry=cruise_spinner1,
        parametric_coords=cruise_motor1_tip_parametric,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    cruise_motor2_axis = AxisLsdoGeo(
        name= 'Cruise Motor 2 Axis',
        geometry=cruise_spinner2,
        parametric_coords=cruise_motor2_tip_parametric,
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    cruise_motor_axes = [cruise_motor1_axis, cruise_motor2_axis]


    fd_axis = Axis(
        name='Flight Dynamics Body Fixed Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    wind_axis = Axis(
        name='Wind Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    return openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geometry, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis

openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis,geometry,left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis = axes_create()






## Aircraft Component Creation

parameterization_solver = ParameterizationSolver()
ffd_geometric_variables = GeometricVariables()




Aircraft = AircraftComp(geometry=geometry, compute_surface_area_flag=False, 
                        parameterization_solver=parameterization_solver,
                        ffd_geometric_variables=ffd_geometric_variables)



Fuselage = FuseComp(
    length=csdl.Variable(name="length", shape=(1, ), value=8.2242552),
    max_height=csdl.Variable(name="max_height", shape=(1, ), value=1.09236312),
    max_width=csdl.Variable(name="max_width", shape=(1, ), value=1.24070602),
    geometry=fuselage, skip_ffd=False, 
    parameterization_solver=parameterization_solver,
    ffd_geometric_variables=ffd_geometric_variables)


Aircraft.add_subcomponent(Fuselage)



aileronL.rotate(left_aileron_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
aileronR.rotate(right_aileron_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
flapL.rotate(left_flap_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
flapR.rotate(right_flap_le_center, np.array([0., 1., 0.]), angles=np.deg2rad(0))
htALL.rotate(ht_qc, np.array([0., 1., 0.]), angles=np.deg2rad(0))
rudder.rotate(rudder_le_mid, np.array([0., 0., 1.]), angles=np.deg2rad(0))



wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=15)
wing_span = csdl.Variable(name="wingspan", shape=(1, ), value=9.6)
wing_sweep = csdl.Variable(name="wing_sweep", shape=(1, ), value=0)
wing_dihedral = csdl.Variable(name="wing_dihedral", shape=(1, ), value=0)

Wing = WingComp(AR=wing_AR,
                span=wing_span,
                sweep=wing_sweep,
                dihedral=wing_dihedral,
                geometry=wingALL,
                parametric_geometry=wing_parametric_geometry,
                tight_fit_ffd=False, 
                orientation='horizontal', 
                name='Wing', parameterization_solver=parameterization_solver, 
                ffd_geometric_variables=ffd_geometric_variables,
                do_lift_model=True,
                wing_axis=wing_axis,
                )

Aircraft.add_subcomponent(Wing)
wing_qc_fuse_connection = geometry.evaluate(wing_qc_center_parametric) - geometry.evaluate(fuselage_wing_qc_center_parametric)
parameterization_solver.add_variable(computed_value=wing_qc_fuse_connection, desired_value=wing_qc_fuse_connection.value)




HorTailArea = ht_span*ht_chord
htAR = ht_span**2/HorTailArea
HorTail_AR = csdl.Variable(name="HT_AR", shape=(1, ), value=4)
HT_span = csdl.Variable(name="HT_span", shape=(1, ), value=3.14986972)
HT_sweep = csdl.Variable(name="HT_sweep", shape=(1, ), value=0)

HorTail = WingComp(AR=HorTail_AR, span=HT_span, sweep=HT_sweep,
                   geometry=htALL, parametric_geometry=ht_parametric_geometry,
                   tight_fit_ffd=False, skip_ffd=False,
                   name='Horizontal Tail', orientation='horizontal', 
                   parameterization_solver=parameterization_solver,
                   ffd_geometric_variables=ffd_geometric_variables,
                   do_lift_model=True,
                   wing_axis=ht_tail_axis)
Aircraft.add_subcomponent(HorTail)


tail_moment_arm_computed = csdl.norm(geometry.evaluate(ht_qc_center_parametric) - geometry.evaluate(wing_qc_center_parametric))
h_tail_fuselage_connection = geometry.evaluate(ht_te_center_parametric) - geometry.evaluate(fuselage_tail_te_center_parametric)
# parameterization_solver.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm_computed.value)
parameterization_solver.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)



vt_AR= csdl.Variable(name="VT_AR", shape=(1, ), value=1.998)
VT_span = csdl.Variable(name="VT_span", shape=(1, ), value=2.3761728)
VT_actuation_angle = csdl.Variable(name="VT_actuation_angle", shape=(1, ), value=0)
VT_sweep = csdl.Variable(name="VT_sweep", shape=(1, ), value=-40)


# VertTail = WingComp(AR=vt_AR, span=VT_span, sweep=VT_sweep,
#                     geometry=vtALL, parametric_geometry=vt_parametric_geometry,
#                     tight_fit_ffd=False, 
#                     name='Vertical Tail', orientation='vertical',
#                     parameterization_solver=parameterization_solver,
#                     ffd_geometric_variables=ffd_geometric_variables,do_lift_model=False,wing_axis=vt_tail_axis)
# Aircraft.add_subcomponent(VertTail)

vtail_fuselage_connection = geometry.evaluate(fuselage_rear_pts_parametric) - geometry.evaluate(vt_qc_base_parametric)
parameterization_solver.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

lift_rotors = []
for i in range(1, 13):
    HL_motor = Component(name=f'HL Motor {i}', prop_axis=HL_motor_axes[i-1], do_prop_model=True)
    lift_rotors.append(HL_motor)
    Aircraft.add_subcomponent(HL_motor)

cruise_motors = []
for i in range(1, 3):
    cruise_motor = Component(name=f'Cruise Motor {i}',prop_axis=cruise_motor_axes[i-1], do_prop_model=True)
    cruise_motors.append(cruise_motor)
    Aircraft.add_subcomponent(cruise_motor)

Battery = Component(name='Battery')
Aircraft.add_subcomponent(Battery)

LandingGear = Component(name='Landing Gear')
Aircraft.add_subcomponent(LandingGear)

# parameterization_solver.evaluate(ffd_geometric_variables)
# geometry.plot(camera=dict(pos=(12, 15, -12),  # Camera position 
#                          focal_point=(-Fuselage.parameters.length.value/2, 0, 0),  # Point camera looks at
#                          viewup=(0, 0, -1)),    # Camera up direction
#                          title= f'X-57 Maxwell Aircraft Geometry\nWing Span: {Wing.parameters.span.value[0]:.2f} m\nWing AR: {Wing.parameters.AR.value[0]:.2f}\nWing Area S: {Wing.parameters.S_ref.value[0]:.2f} m^2\nWing Sweep: {Wing.parameters.sweep.value[0]:.2f} deg',
#                         #  title=f'X-57 Maxwell Aircraft Geometry\nFuselage Length: {Fuselage.parameters.length.value[0]:.2f} m\nFuselage Height: {Fuselage.parameters.max_height.value[0]:.2f} m\nFuselage Width: {Fuselage.parameters.max_width.value[0]:.2f} m',
#                          screenshot= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57'/ 'images' / f'x_57_{Wing.parameters.span.value[0]}_AR_{Wing.parameters.AR.value[0]}_S_ref_{Wing.parameters.S_ref.value[0]}_sweep_{Wing.parameters.sweep.value[0]}.png')




### FORCES AND MOMENTS MODELLING


x_57_states = AircraftStates(axis=fd_axis,u=Q_(67, 'mph')) # stall speed
x_57_mi = MassMI(axis=fd_axis,
                 Ixx=Q_(4314.08, 'kg*(m*m)'),
                 Ixy=Q_(-232.85, 'kg*(m*m)'),
                 Ixz=Q_(-2563.29, 'kg*(m*m)'),
                 Iyy=Q_(18656.93, 'kg*(m*m)'),
                 Iyz=Q_(-62.42, 'kg*(m*m)'),
                 Izz=Q_(22340.21, 'kg*(m*m)'),
                 )



x57_mass_properties = MassProperties(mass=Q_(1360.77, 'kg'),
                                      inertia=x_57_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))

atmospheric_states = x_57_states.atmospheric_states
x57_controls = AircraftControlSystem(engine_count=12,symmetrical=True)


x57_aircraft = Component(name='X-57')
x57_aircraft.quantities.mass_properties = x57_mass_properties
HL_radius_x57 = csdl.Variable(shape=(1,), value=1.89/2) # HL propeller radius in ft
cruise_radius_x57 = csdl.Variable(shape=(1,), value=5/2) # cruise propeller radius in ft

e_x57 = csdl.Variable(shape=(1,), value=0.87) # Oswald efficiency factor
CD0_x57 = csdl.Variable(shape=(1,), value=0.001) # Zero-lift drag coefficient
incidence_x57 = csdl.Variable(shape=(1,), value=2*np.pi/180) # Wing incidence angle in radians



## Aerodynamic Forces - from Modification IV

lift_models = []
for comp in Aircraft.comps.values():
    if isinstance(comp, WingComp) and getattr(comp, 'do_lift_model', True):
        # Build the lift model with the wing's parameters
        lift_model = LiftModel(
            AR=comp.parameters.AR,
            e=e_x57,
            CD0=CD0_x57,
            S=comp.parameters.S_ref,
            incidence=incidence_x57
        )
        lift_models.append(lift_model)
    

# wings_aero[0].plot_aerodynamics(
#     velocity_range=(20, 200),
#     alpha_range=(-5, 15),
#     AR_values=[15],
#     e_values=[0.75, 0.85, 0.95]
# )





# # # Rotor Forces



# HL_motor_props[1].plot_propulsion(
#     J_range=(0, 1.6), # Compare different advance ratios
#     velocity_range=(0, 200), # Compare different velocity ranges in mph
#     ref_velocities=[50, 75, 80],  # Compare different reference velocities
#     rpm_ranges=[(3428,4702)],  # Compare different RPM ranges
#     radius_values=[0.5, 1, 1.5],  # Different radii in ft
#     ref_throttles=[1.0, 1.0, 1.0],  # Different throttle settings
#     title=f'High-Lift Motor {1}'
# )


# cruise_motor_props[1].plot_propulsion(
#     J_range=(0, 1.8), # Compare different advance ratios
#     velocity_range=(0, 200), # Compare different velocity ranges in mph
#     ref_velocities=[50, 75, 80],  # Compare different reference velocities
#     rpm_ranges=[(1150,2250)],  # Compare different RPM ranges
#     radius_values=[2, 2.5, 3],  # Different radii in ft
#     ref_throttles=[1.0, 1.0, 1.0],  # Different throttle settings
#     title=f'Cruise Motor {1}'
# )
# # plt.show()


all_forces = []
all_moments = []

Wing.quantities.mass_properties.mass = Q_(152.88, 'kg')
Wing.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
wf, wm = Wing.compute_total_loads(fd_state=x_57_states, load_axis=wing_axis, controls=x57_controls, lift_model=lift_models[0], fd_axis=fd_axis)
all_forces.append(wf)
all_moments.append(wm)

Fuselage.quantities.mass_properties.mass = Q_(235.87, 'kg')
Fuselage.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
ff, fm = Fuselage.compute_total_loads(fd_state=x_57_states, load_axis=openvsp_axis, controls=x57_controls, fd_axis=fd_axis)
all_forces.append(ff)
all_moments.append(fm)

HorTail.quantities.mass_properties.mass = Q_(27.3/2, 'kg')
HorTail.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
hf, hm = HorTail.compute_total_loads(fd_state=x_57_states, load_axis=ht_tail_axis, controls=x57_controls, lift_model=lift_models[1], fd_axis=fd_axis)
all_forces.append(hf)
all_moments.append(hm)


# VertTail.quantities.mass_properties.mass = Q_(27.3/2, 'kg')
# VertTail.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
# VertTail.compute_total_loads(fd_state=x_57_states, load_axis=vt_tail_axis, controls=x57_controls, fd_axis=fd_axis)


Battery.quantities.mass_properties.mass = Q_(390.08, 'kg')
Battery.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
bf, bm = Battery.compute_total_loads(fd_state=x_57_states, load_axis=openvsp_axis, controls=x57_controls, fd_axis=fd_axis)
all_forces.append(bf)
all_moments.append(bm)

LandingGear.quantities.mass_properties.mass = Q_(61.15, 'kg')
LandingGear.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
lfg, lgm = LandingGear.compute_total_loads(fd_state=x_57_states, load_axis=openvsp_axis, controls=x57_controls, fd_axis=fd_axis)
all_forces.append(lfg)
all_moments.append(lgm)

HL_motor_cgs = [
    [-15.39, -34.98, -4.2],
    [-13.42, -57.66, -4.2],
    [-15.39, -80.34, -4.2],
    [-11.92, -103.02, -4.2],
    [-13.90, -125.7, -4.2],
    [-10.41, -148.38, -4.2],
    [-15.39,  34.98, -4.2],
    [-13.42,  57.66, -4.2],
    [-15.39,  80.34, -4.2],
    [-11.92, 103.02, -4.2],
    [-13.90, 125.7, -4.2],
    [-10.41, 148.38, -4.2],
]

for i, HL_motor in enumerate(lift_rotors):
    HL_motor.quantities.mass_properties.mass = Q_(81.65/12, 'kg')
    HL_motor.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array(HL_motor_cgs[i]), 'in'), axis=fd_axis)
    HLf, HLm = HL_motor.compute_total_loads(fd_state=x_57_states, load_axis=HL_motor_axes[i-1], controls=x57_controls, radius=HL_radius_x57, prop_curve=HLPropCurve(), fd_axis=fd_axis)
    all_forces.append(HLf)
    all_moments.append(HLm)

cruise_motor_cgs = [
    [-13.01, -189.74, -0.958],
    [-13.01,  189.74, -0.958],
]

for i, cruise_motor in enumerate(cruise_motors):
    cruise_motor.quantities.mass_properties.mass = Q_(106.14/2, 'kg')
    cruise_motor.quantities.mass_properties.cg_vector = Vector(vector=Q_(np.array(cruise_motor_cgs[i]), 'in'), axis=fd_axis)
    cmf, cmm = cruise_motor.compute_total_loads(fd_state=x_57_states, load_axis=cruise_motor_axes[i-1], controls=x57_controls, radius=cruise_radius_x57, prop_curve=CruisePropCurve(), fd_axis=fd_axis)          
    all_forces.append(cmf)
    all_moments.append(cmm)

complete_forces = np.sum(all_forces, axis=0)
complete_moments = np.sum(all_moments, axis=0)
print('Total Aircraft Forces in FD Axis:')
print(complete_forces, 'N')
print('Total Aircraft Moments in FD Axis:')
print(complete_moments, 'N*m')



recorder.stop()