import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
import lsdo_geo as lg

recorder = csdl.Recorder(inline=True)
recorder.start()

in2m=0.0254
ft2m = 0.3048

# geometry = lg.import_geometry("C:/Users/joeyg/OneDrive/Documents/GitHub/Sarojini_Research/MyAirframe/x57.stp", scale=in2ft, parallelize=False)
# geometry.plot()

from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER
from flight_simulator.core.vehicle.component import Component
from flight_simulator.core.loads.mass_properties import MassProperties

geometry = import_geometry(
    file_name="x57.stp",
    file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'x57',
    refit=False,
    scale=in2m,
    rotate_to_body_fixed_frame=True
)
# geometry.plot()



Complete_Aircraft = Component(name='Complete Aircraft')


# region Declaring all components
# Wing, tails, fuselage
wing = geometry.declare_component(function_search_names=['Wing'], name='wing')
# wing.plot()
aileronR = geometry.declare_component(function_search_names=['Rt_Aileron'], name='aileronR')
# aileronR.plot()
aileronL = geometry.declare_component(function_search_names=['Lt_Aileron'], name='aileronL')
flap = geometry.declare_component(function_search_names=['Flap'], name='flap')

Complete_Wing = Component(name='Complete Wing')
Total_Wing = Component(name='Main Wing', geometry=wing)
Total_Wing.add_subcomponent(Component(name='Right Aileron', geometry=aileronR))
Total_Wing.add_subcomponent(Component(name='Left Aileron', geometry=aileronL))
Total_Wing.add_subcomponent(Component(name='Flap', geometry=flap))
Complete_Wing.add_subcomponent(Total_Wing)
# Complete_Wing.visualize_component_hierarchy(show=True)

Complete_Aircraft.add_subcomponent(Complete_Wing)




h_tail = geometry.declare_component(function_search_names=['HorzStab'], name='h_tail')
trimTab = geometry.declare_component(function_search_names=['TrimTab'], name='trimTab')
# trimTab.plot()

vertTail = geometry.declare_component(function_search_names=['VertTail'], name='vertTail')
rudder = geometry.declare_component(function_search_names=['Rudder'], name='rudder')
# rudder.plot()


Total_Tail = Component(name='Complete Tail')
HT_comp = Component(name="Horizontal Tail", geometry=h_tail)
HT_comp.add_subcomponent(Component(name='Trim Tab',geometry=trimTab))
VT_comp = Component(name="Vertical Tail", geometry=vertTail)
VT_comp.add_subcomponent(Component(name='Rudder',geometry=rudder))
Total_Tail.add_subcomponent(HT_comp)
Total_Tail.add_subcomponent(VT_comp)
# Total_Tail.visualize_component_hierarchy(show=True)

Complete_Aircraft.add_subcomponent(Total_Tail)



fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
# fuselage.plot()

fuselage_comp = Component(name="Fuselage", geometry=fuselage)


gear_pod = geometry.declare_component(function_search_names=['GearPod'], name='gear_pod')
fuselage_comp.add_subcomponent(Component(name="Gear Pod", geometry=gear_pod))

Complete_Aircraft.add_subcomponent(fuselage_comp)



# Prop buildup
pylon7 = geometry.declare_component(function_search_names=['Pylon_07'], name='pylon7')
# pylon7.plot()
pylon8 = geometry.declare_component(function_search_names=['Pylon_08'], name='pylon8')
pylon9 = geometry.declare_component(function_search_names=['Pylon_09'], name='pylon9')
pylon10 = geometry.declare_component(function_search_names=['Pylon_10'], name='pylon10')
pylon11 = geometry.declare_component(function_search_names=['Pylon_11'], name='pylon11')
pylon12 = geometry.declare_component(function_search_names=['Pylon_12'], name='pylon12')

nacelle7 = geometry.declare_component(function_search_names=['HLNacelle_7_Tail'], name='nacelle7')
# nacelle7.plot()
nacelle8 = geometry.declare_component(function_search_names=['HLNacelle_8_Tail'], name='nacelle8')
nacelle9 = geometry.declare_component(function_search_names=['HLNacelle_9_Tail'], name='nacelle9')
nacelle10 = geometry.declare_component(function_search_names=['HLNacelle_10_Tail'], name='nacelle10')
nacelle11 = geometry.declare_component(function_search_names=['HLNacelle_11_Tail'], name='nacelle11')
nacelle12 = geometry.declare_component(function_search_names=['HLNacelle_12_Tail'], name='nacelle12')


spinner = geometry.declare_component(function_search_names=['HL_Spinner'], name='spinner')
# spinner.plot()
prop = geometry.declare_component(function_search_names=['HL-Prop'], name='prop')
# prop.plot()
motor = geometry.declare_component(function_search_names=['HL_Motor'], name='motor')
# motor.plot()
motor_interface = geometry.declare_component(function_search_names=['HL_Motor_Controller_Interface'], name='motor_interface')
# motor_interface.plot()




Total_Prop_Sys = Component(name='Complete Propulsion System')

Motor1 = Component(name='Propulsor 1')
Motor1.add_subcomponent(Component(name='Pylon 1', geometry=pylon7))
Motor1.add_subcomponent(Component(name='Nacelle 1',geometry=nacelle7))
Motor1.add_subcomponent(Component(name='Spinner 1',geometry=spinner))
Motor1.add_subcomponent(Component(name='Prop 1',geometry=prop))
Motor1.add_subcomponent(Component(name='Motor 1',geometry=motor))
Motor1.add_subcomponent(Component(name='Motor Interface 1',geometry=motor_interface))

Motor2 = Component(name='Propulsor 2')
Motor2.add_subcomponent(Component(name='Pylon 2', geometry=pylon8))
Motor2.add_subcomponent(Component(name='Nacelle 2',geometry=nacelle8))
Motor2.add_subcomponent(Component(name='Spinner 2',geometry=spinner))
Motor2.add_subcomponent(Component(name='Prop 2',geometry=prop))
Motor2.add_subcomponent(Component(name='Motor 2',geometry=motor))
Motor2.add_subcomponent(Component(name='Motor Interface 2',geometry=motor_interface))

Motor3 = Component(name='Propulsor 3')
Motor3.add_subcomponent(Component(name='Pylon 3', geometry=pylon9))
Motor3.add_subcomponent(Component(name='Nacelle 3',geometry=nacelle9))
Motor3.add_subcomponent(Component(name='Spinner 3',geometry=spinner))
Motor3.add_subcomponent(Component(name='Prop 3',geometry=prop))
Motor3.add_subcomponent(Component(name='Motor 3',geometry=motor))
Motor3.add_subcomponent(Component(name='Motor Interface 3',geometry=motor_interface))

Motor4 = Component(name='Propulsor 4')
Motor4.add_subcomponent(Component(name='Pylon 4', geometry=pylon10))
Motor4.add_subcomponent(Component(name='Nacelle 4',geometry=nacelle10))
Motor4.add_subcomponent(Component(name='Spinner 4',geometry=spinner))
Motor4.add_subcomponent(Component(name='Prop 4',geometry=prop))
Motor4.add_subcomponent(Component(name='Motor 4',geometry=motor))
Motor4.add_subcomponent(Component(name='Motor Interface 4',geometry=motor_interface))

Motor5 = Component(name='Propulsor 5')
Motor5.add_subcomponent(Component(name='Pylon 5', geometry=pylon11))
Motor5.add_subcomponent(Component(name='Nacelle 5',geometry=nacelle11))
Motor5.add_subcomponent(Component(name='Spinner 5',geometry=spinner))
Motor5.add_subcomponent(Component(name='Prop 5',geometry=prop))
Motor5.add_subcomponent(Component(name='Motor 5',geometry=motor))
Motor5.add_subcomponent(Component(name='Motor Interface 5',geometry=motor_interface))

Motor6 = Component(name='Propulsor 6')
Motor6.add_subcomponent(Component(name='Pylon 6', geometry=pylon12))
Motor6.add_subcomponent(Component(name='Nacelle 6',geometry=nacelle12))
Motor6.add_subcomponent(Component(name='Spinner 6',geometry=spinner))
Motor6.add_subcomponent(Component(name='Prop 6',geometry=prop))
Motor6.add_subcomponent(Component(name='Motor 6',geometry=motor))
Motor6.add_subcomponent(Component(name='Motor Interface 6',geometry=motor_interface))

Motor7 = Component(name='Propulsor 7')
Motor7.add_subcomponent(Component(name='Pylon 7', geometry=pylon7))
Motor7.add_subcomponent(Component(name='Nacelle 7',geometry=nacelle7))
Motor7.add_subcomponent(Component(name='Spinner 7',geometry=spinner))
Motor7.add_subcomponent(Component(name='Prop 7',geometry=prop))
Motor7.add_subcomponent(Component(name='Motor 7',geometry=motor))
Motor7.add_subcomponent(Component(name='Motor Interface 7',geometry=motor_interface))

Motor8 = Component(name='Propulsor 8')
Motor8.add_subcomponent(Component(name='Pylon 8', geometry=pylon8))
Motor8.add_subcomponent(Component(name='Nacelle 8',geometry=nacelle8))
Motor8.add_subcomponent(Component(name='Spinner 8',geometry=spinner))
Motor8.add_subcomponent(Component(name='Prop 8',geometry=prop))
Motor8.add_subcomponent(Component(name='Motor 8',geometry=motor))
Motor8.add_subcomponent(Component(name='Motor Interface 8',geometry=motor_interface))

Motor9 = Component(name='Propulsor 9')
Motor9.add_subcomponent(Component(name='Pylon 9', geometry=pylon9))
Motor9.add_subcomponent(Component(name='Nacelle 9',geometry=nacelle9))
Motor9.add_subcomponent(Component(name='Spinner 9',geometry=spinner))
Motor9.add_subcomponent(Component(name='Prop 9',geometry=prop))
Motor9.add_subcomponent(Component(name='Motor 9',geometry=motor))
Motor9.add_subcomponent(Component(name='Motor Interface 9',geometry=motor_interface))

Motor10 = Component(name='Propulsor 10')
Motor10.add_subcomponent(Component(name='Pylon 10', geometry=pylon10))
Motor10.add_subcomponent(Component(name='Nacelle 10',geometry=nacelle10))
Motor10.add_subcomponent(Component(name='Spinner 10',geometry=spinner))
Motor10.add_subcomponent(Component(name='Prop 10',geometry=prop))
Motor10.add_subcomponent(Component(name='Motor 10',geometry=motor))
Motor10.add_subcomponent(Component(name='Motor Interface 10',geometry=motor_interface))

Motor11 = Component(name='Propulsor 11')
Motor11.add_subcomponent(Component(name='Pylon 11', geometry=pylon11))
Motor11.add_subcomponent(Component(name='Nacelle 11',geometry=nacelle11))
Motor11.add_subcomponent(Component(name='Spinner 11',geometry=spinner))
Motor11.add_subcomponent(Component(name='Prop 11',geometry=prop))
Motor11.add_subcomponent(Component(name='Motor 11',geometry=motor))
Motor11.add_subcomponent(Component(name='Motor Interface 11',geometry=motor_interface))

Motor12 = Component(name='Propulsor 12')
Motor12.add_subcomponent(Component(name='Pylon 12', geometry=pylon12))
Motor12.add_subcomponent(Component(name='Nacelle 12',geometry=nacelle12))
Motor12.add_subcomponent(Component(name='Spinner 12',geometry=spinner))
Motor12.add_subcomponent(Component(name='Prop 12',geometry=prop))
Motor12.add_subcomponent(Component(name='Motor 12',geometry=motor))
Motor12.add_subcomponent(Component(name='Motor Interface 12',geometry=motor_interface))

Total_Prop_Sys.add_subcomponent(Motor1)
Total_Prop_Sys.add_subcomponent(Motor2)
Total_Prop_Sys.add_subcomponent(Motor3)
Total_Prop_Sys.add_subcomponent(Motor4)
Total_Prop_Sys.add_subcomponent(Motor5)
Total_Prop_Sys.add_subcomponent(Motor6)
Total_Prop_Sys.add_subcomponent(Motor7)
Total_Prop_Sys.add_subcomponent(Motor8)
Total_Prop_Sys.add_subcomponent(Motor9)
Total_Prop_Sys.add_subcomponent(Motor10)
Total_Prop_Sys.add_subcomponent(Motor11)
Total_Prop_Sys.add_subcomponent(Motor12)



cruise_spinner =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner'], name='cruise_spinner')
# cruise_spinner.plot()
cruise_motor =  geometry.declare_component(function_search_names=['CruiseNacelle-Motor'], name='cruise_motor')
# cruise_motor.plot()
cruise_nacelle =  geometry.declare_component(function_search_names=['CruiseNacelle-Tail'], name='cruise_nacelle')
# cruise_nacelle.plot()
cruise_prop = geometry.declare_component(function_search_names=['Cruise-Prop'], name='cruise_prop')
# cruise_prop.plot()

CruiseMotor1 = Component(name='Cruise Propulsor 1')
CruiseMotor1.add_subcomponent(Component(name='Cruise Nacelle 1',geometry=cruise_nacelle))
CruiseMotor1.add_subcomponent(Component(name='Cruise Spinner 1',geometry=cruise_spinner))
CruiseMotor1.add_subcomponent(Component(name='Cruise Prop 1',geometry=cruise_prop))
CruiseMotor1.add_subcomponent(Component(name='Cruise Motor 1',geometry=cruise_motor))

CruiseMotor2 = Component(name='Cruise Propulsor 2')
CruiseMotor2.add_subcomponent(Component(name='Cruise Nacelle 2',geometry=cruise_nacelle))
CruiseMotor2.add_subcomponent(Component(name='Cruise Spinner 2',geometry=cruise_spinner))
CruiseMotor2.add_subcomponent(Component(name='Cruise Prop 2',geometry=cruise_prop))
CruiseMotor2.add_subcomponent(Component(name='Cruise Motor 2',geometry=cruise_motor))

Total_Prop_Sys.add_subcomponent(CruiseMotor1)
Total_Prop_Sys.add_subcomponent(CruiseMotor2)
# Total_Prop_Sys.visualize_component_hierarchy(show=True)

Complete_Aircraft.add_subcomponent(Total_Prop_Sys)
# Complete_Aircraft.visualize_component_hierarchy(show=True)







# Wing Region Info
wing_le_left = geometry.evaluate(wing.project(np.array([-12.356, -16, -5.5])*ft2m, plot=False))
wing_le_right = geometry.evaluate(wing.project(np.array([-12.356, 16, -5.5])*ft2m, plot=False))
wing_le_center_parametric = wing.project(np.array([-12.356, 0., -5.5])*ft2m, plot=False)
wing_le_center = geometry.evaluate(wing_le_center_parametric)
wing_te_left = geometry.evaluate(wing.project(np.array([-14.25, -16, -5.5])*ft2m, plot=False))
wing_te_right = geometry.evaluate(wing.project(np.array([-14.25, 16, -5.5])*ft2m, plot=False))
wing_te_center = geometry.evaluate(wing.project(np.array([-14.25, 0, -5.5])*ft2m, plot=False))
wing_qc = geometry.evaluate(wing.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False))

wingspan = csdl.norm(
    wing_le_left - wing_le_right
)
print("Wingspan: ", wingspan.value)

# HT Region Info
ht_le_left = geometry.evaluate(h_tail.project(np.array([-26.5, -5.25, -5.5])*ft2m, plot=False))
ht_le_center_parametric = h_tail.project(np.array([-27, 0., -5.5])*ft2m, plot=False)
ht_le_center = geometry.evaluate(ht_le_center_parametric)
ht_le_right = geometry.evaluate(h_tail.project(np.array([-26.5, 5.25, -5.5])*ft2m, plot=False))
ht_te_left = geometry.evaluate(h_tail.project(np.array([-30, -5.25, -5.5])*ft2m, plot=False))
ht_te_center = geometry.evaluate(h_tail.project(np.array([-30, 0., -5.5])*ft2m, plot=False))
ht_te_right = geometry.evaluate(h_tail.project(np.array([-30, 5.25, -5.5])*ft2m, plot=False))
ht_qc = geometry.evaluate(h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))

HTspan = csdl.norm(
    ht_le_left - ht_le_right
)
print("HT span: ", HTspan.value)

# VT Region Info
vt_le_base = geometry.evaluate(vertTail.project(np.array([-23, 0, -5.5])*ft2m, plot=False))
vt_le_mid_parametric = vertTail.project(np.array([-26, 0., -8])*ft2m, plot=False)
vt_le_mid = geometry.evaluate(vt_le_mid_parametric)
vt_le_tip = geometry.evaluate(vertTail.project(np.array([-28.7, 0, -11])*ft2m, plot=False))
vt_te_base = geometry.evaluate(vertTail.project(np.array([-27.75, 0, -5.5])*ft2m, plot=False))
vt_te_mid= geometry.evaluate(vertTail.project(np.array([-28.7, 0, -8])*ft2m, plot=False))
vt_te_tip = geometry.evaluate(vertTail.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False))
vt_qc = geometry.evaluate(vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m, plot=False))

rudder_le_mid_parametric = rudder.project(np.array([-28.7, 0., -8.])*ft2m, plot=False)
rudder_le_mid = geometry.evaluate(rudder_le_mid_parametric)

VTspan = csdl.norm(
    vt_le_base - vt_le_tip
)
print("VT span: ", VTspan.value)

# Fuselage Region Info
fuselage_wing_qc = geometry.evaluate(fuselage.project(np.array([-12.356+(0.25*(-14.25+12.356))*ft2m, 0., -5.5]), plot=False))
fuselage_wing_te_center = geometry.evaluate(fuselage.project(np.array([-14.25, 0., -5.5])*ft2m, plot=False))
fuselage_tail_qc = geometry.evaluate(fuselage.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))
fuselage_tail_te_center = geometry.evaluate(fuselage.project(np.array([-30, 0., -5.5])*ft2m, plot=False))

# Propeller Region Info
left_outermost_disk_pt =  np.array([-12.5, -13, -7.355])*ft2m
right_outermost_disk_pt = np.array([-12.5, 13, -7.355])*ft2m
left_outer_disk_pt =  np.array([-12.35, -10, -7.355])*ft2m
right_outer_disk_pt = np.array([-12.35, 10, -7.355])*ft2m
left_inner_disk_pt = np.array([-12.2, -7, -7.659])*ft2m
right_inner_disk_pt = np.array([-12.2, 7, -7.659])*ft2m
left_innermost_disk_pt = np.array([-12, -4, -7.659])*ft2m
right_innermost_disk_pt = np.array([-12, 4, -7.659])*ft2m


left_outermost_disk_on_wing = wing.project(left_outermost_disk_pt, plot=False)
l_om_disk = geometry.evaluate(left_outermost_disk_on_wing)
# print('From aircraft, Left Outermost Disk (ft): ', l_om_disk.value)

right_outermost_disk_on_wing = wing.project(right_outermost_disk_pt, plot=False)
r_om_disk = geometry.evaluate(right_outermost_disk_on_wing)
# print('From aircraft, Right Outermost Disk (ft): ', r_om_disk.value)

left_outer_disk_on_wing = wing.project(left_outer_disk_pt, plot=False)
l_o_disk = geometry.evaluate(left_outer_disk_on_wing)
# print('From aircraft, Left Outer Disk (ft): ', l_o_disk.value)

right_outer_disk_on_wing = wing.project(right_outer_disk_pt, plot=False)
r_o_disk = geometry.evaluate(right_outer_disk_on_wing)
# print('From aircraft, Right Outer Disk (ft): ', r_o_disk.value)

left_inner_disk_on_wing = wing.project(left_inner_disk_pt, plot=False)
l_i_disk = geometry.evaluate(left_inner_disk_on_wing)
# print('From aircraft, Left Inner Disk (ft): ', l_i_disk.value)

right_inner_disk_on_wing = wing.project(right_inner_disk_pt, plot=False)
r_i_disk = geometry.evaluate(right_inner_disk_on_wing)
# print('From aircraft, Right Inner Disk (ft): ', r_i_disk.value)

left_innermost_disk_on_wing = wing.project(left_innermost_disk_pt, plot=False)
l_im_disk = geometry.evaluate(left_innermost_disk_on_wing)
# print('From aircraft, Left Innermost Disk (ft): ', l_im_disk.value)

right_innermost_disk_on_wing = wing.project(right_innermost_disk_pt, plot=False)
r_im_disk = geometry.evaluate(right_innermost_disk_on_wing)
# print('From aircraft, Right Innermost Disk (ft): ', r_im_disk.value)

fuselage_nose_guess = np.array([-1.75, 0, -4])*ft2m
fuselage_rear_guess = np.array([-29.5, 0, -5.5])*ft2m
fuselage_nose_pts_parametric = fuselage.project(fuselage_nose_guess, grid_search_density_parameter=20, plot=False)
fuselage_nose = geometry.evaluate(fuselage_nose_pts_parametric)
fuselage_rear_pts_parametric = fuselage.project(fuselage_rear_guess, plot=False)
fuselage_rear = geometry.evaluate(fuselage_rear_pts_parametric)

# For Cruise Motor Hub Region
cruise_motor_tip_guess = np.array([-13, -15.83, -5.5])*ft2m
cruise_motor_tip_parametric = cruise_spinner.project(cruise_motor_tip_guess, plot=False)
cruise_motor_tip = geometry.evaluate(cruise_motor_tip_parametric)
print('From aircraft, cruise motor hub tip (ft): ', cruise_motor_tip.value)

cruise_motor_base_guess = cruise_motor_tip + np.array([-1.67, 0, 0])*ft2m
cruise_motor_base_parametric = cruise_spinner.project(cruise_motor_base_guess, plot=False)
cruise_motor_base= geometry.evaluate(cruise_motor_base_parametric)
print('From aircraft, cruise motor hub base (ft): ', cruise_motor_base.value)


## AXIS/AXISLSDOGEO CREATION

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from typing import Union
from dataclasses import dataclass
from flight_simulator import ureg


# OpenVSP Model Axis
openvsp_axis = Axis(
    name='OpenVSP Axis',
    x = np.array([0, ]) * ureg.foot,
    y = np.array([0, ])* ureg.foot,
    z = np.array([0, ])* ureg.foot,
    origin=ValidOrigins.OpenVSP.value
)



# Cruise Motor Region

@dataclass
class CruiseMotorRotation(csdl.VariableGroup):
    cant : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree
    pitch : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(15), name='CruiseMotorPitchAngle')
    yaw : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree
cruise_spinner_rotation = CruiseMotorRotation()
# cruise_spinner.rotate(cruise_motor_base, np.array([0., 1., 0.]), angles=cruise_spinner_rotation.pitch)

cruise_motor_axis = AxisLsdoGeo(
    name= 'Cruise Motor Axis',
    geometry=cruise_spinner,
    parametric_coords=cruise_motor_tip_parametric,
    sequence=np.array([3,2,1]),
    phi=cruise_spinner_rotation.cant,
    theta=cruise_spinner_rotation.pitch,
    psi=cruise_spinner_rotation.yaw,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Cruise motor axis translation (ft): ', cruise_motor_axis.translation.value)
print('Cruise motor axis rotation (deg): ', np.rad2deg(cruise_motor_axis.euler_angles_vector.value))



# Wing Axis Region

wing_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(45), name='Wing incidence')
wing.rotate(wing_le_center, np.array([0., 1., 0.]), angles=wing_incidence)

wing_axis = AxisLsdoGeo(
    name='Wing Axis',
    geometry=wing,
    parametric_coords=wing_le_center_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=wing_incidence,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Wing axis translation (ft): ', wing_axis.translation.value)
print('Wing axis rotation (deg): ', np.rad2deg(wing_axis.euler_angles_vector.value))
# geometry.plot()


## Distributed Propulsion Motors Axes

l_om_axis = AxisLsdoGeo(
    name= 'Left Outermost Motor Axis',
    geometry=wing,
    parametric_coords=left_outermost_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Left Outermost motor axis translation (ft): ', l_om_axis.translation.value)
print('Left Outermost motor axis rotation (deg): ', np.rad2deg(l_om_axis.euler_angles_vector.value))

r_om_axis = AxisLsdoGeo(
    name= 'Right Outermost Motor Axis',
    geometry=wing,
    parametric_coords=right_outermost_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Right Outermost motor axis translation (ft): ', r_om_axis.translation.value)
print('Right Outermost motor axis rotation (deg): ', np.rad2deg(r_om_axis.euler_angles_vector.value))

l_o_axis = AxisLsdoGeo(
    name= 'Left Outer Motor Axis',
    geometry=wing,
    parametric_coords=left_outer_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Left Outer motor axis translation (ft): ', l_o_axis.translation.value)
print('Left Outer motor axis rotation (deg): ', np.rad2deg(l_o_axis.euler_angles_vector.value))

r_o_axis = AxisLsdoGeo(
    name= 'Right Outer Motor Axis',
    geometry=wing,
    parametric_coords=right_outer_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Right Outer motor axis translation (ft): ', r_o_axis.translation.value)
print('Right Outer motor axis rotation (deg): ', np.rad2deg(r_o_axis.euler_angles_vector.value))

l_i_axis = AxisLsdoGeo(
    name= 'Left Inner Motor Axis',
    geometry=wing,
    parametric_coords=left_inner_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Left Inner motor axis translation (ft): ', l_i_axis.translation.value)
print('Left Inner motor axis rotation (deg): ', np.rad2deg(l_i_axis.euler_angles_vector.value))

r_i_axis = AxisLsdoGeo(
    name= 'Right Inner Motor Axis',
    geometry=wing,
    parametric_coords=right_inner_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Right Inner motor axis translation (ft): ', r_i_axis.translation.value)
print('Right Inner motor axis rotation (deg): ', np.rad2deg(r_i_axis.euler_angles_vector.value))

l_im_axis = AxisLsdoGeo(
    name= 'Left Innermost Motor Axis',
    geometry=wing,
    parametric_coords=left_innermost_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Left Innermost motor axis translation (ft): ', l_im_axis.translation.value)
print('Left Innermost motor axis rotation (deg): ', np.rad2deg(l_im_axis.euler_angles_vector.value))

r_im_axis = AxisLsdoGeo(
    name= 'Right Innermost Motor Axis',
    geometry=wing,
    parametric_coords=right_innermost_disk_on_wing,
    sequence=np.array([3,2,1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Right Innermost motor axis translation (ft): ', r_im_axis.translation.value)
print('Right Innermost motor axis rotation (deg): ', np.rad2deg(r_im_axis.euler_angles_vector.value))



## Tail Region Axis


ht_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(45), name='HT incidence')
h_tail.rotate(ht_le_center, np.array([0., 1., 0.]), angles=ht_incidence)


ht_tail_axis = AxisLsdoGeo(
    name='Horizontal Tail Axis',
    geometry=h_tail,
    parametric_coords=ht_le_center_parametric,
    sequence=np.array([3, 2, 1]),
    phi = ht_incidence,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('HT axis translation (ft): ', ht_tail_axis.translation.value)
print('HT axis rotation (deg): ', np.rad2deg(ht_tail_axis.euler_angles_vector.value))
# geometry.plot()






rudder_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(45), name='VT incidence')
rudder.rotate(rudder_le_mid, np.array([0., 0., 1.]), angles=rudder_incidence)


vt_tail_axis = AxisLsdoGeo(
    name='Vertical Tail Axis',
    geometry=rudder,
    parametric_coords=rudder_le_mid_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=rudder_incidence,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('VT axis translation (ft): ', vt_tail_axis.translation.value)
print('VT axis rotation (deg): ', np.rad2deg(vt_tail_axis.euler_angles_vector.value))
geometry.plot()



inertial_axis = Axis(
    name='Inertial Axis',
    x=np.array([0, ]) * ureg.meter,
    y=np.array([0, ]) * ureg.meter,
    z=np.array([0, ]) * ureg.meter,
    origin=ValidOrigins.Inertial.value
)



fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    x = np.array([0, ]) * ureg.meter,
    y = np.array([0, ]) * ureg.meter,
    z = np.array([0, ]) * ureg.meter,
    phi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='phi'),
    theta=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(4.), ]), name='theta'),
    psi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='psi'),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
print('Body-fixed angles (deg)', np.rad2deg(fd_axis.euler_angles_vector.value))



# Aircraft Wind Axis
@dataclass
class WindAxisRotations(csdl.VariableGroup):
    mu : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree # bank
    gamma : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(2), name='Flight path angle')
    xi : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree  # Heading
wind_axis_rotations = WindAxisRotations()

wind_axis = Axis(
    name='Wind Axis',
    x = np.array([0, ]) * ureg.meter,
    y = np.array([0, ]) * ureg.meter,
    z = np.array([0, ]) * ureg.meter,
    phi=wind_axis_rotations.mu,
    theta=wind_axis_rotations.gamma,
    psi=wind_axis_rotations.xi,
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
print('Wind axis angles (deg)', np.rad2deg(wind_axis.euler_angles_vector.value))



## FORCES AND MOMENTS

from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.euler_rotations import build_rotation_matrix


velocity_vector_in_wind = Vector(vector=csdl.Variable(shape=(3,), value=np.array([-1, 0, 0]), name='wind_vector'), axis=wind_axis)
print('Unit wind vector in wind axis: ', velocity_vector_in_wind.vector.value)

R_wind_to_inertial = build_rotation_matrix(wind_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_inertial =  Vector(csdl.matvec(R_wind_to_inertial, velocity_vector_in_wind.vector), axis=inertial_axis)
print('Unit wind vector in inertial axis: ', wind_vector_in_inertial.vector.value)

R_body_to_inertial = build_rotation_matrix(fd_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_body =  Vector(csdl.matvec(csdl.transpose(R_body_to_inertial), wind_vector_in_inertial.vector), axis=fd_axis)
print('Unit wind vector in body axis: ', wind_vector_in_body.vector.value)

R_wing_to_openvsp = build_rotation_matrix(wing_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_wing =  Vector(csdl.matvec(csdl.transpose(R_wing_to_openvsp), wind_vector_in_body.vector), axis=wing_axis)
print('Unit wind vector in wing axis: ', wind_vector_in_wing.vector.value)
alpha = csdl.arctan(wind_vector_in_wing.vector[2]/wind_vector_in_wing.vector.value[0])
print('Effective angle of attack (deg): ', np.rad2deg(alpha.value))


## Aerodynamic Forces


CL = 2*np.pi*alpha
e = 0.87
AR = 12
CD = 0.001 + 1/(np.pi*e*AR) * CL**2
rho = 1.225
S = 50*ft2m
V = 10
L = 0.5*rho*V**2*CL*S
D = 0.5*rho*V**2*CD*S

aero_force = csdl.Variable(shape=(3, ), value=0.)
aero_force = aero_force.set(csdl.slice[0], -D)
aero_force = aero_force.set(csdl.slice[2], -L)

aero_force_vector_in_wind = Vector(vector=aero_force, axis=wind_axis)
print('Aero force vector in wind-axis: ', aero_force_vector_in_wind.vector.value)
aero_force_vector_in_inertial =  Vector(csdl.matvec(R_wind_to_inertial, aero_force_vector_in_wind.vector), axis=inertial_axis)
print('Aero force vector in inertial-axis: ', aero_force_vector_in_inertial.vector.value)
aero_force_vector_in_body =  Vector(csdl.matvec(csdl.transpose(R_body_to_inertial), aero_force_vector_in_inertial.vector), axis=fd_axis)
print('Aero force vector in body-axis: ', aero_force_vector_in_body.vector.value)
aero_force_vector_in_wing =  Vector(csdl.matvec(csdl.transpose(R_wing_to_openvsp), aero_force_vector_in_body.vector), axis=fd_axis)
print('Aero force vector in wing-axis: ', aero_force_vector_in_wing.vector.value)


# Rotor Forces
cruise_motor_force = Vector(vector=np.array([0, 0, 0])*ureg.lbf, axis=cruise_motor_axis)
cruise_motor_moment = Vector(vector=np.array([0, 0, 0])*ureg.lbf*ureg.inch, axis=cruise_motor_axis)
cruise_motor_loads = ForcesMoments(force=cruise_motor_force, moment=cruise_motor_moment)




# cruise_motor_hub_rotation = csdl.Variable(value=np.deg2rad(15))
# cruise_spinner.rotate(cruise_motor_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation)
# print("Cruise Motor Translation: ", cruise_motor_axis.translation.value)


thrust_axis = cruise_motor_tip - cruise_motor_base
print('Thrust Axis: ', thrust_axis.value)


##  FFD Stuff

# Region Parameterization
constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# Region Parameterization Setup
parameterization_solver = lg.ParameterizationSolver()
parameterization_design_parameters = lg.GeometricVariables()

## Wing Region FFD Setup

wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing,num_coefficients=(2,11,2),degree=(1,3,1))
wing_ffd_block_sect_param = lg.VolumeSectionalParameterization(name='wing_sect_param',parameterized_points=wing_ffd_block.coefficients,principal_parametric_dimension=1)


## HT FFD Setup
h_tail_ffd_block = lg.construct_ffd_block_around_entities(name='h_tail_ffd_block', entities=h_tail, num_coefficients=(2,11,2), degree=(1,3,1))
h_tail_ffd_block_sect_param = lg.VolumeSectionalParameterization(name='h_tail_sectional_param',
                                                                            parameterized_points=h_tail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)


## Fuselage FFD Setup

fuselage_ffd_block = lg.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=fuselage, num_coefficients=(2,2,2), degree=(1,1,1))
fuselage_ffd_block_sectional_param = lg.VolumeSectionalParameterization(name='fuselage_sectional_param',
                                                                            parameterized_points=fuselage_ffd_block.coefficients,
                                                                            principal_parametric_dimension=0)



recorder.stop()