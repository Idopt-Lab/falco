import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
import lsdo_geo as lg
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER
from flight_simulator.core.component import Component, Configuration
from flight_simulator.core.condition import Condition
from flight_simulator.core.loads.mass_properties import MassProperties
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from typing import Union
from dataclasses import dataclass
from flight_simulator import ureg
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.euler_rotations import build_rotation_matrix


debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=debug)
recorder.start()
run_ffd = True
# run_optimization = True


in2m=0.0254
ft2m = 0.3048


geometry = import_geometry(
    file_name="x57.stp",
    file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57',
    refit=False,
    scale=in2m,
    rotate_to_body_fixed_frame=True
)

def define_base_geometry():
    wing = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4'], name='wing')
    aileronR = geometry.declare_component(function_search_names=['Rt_Aileron'], name='aileronR')
    aileronL = geometry.declare_component(function_search_names=['Lt_Aileron'], name='aileronL')
    flap = geometry.declare_component(function_search_names=['Flap'], name='flap')
    h_tail = geometry.declare_component(function_search_names=['HorzStab'], name='h_tail')
    trimTab = geometry.declare_component(function_search_names=['TrimTab'], name='trimTab')
    vertTail = geometry.declare_component(function_search_names=['VertTail'], name='vertTail')
    rudder = geometry.declare_component(function_search_names=['Rudder'], name='rudder')
    fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
    gear_pod = geometry.declare_component(function_search_names=['GearPod'], name='gear_pod')

    pylon1 = geometry.declare_component(function_search_names=['Pylon_07'], name='pylon1')
    pylon2 = geometry.declare_component(function_search_names=['Pylon_08'], name='pylon2')
    pylon3 = geometry.declare_component(function_search_names=['Pylon_09'], name='pylon3')
    pylon4 = geometry.declare_component(function_search_names=['Pylon_10'], name='pylon4')
    pylon5 = geometry.declare_component(function_search_names=['Pylon_11'], name='pylon5')
    pylon6 = geometry.declare_component(function_search_names=['Pylon_12'], name='pylon6')
    pylon7 = geometry.declare_component(function_search_names=['Pylon_07'], name='pylon7')
    pylon8 = geometry.declare_component(function_search_names=['Pylon_08'], name='pylon8')
    pylon9 = geometry.declare_component(function_search_names=['Pylon_09'], name='pylon9')
    pylon10 = geometry.declare_component(function_search_names=['Pylon_10'], name='pylon10')
    pylon11 = geometry.declare_component(function_search_names=['Pylon_11'], name='pylon11')
    pylon12 = geometry.declare_component(function_search_names=['Pylon_12'], name='pylon12')

    nacelle7 = geometry.declare_component(function_search_names=['HLNacelle_7_Tail'], name='nacelle7')
    nacelle8 = geometry.declare_component(function_search_names=['HLNacelle_8_Tail'], name='nacelle8')
    nacelle9 = geometry.declare_component(function_search_names=['HLNacelle_9_Tail'], name='nacelle9')
    nacelle10 = geometry.declare_component(function_search_names=['HLNacelle_10_Tail'], name='nacelle10')
    nacelle11 = geometry.declare_component(function_search_names=['HLNacelle_11_Tail'], name='nacelle11')
    nacelle12 = geometry.declare_component(function_search_names=['HLNacelle_12_Tail'], name='nacelle12')

    spinner1 = geometry.declare_component(function_search_names=['HL_Spinner12, 0'], name='spinner1')
    spinner2 = geometry.declare_component(function_search_names=['HL_Spinner11, 0'], name='spinner2')
    spinner3 = geometry.declare_component(function_search_names=['HL_Spinner10, 0'], name='spinner3')
    spinner4 = geometry.declare_component(function_search_names=['HL_Spinner9, 0'], name='spinner4')
    spinner5 = geometry.declare_component(function_search_names=['HL_Spinner8, 0'], name='spinner5')
    spinner6 = geometry.declare_component(function_search_names=['HL_Spinner7, 0'], name='spinner6')
    spinner7 = geometry.declare_component(function_search_names=['HL_Spinner7, 1'], name='spinner7')
    spinner8 = geometry.declare_component(function_search_names=['HL_Spinner8, 1'], name='spinner8')
    spinner9 = geometry.declare_component(function_search_names=['HL_Spinner9, 1'], name='spinner9')
    spinner10 = geometry.declare_component(function_search_names=['HL_Spinner10, 1'], name='spinner10')
    spinner11 = geometry.declare_component(function_search_names=['HL_Spinner11, 1'], name='spinner11')
    spinner12 = geometry.declare_component(function_search_names=['HL_Spinner12, 1'], name='spinner12')

    
    prop = geometry.declare_component(function_search_names=['HL-Prop'], name='prop')
    motor = geometry.declare_component(function_search_names=['HL_Motor'], name='motor')
    motor_interface = geometry.declare_component(function_search_names=['HL_Motor_Controller_Interface'], name='motor_interface')

    cruise_spinner =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner'], name='cruise_spinner')
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
    CM1_components = [cruise_nacelle, cruise_spinner, cruise_prop, cruise_motor]
    CM2_components = [cruise_nacelle, cruise_spinner, cruise_prop, cruise_motor]
    total_HL_motor_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components]
    total_prop_sys_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components, CM1_components, CM2_components]
    
    return wing, aileronR, aileronL, flap, h_tail, trimTab, vertTail, rudder, fuselage, gear_pod, pylon1, pylon2, pylon3, pylon4, pylon5, pylon6, pylon7, pylon8, pylon9, pylon10, pylon11, pylon12, nacelle7, nacelle8, nacelle9, nacelle10, nacelle11, nacelle12, spinner1, spinner2, spinner3, spinner4, spinner5, spinner6, spinner7, spinner8, spinner9, spinner10, spinner11, spinner12, prop, motor, motor_interface, cruise_spinner, cruise_motor, cruise_nacelle, cruise_prop, M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components, CM1_components, CM2_components, total_HL_motor_components, total_prop_sys_components
wing, aileronR, aileronL, flap, h_tail, trimTab, vertTail, rudder, fuselage, gear_pod, pylon1, pylon2, pylon3, pylon4, pylon5, pylon6, pylon7, pylon8, pylon9, pylon10, pylon11, pylon12, nacelle7, nacelle8, nacelle9, nacelle10, nacelle11, nacelle12, spinner1, spinner2, spinner3, spinner4, spinner5, spinner6, spinner7, spinner8, spinner9, spinner10, spinner11, spinner12, prop, motor, motor_interface, cruise_spinner, cruise_motor, cruise_nacelle, cruise_prop, M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components, CM1_components, CM2_components, total_HL_motor_components, total_prop_sys_components = define_base_geometry()


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

wing_qc = geometry.evaluate(wing.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False))

wing_te_center_flaps_guess = np.array([-14, 0., -7.3])*ft2m
wing_te_center_flaps = geometry.evaluate(wing.project(wing_te_center_flaps_guess, plot=False))

wing_te_center_ailerons_guess = np.array([-13.5, 0., -7.3])*ft2m
wing_te_center_ailerons = geometry.evaluate(wing.project(wing_te_center_ailerons_guess, plot=False))

WingRegionGeoGuess = [wing_le_left_guess,wing_le_right_guess,wing_le_center_guess,wing_te_left_guess,wing_te_right_guess,wing_te_center_guess,wing_te_center_flaps_guess,wing_te_center_ailerons_guess]

# HT Region Info
ht_le_left = geometry.evaluate(h_tail.project(np.array([-26.5, -5.25, -5.5])*ft2m, plot=False))
ht_le_center_parametric = h_tail.project(np.array([-27, 0., -5.5])*ft2m, plot=False)
ht_le_center = geometry.evaluate(ht_le_center_parametric)
ht_le_right = geometry.evaluate(h_tail.project(np.array([-26.5, 5.25, -5.5])*ft2m, plot=False))
ht_te_left = geometry.evaluate(h_tail.project(np.array([-30, -5.25, -5.5])*ft2m, plot=False))

ht_te_center_guess = np.array([-30, 0., -5.5])*ft2m
ht_te_center = geometry.evaluate(h_tail.project(ht_te_center_guess, plot=False))
ht_te_right = geometry.evaluate(h_tail.project(np.array([-30, 5.25, -5.5])*ft2m, plot=False))
ht_qc = geometry.evaluate(h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))

trimTab_le_center_parametric = trimTab.project(np.array([-29.4, 0, -5.5])*ft2m, plot=False)
trimTab_le_center = geometry.evaluate(trimTab_le_center_parametric)

# VT Region Info
vt_le_base = geometry.evaluate(vertTail.project(np.array([-23, 0, -5.5])*ft2m, plot=False))
vt_le_mid_parametric = vertTail.project(np.array([-26, 0., -8])*ft2m, plot=False)
vt_le_mid = geometry.evaluate(vt_le_mid_parametric)
vt_le_tip = geometry.evaluate(vertTail.project(np.array([-28.7, 0, -11])*ft2m, plot=False))
vt_te_base = geometry.evaluate(vertTail.project(np.array([-27.75, 0, -5.5])*ft2m, plot=False))

vt_te_mid_guess = np.array([-28.7, 0., -8])*ft2m
vt_te_mid= geometry.evaluate(vertTail.project(vt_te_mid_guess, plot=False))
vt_te_tip = geometry.evaluate(vertTail.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False))
vt_qc = geometry.evaluate(vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m, plot=False))

rudder_le_mid_parametric = rudder.project(np.array([-28.7, 0., -8.])*ft2m, plot=False)
rudder_le_mid = geometry.evaluate(rudder_le_mid_parametric)

# Fuselage Region Info
fuselage_wing_qc = geometry.evaluate(fuselage.project(np.array([-12.356+(0.25*(-14.25+12.356))*ft2m, 0., -5.5]), plot=False))
fuselage_wing_te_center = geometry.evaluate(fuselage.project(np.array([-14.25, 0., -5.5])*ft2m, plot=False))
fuselage_tail_qc = geometry.evaluate(fuselage.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))
fuselage_tail_te_center = geometry.evaluate(fuselage.project(np.array([-30, 0., -5.5])*ft2m, plot=False))


# Propeller Region Info
M1_disk_pt =  np.array([-12.5, 14, -7.355])*ft2m
M12_disk_pt = np.array([-12.5, -14, -7.355])*ft2m
M2_disk_pt =  np.array([-12.35, 12, -7.355])*ft2m
M11_disk_pt = np.array([-12.35, -12, -7.355])*ft2m
M3_disk_pt = np.array([-12.2, 10, -7.659])*ft2m
M10_disk_pt = np.array([-12.2, -10, -7.659])*ft2m
M4_disk_pt = np.array([-12, 8, -7.659])*ft2m
M9_disk_pt = np.array([-12, -8, -7.659])*ft2m
M5_disk_pt = np.array([-11.8, 6, -7.659])*ft2m
M8_disk_pt = np.array([-11.8, -6, -7.659])*ft2m
M6_disk_pt = np.array([-11.6, 4, -7.659])*ft2m
M7_disk_pt = np.array([-11.6, -4, -7.659])*ft2m

M1_disk_on_wing = spinner1.project(M1_disk_pt, plot=False)
M1_disk = geometry.evaluate(M1_disk_on_wing)
# print('From aircraft, Left Outermost Disk (ft): ', M1_disk.value)

M2_disk_on_wing = spinner2.project(M2_disk_pt, plot=False)
M2_disk = geometry.evaluate(M2_disk_on_wing)
# print('From aircraft, Left Outer Disk (ft): ', M2_disk.value)

M3_disk_on_wing = spinner3.project(M3_disk_pt, plot=False)
M3_disk = geometry.evaluate(M3_disk_on_wing)
# print('From aircraft, Left Inner Disk (ft): ', M3_disk.value)

M4_disk_on_wing = spinner4.project(M4_disk_pt, plot=False)
M4_disk = geometry.evaluate(M4_disk_on_wing)
# print('From aircraft, Left Innermost Disk (ft): ', M4_disk.value)

M5_disk_on_wing = spinner5.project(M5_disk_pt, plot=False)
M5_disk = geometry.evaluate(M5_disk_on_wing)
# print('From aircraft, Left Disk (ft): ', M5_disk.value)

M6_disk_on_wing = spinner6.project(M6_disk_pt, plot=False)
M6_disk = geometry.evaluate(M6_disk_on_wing)
# print('From aircraft, Right Disk (ft): ', M6_disk.value)

M7_disk_on_wing = spinner7.project(M7_disk_pt, plot=False)
M7_disk = geometry.evaluate(M7_disk_on_wing)
# print('From aircraft, Right Disk (ft): ', M7_disk.value)

M8_disk_on_wing = spinner8.project(M8_disk_pt, plot=False)
M8_disk = geometry.evaluate(M8_disk_on_wing)
# print('From aircraft, Right Disk (ft): ', M8_disk.value)

M9_disk_on_wing = spinner9.project(M9_disk_pt, plot=False)
M9_disk = geometry.evaluate(M9_disk_on_wing)
# print('From aircraft, Right Innermost Disk (ft): ', M9_disk.value)

M10_disk_on_wing = spinner10.project(M10_disk_pt, plot=False)
M10_disk = geometry.evaluate(M10_disk_on_wing)
# print('From aircraft, Right Inner Disk (ft): ', M10_disk.value)

M11_disk_on_wing = spinner11.project(M11_disk_pt, plot=False)
M11_disk = geometry.evaluate(M11_disk_on_wing)
# print('From aircraft, Right Outer Disk (ft): ', M11_disk.value)

M12_disk_on_wing = spinner12.project(M12_disk_pt, plot=False)
M12_disk = geometry.evaluate(M12_disk_on_wing)
# print('From aircraft, Right Outermost Disk (ft): ', M12_disk.value)


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
# print('From aircraft, cruise motor hub tip (ft): ', cruise_motor_tip.value)

cruise_motor_base_guess = cruise_motor_tip + np.array([-1.67, 0, 0])*ft2m
cruise_motor_base_parametric = cruise_spinner.project(cruise_motor_base_guess, plot=False)
cruise_motor_base= geometry.evaluate(cruise_motor_base_parametric)
# print('From aircraft, cruise motor hub base (ft): ', cruise_motor_base.value)



## AXIS/AXISLSDOGEO CREATION


def axes_create():

    # OpenVSP Model Axis
    openvsp_axis = Axis(
        name='OpenVSP Axis',
        x = np.array([0, ]) * ureg.foot,
        y = np.array([0, ])* ureg.foot,
        z = np.array([0, ])* ureg.foot,
        origin=ValidOrigins.OpenVSP.value
    )
    
    wing_axis = AxisLsdoGeo(
    name='Wing Axis',
    geometry=wing,
    parametric_coords=wing_le_center_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
    )
    # print('Wing axis translation (ft): ', wing_axis.translation.value)
    # print('Wing axis rotation (deg): ', np.rad2deg(wing_axis.euler_angles_vector.value))
    # geometry.plot()

    ## Tail Region Axis


    # ht_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='HT incidence')
    # trimTab_deflection = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='Trim Tab Deflection')

    # h_tail.rotate(ht_le_center, np.array([0., 1., 0.]), angles=ht_incidence)
    # trimTab.rotate(ht_le_center, np.array([0., 1., 0.]), angles=ht_incidence)

    ht_tail_axis = AxisLsdoGeo(
        name='Horizontal Tail Axis',
        geometry=h_tail,
        parametric_coords=ht_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi = np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('HT axis translation (ft): ', ht_tail_axis.translation.value)
    # print('HT axis rotation (deg): ', np.rad2deg(ht_tail_axis.euler_angles_vector.value))

    trimTab_axis = AxisLsdoGeo(
        name='Trim Tab Axis',
        geometry=trimTab,
        parametric_coords=trimTab_le_center_parametric,
        sequence=np.array([3, 2, 1]),
        phi = np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=ht_tail_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    # trimTab.rotate(trimTab_le_center, np.array([0.,1.,0.]),angles=trimTab_deflection)


    # print('Trim Tab axis translation (ft): ', trimTab_axis.translation.value)
    # print('Trim Tab axis rotation (deg): ', np.rad2deg(trimTab_axis.euler_angles_vector.value))
    # geometry.plot()



    # rudder_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='VT incidence')
    # rudder.rotate(rudder_le_mid, np.array([0., 0., 1.]), angles=rudder_incidence)


    vt_tail_axis = AxisLsdoGeo(
        name='Vertical Tail Axis',
        geometry=rudder,
        parametric_coords=rudder_le_mid_parametric,
        sequence=np.array([3, 2, 1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('VT axis translation (ft): ', vt_tail_axis.translation.value)
    # print('VT axis rotation (deg): ', np.rad2deg(vt_tail_axis.euler_angles_vector.value))
    # geometry.plot()

    ## Distributed Propulsion Motors Axes

    M1_axis = AxisLsdoGeo(
        name= 'Motor 1 Axis',
        geometry=spinner1,
        parametric_coords=M1_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Left Outermost motor axis translation (ft): ', l_om_axis.translation.value)
    # print('Left Outermost motor axis rotation (deg): ', np.rad2deg(l_om_axis.euler_angles_vector.value))

    M2_axis = AxisLsdoGeo(
        name= 'Motor 2 Axis',
        geometry=spinner2,
        parametric_coords=M2_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Left Outer motor axis translation (ft): ', l_o_axis.translation.value)
    # print('Left Outer motor axis rotation (deg): ', np.rad2deg(l_o_axis.euler_angles_vector.value))

    M3_axis = AxisLsdoGeo(
        name= 'Motor 3 Axis',
        geometry=spinner3,
        parametric_coords=M3_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M4_axis = AxisLsdoGeo(
        name= 'Motor 4 Axis',
        geometry=spinner4,
        parametric_coords=M4_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M5_axis = AxisLsdoGeo(
        name= 'Motor 5 Axis',
        geometry=spinner5,
        parametric_coords=M5_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M6_axis = AxisLsdoGeo(
        name= 'Motor 6 Axis',
        geometry=spinner6,
        parametric_coords=M6_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M7_axis = AxisLsdoGeo(
        name= 'Motor 7 Axis',
        geometry=spinner7,
        parametric_coords=M7_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M8_axis = AxisLsdoGeo(
        name= 'Motor 8 Axis',
        geometry=spinner8,
        parametric_coords=M8_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M9_axis = AxisLsdoGeo(
        name= 'Motor 9 Axis',
        geometry=spinner9,
        parametric_coords=M9_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M10_axis = AxisLsdoGeo(
        name= 'Motor 10 Axis',
        geometry=spinner10,
        parametric_coords=M10_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M11_axis = AxisLsdoGeo(
        name= 'Motor 11 Axis',
        geometry=spinner11,
        parametric_coords=M11_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M12_axis = AxisLsdoGeo(
        name= 'Motor 12 Axis',
        geometry=spinner12,
        parametric_coords=M12_disk_on_wing,
        sequence=np.array([3,2,1]),
        phi=np.array([0, ]) * ureg.degree,
        theta=np.array([0, ]) * ureg.degree,
        psi=np.array([0, ]) * ureg.degree,
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )
    # print('Right Outermost motor axis translation (ft): ', r_om_axis.translation.value)
    # print('Right Outermost motor axis rotation (deg): ', np.rad2deg(r_om_axis.euler_angles_vector.value))


    # Cruise Motor Region

    @dataclass
    class CruiseMotorRotation(csdl.VariableGroup):
        cant : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree
        pitch : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(0), name='CruiseMotorPitchAngle')
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
    # print('Cruise motor axis translation (ft): ', cruise_motor_axis.translation.value)
    # print('Cruise motor axis rotation (deg): ', np.rad2deg(cruise_motor_axis.euler_angles_vector.value))

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
        theta=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='theta'),
        psi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='psi'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )
    # print('Body-fixed angles (deg)', np.rad2deg(fd_axis.euler_angles_vector.value))



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
    # print('Wind axis angles (deg)', np.rad2deg(wind_axis.euler_angles_vector.value))

    return openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis, cruise_motor_axis, inertial_axis, fd_axis, wind_axis, geometry

openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis, cruise_motor_axis, inertial_axis, fd_axis, wind_axis,geometry = axes_create()




velocity_vector_in_wind = Vector(vector=csdl.Variable(shape=(3,), value=np.array([-1, 0, 0]), name='wind_vector'), axis=wind_axis)
# print('Unit wind vector in wind axis: ', velocity_vector_in_wind.vector.value)

R_wind_to_inertial = build_rotation_matrix(wind_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_inertial =  Vector(csdl.matvec(R_wind_to_inertial, velocity_vector_in_wind.vector), axis=inertial_axis)
# print('Unit wind vector in inertial axis: ', wind_vector_in_inertial.vector.value)

R_body_to_inertial = build_rotation_matrix(fd_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_body =  Vector(csdl.matvec(csdl.transpose(R_body_to_inertial), wind_vector_in_inertial.vector), axis=fd_axis)
# print('Unit wind vector in body axis: ', wind_vector_in_body.vector.value)

R_wing_to_openvsp = build_rotation_matrix(wing_axis.euler_angles_vector, np.array([3, 2, 1]))
wind_vector_in_wing =  Vector(csdl.matvec(csdl.transpose(R_wing_to_openvsp), wind_vector_in_body.vector), axis=wing_axis)
# print('Unit wind vector in wing axis: ', wind_vector_in_wing.vector.value)
alpha = csdl.arctan(wind_vector_in_wing.vector[2]/wind_vector_in_wing.vector.value[0])
print('Effective angle of attack (deg): ', np.rad2deg(alpha.value))


## Aerodynamic Forces


CL = 2*np.pi*alpha
e = 0.87
AR = 12
CD = 0.001 + 1/(np.pi*e*AR) * CL**2
rho = 1.225
S = 50
V = 35
L = 0.5*rho*V**2*CL*S
D = 0.5*rho*V**2*CD*S

aero_force = csdl.Variable(shape=(3, ), value=0.)
aero_moment = csdl.Variable(shape=(3, ), value=0.)
aero_force = aero_force.set(csdl.slice[0], -D)
aero_force = aero_force.set(csdl.slice[2], -L)


aero_force_vector_in_wing = Vector(vector=aero_force, axis=wing_axis)
aero_moment_vector_in_wing = Vector(vector=aero_moment, axis=wing_axis)
aero_force_moment_in_wing = ForcesMoments(force=aero_force_vector_in_wing, moment=aero_moment_vector_in_wing)

print('Aero Force in Wing Axis: ', aero_force_vector_in_wing.vector.value)
print('Aero Moment in Wing Axis: ', aero_moment_vector_in_wing.vector.value)


aero_force_moment_in_body = aero_force_moment_in_wing.rotate_to_axis(fd_axis)
aero_force_in_body = aero_force_moment_in_body.F
aero_moment_in_body = aero_force_moment_in_body.M
print('Aero Force in Body Axis: ', aero_force_in_body.vector.value)
print('Aero Moment in Body Axis: ', aero_moment_in_body.vector.value)


# Rotor Forces

cruise_motor_thrust=400
cruise_motor_prop_force = csdl.Variable(shape=(3, ), value=0.)
cruise_motor_prop_moment = csdl.Variable(shape=(3, ), value=0.)
cruise_motor_prop_force = cruise_motor_prop_force.set(csdl.slice[1], cruise_motor_thrust)


prop_force_vector_in_cruise_motor = Vector(vector=cruise_motor_prop_force, axis=cruise_motor_axis)
prop_moment_vector_in_cruise_motor = Vector(vector=cruise_motor_prop_moment, axis=cruise_motor_axis)
prop_force_moment_in_cruise_motor = ForcesMoments(force=prop_force_vector_in_cruise_motor, moment=prop_moment_vector_in_cruise_motor)
print('Prop Force in Cruise Motor Axis: ', prop_force_vector_in_cruise_motor.vector.value)
print('Prop Moment in Cruise Motor Axis: ', prop_moment_vector_in_cruise_motor.vector.value)


cruise_motor_prop_force_moment_in_body = prop_force_moment_in_cruise_motor.rotate_to_axis(fd_axis)
cruise_motor_prop_force_in_body = cruise_motor_prop_force_moment_in_body.F
cruise_motor_prop_moment_in_body = cruise_motor_prop_force_moment_in_body.M
print('Cruise Motor Prop Force in Body Axis: ', cruise_motor_prop_force_in_body.vector.value)
print('Cruise Motor Prop Moment in Body Axis: ', cruise_motor_prop_moment_in_body.vector.value)






thrust_axis = cruise_motor_tip - cruise_motor_base
# print('Thrust Axis: ', thrust_axis.value)



#FFD Stuff

# #Region Parameterization
# constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
# linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
# linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
# cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# # FFD Blocks
# wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2,11,2), degree=(1,3,1))
# aileronL_ffd_block = lg.construct_ffd_block_around_entities(name='left_aileron_ffd_block', entities=aileronL, num_coefficients=(2,11,2), degree=(1,3,1))
# aileronR_ffd_block = lg.construct_ffd_block_around_entities(name='right_aileron_ffd_block', entities=aileronR, num_coefficients=(2,11,2), degree=(1,3,1))
# flap_ffd_block = lg.construct_ffd_block_around_entities(name='flap_ffd_block', entities=flap, num_coefficients=(2,11,2), degree=(1,3,1))
# h_tail_ffd_block = lg.construct_ffd_block_around_entities(name='h_tail_ffd_block', entities=h_tail, num_coefficients=(2,11,2), degree=(1,3,1))
# trimTab_ffd_block = lg.construct_ffd_block_around_entities(name='trimTab_ffd_block', entities=trimTab, num_coefficients=(2,11,2), degree=(1,3,1))
# vertTail_ffd_block = lg.construct_ffd_block_around_entities(name='v_tail_ffd_block', entities=vertTail, num_coefficients=(2,11,2), degree=(1,3,1))
# rudder_ffd_block = lg.construct_ffd_block_around_entities(name='rudder_ffd_block', entities=rudder, num_coefficients=(2,11,2), degree=(1,3,1))
# fuselage_ffd_block = lg.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=fuselage, num_coefficients=(2,2,2), degree=(1,1,1))




# # Region Parameterization Setup
# parameterization_solver = lg.ParameterizationSolver()
# parameterization_design_parameters = lg.GeometricVariables()

# # Wing Region FFD Setup

# wing_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='wing_sect_param',parameterized_points=wing_ffd_block.coefficients,principal_parametric_dimension=1)

# wing_chord_stretch_coefficients = csdl.Variable(name='wing_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
# wing_chord_stretch_b_spline = lfs.Function(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
#                                         coefficients=wing_chord_stretch_coefficients)

# wing_wingspan_stretch_coefficients = csdl.Variable(name='wing_wingspan_stretch_coefficients', value=np.array([-0., 0.]))
# wing_wingspan_stretch_b_spline = lfs.Function(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                         coefficients=wing_wingspan_stretch_coefficients)

# wing_twist_coefficients = csdl.Variable(name='wing_twist_coefficients', value=np.array([-0, 0, -0, 0, -0])*np.pi/180)
# wing_twist_b_spline = lfs.Function(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#                                         coefficients=wing_twist_coefficients)

# wing_sweep_coefficients = csdl.Variable(name='wing_sweep_coefficients', value=np.array([0., 0.0, 0.]))
# wing_sweep_b_spline = lfs.Function(space=linear_b_spline_curve_3_dof_space,
#                                             coefficients=wing_sweep_coefficients, name='wing_sweep_b_spline')

# wing_translation_x_coefficients = csdl.Variable(name='wing_translation_x_coefficients', value=np.array([0.]))
# wing_translation_x_b_spline = lfs.Function(name='wing_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=wing_translation_x_coefficients)

# wing_translation_z_coefficients = csdl.Variable(name='wing_translation_z_coefficients', value=np.array([0.]))
# wing_translation_z_b_spline = lfs.Function(name='wing_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=wing_translation_z_coefficients)

# parameterization_solver.add_parameter(parameter=wing_chord_stretch_coefficients)
# parameterization_solver.add_parameter(parameter=wing_wingspan_stretch_coefficients, cost=1.e3)
# parameterization_solver.add_parameter(parameter=wing_twist_coefficients)
# parameterization_solver.add_parameter(parameter=wing_translation_x_coefficients)
# parameterization_solver.add_parameter(parameter=wing_translation_z_coefficients)

# section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
# sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = lg.VolumeSectionalParameterizationInputs(
#     stretches={0: sectional_wing_chord_stretch},
#     translations={1: sectional_wing_wingspan_stretch, 0: sectional_wing_translation_x, 2: sectional_wing_translation_z, 0: sectional_wing_sweep},
#     rotations={1: sectional_wing_twist}
# )


# wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
# wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
# wing.set_coefficients(wing_coefficients)

# # Wing Region Design Parameters

# wing_span_computed = csdl.norm(geometry.evaluate(wing_le_right_parametric) - geometry.evaluate(wing_le_left_parametric))
# wing_root_chord_computed = csdl.norm(geometry.evaluate(wing_te_center_parametric) - geometry.evaluate(wing_le_center_parametric))
# wing_tip_chord_left_computed = csdl.norm(geometry.evaluate(wing_te_left_parametric) - geometry.evaluate(wing_le_left_parametric))
# wing_tip_chord_right_computed = csdl.norm(geometry.evaluate(wing_te_right_parametric) - geometry.evaluate(wing_le_right_parametric))

# wing_span = csdl.Variable(name='wing_span', value=np.array([50.]))
# wing_root_chord = csdl.Variable(name='wing_root_chord', value=np.array([5.]))
# wing_tip_chord = csdl.Variable(name='wing_tip_chord_left', value=np.array([1.]))

# parameterization_design_parameters.add_variable(computed_value=wing_span_computed, desired_value=wing_span)
# parameterization_design_parameters.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
# parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_left_computed, desired_value=wing_tip_chord)
# parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_right_computed, desired_value=wing_tip_chord)

# # HT FFD Setup
# h_tail_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='h_tail_sectional_param',
#                                                                             parameterized_points=h_tail_ffd_block.coefficients,
#                                                                             principal_parametric_dimension=1)

# h_tail_chord_stretch_coefficients = csdl.Variable(name='h_tail_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
# h_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
#                                         coefficients=h_tail_chord_stretch_coefficients)

# h_tail_span_stretch_coefficients = csdl.Variable(name='h_tail_span_stretch_coefficients', value=np.array([-0., 0.]))
# h_tail_span_stretch_b_spline = lfs.Function(name='h_tail_span_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                         coefficients=h_tail_span_stretch_coefficients)

# h_tail_sweep_coefficients = csdl.Variable(name='h_tail_sweep_coefficients', value=np.array([0.0, 0.0, 0.0]))
# h_tail_sweep_b_spline = lfs.Function(space=linear_b_spline_curve_3_dof_space,
#                                             coefficients=h_tail_sweep_coefficients, name='h_tail_sweep_b_spline')

# h_tail_twist_coefficients = csdl.Variable(name='h_tail_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
# h_tail_twist_b_spline = lfs.Function(name='h_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#                                         coefficients=h_tail_twist_coefficients)

# h_tail_translation_x_coefficients = csdl.Variable(name='h_tail_translation_x_coefficients', value=np.array([0.]))
# h_tail_translation_x_b_spline = lfs.Function(name='h_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=h_tail_translation_x_coefficients)

# h_tail_translation_z_coefficients = csdl.Variable(name='h_tail_translation_z_coefficients', value=np.array([0.]))
# h_tail_translation_z_b_spline = lfs.Function(name='h_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=h_tail_translation_z_coefficients)

# parameterization_solver.add_parameter(parameter=h_tail_chord_stretch_coefficients)
# parameterization_solver.add_parameter(parameter=h_tail_span_stretch_coefficients)
# parameterization_solver.add_parameter(parameter=h_tail_twist_coefficients)
# parameterization_solver.add_parameter(parameter=h_tail_translation_x_coefficients)
# parameterization_solver.add_parameter(parameter=h_tail_translation_z_coefficients)

# ## Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver

# section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
# sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_sweep = h_tail_sweep_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = lg.VolumeSectionalParameterizationInputs(
#     stretches={0: sectional_h_tail_chord_stretch},
#     translations={1: sectional_h_tail_span_stretch, 0: sectional_h_tail_translation_x, 2: sectional_h_tail_translation_z, 0: sectional_h_tail_sweep},
#     rotations={1: sectional_h_tail_twist}
# )

# h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
# h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)
# h_tail.set_coefficients(coefficients=h_tail_coefficients)
# # # geometry.plot()


# # HT Region design parameterization inputs
# h_tail_span_computed = csdl.norm(ht_le_right- ht_le_right)
# h_tail_root_chord_computed = csdl.norm(ht_te_center - ht_le_center)
# h_tail_tip_chord_left_computed = csdl.norm(ht_te_left - ht_le_left)
# h_tail_tip_chord_right_computed = csdl.norm(ht_te_right - ht_le_right)

# h_tail_span = csdl.Variable(name='h_tail_span', value=np.array([12.]))
# h_tail_root_chord = csdl.Variable(name='h_tail_root_chord', value=np.array([3.]))
# h_tail_tip_chord = csdl.Variable(name='h_tail_tip_chord_left', value=np.array([2.]))

# parameterization_design_parameters.add_variable(computed_value=h_tail_span_computed, desired_value=h_tail_span)
# parameterization_design_parameters.add_variable(computed_value=h_tail_root_chord_computed, desired_value=h_tail_root_chord)
# parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_left_computed, desired_value=h_tail_tip_chord)
# parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_right_computed, desired_value=h_tail_tip_chord)

# geometry.plot()

# # High Lift Rotors setup
# lift_rotor_ffd_blocks = []
# lift_rotor_sectional_parameterizations = []
# lift_rotor_parameterization_b_splines = []
# for i, component_set in enumerate(total_HL_motor_components):
#     rotor_ffd_block = lg.construct_ffd_block_around_entities(name=f'{component_set[0].name[:3]}_rotor_ffd_block', entities=component_set, num_coefficients=(2,2,2), degree=(1,1,1))
#     rotor_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name=f'{component_set[0].name[:3]}_rotor_sectional_parameterization',
#                                                                                 parameterized_points=rotor_ffd_block.coefficients,
#                                                                                 principal_parametric_dimension=2)
    
#     rotor_stretch_coefficient = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_stretch_coefficient', value=wing_wingspan_stretch_coefficients.value)
#     lift_rotor_sectional_stretch_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_sectional_stretch_x_b_spline', space=linear_b_spline_curve_2_dof_space,
#                                                 coefficients=rotor_stretch_coefficient)
    
#     rotor_twist_coefficients = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_twist_coefficients', value=wing_twist_coefficients.value)
#     rotor_twist_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_twist_b_spline', space=cubic_b_spline_curve_5_dof_space, coefficients=rotor_twist_coefficients)

#     rotor_translation_x_coefficients = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_translation_x_coefficients', value=np.array([0.]))
#     rotor_translation_x_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=rotor_translation_x_coefficients)

#     rotor_translation_z_coefficients = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_translation_z_coefficients', value=np.array([0.]))
#     rotor_translation_z_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=rotor_translation_z_coefficients)
    
#     lift_rotor_ffd_blocks.append(rotor_ffd_block)
#     lift_rotor_sectional_parameterizations.append(rotor_ffd_block_sectional_parameterization)
#     lift_rotor_parameterization_b_splines.append(lift_rotor_sectional_stretch_b_spline)                 

#     parameterization_solver.add_parameter(parameter=rotor_stretch_coefficient)

# for i, component_set in enumerate(total_HL_motor_components):
#     rotor_ffd_block = lift_rotor_ffd_blocks[i]
#     rotor_ffd_block_sectional_parameterization = lift_rotor_sectional_parameterizations[i]
#     rotor_stretch_b_spline = lift_rotor_parameterization_b_splines[i]

#     section_parametric_coordinates = np.linspace(0., 1., rotor_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     sectional_stretch = rotor_stretch_b_spline.evaluate(section_parametric_coordinates)
#     sectional_twist = rotor_twist_b_spline.evaluate(section_parametric_coordinates)
#     sectional_translation_x = rotor_translation_x_b_spline.evaluate(section_parametric_coordinates)
#     sectional_translation_z = rotor_translation_z_b_spline.evaluate(section_parametric_coordinates)

#     sectional_parameters = lg.VolumeSectionalParameterizationInputs(
#         stretches={0: sectional_stretch, 1:sectional_stretch},
#         rotations={1: sectional_twist},
#     )

#     rotor_ffd_block_coefficients = rotor_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
#     rotor_coefficients = rotor_ffd_block.evaluate(rotor_ffd_block_coefficients, plot=False)
#     for i, component in enumerate(component_set):
#         component.set_coefficients(rotor_coefficients[i])

# geometry.plot()

# # Tail Moment Arm Region 
# tail_moment_arm_computed = csdl.norm(ht_qc - wing_qc)
# tail_moment_arm = csdl.Variable(name='tail_moment_arm', value=np.array([25.]))
# parameterization_design_parameters.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm)

# wing_fuselage_connection = wing_te_center - fuselage_wing_te_center
# h_tail_fuselage_connection = ht_te_center - fuselage_tail_te_center
# parameterization_design_parameters.add_variable(computed_value=wing_fuselage_connection, desired_value=wing_fuselage_connection.value)
# parameterization_design_parameters.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)


# # VT FFD Setup

# v_tail_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='v_tail_sectional_param',
#                                                                             parameterized_points=vertTail_ffd_block.coefficients,
#                                                                             principal_parametric_dimension=1)

# v_tail_chord_stretch_coefficients = csdl.Variable(name='v_tail_chord_stretch_coefficients', value=np.array([0., 0.]))
# v_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                         coefficients=v_tail_chord_stretch_coefficients)

# v_tail_span_stretch_coefficients = csdl.Variable(name='v_tail_span_stretch_coefficients', value=np.array([0.]))
# v_tail_span_stretch_b_spline = lfs.Function(name='v_tail_span_stretch_b_spline', space=constant_b_spline_curve_1_dof_space, 
#                                         coefficients=v_tail_span_stretch_coefficients)

# v_tail_twist_coefficients = csdl.Variable(name='v_tail_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
# v_tail_twist_b_spline = lfs.Function(name='v_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#                                         coefficients=v_tail_twist_coefficients)

# v_tail_translation_x_coefficients = csdl.Variable(name='v_tail_translation_x_coefficients', value=np.array([0]))
# v_tail_translation_x_b_spline = lfs.Function(name='v_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=v_tail_translation_x_coefficients)

# v_tail_translation_z_coefficients = csdl.Variable(name='v_tail_translation_z_coefficients', value=np.array([-0.5*v_tail_span_stretch_coefficients.value]))
# v_tail_translation_z_b_spline = lfs.Function(name='v_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=v_tail_translation_z_coefficients)

# parameterization_solver.add_parameter(parameter=v_tail_chord_stretch_coefficients)
# parameterization_solver.add_parameter(parameter=v_tail_span_stretch_coefficients)
# parameterization_solver.add_parameter(parameter=v_tail_twist_coefficients)
# parameterization_solver.add_parameter(parameter=v_tail_translation_x_coefficients)
# parameterization_solver.add_parameter(parameter=v_tail_translation_z_coefficients)

# section_parametric_coordinates = np.linspace(0., 1., v_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
# sectional_v_tail_chord_stretch = v_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_v_tail_span_stretch = v_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_v_tail_twist = v_tail_twist_b_spline.evaluate(section_parametric_coordinates)
# sectional_v_tail_translation_x = v_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
# sectional_v_tail_translation_z = v_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = lg.VolumeSectionalParameterizationInputs(
#     stretches={0: sectional_v_tail_chord_stretch, 2: sectional_v_tail_span_stretch},
#     translations={0: sectional_v_tail_translation_x, 2: sectional_v_tail_translation_z},
#     rotations={1: sectional_v_tail_twist}
# )

# v_tail_ffd_block_coefficients = v_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
# v_tail_coefficients = vertTail_ffd_block.evaluate(v_tail_ffd_block_coefficients, plot=False)
# vertTail.set_coefficients(coefficients=v_tail_coefficients)
# geometry.plot()

# # Vertical Tail Connection
# vtail_fuselage_connection_point = geometry.evaluate(vertTail.project(np.array([30.543, 0., 8.231])))
# vtail_fuselage_connection = geometry.evaluate(fuselage_rear_pts_parametric) - vtail_fuselage_connection_point   
# parameterization_design_parameters.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

# # Fuselage FFD Setup

# fuselage_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='fuselage_sectional_param',
#                                                                             parameterized_points=fuselage_ffd_block.coefficients,
#                                                                             principal_parametric_dimension=0)

# fuselage_stretch_coefficients = csdl.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
# fuselage_stretch_b_spline = lfs.Function(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                         coefficients=fuselage_stretch_coefficients)

# parameterization_solver.add_parameter(parameter=fuselage_stretch_coefficients)

# # Fuselage Parameterization Evaluation for Parameterization Solver

# section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
# sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = lg.VolumeSectionalParameterizationInputs(
#     translations={0: sectional_fuselage_stretch}
# )

# fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
# fuselage_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
# fuselage.set_coefficients(coefficients=fuselage_coefficients)
# # geometry.plot() 

# # geometry.plot()
# # parameterization_solver.evaluate(parameterization_design_parameters)
# # geometry.plot()





# def hierarchy():
#     Aircraft = Component(name='Complete Aircraft', geometry=geometry, compute_surface_area_flag=False)
#     airframe = Component(name='Complete Aircraft', geometry=geometry, compute_surface_area_flag=False)

#     base_config = Configuration(system=airframe)

#     Complete_Wing = Component(name='Complete Wing')
#     Wing = Component(name='Main Wing', geometry=wing)
#     LeftAil = Component(name='Left Aileron', geometry=aileronL)
#     RightAil = Component(name='Right Aileron', geometry=aileronR)
#     Flap = Component(name='Flap', geometry=flap)
#     Wing.add_subcomponent(LeftAil)
#     Wing.add_subcomponent(RightAil)
#     Wing.add_subcomponent(Flap)
#     Complete_Wing.add_subcomponent(Wing)
#     Aircraft.add_subcomponent(Complete_Wing)

#     Empennage = Component(name='Empennage')
#     HorTail = Component(name="Horizontal Tail", geometry=h_tail)
#     TrimTab = Component(name='Trim Tab', geometry=trimTab)
#     HorTail.add_subcomponent(TrimTab)
#     VertTail = Component(name="Vertical Tail", geometry=vertTail)
#     Rudder = Component(name='Rudder', geometry=rudder)
#     VertTail.add_subcomponent(Rudder)
#     Empennage.add_subcomponent(HorTail)
#     Empennage.add_subcomponent(VertTail)

#     Aircraft.add_subcomponent(Empennage)

#     Fuselage = Component(name="Fuselage", geometry=fuselage)
#     Aircraft.add_subcomponent(Fuselage)

#     Total_Prop_Sys = Component(name='Complete Propulsion System')
#     for i in range(1, 13):
#         Motor = Component(name=f'Propulsor {i}')
#         Motor.add_subcomponent(Component(name=f'Nacelle {i}'))
#         Motor.add_subcomponent(Component(name=f'Pylon {i}', geometry=eval(f'pylon{i}')))
#         Motor.add_subcomponent(Component(name=f'Motor {i}'))
#         Motor.add_subcomponent(Component(name=f'Motor Interface {i}'))
#         Motor.add_subcomponent(Component(name=f'Prop {i}'))
#         Motor.add_subcomponent(Component(name=f'Spinner {i}', geometry=eval(f'spinner{i}')))
#         Total_Prop_Sys.add_subcomponent(Motor)


#     for i in range(1, 3):
#         CruiseMotor = Component(name=f'Cruise Propulsor {i}')
#         CruiseMotor.add_subcomponent(Component(name=f'Cruise Spinner {i}', geometry=cruise_spinner))
#         CruiseMotor.add_subcomponent(Component(name=f'Cruise Nacelle {i}'))
#         CruiseMotor.add_subcomponent(Component(name=f'Cruise Prop {i}'))
#         CruiseMotor.add_subcomponent(Component(name=f'Cruise Motor {i}'))
#         Total_Prop_Sys.add_subcomponent(CruiseMotor)

#     Aircraft.add_subcomponent(Total_Prop_Sys)

    # base_config.connect_component_geometries(Wing, LeftAil, connection_point=wing_te_center_ailerons.value)
    # base_config.connect_component_geometries(Wing, RightAil, connection_point=wing_te_center_ailerons.value)
    # base_config.connect_component_geometries(Wing, Flap, connection_point=wing_te_center_flaps.value)

    # base_config.connect_component_geometries(HorTail, TrimTab, connection_point=ht_te_center.value)
    # base_config.connect_component_geometries(VertTail, Rudder, connection_point=vt_te_mid.value)
    
    # base_config.connect_component_geometries(Fuselage, Wing, connection_point=0.75*wing_le_center.value + 0.25*wing_te_center.value)
    # base_config.connect_component_geometries(Fuselage, HorTail, connection_point=ht_te_center.value)
    # base_config.connect_component_geometries(Fuselage, VertTail, connection_point=vt_le_base.value)


    # return Aircraft, base_config
# Aircraft, BaseConfig = hierarchy()

# BaseConfig.setup_geometry(plot=True)






from flight_simulator.core.vehicle.components.wing import Wing
from flight_simulator.core.vehicle.components.fuselage import Fuselage
from flight_simulator.core.vehicle.components.aircraft import Aircraft as AircraftComp

def define_base_config():

    # Aircraft = AircraftComp(geometry=geometry, compute_surface_area_flag=False)
    # Aircraft.geometry=geometry

    Aircraft = Component(name='Complete Aircraft', geometry=geometry, compute_surface_area_flag=False)

    base_config = Configuration(system=Aircraft)

    fuselage_length = csdl.Variable(name="fuselage_length", shape=(1, ), value=csdl.norm(fuselage_rear_guess[0] - fuselage_nose_guess[0]).value)
    Fuselage_comp = Fuselage(
        length=fuselage_length, geometry=fuselage, skip_ffd=False)
    Fuselage_comp.geometry = fuselage

    wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=AR)
    wing_S_ref = csdl.Variable(name="wing_S_ref", shape=(1, ), value=S)
    wing_span = csdl.Variable(name="wingspan", shape=(1, ), value=csdl.norm(
        geometry.evaluate(wing_le_left_parametric) - geometry.evaluate(wing_le_right_parametric)
    ).value)

    Wing_comp = Wing(AR=wing_AR,S_ref=wing_S_ref,
                                        geometry=wing,
                                        tight_fit_ffd=False, orientation='horizontal', name='WingComp')

    flap_AR = csdl.Variable(name="flap_AR", shape=(1, ), value=12.12)
    flap_S_ref = csdl.Variable(name="flap_S_ref", shape=(1, ), value=4)
    Flaps_comp = Wing(AR=flap_AR, S_ref=flap_S_ref,
                                        geometry=flap,tight_fit_ffd=False, orientation='horizontal', name='FlapsComp')

    aileron_AR = csdl.Variable(name="aileron_AR", shape=(1, ), value=12.12)
    aileron_S_ref = csdl.Variable(name="aileron_S_ref", shape=(1, ), value=4)
    Left_Aileron_comp = Wing(AR=aileron_AR, S_ref=aileron_S_ref,
                                        geometry=aileronL,tight_fit_ffd=False, name='LeftAileronComp',orientation='horizontal')
    Right_Aileron_comp = Wing(AR=aileron_AR, S_ref=aileron_S_ref,
                                        geometry=aileronR,tight_fit_ffd=False, name='RightAileronComp',orientation='horizontal')

    HT_comp = Wing(AR=12, S_ref=50, geometry=h_tail, tight_fit_ffd=False, name='HTComp', orientation='horizontal')
    TrimTab_comp = Wing(AR=12, S_ref=4, geometry=trimTab, tight_fit_ffd=False, name='TrimTabComp', orientation='horizontal')

    VT_comp = Wing(AR=12, S_ref=50, geometry=vertTail, tight_fit_ffd=False, name='VTComp', orientation='vertical')
    Rudder_comp = Wing(AR=12, S_ref=4, geometry=rudder, tight_fit_ffd=False, name='RudderComp', orientation='vertical')

    base_config.connect_component_geometries(Wing_comp, Flaps_comp, connection_point=wing_te_center_flaps.value)
    base_config.connect_component_geometries(Wing_comp, Left_Aileron_comp, connection_point=wing_te_center_ailerons.value)
    base_config.connect_component_geometries(Wing_comp, Right_Aileron_comp, connection_point=wing_te_center_ailerons.value)
    base_config.connect_component_geometries(Wing_comp, Fuselage_comp, connection_point=0.75*wing_le_center.value + 0.25*wing_te_center.value)

    base_config.connect_component_geometries(HT_comp, TrimTab_comp, connection_point=ht_te_center.value)
    base_config.connect_component_geometries(VT_comp, Rudder_comp, connection_point=vt_te_mid.value)

    base_config.connect_component_geometries(HT_comp, Fuselage_comp, connection_point=ht_te_center.value)
    base_config.connect_component_geometries(VT_comp, Fuselage_comp, connection_point=vt_le_base.value)


    
    return base_config
BaseConfig = define_base_config()


BaseConfig.setup_geometry(plot=True)

recorder.stop()