import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
from flight_simulator import ureg, REPO_ROOT_FOLDER
from typing import Union
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import ForcesMoments, Vector
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator.utils.euler_rotations import build_rotation_matrix
from flight_simulator.core.vehicle.component import Component
import lsdo_geo as lg

plot_flag=False
# Exported stl from OpenVSP as feet instead of meters or inches so converting to meters
ft2m=0.3048

recorder=csdl.Recorder(inline=True)
recorder.start()

geometry=import_geometry(
    "Wisk_V6.stp",
    file_path= REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'wiskgen6',
    refit=False,
    scale=ft2m,
    rotate_to_body_fixed_frame=True
)
if plot_flag:
    geometry.plot()

Aircraft = Component(name='Aircraft')

# Define Aircraft Components
## Wing
wing = geometry.declare_component(function_search_names=['Wing'], name='wing')
wing.plot()

# if plot_flag:
    # wing.plot()
Aircraft.add_subcomponent(Component(name='Wing',geometry=wing))

## Tail(s)
h_tail = geometry.declare_component(function_search_names=['HTail'], name='h_tail')
v_tail = geometry.declare_component(function_search_names=['VTail'], name='v_tail')
# if plot_flag:
if plot_flag:
    h_tail.plot()
    v_tail.plot()
Aircraft.add_subcomponent(Component(name='Horizontal Tail',geometry=h_tail))
Aircraft.add_subcomponent(Component(name='Vertical Tail',geometry=v_tail))

## Fuselage
fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
Aircraft.add_subcomponent(Component(name='Fuselage',geometry=fuselage))

# Landing Gear
fwd_landing_gear_pylon = geometry.declare_component(function_search_names=['FWD_LG'], name='fwd_landing_gear_pylon')
aft_landing_gear_pylon = geometry.declare_component(function_search_names=['AFT_LG'], name='aft_landing_gear_pylon')
base_landing_gear = geometry.declare_component(function_search_names=['LG_BASE'], name='base_landing_gear')

Landing_Gear = Component(name='Landing Gear')
Landing_Gear.add_subcomponent(Component(name=' Landing Gear Forward Pylon',geometry=fwd_landing_gear_pylon))
Landing_Gear.add_subcomponent(Component(name='Landing Gear Aft Pylon',geometry=aft_landing_gear_pylon))
Landing_Gear.add_subcomponent(Component(name='Landing Gear Base',geometry=base_landing_gear))
Aircraft.add_subcomponent(Landing_Gear)

# Pylons for Propulsors
pylon_ob_left = geometry.declare_component(function_search_names=['OB_SUPPORT_L'], name='pylon_ob_left')
pylon_mid_left = geometry.declare_component(function_search_names=['MID_SUPPORT_L'], name='pylon_mid_left')
pylon_ib_left = geometry.declare_component(function_search_names=['IB_SUPPORT_L'], name='pylon_ib_left')

pylon_ob_right = geometry.declare_component(function_search_names=['OB_SUPPORT_R'], name='pylon_ob_right')
pylon_mid_right = geometry.declare_component(function_search_names=['MID_SUPPORT_R'], name='pylon_mid_right')
pylon_ib_right = geometry.declare_component(function_search_names=['IB_SUPPORT_R'], name='pylon_ib_right')

# Rotor Hubs for Cruise (FWD) motors
rotor_hub_ob_left_fwd = geometry.declare_component(function_search_names=['ROTOR_HUB_OB_FWD_L'], name='rotor_hub_ob_left_fwd')
rotor_hub_mid_left_fwd = geometry.declare_component(function_search_names=['ROTOR_HUB_MID_FWD_L'], name='rotor_hub_mid_left_fwd')
rotor_hub_ib_left_fwd = geometry.declare_component(function_search_names=['ROTOR_HUB_IB_FWD_L'], name='rotor_hub_ib_left_fwd')
rotor_hub_ob_right_fwd = geometry.declare_component(function_search_names=['ROTOR_HUB_OB_FWD_R'], name='rotor_hub_ob_right_fwd')
rotor_hub_mid_right_fwd = geometry.declare_component(function_search_names=['ROTOR_HUB_MID_FWD_R'], name='rotor_hub_mid_right_fwd')
rotor_hub_ib_right_fwd = geometry.declare_component(function_search_names=['ROTOR_HUB_IB_FWD_R'], name='rotor_hub_ib_right_fwd')

# Rotor Hubs for Lift-only (AFT) motors
rotor_hub_ob_left_aft = geometry.declare_component(function_search_names=['ROTOR_HUB_OB_AFT_L'], name='rotor_hub_ob_left_aft')
rotor_hub_mid_left_aft = geometry.declare_component(function_search_names=['ROTOR_HUB_MID_AFT_L'], name='rotor_hub_mid_left_aft')
rotor_hub_ib_left_aft = geometry.declare_component(function_search_names=['ROTOR_HUB_IB_AFT_L'], name='rotor_hub_ib_left_aft')
rotor_hub_ob_right_aft = geometry.declare_component(function_search_names=['ROTOR_HUB_OB_AFT_R'], name='rotor_hub_ob_right_aft')
rotor_hub_mid_right_aft = geometry.declare_component(function_search_names=['ROTOR_HUB_MID_AFT_R'], name='rotor_hub_mid_right_aft')
rotor_hub_ib_right_aft = geometry.declare_component(function_search_names=['ROTOR_HUB_IB_AFT_R'], name='rotor_hub_ib_right_aft')


Propulsion = Component(name='Propulsion')
LPC_Propulsion = Component(name='Lift+Cruise Propulsion (FWD)')
Propulsion.add_subcomponent(LPC_Propulsion)
Motor_ob_left_fwd = Component(name='Motor Outboard Left FWD')
Motor_mid_left_fwd = Component(name='Motor Middle Left FWD')
Motor_ib_left_fwd = Component(name='Motor Inboard Left FWD')
Motor_ob_right_fwd = Component(name='Motor Outboard Right FWD')
Motor_mid_right_fwd = Component(name='Motor Middle Right FWD')
Motor_ib_right_fwd = Component(name='Motor Inboard Right FWD')
LPC_Propulsion.add_subcomponent(Motor_ob_left_fwd)
LPC_Propulsion.add_subcomponent(Motor_mid_left_fwd)
LPC_Propulsion.add_subcomponent(Motor_ib_left_fwd)
LPC_Propulsion.add_subcomponent(Motor_ob_right_fwd)
LPC_Propulsion.add_subcomponent(Motor_mid_right_fwd)
LPC_Propulsion.add_subcomponent(Motor_ib_right_fwd)
LIFT_Propulsion = Component(name='Lift Only Propulsion (AFT)')
Propulsion.add_subcomponent(LIFT_Propulsion)
Motor_ob_left_aft = Component(name='Motor Outboard Left AFT')
Motor_mid_left_aft = Component(name='Motor Middle Left AFT')
Motor_ib_left_aft = Component(name='Motor Inboard Left AFT')
Motor_ob_right_aft = Component(name='Motor Outboard Right AFT')
Motor_mid_right_aft = Component(name='Motor Middle Right AFT')
Motor_ib_right_aft = Component(name='Motor Inboard Right AFT')
LIFT_Propulsion.add_subcomponent(Motor_ob_left_aft)
LIFT_Propulsion.add_subcomponent(Motor_mid_left_aft)
LIFT_Propulsion.add_subcomponent(Motor_ib_left_aft)
LIFT_Propulsion.add_subcomponent(Motor_ob_right_aft)
LIFT_Propulsion.add_subcomponent(Motor_mid_right_aft)
LIFT_Propulsion.add_subcomponent(Motor_ib_right_aft)
Supports = Component(name='Prop Pylons')
Supports.add_subcomponent(Component("Pylon Outboard Left",geometry=pylon_ob_left))
Supports.add_subcomponent(Component("Pylon Outboard Right",geometry=pylon_ob_right))
Supports.add_subcomponent(Component("Pylon Middle Left",geometry=pylon_mid_left))
Supports.add_subcomponent(Component("Pylon Middle Right",geometry=pylon_mid_right))
Supports.add_subcomponent(Component("Pylon Inboard Left",geometry=pylon_ib_left))
Supports.add_subcomponent(Component("Pylon Inboard Right",geometry=pylon_ib_right))
Propulsion.add_subcomponent(Supports)
Aircraft.add_subcomponent(Propulsion)


# Wing info
wing_root_le_guess = np.array([-8, 0, -5.41])*ft2m
wing_root_le_parametric = wing.project(wing_root_le_guess, plot=plot_flag)
wing_root_le = geometry.evaluate(wing_root_le_parametric)

wing_root_te_guess = np.array([-16, 0, -5.7])*ft2m
wing_root_te_parametric = wing.project(wing_root_te_guess, plot=plot_flag)
wing_root_te = geometry.evaluate(wing_root_te_parametric)

wing_tip_left_le_guess = np.array([-10, -26, -5.5])*ft2m
wing_tip_left_le_parametric = wing.project(wing_tip_left_le_guess,plot=plot_flag)

wing_tip_right_le_guess = np.array([-10, 26, -5.5])*ft2m
wing_tip_right_le_parametric = wing.project(wing_tip_right_le_guess,plot=plot_flag)


wingspan = csdl.norm(
    geometry.evaluate(wing_tip_left_le_parametric) - geometry.evaluate(wing_tip_right_le_parametric)
)
# print('Wingspan: ',wingspan.value)

# Horizontal Tail Info
htail_root_le_guess = np.array([-23, 0, -3.7])*ft2m
htail_root_le_parametric = h_tail.project(htail_root_le_guess, plot=plot_flag)
htail_root_le = geometry.evaluate(htail_root_le_parametric)

htail_root_te_guess = np.array([-28, 0, -3.8])*ft2m
htail_root_te_parametric = h_tail.project(htail_root_te_guess, plot=plot_flag)
htail_root_te = geometry.evaluate(htail_root_te_parametric)

htail_tip_left_le_guess = np.array([-25, -7, -3.7])*ft2m
htail_tip_left_le_parametric = h_tail.project(htail_tip_left_le_guess,plot=plot_flag)

htail_tip_right_le_guess = np.array([-25, 7, -3.7])*ft2m
htail_tip_right_le_parametric = h_tail.project(htail_tip_right_le_guess,plot=plot_flag)

htail_span = csdl.norm(
    geometry.evaluate(htail_tip_left_le_parametric) - geometry.evaluate(htail_tip_right_le_parametric)
)
# print('Horizontal Tail Span: ',htail_span.value)
# print('Horizontal Tail Span (ft): ',htail_span.value /ft2m)

# Vertical Tail Info
vtail_root_le_guess = np.array([-23, 0, -3.6])*ft2m
vtail_root_le_parametric = v_tail.project(vtail_root_le_guess, plot=plot_flag)
vtail_root_le = geometry.evaluate(vtail_root_le_parametric)

vtail_root_te_guess = np.array([-28.4, 0, -3.4])*ft2m
vtail_root_te_parametric = v_tail.project(vtail_root_te_guess, plot=plot_flag)
vtail_root_te = geometry.evaluate(vtail_root_te_parametric)

vtail_tip_le_guess = np.array([-27.6, 0, -10.25])*ft2m
vtail_tip_le_parametric = v_tail.project(vtail_tip_le_guess, plot=plot_flag)

vtail_tip_te_guess = np.array([-29.8, 0, -10.25])*ft2m
vtail_tip_te_parametric = v_tail.project(vtail_tip_te_guess, plot=plot_flag)

vtail_span = csdl.norm(
    vtail_root_te - geometry.evaluate(vtail_tip_te_parametric)
)
# print('Vertical Tail Span: ',vtail_span.value)
# print('Vertical Tail Span (ft): ',vtail_span.value /ft2m)


# Propeller Region Info
pt_ob_left_fwd_top_guess = np.array([-4.262,-22.672,-6])*ft2m
pt_ob_left_fwd_top_parametric = rotor_hub_ob_left_fwd.project(pt_ob_left_fwd_top_guess, plot=plot_flag)
pt_ob_left_fwd_bot_guess = np.array([-4.262,-22.672,-4.9])*ft2m
pt_ob_left_fwd_bot_parametric = rotor_hub_ob_left_fwd.project(pt_ob_left_fwd_bot_guess, plot=plot_flag)
pt_ob_left_fwd_mid_guess=1/2*(rotor_hub_ob_left_fwd.evaluate(pt_ob_left_fwd_top_parametric)+rotor_hub_ob_left_fwd.evaluate(pt_ob_left_fwd_bot_parametric))

pt_mid_left_fwd_top_guess = np.array([-4.262,-15.4,-6])*ft2m
pt_mid_left_fwd_bot_guess = np.array([-4.262,-15.4,-4.9])*ft2m
pt_mid_left_fwd_top_parametric = rotor_hub_mid_left_fwd.project(pt_mid_left_fwd_top_guess, plot=plot_flag)
pt_mid_left_fwd_bot_parametric = rotor_hub_mid_left_fwd.project(pt_mid_left_fwd_bot_guess, plot=plot_flag)

pt_ib_left_fwd_top_guess = np.array([-4.262,-7.4,-6.2])*ft2m
pt_ib_left_fwd_bot_guess = np.array([-4.262,-7.4,-4.9])*ft2m
pt_ib_left_fwd_top_parametric = rotor_hub_ib_left_fwd.project(pt_ib_left_fwd_top_guess, plot=plot_flag)
pt_ib_left_fwd_bot_parametric = rotor_hub_ib_left_fwd.project(pt_ib_left_fwd_bot_guess, plot=plot_flag)

pt_ib_right_fwd_top_guess = np.array([-4.262,7.4,-6.2])*ft2m
pt_ib_right_fwd_bot_guess = np.array([-4.262,7.4,-4.9])*ft2m
pt_ib_right_fwd_top_parametric = rotor_hub_ib_right_fwd.project(pt_ib_right_fwd_top_guess, plot=plot_flag)
pt_ib_right_fwd_bot_parametric = rotor_hub_ib_right_fwd.project(pt_ib_right_fwd_bot_guess, plot=plot_flag)

pt_ob_right_fwd_top_guess = np.array([-4.262,22.672,-6])*ft2m
pt_ob_right_fwd_bot_guess = np.array([-4.262,22.672,-4.9])*ft2m
pt_ob_right_fwd_top_parametric = rotor_hub_ob_right_fwd.project(pt_ob_right_fwd_top_guess, plot=plot_flag)
pt_ob_right_fwd_bot_parametric = rotor_hub_ob_right_fwd.project(pt_ob_right_fwd_bot_guess, plot=plot_flag)

pt_mid_right_fwd_top_guess = np.array([-4.262,15.4,-6])*ft2m
pt_mid_right_fwd_bot_guess = np.array([-4.262,15.4,-4.9])*ft2m
pt_mid_right_fwd_top_parametric = rotor_hub_mid_right_fwd.project(pt_mid_right_fwd_top_guess, plot=plot_flag)
pt_mid_right_fwd_bot_parametric = rotor_hub_mid_right_fwd.project(pt_mid_right_fwd_bot_guess, plot=plot_flag)

pt_ob_left_aft_top_guess = np.array([-17.4,-22.2,-6.2])*ft2m
pt_ob_left_aft_top_parametric = rotor_hub_ob_left_aft.project(pt_ob_left_aft_top_guess, plot=plot_flag)
pt_ob_left_aft_bot_guess = np.array([-17.4,-22.2,-4.8])*ft2m
pt_ob_left_aft_bot_parametric = rotor_hub_ob_left_aft.project(pt_ob_left_aft_bot_guess, plot=plot_flag)

pt_mid_left_aft_top_guess = np.array([-17.4,-15.1,-6.2])*ft2m
pt_mid_left_aft_top_parametric = rotor_hub_mid_left_aft.project(pt_mid_left_aft_top_guess, plot=plot_flag)
pt_mid_left_aft_bot_guess = np.array([-17.4,-15.1,-4.8])*ft2m
pt_mid_left_aft_bot_parametric = rotor_hub_mid_left_aft.project(pt_mid_left_aft_bot_guess, plot=plot_flag)

pt_ib_left_aft_top_guess = np.array([-17.4,-7,-6.2])*ft2m
pt_ib_left_aft_top_parametric = rotor_hub_ib_left_aft.project(pt_ib_left_aft_top_guess, plot=plot_flag)
pt_ib_left_aft_bot_guess = np.array([-17.4,-7,-4.8])*ft2m
pt_ib_left_aft_bot_parametric = rotor_hub_ib_left_aft.project(pt_ib_left_aft_bot_guess, plot=plot_flag)

pt_ib_right_aft_top_guess = np.array([-17.4,7,-6.2])*ft2m
pt_ib_right_aft_top_parametric = rotor_hub_ib_right_aft.project(pt_ib_right_aft_top_guess, plot=plot_flag)
pt_ib_right_aft_bot_guess = np.array([-17.4,7,-4.8])*ft2m
pt_ib_right_aft_bot_parametric = rotor_hub_ib_right_aft.project(pt_ib_right_aft_bot_guess, plot=plot_flag)

pt_ob_right_aft_top_guess = np.array([-17.4,22.2,-6.2])*ft2m
pt_ob_right_aft_top_parametric = rotor_hub_ob_right_aft.project(pt_ob_right_aft_top_guess, plot=plot_flag)
pt_ob_right_aft_bot_guess = np.array([-17.4,22.2,-4.8])*ft2m
pt_ob_right_aft_bot_parametric = rotor_hub_ob_right_aft.project(pt_ob_right_aft_bot_guess, plot=plot_flag)

pt_mid_right_aft_top_guess = np.array([-17.4,15.1,-6.2])*ft2m
pt_mid_right_aft_top_parametric = rotor_hub_mid_right_aft.project(pt_mid_right_aft_top_guess, plot=plot_flag)
pt_mid_right_aft_bot_guess = np.array([-17.4,15.1,-4.8])*ft2m
pt_mid_right_aft_bot_parametric = rotor_hub_mid_right_aft.project(pt_mid_right_aft_bot_guess, plot=plot_flag)

# Axis Creation
openvsp_axis = Axis(
    name='OpenVSP Axis',
    x = np.array([0, ])*ureg.meter,
    y = np.array([0, ])*ureg.meter,
    z = np.array([0, ])*ureg.meter,
    origin = ValidOrigins.OpenVSP.value
)

wing_axis = AxisLsdoGeo(
    name='Wing Axis',
    geometry=wing,
    parametric_coords = wing_root_le_parametric,
    sequence = np.array([3,2,1]),
    phi=np.array([0, ])*ureg.degree,
    theta=np.array([0, ])*ureg.degree,
    psi=np.array([0, ])*ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
# print('Wing Axis Translation (m): ', wing_axis.translation.value)
# print('Wing Axis Rotation (deg): ', np.rad2deg(wing_axis.euler_angles_vector.value))

vtail_deflection = csdl.Variable(shape=(1, ), value=np.deg2rad(45), name='Vtail Deflection')
v_tail.rotate(vtail_root_le, np.array([0., 0., 1.]), angles=vtail_deflection)

vtail_axis = AxisLsdoGeo(
    name='Vtail Axis',
    geometry=v_tail,
    parametric_coords = vtail_root_le_parametric,
    sequence = np.array([3,2,1]),
    phi=np.array([0, ])*ureg.degree,
    theta=np.array([0, ])*ureg.degree,
    psi=vtail_deflection,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
# geometry.plot()

# print('Vtail Axis Translation (m): ', vtail_axis.translation.value)
# print('Vtail Axis Rotation (deg): ', np.rad2deg(vtail_axis.euler_angles_vector.value))

htail_deflection = csdl.Variable(shape=(1, ), value=np.deg2rad(45), name='Htail Deflection')
h_tail.rotate(htail_root_le, np.array([0., 1., 0.]), angles=htail_deflection)

htail_axis = AxisLsdoGeo(
    name='Htail Axis',
    geometry=h_tail,
    parametric_coords = htail_root_le_parametric,
    sequence = np.array([3,2,1]),
    phi=np.array([0, ])*ureg.degree,
    theta=htail_deflection,
    psi=np.array([0, ])*ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
# print('Htail Axis Translation (m): ', htail_axis.translation.value)
# print('Htail Axis Rotation (deg): ', np.rad2deg(htail_axis.euler_angles_vector.value))
# geometry.plot()

pt_axis_ob_left_fwd = AxisLsdoGeo(
    name='PT Axis Outboard Left FWD',
    geometry=rotor_hub_ob_left_fwd,
    parametric_coords = pt_ob_left_fwd_bot_parametric,
    sequence = np.array([3,2,1]),
    phi=np.array([0, ])*ureg.degree,
    theta=np.array([0, ])*ureg.degree,  
    psi=np.array([0, ])*ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

# print('Prop Axis Outboard Left FWD Translation (m): ', pt_axis_ob_left_fwd.translation.value)
# print('Prop Axis Outboard Left FWD Rotation (deg): ', np.rad2deg(pt_axis_ob_left_fwd.euler_angles_vector.value))

pt_axis_mid_left_fwd = AxisLsdoGeo(
    name='PT Axis Middle Left FWD',
    geometry=rotor_hub_mid_left_fwd,
    parametric_coords=pt_mid_left_fwd_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ib_left_fwd = AxisLsdoGeo(
    name='PT Axis Inboard Left FWD',
    geometry=rotor_hub_ib_left_fwd,
    parametric_coords=pt_ib_left_fwd_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ob_right_fwd = AxisLsdoGeo(
    name='PT Axis Outboard Right FWD',
    geometry=rotor_hub_ob_right_fwd,
    parametric_coords=pt_ob_right_fwd_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_mid_right_fwd = AxisLsdoGeo(
    name='PT Axis Middle Right FWD',
    geometry=rotor_hub_mid_right_fwd,
    parametric_coords=pt_mid_right_fwd_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ib_right_fwd = AxisLsdoGeo(
    name='PT Axis Inboard Right FWD',
    geometry=rotor_hub_ib_right_fwd,
    parametric_coords=pt_ib_right_fwd_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ob_left_aft = AxisLsdoGeo(
    name='PT Axis Outboard Left AFT',
    geometry=rotor_hub_ob_left_aft,
    parametric_coords=pt_ob_left_aft_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_mid_left_aft = AxisLsdoGeo(
    name='PT Axis Middle Left AFT',
    geometry=rotor_hub_mid_left_aft,
    parametric_coords=pt_mid_left_aft_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ib_left_aft = AxisLsdoGeo(
    name='PT Axis Inboard Left AFT',
    geometry=rotor_hub_ib_left_aft,
    parametric_coords=pt_ib_left_aft_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ob_right_aft = AxisLsdoGeo(
    name='PT Axis Outboard Right AFT',
    geometry=rotor_hub_ob_right_aft,
    parametric_coords=pt_ob_right_aft_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_mid_right_aft = AxisLsdoGeo(
    name='PT Axis Middle Right AFT',
    geometry=rotor_hub_mid_right_aft,
    parametric_coords=pt_mid_right_aft_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

pt_axis_ib_right_aft = AxisLsdoGeo(
    name='PT Axis Inboard Right AFT',
    geometry=rotor_hub_ib_right_aft,
    parametric_coords=pt_ib_right_aft_bot_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=np.array([0, ]) * ureg.degree,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)

# Non dependent on Aircraft FD axes
# Choosing Inertial aaxis as OpenVSP [0,0,0]
inertial_axis = Axis(
    name='Inertial Axis',
    x=np.array([0, ])*ureg.meter,
    y=np.array([0, ])*ureg.meter,
    z=np.array([0, ])*ureg.meter,
    origin = ValidOrigins.Inertial.value
)


from flight_simulator.core.dynamics.aircraft_states import AircaftStates

# ac_states = AircaftStates()
# ac_states.phi = csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='phi')
# ac_states.theta = csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='theta')
# ac_states.psi = csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='psi')

fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    x=np.array([0, ])*ureg.meter,
    y=np.array([0, ])*ureg.meter,
    z=np.array([5000, ])*ureg.meter,
    phi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='phi'),
    theta=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='theta'),
    psi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='psi'),
    sequence=np.array([3,2,1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
# print('Body-Fixed Angles (deg)', np.rad2deg(fd_axis.euler_angles_vector.value))

@dataclass
class WindAxisRotations(csdl.VariableGroup):
    mu : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree # bank
    gamma : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(2), name='Flight path angle')
    xi : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree  # Heading
wind_axis_rotations = WindAxisRotations()

wind_axis = Axis(
    name='Wind Axis',
    x=np.array([0, ])*ureg.meter,
    y=np.array([0, ])*ureg.meter,
    z=np.array([0, ])*ureg.meter,
    phi=wind_axis_rotations.mu,
    theta=wind_axis_rotations.gamma,
    psi=wind_axis_rotations.xi,
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
# print('Wind axis angles (deg)', np.rad2deg(wind_axis.euler_angles_vector.value))
# endregion

# endregion

# region Forces and Moments

# region Aero forces

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
# print('Effective angle of attack (deg): ', np.rad2deg(alpha.value))

CL = 2*np.pi*alpha
CD = 0.001 + 1/(np.pi*0.87*12) * CL**2
rho = 1.225
S = 203*ft2m
V = 35
L = 0.5*rho*V**2*CL*S
D = 0.5*rho*V**2*CD*S

aero_force = csdl.Variable(shape=(3, ), value=0.)
aero_force = aero_force.set(csdl.slice[0], -D)
aero_force = aero_force.set(csdl.slice[2], -L)

aero_force_vector_in_wind = Vector(vector=aero_force, axis=wind_axis)
# print('Aero force vector in wind-axis: ', aero_force_vector_in_wind.vector.value)
aero_force_vector_in_inertial =  Vector(csdl.matvec(R_wind_to_inertial, aero_force_vector_in_wind.vector), axis=inertial_axis)
# print('Aero force vector in inertial-axis: ', aero_force_vector_in_inertial.vector.value)
aero_force_vector_in_body =  Vector(csdl.matvec(csdl.transpose(R_body_to_inertial), aero_force_vector_in_inertial.vector), axis=fd_axis)
# print('Aero force vector in body-axis: ', aero_force_vector_in_body.vector.value)
aero_force_vector_in_wing =  Vector(csdl.matvec(csdl.transpose(R_wing_to_openvsp), aero_force_vector_in_body.vector), axis=fd_axis)
# print('Aero force vector in wing-axis: ', aero_force_vector_in_wing.vector.value)


## FFD

# region Parameterization

constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# region Parameterization Setup
parameterization_solver = lg.ParameterizationSolver()
parameterization_design_parameters = lg.GeometricVariables()

# region Wing Parameterization setup
wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2,11,2), degree=(1,3,1))
wing_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='wing_sectional_parameterization',
                                                                            parameterized_points=wing_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

wing_chord_stretch_coefficients = csdl.Variable(name='wing_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
wing_chord_stretch_b_spline = lfs.Function(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=wing_chord_stretch_coefficients)

wing_wingspan_stretch_coefficients = csdl.Variable(name='wing_wingspan_stretch_coefficients', value=np.array([-0., 0.]))
wing_wingspan_stretch_b_spline = lfs.Function(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=wing_wingspan_stretch_coefficients)

wing_twist_coefficients = csdl.Variable(name='wing_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
wing_twist_b_spline = lfs.Function(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                          coefficients=wing_twist_coefficients)

wing_translation_x_coefficients = csdl.Variable(name='wing_translation_x_coefficients', value=np.array([0.]))
wing_translation_x_b_spline = lfs.Function(name='wing_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=wing_translation_x_coefficients)

wing_translation_z_coefficients = csdl.Variable(name='wing_translation_z_coefficients', value=np.array([0.]))
wing_translation_z_b_spline = lfs.Function(name='wing_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=wing_translation_z_coefficients)

parameterization_solver.add_parameter(parameter=wing_chord_stretch_coefficients)
parameterization_solver.add_parameter(parameter=wing_wingspan_stretch_coefficients, cost=1.e3)
parameterization_solver.add_parameter(parameter=wing_twist_coefficients)
parameterization_solver.add_parameter(parameter=wing_translation_x_coefficients)
parameterization_solver.add_parameter(parameter=wing_translation_z_coefficients)

# region Horizontal Stabilizer setup
h_tail_ffd_block = lg.construct_ffd_block_around_entities(name='h_tail_ffd_block', entities=h_tail, num_coefficients=(2,11,2), degree=(1,3,1))
h_tail_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='h_tail_sectional_parameterization',
                                                                            parameterized_points=h_tail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

h_tail_chord_stretch_coefficients = csdl.Variable(name='h_tail_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
h_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=h_tail_chord_stretch_coefficients)

h_tail_span_stretch_coefficients = csdl.Variable(name='h_tail_span_stretch_coefficients', value=np.array([-0., 0.]))
h_tail_span_stretch_b_spline = lfs.Function(name='h_tail_span_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=h_tail_span_stretch_coefficients)

h_tail_twist_coefficients = csdl.Variable(name='h_tail_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
h_tail_twist_b_spline = lfs.Function(name='h_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                          coefficients=h_tail_twist_coefficients)

h_tail_translation_x_coefficients = csdl.Variable(name='h_tail_translation_x_coefficients', value=np.array([0.]))
h_tail_translation_x_b_spline = lfs.Function(name='h_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=h_tail_translation_x_coefficients)
h_tail_translation_z_coefficients = csdl.Variable(name='h_tail_translation_z_coefficients', value=np.array([0.]))
h_tail_translation_z_b_spline = lfs.Function(name='h_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=h_tail_translation_z_coefficients)

parameterization_solver.add_parameter(parameter=h_tail_chord_stretch_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_span_stretch_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_twist_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_translation_x_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_translation_z_coefficients)
# endregion Horizontal Stabilizer setup

# region Fuselage setup
fuselage_ffd_block = lg.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=fuselage, num_coefficients=(2,2,2), degree=(1,1,1))
fuselage_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='fuselage_sectional_parameterization',
                                                                            parameterized_points=fuselage_ffd_block.coefficients,
                                                                            principal_parametric_dimension=0)
# fuselage_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_fuselage_stretch', axis=0)

fuselage_stretch_coefficients = csdl.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
fuselage_stretch_b_spline = lfs.Function(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=fuselage_stretch_coefficients)

parameterization_solver.add_parameter(parameter=fuselage_stretch_coefficients)
# endregion

# region Lift Rotors setup
lift_rotor_ffd_blocks = []
lift_rotor_sectional_parameterizations = []
lift_rotor_parameterization_b_splines = []
lift_rotor_related_components = [rotor_hub_ob_left_fwd, rotor_hub_mid_left_fwd, rotor_hub_ib_left_fwd, rotor_hub_ob_right_fwd, rotor_hub_mid_right_fwd, rotor_hub_ib_right_fwd]
for i, component_set in enumerate(lift_rotor_related_components):
    rotor_ffd_block = lg.construct_ffd_block_around_entities(name=f'{component_set[0].name[:3]}_rotor_ffd_block', entities=component_set, num_coefficients=(2,2,2), degree=(1,1,1))
    rotor_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name=f'{component_set[0].name[:3]}_rotor_sectional_parameterization',
                                                                                parameterized_points=rotor_ffd_block.coefficients,
                                                                                principal_parametric_dimension=2)
    
    rotor_stretch_coefficient = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_stretch_coefficient', shape=(1,), value=0.)
    lift_rotor_sectional_stretch_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_sectional_stretch_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                                coefficients=rotor_stretch_coefficient)
    
    lift_rotor_ffd_blocks.append(rotor_ffd_block)
    lift_rotor_sectional_parameterizations.append(rotor_ffd_block_sectional_parameterization)
    lift_rotor_parameterization_b_splines.append(lift_rotor_sectional_stretch_b_spline)                 

    parameterization_solver.add_parameter(parameter=rotor_stretch_coefficient)
# endregion Lift Rotors setup


# region Wing Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_wing_chord_stretch},
    translations={1: sectional_wing_wingspan_stretch, 0: sectional_wing_translation_x, 2: sectional_wing_translation_z}
)

wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
wing.set_coefficients(wing_coefficients)

# endregion Wing Parameterization Evaluation for Parameterization Solver

# region Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_h_tail_chord_stretch},
    translations={1: sectional_h_tail_span_stretch, 0: sectional_h_tail_translation_x, 2: sectional_h_tail_translation_z}
)

h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)
h_tail.set_coefficients(coefficients=h_tail_coefficients)
geometry.plot()
# endregion

# region Fuselage Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = {'sectional_fuselage_stretch':sectional_fuselage_stretch}
sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    translations={0: sectional_fuselage_stretch}
)

fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
fuselage_and_nose_hub_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
fuselage_coefficients = fuselage_and_nose_hub_coefficients[0]
nose_hub_coefficients = fuselage_and_nose_hub_coefficients[1]

fuselage.set_coefficients(coefficients=fuselage_coefficients)
nose_hub.set_coefficients(coefficients=nose_hub_coefficients)
# geometry.plot()

# endregion

# region Lift Rotors rigid body translation
for i, component_set in enumerate(lift_rotor_related_components):
    # disk = component_set[0]
    # blade_1 = component_set[1]
    # blade_2 = component_set[2]
    # hub = component_set[3]

    boom = boom_components[i]

    # Add rigid body translation
    rigid_body_translation = csdl.Variable(shape=(3,), value=0., name=f'{component_set[0].name[:3]}_rotor_rigid_body_translation')

    for component in component_set:
        for function in component.functions.values():
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

    for function in boom.functions.values():
        function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

    parameterization_solver.add_parameter(parameter=rigid_body_translation)
# endregion Lift Rotors rigid body translation

# region pusher rigid body translation
rigid_body_translation = csdl.Variable(shape=(3,), value=0., name='pp_rotor_rigid_body_translation')
for component in pp_components:
    for function in component.functions.values():
        function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

parameterization_solver.add_parameter(parameter=rigid_body_translation)
# endregion pusher rigid body translation

# region Vertical Stabilizer rigid body translation
rigid_body_translation = csdl.Variable(shape=(3,), value=0., name='pp_rotor_rigid_body_translation')
for function in v_tail.functions.values():
    function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

parameterization_solver.add_parameter(parameter=rigid_body_translation)
# endregion Vertical Stabilizer rigid body translation

# endregion Parameterization Solver Setup Evaluations

# Aircraft.visualize_component_hierarchy(show=True)
# geometry.plot()
# region Define Design Parameters

# region wing design parameters
wing_span_computed = csdl.norm(geometry.evaluate(wing_le_right) - geometry.evaluate(wing_le_left))
wing_root_chord_computed = csdl.norm(geometry.evaluate(wing_te_center) - geometry.evaluate(wing_le_center))
wing_tip_chord_left_computed = csdl.norm(geometry.evaluate(wing_te_left) - geometry.evaluate(wing_le_left))
wing_tip_chord_right_computed = csdl.norm(geometry.evaluate(wing_te_right) - geometry.evaluate(wing_le_right))

wing_span = csdl.Variable(name='wing_span', value=np.array([50.]))
wing_root_chord = csdl.Variable(name='wing_root_chord', value=np.array([5.]))
wing_tip_chord = csdl.Variable(name='wing_tip_chord_left', value=np.array([1.]))

parameterization_design_parameters.add_variable(computed_value=wing_span_computed, desired_value=wing_span)
parameterization_design_parameters.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_left_computed, desired_value=wing_tip_chord)
parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_right_computed, desired_value=wing_tip_chord)
# endregion wing design parameters

# region h_tail design parameterization inputs
h_tail_span_computed = csdl.norm(geometry.evaluate(tail_le_right) - geometry.evaluate(tail_le_left))
h_tail_root_chord_computed = csdl.norm(geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
h_tail_tip_chord_left_computed = csdl.norm(geometry.evaluate(tail_te_left) - geometry.evaluate(tail_le_left))
h_tail_tip_chord_right_computed = csdl.norm(geometry.evaluate(tail_te_right) - geometry.evaluate(tail_le_right))

h_tail_span = csdl.Variable(name='h_tail_span', value=np.array([12.]))
h_tail_root_chord = csdl.Variable(name='h_tail_root_chord', value=np.array([3.]))
h_tail_tip_chord = csdl.Variable(name='h_tail_tip_chord_left', value=np.array([2.]))

parameterization_design_parameters.add_variable(computed_value=h_tail_span_computed, desired_value=h_tail_span)
parameterization_design_parameters.add_variable(computed_value=h_tail_root_chord_computed, desired_value=h_tail_root_chord)
parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_left_computed, desired_value=h_tail_tip_chord)
parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_right_computed, desired_value=h_tail_tip_chord)
# endregion h_tail design parameterization inputs

# region tail moment arm variables
tail_moment_arm_computed = csdl.norm(geometry.evaluate(tail_qc) - geometry.evaluate(wing_qc))
tail_moment_arm = csdl.Variable(name='tail_moment_arm', value=np.array([25.]))
parameterization_design_parameters.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm)

wing_fuselage_connection = geometry.evaluate(wing_te_center) - geometry.evaluate(fuselage_wing_te_center)
h_tail_fuselage_connection = geometry.evaluate(tail_te_center) - geometry.evaluate(fuselage_tail_te_center)
parameterization_design_parameters.add_variable(computed_value=wing_fuselage_connection, desired_value=wing_fuselage_connection.value)
parameterization_design_parameters.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)

# endregion tail moment arm variables

# region v-tail connection
vtail_fuselage_connection_point = geometry.evaluate(v_tail.project(np.array([30.543, 0., 8.231])))
vtail_fuselage_connection = geometry.evaluate(fueslage_rear_points_parametric) - vtail_fuselage_connection_point
parameterization_design_parameters.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

# endregion v-tail connection

# region lift + pusher rotor parameterization inputs
pusher_fuselage_connection = geometry.evaluate(fueslage_rear_points_parametric) - geometry.evaluate(fuselage_rear_point_on_pusher_disk_parametric)
parameterization_design_parameters.add_variable(computed_value=pusher_fuselage_connection, desired_value=pusher_fuselage_connection.value)

flo_radius = fro_radius = front_outer_radius = csdl.Variable(name='front_outer_radius', value=10/2)
fli_radius = fri_radius = front_inner_radius = csdl.Variable(name='front_inner_radius', value=10/2)
rlo_radius = rro_radius = rear_outer_radius = csdl.Variable(name='rear_outer_radius', value=10/2)
rli_radius = rri_radius = rear_inner_radius = csdl.Variable(name='rear_inner_radius', value=10/2)
dv_radius_list = [rlo_radius, rli_radius, rri_radius, rro_radius, flo_radius, fli_radius, fri_radius, fro_radius]

boom_points = [boom_rlo, boom_rli, boom_rri, boom_rro, boom_flo, boom_fli, boom_fri, boom_fro]
boom_points_on_wing = [wing_boom_rlo, wing_boom_rli, wing_boom_rri, wing_boom_rro, wing_boom_flo, wing_boom_fli, wing_boom_fri, wing_boom_fro]
rotor_prefixes = ['rlo', 'rli', 'rri', 'rro', 'flo', 'fli', 'fri', 'fro']

for i in range(len(boom_points)):
    boom_connection = geometry.evaluate(boom_points[i]) - geometry.evaluate(boom_points_on_wing[i])

    parameterization_design_parameters.add_variable(computed_value=boom_connection, desired_value=boom_connection.value)
    
    component_rotor_edges = rotor_edges[i]
    radius_computed = csdl.norm(geometry.evaluate(component_rotor_edges[0]) - geometry.evaluate(component_rotor_edges[1]))/2
    parameterization_design_parameters.add_variable(computed_value=radius_computed, desired_value=dv_radius_list[i])

# endregion lift + pusher rotor parameterization inputs

# endregion Define Design Parameters

# geometry.plot()
parameterization_solver.evaluate(parameterization_design_parameters)
# geometry.plot()

# endregion


recorder.stop()
