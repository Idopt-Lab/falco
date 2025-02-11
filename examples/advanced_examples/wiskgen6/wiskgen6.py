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
from flight_simulator.core.vehicle.component import Component, Configuration
import lsdo_geo as lg
import vedo

plot_flag=False
# Exported stl from OpenVSP as feet instead of meters or inches so converting to meters
ft2m=0.3048

recorder=csdl.Recorder(inline=True)
recorder.start()


def define_base_geometry():
    geometry=import_geometry(
        "Wisk_V6.stp",
        file_path= REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'wiskgen6',
        refit=False,
        scale=ft2m,
        rotate_to_body_fixed_frame=True
    )
    if plot_flag:
        geometry.plot()
    # Define Aircraft Components
    ## Wing
    wing = geometry.declare_component(function_search_names=['Wing'], name='wing')
    ## Tail(s)
    h_tail = geometry.declare_component(function_search_names=['HTail'], name='h_tail')
    v_tail = geometry.declare_component(function_search_names=['VTail'], name='v_tail')
    ## Fuselage
    fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
    # Landing Gear
    fwd_landing_gear_pylon = geometry.declare_component(function_search_names=['FWD_LG'], name='fwd_landing_gear_pylon')
    aft_landing_gear_pylon = geometry.declare_component(function_search_names=['AFT_LG'], name='aft_landing_gear_pylon')
    base_landing_gear = geometry.declare_component(function_search_names=['LG_BASE'], name='base_landing_gear')
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
    
    return geometry, wing, h_tail, v_tail, fuselage, fwd_landing_gear_pylon, aft_landing_gear_pylon, base_landing_gear, \
           pylon_ob_left, pylon_mid_left, pylon_ib_left, pylon_ob_right, pylon_mid_right, pylon_ib_right, \
           rotor_hub_ob_left_fwd, rotor_hub_mid_left_fwd, rotor_hub_ib_left_fwd, rotor_hub_ob_right_fwd, rotor_hub_mid_right_fwd, rotor_hub_ib_right_fwd, \
           rotor_hub_ob_left_aft, rotor_hub_mid_left_aft, rotor_hub_ib_left_aft, rotor_hub_ob_right_aft, rotor_hub_mid_right_aft, rotor_hub_ib_right_aft

geometry, wing, h_tail, v_tail, fuselage, fwd_landing_gear_pylon, aft_landing_gear_pylon, base_landing_gear, \
pylon_ob_left, pylon_mid_left, pylon_ib_left, pylon_ob_right, pylon_mid_right, pylon_ib_right, \
rotor_hub_ob_left_fwd, rotor_hub_mid_left_fwd, rotor_hub_ib_left_fwd, rotor_hub_ob_right_fwd, rotor_hub_mid_right_fwd, rotor_hub_ib_right_fwd, \
rotor_hub_ob_left_aft, rotor_hub_mid_left_aft, rotor_hub_ib_left_aft, rotor_hub_ob_right_aft, rotor_hub_mid_right_aft, rotor_hub_ib_right_aft = define_base_geometry()



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





# # Region Axis Creation
def define_axes():
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

    vtail_deflection = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='Vtail Deflection')
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

    htail_deflection = csdl.Variable(shape=(1, ), value=np.deg2rad(0), name='Htail Deflection')
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
    return [
        openvsp_axis, 
        wing_axis, vtail_axis, htail_axis,
        pt_axis_ob_left_fwd, pt_axis_mid_left_fwd, pt_axis_ib_left_fwd,
        pt_axis_ob_right_fwd, pt_axis_mid_right_fwd, pt_axis_ib_right_fwd,
        pt_axis_ob_left_aft, pt_axis_mid_left_aft, pt_axis_ib_left_aft,
        pt_axis_ob_right_aft, pt_axis_mid_right_aft, pt_axis_ib_right_aft,
        inertial_axis, fd_axis, wind_axis
    ]

openvsp_axis, wing_axis, vtail_axis, htail_axis, \
pt_axis_ob_left_fwd, pt_axis_mid_left_fwd, pt_axis_ib_left_fwd, \
pt_axis_ob_right_fwd, pt_axis_mid_right_fwd, pt_axis_ib_right_fwd, \
pt_axis_ob_left_aft, pt_axis_mid_left_aft, pt_axis_ib_left_aft, \
pt_axis_ob_right_aft, pt_axis_mid_right_aft, pt_axis_ib_right_aft, \
inertial_axis, fd_axis, wind_axis = define_axes()


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


# thrust_axis = geometry.evaluate(pt_axis_ob_left_fwd)










## FFD



# # region Parameterization

constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# # region Parameterization Setup
# parameterization_solver = lg.ParameterizationSolver()
# parameterization_design_parameters = lg.GeometricVariables()

# # region Wing Parameterization setup
wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=[wing], num_coefficients=(2,11,2), degree=(1,3,1))

wing_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='wing_sectional_parameterization',
                                                                            parameterized_points=wing_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)


wing_chord_stretch_coefficients = csdl.Variable(name='wing_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
wing_chord_stretch_b_spline = lfs.Function(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=wing_chord_stretch_coefficients)

wing_wingspan_stretch_amount = csdl.Variable(name='wing_span_stretch_amount', value=np.array([0.]))
wing_wingspan_stretch_coefficients = csdl.Variable(name='wing_wingspan_stretch_coefficients', value=np.array([-wing_wingspan_stretch_amount.value/2, wing_wingspan_stretch_amount.value/2]))
wing_wingspan_stretch_b_spline = lfs.Function(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=wing_wingspan_stretch_coefficients)

wing_twist_coefficients = csdl.Variable(name='wing_twist_coefficients', value=np.array([0,0,0,0,0])*np.pi/180)
wing_twist_b_spline = lfs.Function(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                          coefficients=wing_twist_coefficients)

wing_sweep_amount = csdl.Variable(name='wing_sweep_amount', value=np.array([0.]))
wing_sweep_coefficients = csdl.Variable(name='wing_sweep_coefficients', value=np.array([-wing_sweep_amount.value[0]*np.pi/180, wing_sweep_amount.value[0]*np.pi/180, -wing_sweep_amount.value[0]*np.pi/180]))
wing_sweep_b_spline = lfs.Function(name='wing_sweep_b_spline',space=linear_b_spline_curve_3_dof_space, coefficients=wing_sweep_coefficients)

wing_translation_x_coefficients = csdl.Variable(name='wing_translation_x_coefficients', value=np.array([0.]))
wing_translation_x_b_spline = lfs.Function(name='wing_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=wing_translation_x_coefficients)

wing_translation_z_coefficients = csdl.Variable(name='wing_translation_z_coefficients', value=np.array([0.]))
wing_translation_z_b_spline = lfs.Function(name='wing_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=wing_translation_z_coefficients)


wing_section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(wing_section_parametric_coordinates)
sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(wing_section_parametric_coordinates)
sectional_wing_twist = wing_twist_b_spline.evaluate(wing_section_parametric_coordinates)
sectional_wing_sweep = wing_sweep_b_spline.evaluate(wing_section_parametric_coordinates)
sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(wing_section_parametric_coordinates)
sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(wing_section_parametric_coordinates)

wing_sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_wing_chord_stretch},
    translations={1: sectional_wing_wingspan_stretch, 0: sectional_wing_translation_x, 2: sectional_wing_translation_z, 0: sectional_wing_sweep},
    rotations={1: sectional_wing_twist}
    )

# sectional_parameters = lg.VolumeSectionalParameterizationInputs()
# sectional_parameters.add_sectional_stretch(axis=0, stretch=sectional_wing_chord_stretch)
# sectional_parameters.add_sectional_translation(axis=1, translation=sectional_wing_wingspan_stretch)
# sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
# sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)






# # region Horizontal Stabilizer setup
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


htail_section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(htail_section_parametric_coordinates)
sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(htail_section_parametric_coordinates)
sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(htail_section_parametric_coordinates)
sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(htail_section_parametric_coordinates)
sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(htail_section_parametric_coordinates)

htail_sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_h_tail_chord_stretch},
    translations={1: sectional_h_tail_span_stretch, 0: sectional_h_tail_translation_x, 2: sectional_h_tail_translation_z}
)





# region Vertical Stabilizer setup
v_tail_ffd_block = lg.construct_ffd_block_around_entities(name='v_tail_ffd_block', entities=v_tail, num_coefficients=(2,11,2), degree=(1,3,1))
v_tail_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='v_tail_sectional_parameterization',
                                                                            parameterized_points=v_tail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)


vtailchordstretch=np.array([-0., 0.])
v_tail_chord_stretch_coefficients = csdl.Variable(name='v_tail_chord_stretch_coefficients', value=vtailchordstretch)
v_tail_chord_stretch_b_spline = lfs.Function(name='v_tail_chord_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                            coefficients=v_tail_chord_stretch_coefficients)
# geometry.plot(color='red')
vtailspanstrech=np.array([0.])
v_tail_span_stretch_coefficients = csdl.Variable(name='v_tail_span_stretch_coefficients', value=vtailspanstrech)
v_tail_span_stretch_b_spline = lfs.Function(name='v_tail_span_stretch_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                          coefficients=v_tail_span_stretch_coefficients)
v_tail_translation_x_coefficients = csdl.Variable(name='v_tail_translation_x_coefficients', value=np.array([-vtailchordstretch[1]/4]))
v_tail_translation_x_b_spline = lfs.Function(name='v_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=v_tail_translation_x_coefficients)
v_tail_translation_z_coefficients = csdl.Variable(name='v_tail_translation_z_coefficients', value=np.array([-0.5*vtailspanstrech]))
v_tail_translation_z_b_spline = lfs.Function(name='v_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=v_tail_translation_z_coefficients)


vtail_section_parametric_coordinates = np.linspace(0., 1., v_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_v_tail_chord_stretch = v_tail_chord_stretch_b_spline.evaluate(vtail_section_parametric_coordinates)
sectional_v_tail_span_stretch = v_tail_span_stretch_b_spline.evaluate(vtail_section_parametric_coordinates)
sectional_v_tail_translation_x = v_tail_translation_x_b_spline.evaluate(vtail_section_parametric_coordinates)
sectional_v_tail_translation_z = v_tail_translation_z_b_spline.evaluate(vtail_section_parametric_coordinates)

vtail_sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_v_tail_chord_stretch,2: sectional_v_tail_span_stretch},
    translations={0: sectional_v_tail_translation_x, 2: sectional_v_tail_translation_z}
)




## region lift only rotor setup
# lift_rotors = [rotor_hub_ib_left_aft, rotor_hub_mid_left_aft, rotor_hub_ob_left_aft, rotor_hub_ib_right_aft, rotor_hub_mid_right_aft, rotor_hub_ob_right_aft]
# lift_rotor_ffd_blocks = []
# lift_rotor_sectional_parameterizations = []
# lift_rotor_parameterization_b_splines = []
# for i, comp_set in enumerate(lift_rotors):
#     lift_rotor_ffd_block = lg.construct_ffd_block_around_entities(name=f'{comp_set[0].name[:11]}_lift_rotor_ffd_block', entities=comp_set, num_coefficients=(2,2,2), degree=(1,1,1))
#     lift_rotor_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name=f'{comp_set[0].name[:11]}_lift_rotor_sectional_parameterization',
#                                                                                 parameterized_points=lift_rotor_ffd_block.coefficients,
#                                                                                 principal_parametric_dimension=2)
    
#     lift_rotor_stretch_coefficient = csdl.Variable(name=f'{comp_set[0].name[:11]}_lift_rotor_stretch_coefficient', shape=(2,), value=0)
#     lift_rotor_ffd_blocks.append(rotor_ffd_block)
#     lift_rotor_sectional_parameterizations.append(rotor_ffd_block_sectional_parameterization)
#     lift_rotor_parameterization_b_splines.append(lift_rotor_sectional_stretch_b_spline) 

# region Lift Rotors Parameterization Evaluation for Parameterization Solver

pylons = [pylon_ob_left, pylon_mid_left, pylon_ib_left, pylon_ob_right, pylon_mid_right, pylon_ib_right]
# pylonGroups = [Pylon_Outboard_Left, Pylon_Middle_Left, Pylon_Inboard_Left, Pylon_Outboard_Right, Pylon_Middle_Right, Pylon_Inboard_Right]

pylon_ffd_blocks = []
pylon_sectional_parameterizations = []
pylon_parameterization_b_splines = []
for i, comp in enumerate(pylons):
    pylon_ffd_block = lg.construct_ffd_block_around_entities(name=f'{comp.name[:8]}_pylon_ffd_block', entities=comp, num_coefficients=(2,2,2), degree=(1,1,1))
    pylon_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name=f'{comp.name[:8]}_pylon_sectional_parameterization',
                                                                                parameterized_points=pylon_ffd_block.coefficients,
                                                                                principal_parametric_dimension=2)
    
    pylon_stretch_coefficient = csdl.Variable(name=f'{comp.name[:8]}_pylon_stretch_coefficient', shape=(2,), value=0)
    pylon_sectional_stretch_b_spline = lfs.Function(name=f'{comp.name[:8]}_pylon_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                    coefficients=pylon_stretch_coefficient)
    pylon_ffd_blocks.append(pylon_ffd_block)
    pylon_sectional_parameterizations.append(pylon_ffd_block_sectional_parameterization)
    pylon_parameterization_b_splines.append(pylon_sectional_stretch_b_spline)

    pylon_section_parametric_coordinates = np.linspace(0., 1., pylon_sectional_parameterizations[i].num_sections).reshape((-1, 1))
    sectional_pylon_stretch = pylon_parameterization_b_splines[i].evaluate(pylon_section_parametric_coordinates)

    pylon_sectional_parameters = lg.VolumeSectionalParameterizationInputs(
        stretches={0: sectional_pylon_stretch}
    )



# for i, component_set in enumerate(lift_rotor_related_components):
#     rotor_ffd_block = lift_rotor_ffd_blocks[i]
#     rotor_ffd_block_sectional_parameterization = lift_rotor_sectional_parameterizations[i]
#     rotor_stretch_b_spline = lift_rotor_parameterization_b_splines[i]

#     section_parametric_coordinates = np.linspace(0., 1., rotor_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     sectional_stretch = rotor_stretch_b_spline.evaluate(section_parametric_coordinates)

#     sectional_parameters = lsdo_geo.VolumeSectionalParameterizationInputs(
#         stretches={0: sectional_stretch, 1:sectional_stretch}
#     )

#     rotor_ffd_block_coefficients = rotor_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
#     rotor_coefficients = rotor_ffd_block.evaluate(rotor_ffd_block_coefficients, plot=False)
#     for i, component in enumerate(component_set):
#         component.set_coefficients(rotor_coefficients[i])
    

#     # Add rigid body translation
#     rigid_body_translation = csdl.Variable(shape=(3,), value=0., name=f'{component_set[0].name[:11]}_rotor_rigid_body_translation')

#     for component in component_set:
#         for function in component.functions.values():
#             function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

#     for function in pylon.functions.values():
#         function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')





# # region Fuselage setup


fuselage_ffd_block = lg.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=fuselage, num_coefficients=(2,2,2), degree=(1,1,1))
fuselage_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='fuselage_sectional_parameterization',
                                                                            parameterized_points=fuselage_ffd_block.coefficients,
                                                                            principal_parametric_dimension=0)

fuselage_stretch_coefficients = csdl.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
fuselage_stretch_b_spline = lfs.Function(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=fuselage_stretch_coefficients)

section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    translations={0: sectional_fuselage_stretch}
)



# fuselage_tailing_point=fuselage.project(fuselage_ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])))
# fuselage_leading_point=fuselage.project(fuselage_ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])))




# This keeps the pylons in place (not desired)
# for i, PylonComp in enumerate(pylonGroups):
#     config.connect_component_geometries(comp_1=Wing,comp_2=PylonComp,comp_1_ffd_block=wing_ffd_block,comp_2_ffd_block=pylon_ffd_blocks[i])
 

wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(wing_sectional_parameters, plot=False)
wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
wing.set_coefficients(wing_coefficients)
h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(htail_sectional_parameters, plot=False)
h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)
h_tail.set_coefficients(coefficients=h_tail_coefficients)
v_tail_ffd_block_coefficients = v_tail_ffd_block_sectional_parameterization.evaluate(vtail_sectional_parameters, plot=False)
v_tail_coefficients = v_tail_ffd_block.evaluate(v_tail_ffd_block_coefficients, plot=False)
v_tail.set_coefficients(coefficients=v_tail_coefficients)
for i, comp in enumerate(pylons):
    pylon_ffd_block_coefficients = pylon_sectional_parameterizations[i].evaluate(pylon_sectional_parameters, plot=False)
    pylon_coefficients = pylon_ffd_blocks[i].evaluate(pylon_ffd_block_coefficients, plot=False)
    pylons[i].set_coefficients(pylon_coefficients)
# fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
# fuselage_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
# fuselage.set_coefficients(coefficients=fuselage_coefficients)






def define_heirarchy():
    Aircraft = Component(name='Aircraft', geometry=geometry)
    config = Configuration(system=Aircraft)
    # Wing, Tails, Fuselage
    Wing = Component(name='Wing',geometry=wing)
    HorizTail = Component(name='Horizontal Tail',geometry=h_tail)
    VertTail = Component(name='Vertical Tail',geometry=v_tail)
    Fuselage = Component(name='Fuselage',geometry=fuselage)
    Aircraft.add_subcomponent(Wing)
    Aircraft.add_subcomponent(HorizTail)
    Aircraft.add_subcomponent(VertTail)
    Aircraft.add_subcomponent(Fuselage)
    # Landing Gears
    # Landing Gear Components
    Landing_Gear = Component(name='Landing Gear')
    Fwd_Landing_Gear_Pylon = Component(name='Forward Landing Gear Pylon', geometry=fwd_landing_gear_pylon)
    Aft_Landing_Gear_Pylon = Component(name='Aft Landing Gear Pylon', geometry=aft_landing_gear_pylon)
    Base_Landing_Gear = Component(name='Base Landing Gear', geometry=base_landing_gear)
    Landing_Gear.add_subcomponent(Fwd_Landing_Gear_Pylon)
    Landing_Gear.add_subcomponent(Aft_Landing_Gear_Pylon)
    Landing_Gear.add_subcomponent(Base_Landing_Gear)
    Aircraft.add_subcomponent(Landing_Gear)
    # Propulsion and Pylons
    Propulsion = Component(name='Propulsion')
    LPC_Propulsion = Component(name='Lift+Cruise Propulsion (FWD)')
    Propulsion.add_subcomponent(LPC_Propulsion)
    Motor_ob_left_fwd = Component(name='Motor Outboard Left FWD', geometry=rotor_hub_ob_left_fwd)
    Motor_mid_left_fwd = Component(name='Motor Middle Left FWD', geometry=rotor_hub_mid_left_fwd)
    Motor_ib_left_fwd = Component(name='Motor Inboard Left FWD', geometry=rotor_hub_ib_left_fwd)
    Motor_ob_right_fwd = Component(name='Motor Outboard Right FWD', geometry=rotor_hub_ob_right_fwd)
    Motor_mid_right_fwd = Component(name='Motor Middle Right FWD', geometry=rotor_hub_mid_right_fwd)
    Motor_ib_right_fwd = Component(name='Motor Inboard Right FWD', geometry=rotor_hub_ib_right_fwd)
    LPC_Propulsion.add_subcomponent(Motor_ob_left_fwd)
    LPC_Propulsion.add_subcomponent(Motor_mid_left_fwd)
    LPC_Propulsion.add_subcomponent(Motor_ib_left_fwd)
    LPC_Propulsion.add_subcomponent(Motor_ob_right_fwd)
    LPC_Propulsion.add_subcomponent(Motor_mid_right_fwd)
    LPC_Propulsion.add_subcomponent(Motor_ib_right_fwd)
    LIFT_Propulsion = Component(name='Lift Only Propulsion (AFT)')
    Propulsion.add_subcomponent(LIFT_Propulsion)
    Motor_ob_left_aft = Component(name='Motor Outboard Left AFT', geometry=rotor_hub_ob_left_aft)
    Motor_mid_left_aft = Component(name='Motor Middle Left AFT', geometry=rotor_hub_mid_left_aft)
    Motor_ib_left_aft = Component(name='Motor Inboard Left AFT', geometry=rotor_hub_ib_left_aft)
    Motor_ob_right_aft = Component(name='Motor Outboard Right AFT', geometry=rotor_hub_ob_right_aft)
    Motor_mid_right_aft = Component(name='Motor Middle Right AFT', geometry=rotor_hub_mid_right_aft)
    Motor_ib_right_aft = Component(name='Motor Inboard Right AFT', geometry=rotor_hub_ib_right_aft)
    LIFT_Propulsion.add_subcomponent(Motor_ob_left_aft)
    LIFT_Propulsion.add_subcomponent(Motor_mid_left_aft)
    LIFT_Propulsion.add_subcomponent(Motor_ib_left_aft)
    LIFT_Propulsion.add_subcomponent(Motor_ob_right_aft)
    LIFT_Propulsion.add_subcomponent(Motor_mid_right_aft)
    LIFT_Propulsion.add_subcomponent(Motor_ib_right_aft)
    Supports = Component(name='Prop Pylons')
    Pylon_Outboard_Left = Component(name='Pylon Outboard Left', geometry=pylon_ob_left)
    Pylon_Outboard_Right = Component(name='Pylon Outboard Right', geometry=pylon_ob_right)
    Pylon_Middle_Left = Component(name='Pylon Middle Left', geometry=pylon_mid_left)
    Pylon_Middle_Right = Component(name='Pylon Middle Right', geometry=pylon_mid_right)
    Pylon_Inboard_Left = Component(name='Pylon Inboard Left', geometry=pylon_ib_left)
    Pylon_Inboard_Right = Component(name='Pylon Inboard Right', geometry=pylon_ib_right)
    Supports.add_subcomponent(Pylon_Outboard_Left)
    Supports.add_subcomponent(Pylon_Outboard_Right)
    Supports.add_subcomponent(Pylon_Middle_Left)
    Supports.add_subcomponent(Pylon_Middle_Right)
    Supports.add_subcomponent(Pylon_Inboard_Left)
    Supports.add_subcomponent(Pylon_Inboard_Right)
    Propulsion.add_subcomponent(Supports)
    Aircraft.add_subcomponent(Propulsion)


    config.connect_component_geometries(Fuselage,HorizTail,connection_point=htail_root_te_guess)
    config.connect_component_geometries(Fuselage,VertTail,connection_point=vtail_root_te_guess)
    # config.connect_component_geometries(Motor_ib_left_aft,Pylon_Inboard_Left,connection_point=rotor_hub_ib_left_aft.evaluate(pt_ib_left_aft_bot_parametric))
    # config.connect_component_geometries(Motor_mid_left_aft,Pylon_Middle_Left,connection_point=rotor_hub_mid_left_aft.evaluate(pt_mid_left_aft_bot_parametric))
    # config.connect_component_geometries(Motor_ob_left_aft,Pylon_Outboard_Left,connection_point=rotor_hub_ob_left_aft.evaluate(pt_ob_left_aft_bot_parametric))
    # config.connect_component_geometries(Motor_ib_right_aft,Pylon_Inboard_Right,connection_point=rotor_hub_ib_right_aft.evaluate(pt_ib_right_aft_bot_parametric))
    # config.connect_component_geometries(Motor_mid_right_aft,Pylon_Middle_Right,connection_point=rotor_hub_mid_right_aft.evaluate(pt_mid_right_aft_bot_parametric))
    # config.connect_component_geometries(Motor_ob_right_aft,Pylon_Outboard_Right,connection_point=rotor_hub_ob_right_aft.evaluate(pt_ob_right_aft_bot_parametric))
    # config.connect_component_geometries(Motor_ib_left_fwd,Pylon_Inboard_Left,connection_point=rotor_hub_ib_left_fwd.evaluate(pt_ib_left_fwd_bot_parametric))
    # config.connect_component_geometries(Motor_mid_left_fwd,Pylon_Middle_Left,connection_point=rotor_hub_mid_left_fwd.evaluate(pt_mid_left_fwd_bot_parametric))
    # config.connect_component_geometries(Motor_ob_left_fwd,Pylon_Outboard_Left,connection_point=rotor_hub_ob_left_fwd.evaluate(pt_ob_left_fwd_bot_parametric))
    # config.connect_component_geometries(Motor_ib_right_fwd,Pylon_Inboard_Right,connection_point=rotor_hub_ib_right_fwd.evaluate(pt_ib_right_fwd_bot_parametric))
    # config.connect_component_geometries(Motor_mid_right_fwd,Pylon_Middle_Right,connection_point=rotor_hub_mid_right_fwd.evaluate(pt_mid_right_fwd_bot_parametric))
    # config.connect_component_geometries(Motor_ob_right_fwd,Pylon_Outboard_Right,connection_point=rotor_hub_ob_right_fwd.evaluate(pt_ob_right_fwd_bot_parametric))

    return Aircraft, config

Aircraft, config = define_heirarchy()

# print(rotor_hub_ib_left_aft.evaluate(pt_ib_left_aft_bot_parametric))

config.setup_geometry(plot=True)


recorder.stop()
