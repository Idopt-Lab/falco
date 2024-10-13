import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,
    construct_ffd_block_around_entities,
    construct_ffd_block_from_corners
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
import lsdo_geo

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.utils.axis import Axis
from flight_simulator.utils.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.utils.forces_moments import Vector, ForcesMoments


plot_flag = True
ft2m = 0.3048

# Every CSDl code starts with a recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# region Base Geometry
# Create all parametric functions on this base geometry

# Import the .stp file of the X-57
geometry = lsdo_geo.import_geometry(
    "x57.stp",
    parallelize=False,
    scale=ft2m
)
geometry.plot()

# region Define geometry components

# Cruise Motor
cruise_motor_hub = geometry.declare_component(function_search_names=['CruiseNacelle-Spinner'])
cruise_motor_hub.plot()
# Wing
wing = geometry.declare_component(function_search_names=['CleanWing'])
wing.plot()
# endregion

# region Wing Info
wing_root_le_guess = np.array([120., 0,   87.649])*ft2m
wing_root_le_parametric = wing.project(wing_root_le_guess, plot=plot_flag)
wing_root_le = geometry.evaluate(wing_root_le_parametric)

wing_root_te_guess = np.array([180., 0,   87.649])*ft2m
wing_root_te_parametric = wing.project(wing_root_te_guess, plot=plot_flag)
wing_root_te = geometry.evaluate(wing_root_te_parametric)
# endregion

# region Cruise Motor Hub Info
cruise_motor_hub_tip_guess = np.array([120., -189.741,   87.649])*ft2m
cruise_motor_hub_tip_parametric = cruise_motor_hub.project(cruise_motor_hub_tip_guess, plot=plot_flag)
cruise_motor_hub_tip = geometry.evaluate(cruise_motor_hub_tip_parametric)
print(cruise_motor_hub_tip.value)

cruise_motor_hub_base_guess = cruise_motor_hub_tip.value + np.array([20., 0., 0.])*ft2m
cruise_motor_hub_base_parametric = cruise_motor_hub.project(cruise_motor_hub_base_guess, plot=plot_flag)
cruise_motor_hub_base = geometry.evaluate(cruise_motor_hub_base_parametric)
print(cruise_motor_hub_base.value)
# endregion

# endregion


# region Use Geometry to define quantities I need

# region Inertial Axis
# I am picking the inertial axis location as the OpenVSP (0,0,0)
inertial_axis = Axis(
    name='Inertial Axis',
    translation=np.array([0, 0, 0]) * ureg.meter,
    angles=np.array([0, 0, 0]) * ureg.degree,
    origin='inertial'
)
# endregion

# region Aircraft FD Axis
fd_euler_angles = csdl.Variable(value=np.array([np.deg2rad(0.), np.deg2rad(5.), np.deg2rad(0.)]))
fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    translation=np.array([0, 0, 5000]) * ureg.ft,
    angles=fd_euler_angles,
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin='inertial'
)
print('Flight Dynamics angles (deg)', np.rad2deg(fd_axis.angles.value))
# endregion

# region Cruise Motor
cruise_motor_axis = AxisLsdoGeo(
    name='Cruise Motor Axis',
    geometry=cruise_motor_hub,
    parametric_coords=cruise_motor_hub_base_parametric,
    angles=np.array([0, 0, 0]) * ureg.degree,
    sequence=np.array([3, 2, 1]),
    reference=fd_axis,
    origin='ref'
)
# endregion

# # region Aerodynamic axis
# # Wing aerodynamic axis is the wing quarter chord
# # todo: make this come directly from geometry rather than specifying values like this
# waa_x = wing_root_le.value[0] + (wing_root_te.value[0]-wing_root_le.value[0])/4
# waa_y = 0.
# waa_z = wing_root_le.value[2]
#
# wing_aerodynamic_axis = Axis(
#     name='Inertial Axis',
#     translation=np.array([waa_x, waa_y, waa_z]) * ureg.meter,
#     angles=np.array([0, 0, 0]) * ureg.degree,
#     reference=fd_axis,
#     origin='ref'
# )
# del waa_x, waa_y, waa_z
# # endregion

# endregion


print(cruise_motor_axis.translation.value)


cruise_motor_hub_rotation = csdl.Variable(value=np.deg2rad(15))
cruise_motor_hub.rotate(cruise_motor_hub_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation)
cruise_motor_hub.plot()

print(cruise_motor_axis.translation.value)

exit()







for surface in wing.functions.values():
    surface.coefficients = surface.coefficients.set(csdl.slice[:,:,1], surface.coefficients[:,:,1]*2)
geometry.plot()


thrust_axis = geometry.evaluate(cruise_motor_hub_tip_parametric) - geometry.evaluate(cruise_motor_hub_base_parametric)
print(thrust_axis.value)

cruise_motor_hub_rotation = csdl.Variable(value=np.deg2rad(15))
cruise_motor_hub.rotate(cruise_motor_hub_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation)
cruise_motor_hub.plot()








wing.rotate(cruise_motor_hub_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation)
cruise_motor_hub.plot()

print('Rotated Geometry')
cruise_motor_hub_tip = geometry.evaluate(cruise_motor_hub_tip_parametric)
cruise_motor_hub_base = geometry.evaluate(cruise_motor_hub_base_parametric)
thrust_axis = geometry.evaluate(cruise_motor_hub_tip_parametric) - geometry.evaluate(cruise_motor_hub_base_parametric)
print(cruise_motor_hub_tip.value)
print(cruise_motor_hub_base.value)
print(thrust_axis.value)







# When making a change to the Euler angles, we must use the axis setter to update the angle.
fd_axis.angles = fd_euler_angles.set(csdl.slice[0], np.deg2rad(10.))
print('Flight Dynamics angles (deg)', np.rad2deg(fd_axis.angles.value))

pass