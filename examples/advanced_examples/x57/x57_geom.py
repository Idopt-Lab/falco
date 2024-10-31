from abc import abstractmethod
from symtable import Class

import lsdo_function_spaces as lfs
from lsdo_geo.core.geometry.temp_test import axis_origin

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

from flight_simulator import ureg, Q_, REPO_ROOT_FOLDER
import csdl_alpha as csdl
import numpy as np

from flight_simulator.utils.axis import Axis
from flight_simulator.utils.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.utils.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.import_geometry import import_geometry

plot_flag = True
in2m = 0.0254

# Every CSDl code starts with a recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# region Base Geometry
# Create all parametric functions on this base geometry
geometry = import_geometry(
    file_name="x57.stp",
    file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'x57',
    refit=False,
    scale=in2m,
    rotate_to_body_fixed_frame=True
)
if plot_flag:
    geometry.plot()

# region Define geometry components

# Cruise Motor
cruise_motor_hub = geometry.declare_component(function_search_names=['CruiseNacelle-Spinner'])
if plot_flag:
    cruise_motor_hub.plot()
# Wing
wing = geometry.declare_component(function_search_names=['CleanWing'])
if plot_flag:
    wing.plot()
# endregion

# region Wing Info
wing_root_le_guess = np.array([-120., 0,   -87.649])*in2m
wing_root_le_parametric = wing.project(wing_root_le_guess, plot=plot_flag)
wing_root_le = geometry.evaluate(wing_root_le_parametric)

wing_root_te_guess = np.array([-180., 0,   -87.649])*in2m
wing_root_te_parametric = wing.project(wing_root_te_guess, plot=plot_flag)
wing_root_te = geometry.evaluate(wing_root_te_parametric)

wing_tip_left_le_guess = np.array([-144.87186743367556, -200,   -87.649])*in2m
wing_tip_left_le_parametric = wing.project(wing_tip_left_le_guess, plot=plot_flag)

wing_tip_right_le_guess = np.array([-144.87186743367556, 200,   -87.649])*in2m
wing_tip_right_le_parametric = wing.project(wing_tip_right_le_guess, plot=plot_flag)
# endregion

# region Cruise Motor Hub Info
cruise_motor_hub_tip_guess = np.array([-120., -189.741,   -87.649])*in2m
cruise_motor_hub_tip_parametric = cruise_motor_hub.project(cruise_motor_hub_tip_guess, plot=plot_flag)
cruise_motor_hub_tip = geometry.evaluate(cruise_motor_hub_tip_parametric)
print(cruise_motor_hub_tip.value)

cruise_motor_hub_base_guess = cruise_motor_hub_tip.value + np.array([-20., 0., 0.])*in2m
cruise_motor_hub_base_parametric = cruise_motor_hub.project(cruise_motor_hub_base_guess, plot=plot_flag)
cruise_motor_hub_base = geometry.evaluate(cruise_motor_hub_base_parametric)
print(cruise_motor_hub_base.value)
# endregion

# endregion


# region Use Geometry to define quantities I need

# region OpenVSP Axis
openvsp_axis = Axis(
    name='OpenVSP Axis',
    translation=np.array([0, 0, 0]) * ureg.meter,
)
# endregion

# region Inertial Axis
# I am picking the inertial axis location as the OpenVSP (0,0,0)
inertial_axis = Axis(
    name='Inertial Axis',
    translation=np.array([0, 0, 0]) * ureg.meter,
    reference=openvsp_axis,
)
# endregion

# region Aircraft FD Axis
fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    translation=np.array([0, 0, 5000]) * ureg.ft,
    phi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ])),
    theta=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(4.), ])),
    psi=csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ])),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin='inertial'
)
print('Flight Dynamics angles (deg)', np.rad2deg(fd_axis.euler_angles.value))
# endregion

cruise_motor_hub_rotation = csdl.Variable(value=np.deg2rad(15))
cruise_motor_hub.rotate(cruise_motor_hub_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation)
cruise_motor_hub.plot()
geometry.plot()

# Reference point -> Body frame
# Center of gravity
#
# Intertial/Earth frame
# Wind frame
# Stability frame

# region Cruise Motor
cruise_motor_axis = AxisLsdoGeo(
    name='Cruise Motor Axis',
    geometry=cruise_motor_hub,
    parametric_coords=cruise_motor_hub_base_parametric,
    sequence=np.array([3, 2, 1]),
    reference=fd_axis,
    origin='ref'
)
print('Cruise motor axis translation: ', cruise_motor_axis.translation.value)
# endregion

# region Aerodynamic axis
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
# endregion

wingspan = csdl.norm(
    geometry.evaluate(wing_tip_left_le_parametric) - geometry.evaluate(wing_tip_right_le_parametric)
)
print("Wingspan: ", wingspan.value)

# endregion


cruise_motor_hub_rotation = csdl.Variable(value=np.deg2rad(15))
cruise_motor_hub.rotate(cruise_motor_hub_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation)
if plot_flag:
    cruise_motor_hub.plot()

print(cruise_motor_axis.translation.value)

# # region Geometry Parametrization
#
# # region Create wing parametrization objects
# num_ffd_sections = 3
# num_wing_sections = 2
# wing_ffd_block = construct_tight_fit_ffd_block(
#     entities=wing,
#     num_coefficients=(2, (num_ffd_sections // num_wing_sections + 1), 2),
#     degree=(1,1,1)
# )
# if plot_flag:
#     wing_ffd_block.plot()
#
# ffd_sectional_parameterization = VolumeSectionalParameterization(
#     name="ffd_sectional_parameterization",
#     parameterized_points=wing_ffd_block.coefficients,
#     principal_parametric_dimension=1,
# )
# if plot_flag:
#     ffd_sectional_parameterization.plot()
#
# space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
# wingspan_stretching_b_spline = lfs.Function(
#     space=space_of_linear_2_dof_b_splines,
#     coefficients=csdl.Variable(shape=(2,), value=np.array([-4., 4.])),
#     name='wingspan_stretching_b_spline_coefficients'
# )
# # endregion
#
# # region Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver
# parametric_b_spline_inputs = np.linspace(0.0, 1.0, 3).reshape((-1, 1))
# wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
#     parametric_b_spline_inputs
# )
#
# sectional_parameters = VolumeSectionalParameterizationInputs()
# sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
#
# ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=plot_flag)
#
# wing_coefficients = wing_ffd_block.evaluate(ffd_coefficients, plot=plot_flag)
# wing.set_coefficients(wing_coefficients)
# # endregion
#
# # region Create Newton solver for inner optimization
# wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([20.0]))
# geometry_solver = lsdo_geo.ParameterizationSolver()
# geometry_solver.add_parameter(wingspan_stretching_b_spline.coefficients)
#
# geometric_variables = lsdo_geo.GeometricVariables()
# geometric_variables.add_variable(computed_value=wingspan, desired_value=wingspan_outer_dv)
#
# geometry_solver.evaluate(geometric_variables)
# print("Wingspan: ", wingspan.value)
# geometry.plot()
# # endregion
#
# # endregion

exit()










for surface in wing.functions.values():
    surface.coefficients = surface.coefficients.set(csdl.slice[:,:,1], surface.coefficients[:,:,1]*2)
geometry.plot()


thrust_axis = geometry.evaluate(cruise_motor_hub_tip_parametric) - geometry.evaluate(cruise_motor_hub_base_parametric)
print(thrust_axis.value)










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





