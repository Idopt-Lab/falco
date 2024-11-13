from dataclasses import dataclass

from flight_simulator import ureg, REPO_ROOT_FOLDER
import csdl_alpha as csdl
import numpy as np
from typing import Union


from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator.core.dynamics.aircraft_states import AircaftStates
from flight_simulator.utils.euler_rotations import build_rotation_matrix

plot_flag = True
in2m = 0.0254

# Every CSDl code starts with a recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# region Base Geometry
# Create all parametric functions on this base aircraft
geometry = import_geometry(
    file_name="x57.stp",
    file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'x57',
    refit=False,
    scale=in2m,
    rotate_to_body_fixed_frame=True
)
if plot_flag:
    geometry.plot()

# region Define aircraft components

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
print('From aircraft, cruise motor hub tip (m): ', cruise_motor_hub_tip.value)

cruise_motor_hub_base_guess = cruise_motor_hub_tip.value + np.array([-20., 0., 0.])*in2m
cruise_motor_hub_base_parametric = cruise_motor_hub.project(cruise_motor_hub_base_guess, plot=plot_flag)
cruise_motor_hub_base = geometry.evaluate(cruise_motor_hub_base_parametric)
print('From aircraft, cruise motor hub base (m): ', cruise_motor_hub_base.value)
# endregion

# endregion


# region Use Geometry to define quantities I need

# region OpenVSP Axis
openvsp_axis = Axis(
    name='OpenVSP Axis',
    translation=np.array([0, 0, 0]) * ureg.meter,
    origin=ValidOrigins.OpenVSP.value
)
# endregion

# region Cruise Motor
@dataclass
class CruiseMotorRotation(csdl.VariableGroup):
    cant : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree
    pitch : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(15), name='CruiseMotorPitchAngle')
    yaw : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree

cruise_motor_hub_rotation = CruiseMotorRotation()
cruise_motor_hub.rotate(cruise_motor_hub_base, np.array([0., 1., 0.]), angles=cruise_motor_hub_rotation.pitch)
if plot_flag:
    cruise_motor_hub.plot()
    geometry.plot()

cruise_motor_axis = AxisLsdoGeo(
    name='Cruise Motor Axis',
    geometry=cruise_motor_hub,
    parametric_coords=cruise_motor_hub_base_parametric,
    sequence=np.array([3, 2, 1]),
    phi=cruise_motor_hub_rotation.cant,
    theta=cruise_motor_hub_rotation.pitch,
    psi=cruise_motor_hub_rotation.yaw,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Cruise motor axis translation (m): ', cruise_motor_axis.translation.value)
print('Cruise motor axis rotation (deg): ', np.rad2deg(cruise_motor_axis.euler_angles.value))
# endregion

# region Wing

wing_incidence = csdl.Variable(shape=(1, ), value=np.deg2rad(1.5), name='Wing incidence')
wing.rotate(wing_root_le, np.array([0., 1., 0.]), angles=wing_incidence)
if plot_flag:
    wing.plot()
    geometry.plot()

wing_axis = AxisLsdoGeo(
    name='Wing Axis',
    geometry=wing,
    parametric_coords=wing_root_le_parametric,
    sequence=np.array([3, 2, 1]),
    phi=np.array([0, ]) * ureg.degree,
    theta=wing_incidence,
    psi=np.array([0, ]) * ureg.degree,
    reference=openvsp_axis,
    origin=ValidOrigins.OpenVSP.value
)
print('Wing axis translation (m): ', wing_axis.translation.value)
print('Wing axis rotation (deg): ', np.rad2deg(wing_axis.euler_angles.value))
# endregion

# endregion

# region FD axis that do not depend on aircraft

# region Inertial Axis
# I am picking the inertial axis location as the OpenVSP (0,0,0)
inertial_axis = Axis(
    name='Inertial Axis',
    translation=np.array([0, 0, 0]) * ureg.meter,
    origin=ValidOrigins.Inertial.value
)
# endregion

# region Aircraft FD Axis
ac_states = AircaftStates()
ac_states.phi = csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='phi')
ac_states.theta = csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(4.), ]), name='theta')
ac_states.psi = csdl.Variable(shape=(1, ), value=np.array([np.deg2rad(0.), ]), name='psi')

fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    translation=np.array([0, 0, 5000]) * ureg.ft,
    phi=ac_states.phi,
    theta=ac_states.theta,
    psi=ac_states.psi,
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
print('Body-fixed angles (deg)', np.rad2deg(fd_axis.euler_angles.value))
# endregion

# region Aircraft Wind Axis
@dataclass
class WindAxisRotations(csdl.VariableGroup):
    mu : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree # bank
    gamma : Union[csdl.Variable, np.ndarray, ureg.Quantity] = csdl.Variable(value=np.deg2rad(2), name='Flight path angle')
    xi : Union[csdl.Variable, ureg.Quantity] = np.array([0, ]) * ureg.degree  # Heading
wind_axis_rotations = WindAxisRotations()

wind_axis = Axis(
    name='Wind Axis',
    translation=np.array([0, 0, 0]) * ureg.ft,
    phi=wind_axis_rotations.mu,
    theta=wind_axis_rotations.gamma,
    psi=wind_axis_rotations.xi,
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
print('Wind axis angles (deg)', np.rad2deg(wind_axis.euler_angles.value))
# endregion

# endregion

# region Forces and Moments

# region Aero forces

velocity_vector_in_wind = Vector(vector=csdl.Variable(shape=(3,), value=np.array([-1, 0, 0]), name='wind_vector'), axis=wind_axis)
print('Unit wind vector in wind axis: ', velocity_vector_in_wind.vector.value)

R_wind_to_inertial = build_rotation_matrix(wind_axis.euler_angles, np.array([3, 2, 1]))
wind_vector_in_inertial =  Vector(csdl.matvec(R_wind_to_inertial, velocity_vector_in_wind.vector), axis=inertial_axis)
print('Unit wind vector in inertial axis: ', wind_vector_in_inertial.vector.value)

R_body_to_inertial = build_rotation_matrix(fd_axis.euler_angles, np.array([3, 2, 1]))
wind_vector_in_body =  Vector(csdl.matvec(csdl.transpose(R_body_to_inertial), wind_vector_in_inertial.vector), axis=fd_axis)
print('Unit wind vector in body axis: ', wind_vector_in_body.vector.value)

R_wing_to_openvsp = build_rotation_matrix(wing_axis.euler_angles, np.array([3, 2, 1]))
wind_vector_in_wing =  Vector(csdl.matvec(csdl.transpose(R_wing_to_openvsp), wind_vector_in_body.vector), axis=wing_axis)
print('Unit wind vector in wing axis: ', wind_vector_in_wing.vector.value)
alpha = csdl.arctan(wind_vector_in_wing.vector[2]/wind_vector_in_wing.vector.value[0])
print('Effective angle of attack (deg): ', np.rad2deg(alpha.value))

CL = 2*np.pi*alpha
CD = 0.001 + 1/(np.pi*0.87*12) * CL**2
rho = 1.225
S = 50
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
# endregion

exit()

# region Rotor forces
cruise_motor_force = Vector(vector=np.array([0, 400, 0])*ureg.lbf, axis=cruise_motor_axis)
cruise_motor_moment = Vector(vector=np.array([0, 0, 0])*ureg.lbf*ureg.inch, axis=cruise_motor_axis)
cruise_motor_loads = ForcesMoments(force=cruise_motor_force, moment=cruise_motor_moment)


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
# aircraft.plot()
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





