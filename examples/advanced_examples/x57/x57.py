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


recorder = csdl.Recorder(inline=True)
recorder.start()

geometry = lsdo_geo.import_geometry(
    "x57.stp",
    parallelize=False,
)
geometry.plot()


# Define geometry components
cruise_motor = geometry.declare_component(function_search_names=['CruiseNacelle-Spinner'])
# cruise_motor.plot()

wing = geometry.declare_component(function_search_names=['CleanWing'])
# wing.plot()

cruise_motor_origin = np.array([150., -190., 100.])
cruise_motor_origin_parametric = cruise_motor.project(cruise_motor_origin, plot=True)
cruise_motor_front = np.array([140., -190., 100.])
cruise_motor_front_parametric = cruise_motor.project(cruise_motor_front, plot=True)



cruise_motor_rotation_1 = csdl.Variable(value=np.deg2rad(15))
cruise_motor.rotate(np.array([160., -190., 100.]), np.array([0., 1., 0.]), angles=cruise_motor_rotation_1)
cruise_motor.plot()
thrust_origin = geometry.evaluate(cruise_motor_origin_parametric)
thrust_axis = geometry.evaluate(cruise_motor_front_parametric) - geometry.evaluate(cruise_motor_origin_parametric)

geometry_parameterization_solver = lsdo_geo.ParameterizationSolver()
wing_rotation = csdl.Variable(value=np.deg2rad(15))
wing.rotate(axis_origin=np.array([160., 0., 100.]),axis_vector=np.array([0., 1., 0.]), angles=wing_rotation)
geometry_parameterization_solver.add_parameter(cruise_motor_rotation_1)
geometry_parameterization_solver.add_parameter(wing_rotation)

cruise_motor_mount_on_wing = wing.project(cruise_motor_origin)
cruise_motor_mount_on_wing_back = wing.project(cruise_motor_origin + np.array([20, 0., 0.]))
wing_axis = geometry.evaluate(cruise_motor_mount_on_wing) - geometry.evaluate(cruise_motor_mount_on_wing_back)
alignment_constraint = csdl.vdot(thrust_axis, wing_axis) - 1

wing_rotor_displacement = geometry.evaluate(cruise_motor_origin_parametric) - geometry.evaluate(cruise_motor_mount_on_wing)

geometry_parameterization_variables = lsdo_geo.GeometricVariables()
# geometry_parameterization_variables.add_variable(computed_value=wing_rotor_displacement, desired_value=wing_rotor_displacement.value)
geometry_parameterization_variables.add_variable(computed_value=alignment_constraint, desired_value=0.)

# geometry_parameterization_solver.evaluate(geometry_parameterization_variables)

# geometry.plot()

for surface in wing.functions.values():
    surface.coefficients = surface.coefficients.set(csdl.slice[:,:,1], surface.coefficients[:,:,1]*2)
geometry.plot()



wing_and_motor = geometry.declare_component(function_search_names=['CruiseNacelle-Spinner', 'CleanWing'])

pass

