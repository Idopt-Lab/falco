import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
import lsdo_geo as lg
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER
from flight_simulator.core.vehicle.component import Component, Configuration
from flight_simulator.core.vehicle.condition import Condition
from flight_simulator.core.loads.mass_properties import MassProperties
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from typing import Union, List
from dataclasses import dataclass
from flight_simulator import ureg
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.utils.euler_rotations import build_rotation_matrix

lfs.num_workers = 1

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


wing = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4'], name='wing')

# wing.plot()


wing_le_left_guess = np.array([-12.356, -16, -5.5])*ft2m
wing_le_left_parametric = wing.project(wing_le_left_guess, plot=False)

wing_le_right_guess = np.array([-12.356, 16, -5.5])*ft2m
wing_le_right_parametric = wing.project(wing_le_right_guess, plot=False)

wing_le_center_guess = np.array([-12.356, 0., -5.5])*ft2m
wing_le_center_parametric = wing.project(wing_le_center_guess, plot=False)

wing_te_left_guess = np.array([-14.25, -16, -5.5])*ft2m
wing_te_left_parametric = wing.project(wing_te_left_guess, plot=False)

wing_te_right_guess = np.array([-14.25, 16, -5.5])*ft2m
wing_te_right_parametric = wing.project(wing_te_right_guess, plot=False)

wing_te_center_guess = np.array([-14.25, 0., -5.5])*ft2m
wing_te_center_parametric = wing.project(wing_te_center_guess, plot=False)

wing_qc_center = wing.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -5.5])*ft2m, plot=False)
wing_qc_tip_right = wing.project(np.array([-12.356+(0.25*(-14.25+12.356)), 16., -5.5])*ft2m, plot=False)
wing_qc_tip_left = wing.project(np.array([-12.356+(0.25*(-14.25+12.356)), -16., -5.5])*ft2m, plot=False)


num_ffd_sections = 3
num_wing_secctions = 2
ffd_block = lg.construct_ffd_block_around_entities(entities=wing, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))
# ffd_block = lg.construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, (num_ffd_sections // num_wing_secctions + 1), 2), degree=(1,1,1))
# ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, 3, 2), degree=(1,1,1))
# ffd_block.plot()




from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)


ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)

space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0., 0., 0.])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([0., 0.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0.0, 0.0, 0.0])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([0, 0., 0])*np.pi/180), name='twist_b_spline_coefficients')



parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
    parametric_b_spline_inputs
)

twist_sectional_parameters = twist_b_spline.evaluate(
    parametric_b_spline_inputs
)


sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)    # TODO: Fix plot function
ffd_coefficients.name = 'ffd_coefficients'


geometry_coefficients = ffd_block.evaluate(ffd_coefficients, plot=False)
# print(geometry_coefficients)
wing.set_coefficients(geometry_coefficients)
# wing.plot()

wingspan = csdl.norm(
    wing.evaluate(wing_le_right_parametric) - wing.evaluate(wing_le_left_parametric)
)
root_chord = csdl.norm(
    wing.evaluate(wing_te_center_parametric) - wing.evaluate(wing_le_center_parametric)
)
tip_chord_left = csdl.norm(
    wing.evaluate(wing_te_left_parametric) - wing.evaluate(wing_le_left_parametric)
)
tip_chord_right = csdl.norm(
    wing.evaluate(wing_te_right_parametric) - wing.evaluate(wing_le_right_parametric)
)

spanwise_direction_left = wing.evaluate(wing_qc_tip_left) - wing.evaluate(wing_qc_center)
spanwise_direction_right = wing.evaluate(wing_qc_tip_right) - wing.evaluate(wing_qc_center)
# sweep_angle = csdl.arccos(csdl.vdot(spanwise_direction, np.array([0., -1., 0.])) / csdl.norm(spanwise_direction))
sweep_angle_left = csdl.arccos(-spanwise_direction_left[1] / csdl.norm(spanwise_direction_left))
sweep_angle_right = csdl.arccos(spanwise_direction_right[1] / csdl.norm(spanwise_direction_right))




print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)

# Create Newton solver for inner optimization
chord_stretching_b_spline.coefficients.add_name('chord_stretching_b_spline_coefficients')
wingspan_stretching_b_spline.coefficients.add_name('wingspan_stretching_b_spline_coefficients')
sweep_translation_b_spline.coefficients.add_name('sweep_translation_b_spline_coefficients')


wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([6.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([2.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([45*np.pi/180]))



from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
geometry_solver = ParameterizationSolver()
geometry_solver.add_parameter(chord_stretching_b_spline.coefficients)
geometry_solver.add_parameter(wingspan_stretching_b_spline.coefficients)
geometry_solver.add_parameter(sweep_translation_b_spline.coefficients)

geometric_variables = GeometricVariables()
geometric_variables.add_variable(wingspan, wingspan_outer_dv)
geometric_variables.add_variable(root_chord, root_chord_outer_dv)
geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv)
geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv)
geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv)
geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv)


print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)

wing.plot()
geometry_solver.evaluate(geometric_variables)
wing.plot()
