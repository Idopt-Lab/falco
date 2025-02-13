##  FFD Stuff

# Region Parameterization
constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# FFD Blocks
wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2,11,2), degree=(1,3,1))
aileronL_ffd_block = lg.construct_ffd_block_around_entities(name='left_aileron_ffd_block', entities=aileronL, num_coefficients=(2,11,2), degree=(1,3,1))
aileronR_ffd_block = lg.construct_ffd_block_around_entities(name='right_aileron_ffd_block', entities=aileronR, num_coefficients=(2,11,2), degree=(1,3,1))
flap_ffd_block = lg.construct_ffd_block_around_entities(name='flap_ffd_block', entities=flap, num_coefficients=(2,11,2), degree=(1,3,1))
h_tail_ffd_block = lg.construct_ffd_block_around_entities(name='h_tail_ffd_block', entities=h_tail, num_coefficients=(2,11,2), degree=(1,3,1))
trimTab_ffd_block = lg.construct_ffd_block_around_entities(name='trimTab_ffd_block', entities=trimTab, num_coefficients=(2,11,2), degree=(1,3,1))
vertTail_ffd_block = lg.construct_ffd_block_around_entities(name='v_tail_ffd_block', entities=vertTail, num_coefficients=(2,11,2), degree=(1,3,1))
rudder_ffd_block = lg.construct_ffd_block_around_entities(name='rudder_ffd_block', entities=rudder, num_coefficients=(2,11,2), degree=(1,3,1))
fuselage_ffd_block = lg.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=fuselage, num_coefficients=(2,2,2), degree=(1,1,1))




# Region Parameterization Setup
parameterization_solver = lg.ParameterizationSolver()
parameterization_design_parameters = lg.GeometricVariables()

## Wing Region FFD Setup

wing_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='wing_sect_param',parameterized_points=wing_ffd_block.coefficients,principal_parametric_dimension=1)

wing_chord_stretch_coefficients = csdl.Variable(name='wing_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
wing_chord_stretch_b_spline = lfs.Function(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=wing_chord_stretch_coefficients)

wing_wingspan_stretch_coefficients = csdl.Variable(name='wing_wingspan_stretch_coefficients', value=np.array([-0., 0.]))
wing_wingspan_stretch_b_spline = lfs.Function(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=wing_wingspan_stretch_coefficients)

wing_twist_coefficients = csdl.Variable(name='wing_twist_coefficients', value=np.array([-0, 0, -0, 0, -0])*np.pi/180)
wing_twist_b_spline = lfs.Function(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                          coefficients=wing_twist_coefficients)

wing_sweep_coefficients = csdl.Variable(name='wing_sweep_coefficients', value=np.array([0., 0.0, 0.]))
wing_sweep_b_spline = lfs.Function(space=linear_b_spline_curve_3_dof_space,
                                            coefficients=wing_sweep_coefficients, name='wing_sweep_b_spline')

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

section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_wing_chord_stretch},
    translations={1: sectional_wing_wingspan_stretch, 0: sectional_wing_translation_x, 2: sectional_wing_translation_z, 0: sectional_wing_sweep},
    rotations={1: sectional_wing_twist}
)


wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
wing.set_coefficients(wing_coefficients)

# Wing Region Design Parameters

wing_span_computed = csdl.norm(geometry.evaluate(wing_le_right_parametric) - geometry.evaluate(wing_le_left_parametric))
wing_root_chord_computed = csdl.norm(geometry.evaluate(wing_te_center_parametric) - geometry.evaluate(wing_le_center_parametric))
wing_tip_chord_left_computed = csdl.norm(geometry.evaluate(wing_te_left_parametric) - geometry.evaluate(wing_le_left_parametric))
wing_tip_chord_right_computed = csdl.norm(geometry.evaluate(wing_te_right_parametric) - geometry.evaluate(wing_le_right_parametric))

wing_span = csdl.Variable(name='wing_span', value=np.array([50.]))
wing_root_chord = csdl.Variable(name='wing_root_chord', value=np.array([5.]))
wing_tip_chord = csdl.Variable(name='wing_tip_chord_left', value=np.array([1.]))

parameterization_design_parameters.add_variable(computed_value=wing_span_computed, desired_value=wing_span)
parameterization_design_parameters.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_left_computed, desired_value=wing_tip_chord)
parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_right_computed, desired_value=wing_tip_chord)

# geometry.plot()

# High Lift Rotors setup
lift_rotor_ffd_blocks = []
lift_rotor_sectional_parameterizations = []
lift_rotor_parameterization_b_splines = []
for i, component_set in enumerate(total_HL_motor_components):
    rotor_ffd_block = lg.construct_ffd_block_around_entities(name=f'{component_set[0].name[:3]}_rotor_ffd_block', entities=component_set, num_coefficients=(2,2,2), degree=(1,1,1))
    rotor_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name=f'{component_set[0].name[:3]}_rotor_sectional_parameterization',
                                                                                parameterized_points=rotor_ffd_block.coefficients,
                                                                                principal_parametric_dimension=2)
    
    rotor_stretch_coefficient = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_stretch_coefficient', value=wing_wingspan_stretch_coefficients.value)
    lift_rotor_sectional_stretch_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_sectional_stretch_x_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                coefficients=rotor_stretch_coefficient)
    
    rotor_twist_coefficients = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_twist_coefficients', value=wing_twist_coefficients.value)
    rotor_twist_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_twist_b_spline', space=cubic_b_spline_curve_5_dof_space, coefficients=rotor_twist_coefficients)

    rotor_translation_x_coefficients = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_translation_x_coefficients', value=np.array([0.]))
    rotor_translation_x_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=rotor_translation_x_coefficients)

    rotor_translation_z_coefficients = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_translation_z_coefficients', value=np.array([0.]))
    rotor_translation_z_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=rotor_translation_z_coefficients)
    
    lift_rotor_ffd_blocks.append(rotor_ffd_block)
    lift_rotor_sectional_parameterizations.append(rotor_ffd_block_sectional_parameterization)
    lift_rotor_parameterization_b_splines.append(lift_rotor_sectional_stretch_b_spline)                 

    parameterization_solver.add_parameter(parameter=rotor_stretch_coefficient)

for i, component_set in enumerate(total_HL_motor_components):
    rotor_ffd_block = lift_rotor_ffd_blocks[i]
    rotor_ffd_block_sectional_parameterization = lift_rotor_sectional_parameterizations[i]
    rotor_stretch_b_spline = lift_rotor_parameterization_b_splines[i]

    section_parametric_coordinates = np.linspace(0., 1., rotor_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_stretch = rotor_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_twist = rotor_twist_b_spline.evaluate(section_parametric_coordinates)
    sectional_translation_x = rotor_translation_x_b_spline.evaluate(section_parametric_coordinates)
    sectional_translation_z = rotor_translation_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = lg.VolumeSectionalParameterizationInputs(
        stretches={0: sectional_stretch, 1:sectional_stretch},
        rotations={1: sectional_twist},
    )

    rotor_ffd_block_coefficients = rotor_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    rotor_coefficients = rotor_ffd_block.evaluate(rotor_ffd_block_coefficients, plot=False)
    for i, component in enumerate(component_set):
        component.set_coefficients(rotor_coefficients[i])

# geometry.plot()




## HT FFD Setup
h_tail_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='h_tail_sectional_param',
                                                                            parameterized_points=h_tail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

h_tail_chord_stretch_coefficients = csdl.Variable(name='h_tail_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
h_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=h_tail_chord_stretch_coefficients)

h_tail_span_stretch_coefficients = csdl.Variable(name='h_tail_span_stretch_coefficients', value=np.array([-0., 0.]))
h_tail_span_stretch_b_spline = lfs.Function(name='h_tail_span_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=h_tail_span_stretch_coefficients)

h_tail_sweep_coefficients = csdl.Variable(name='h_tail_sweep_coefficients', value=np.array([0.0, 0.0, 0.0]))
h_tail_sweep_b_spline = lfs.Function(space=linear_b_spline_curve_3_dof_space,
                                            coefficients=h_tail_sweep_coefficients, name='h_tail_sweep_b_spline')

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

## Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver

section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_sweep = h_tail_sweep_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_h_tail_chord_stretch},
    translations={1: sectional_h_tail_span_stretch, 0: sectional_h_tail_translation_x, 2: sectional_h_tail_translation_z, 0: sectional_h_tail_sweep},
    rotations={1: sectional_h_tail_twist}
)

h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)
h_tail.set_coefficients(coefficients=h_tail_coefficients)
# geometry.plot()


# HT Region design parameterization inputs
h_tail_span_computed = csdl.norm(ht_le_right- ht_le_right)
h_tail_root_chord_computed = csdl.norm(ht_te_center - ht_le_center)
h_tail_tip_chord_left_computed = csdl.norm(ht_te_left - ht_le_left)
h_tail_tip_chord_right_computed = csdl.norm(ht_te_right - ht_le_right)

h_tail_span = csdl.Variable(name='h_tail_span', value=np.array([12.]))
h_tail_root_chord = csdl.Variable(name='h_tail_root_chord', value=np.array([3.]))
h_tail_tip_chord = csdl.Variable(name='h_tail_tip_chord_left', value=np.array([2.]))

parameterization_design_parameters.add_variable(computed_value=h_tail_span_computed, desired_value=h_tail_span)
parameterization_design_parameters.add_variable(computed_value=h_tail_root_chord_computed, desired_value=h_tail_root_chord)
parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_left_computed, desired_value=h_tail_tip_chord)
parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_right_computed, desired_value=h_tail_tip_chord)


# Tail Moment Arm Region 
tail_moment_arm_computed = csdl.norm(ht_qc - wing_qc)
tail_moment_arm = csdl.Variable(name='tail_moment_arm', value=np.array([25.]))
parameterization_design_parameters.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm)

wing_fuselage_connection = wing_te_center - fuselage_wing_te_center
h_tail_fuselage_connection = ht_te_center - fuselage_tail_te_center
parameterization_design_parameters.add_variable(computed_value=wing_fuselage_connection, desired_value=wing_fuselage_connection.value)
parameterization_design_parameters.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)


## VT FFD Setup

v_tail_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='v_tail_sectional_param',
                                                                            parameterized_points=vertTail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

v_tail_chord_stretch_coefficients = csdl.Variable(name='v_tail_chord_stretch_coefficients', value=np.array([0., 0.]))
v_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=v_tail_chord_stretch_coefficients)

v_tail_span_stretch_coefficients = csdl.Variable(name='v_tail_span_stretch_coefficients', value=np.array([0.]))
v_tail_span_stretch_b_spline = lfs.Function(name='v_tail_span_stretch_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                          coefficients=v_tail_span_stretch_coefficients)

v_tail_twist_coefficients = csdl.Variable(name='v_tail_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
v_tail_twist_b_spline = lfs.Function(name='v_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                          coefficients=v_tail_twist_coefficients)

v_tail_translation_x_coefficients = csdl.Variable(name='v_tail_translation_x_coefficients', value=np.array([0]))
v_tail_translation_x_b_spline = lfs.Function(name='v_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=v_tail_translation_x_coefficients)

v_tail_translation_z_coefficients = csdl.Variable(name='v_tail_translation_z_coefficients', value=np.array([-0.5*v_tail_span_stretch_coefficients.value]))
v_tail_translation_z_b_spline = lfs.Function(name='v_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=v_tail_translation_z_coefficients)

parameterization_solver.add_parameter(parameter=v_tail_chord_stretch_coefficients)
parameterization_solver.add_parameter(parameter=v_tail_span_stretch_coefficients)
parameterization_solver.add_parameter(parameter=v_tail_twist_coefficients)
parameterization_solver.add_parameter(parameter=v_tail_translation_x_coefficients)
parameterization_solver.add_parameter(parameter=v_tail_translation_z_coefficients)

section_parametric_coordinates = np.linspace(0., 1., v_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_v_tail_chord_stretch = v_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_v_tail_span_stretch = v_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_v_tail_twist = v_tail_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_v_tail_translation_x = v_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_v_tail_translation_z = v_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_v_tail_chord_stretch, 2: sectional_v_tail_span_stretch},
    translations={0: sectional_v_tail_translation_x, 2: sectional_v_tail_translation_z},
    rotations={1: sectional_v_tail_twist}
)

v_tail_ffd_block_coefficients = v_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
v_tail_coefficients = vertTail_ffd_block.evaluate(v_tail_ffd_block_coefficients, plot=False)
vertTail.set_coefficients(coefficients=v_tail_coefficients)
# geometry.plot()

# Vertical Tail Connection
vtail_fuselage_connection_point = geometry.evaluate(vertTail.project(np.array([30.543, 0., 8.231])))
vtail_fuselage_connection = geometry.evaluate(fuselage_rear_pts_parametric) - vtail_fuselage_connection_point   
parameterization_design_parameters.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)




## Fuselage FFD Setup

fuselage_ffd_block_sectional_parameterization = lg.VolumeSectionalParameterization(name='fuselage_sectional_param',
                                                                            parameterized_points=fuselage_ffd_block.coefficients,
                                                                            principal_parametric_dimension=0)

fuselage_stretch_coefficients = csdl.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
fuselage_stretch_b_spline = lfs.Function(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=fuselage_stretch_coefficients)

parameterization_solver.add_parameter(parameter=fuselage_stretch_coefficients)

# Fuselage Parameterization Evaluation for Parameterization Solver

section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lg.VolumeSectionalParameterizationInputs(
    translations={0: sectional_fuselage_stretch}
)

fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
fuselage_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
fuselage.set_coefficients(coefficients=fuselage_coefficients)
# geometry.plot() 

geometry.plot()
parameterization_solver.evaluate(parameterization_design_parameters)
geometry.plot()

