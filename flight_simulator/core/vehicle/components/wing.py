from flight_simulator.core.vehicle.components.component import Component

import lsdo_function_spaces as lfs
from typing import Union, List

import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
import lsdo_geo as lg
from flight_simulator import ureg, Q_


in2m=0.0254
ft2m = 0.3048



@dataclass
class WingParameters(csdl.VariableGroup):
    #TODO: 
        #REPLACE FLOAT AND INT WITH UREG
        #EXPLICTLY CONVERT EVERYTHING TO CSDL VARIABLE
    AR : csdl.Variable
    S_ref : csdl.Variable
    span : csdl.Variable
    sweep : csdl.Variable
    incidence : csdl.Variable
    taper_ratio : csdl.Variable
    dihedral : csdl.Variable
    root_twist_delta : Union[csdl.Variable,None]
    tip_twist_delta : Union[csdl.Variable,None]
    thickness_to_chord : csdl.Variable
    thickness_to_chord_loc : csdl.Variable
    actuate_angle: csdl.Variable = None
    actuate_axis_location: Union[csdl.Variable,None]=None
    MAC: Union[csdl.Variable,None]=None
    S_wet : Union[csdl.Variable,None]=None

    
@dataclass
class WingGeometricQuantities:
    span: csdl.Variable
    center_chord: csdl.Variable
    left_tip_chord: csdl.Variable
    right_tip_chord: csdl.Variable
    sweep_angle_left: csdl.Variable
    sweep_angle_right: csdl.Variable
    dihedral_angle_left: csdl.Variable
    dihedral_angle_right: csdl.Variable


class Wing(Component):
    """The wing component class.

    Parameters
    ----------
    - AR : aspect ratio
    - S_ref : reference area
    - span (None default)
    - dihedral (deg) (None default)
    - sweep (deg) (None default)
    - taper_ratio (None default)

    Note that parameters may be design variables for optimizaiton.
    If a geometry is provided, the geometry parameterization sovler
    will manipulate the geometry through free-form deformation such 
    that the wing geometry satisfies these parameters.

    Attributes
    ----------
    - parameters : data class storing the above parameters
    - geometry : b-spline set or subset containing the wing geometry
    - comps : dictionary for children components
    - quantities : dictionary for storing (solver) data (e.g., field data)
    """

    def __init__(
        self,
        AR : Union[ureg.Quantity, csdl.Variable, None] = None, 
        S_ref : Union[ureg.Quantity, csdl.Variable, None] = None,
        span : Union[ureg.Quantity, csdl.Variable, None] = None, 
        dihedral : Union[ureg.Quantity, csdl.Variable, None] = None, 
        sweep : Union[ureg.Quantity, csdl.Variable, None] = None, 
        taper_ratio : Union[ureg.Quantity, csdl.Variable, None] = None,
        incidence : Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'), 
        root_twist_delta : Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
        tip_twist_delta : Union[ureg.Quantity, csdl.Variable] = Q_(0, 'rad'),
        thickness_to_chord: Union[ureg.Quantity, csdl.Variable] = Q_(0.15, 'dimensionless'),
        thickness_to_chord_loc: Union[ureg.Quantity, csdl.Variable] = Q_(0.3, 'm'),
        actuate_angle: Union[ureg.Quantity, csdl.Variable,None] = None,
        actuate_axis_location: Union[ureg.Quantity, csdl.Variable, None] = Q_(0.25, 'm'),
        geometry : Union[lfs.FunctionSet, None]=None,
        parametric_geometry: List = None,
        tight_fit_ffd: bool = False,
        skip_ffd: bool = False,
        orientation: str = "horizontal",
        parameterization_solver = None,
        ffd_geometric_variables = None,
        **kwargs
    ) -> None:
        parameterization_solver = parameterization_solver
        ffd_geometric_variables = ffd_geometric_variables
        kwargs["do_not_remake_ffd_block"] = True
        super().__init__(geometry=geometry, **kwargs)
        
        
        def define_checks(self):
            self.add_check('AR', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('S_ref', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('span', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('sweep', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('incidence', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('taper_ratio', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('dihedral', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('root_twist_delta', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('tip_twist_delta', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('MAC', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('S_wet', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('actuate_angle', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('actuate_axis_location', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

        def _check_parameters(self, name, value):
            if self._metadata[name]['type'] is not None:
                if type(value) not in self._metadata[name]['type']:
                    raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")
            if self._metadata[name]['variablize']:
                if isinstance(value, ureg.Quantity):
                    value_si = value.to_base_units()
                    value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                    value.add_tag(tag=str(value_si.units))
            if self._metadata[name]['shape'] is not None:
                if value.shape != self._metadata[name]['shape']:
                    raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
            return value



        # Check if wing is over-parameterized
        if all(arg is not None for arg in [AR, S_ref, span]):
            raise Exception("Wing comp over-parameterized: Cannot specifiy AR, S_ref, and span at the same time.")
        # Check if wing is under-parameterized
        if sum(1 for arg in [AR, S_ref, span] if arg is None) >= 2:
            raise Exception("Wing comp under-parameterized: Must specify two out of three: AR, S_ref, and span.")
        
        if orientation == "vertical" and dihedral is not None:
            raise ValueError("Cannot specify dihedral for vertical wing.")
           
        if incidence is not None:
            if incidence != 0.:
                self._incidence = self.apply_incidence(incidence)
            else:
                self._incidence = 0
        

        self._name = f"{self._name}"
        self._tight_fit_ffd = tight_fit_ffd
        self._orientation = orientation
        self.skip_ffd = skip_ffd
        self._skip_ffd = skip_ffd
        self.geometry = geometry

        
        # Assign parameters
        self.parameters : WingParameters =  WingParameters(
            AR=AR,
            S_ref=S_ref,
            span=span,
            sweep=sweep,
            incidence=incidence,
            dihedral=dihedral,
            taper_ratio=taper_ratio,
            root_twist_delta=root_twist_delta,
            tip_twist_delta=tip_twist_delta,
            thickness_to_chord=thickness_to_chord,
            thickness_to_chord_loc=thickness_to_chord_loc,
            actuate_angle=actuate_angle,
            actuate_axis_location=actuate_axis_location,
        )

        if taper_ratio is None:
            taper_ratio = 1
            self.parameters.taper_ratio = csdl.Variable(name=f"{self._name}_taper_ratio", value=taper_ratio)

        if AR is not None and S_ref is not None:
            span = (AR * S_ref)**0.5
            self.parameters.span = span
            root_chord_input = 2 * S_ref/((1 + taper_ratio) * span)
            tip_chord_input = root_chord_input * taper_ratio
            MAC = (2/3) * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio) * root_chord_input
            self.parameters.MAC = MAC

        elif S_ref is not None and span is not None:
            span = self.parameters.span
            AR = self.parameters.span**2 / self.parameters.S_ref
            self.parameters.AR = AR
            root_chord_input = 2 * S_ref/((1 + taper_ratio) * span)
            tip_chord_input = root_chord_input * taper_ratio
            MAC = (2/3) * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio) * root_chord_input
            self.parameters.MAC = MAC

        elif span is not None and AR is not None:
            S_ref = span**2 / AR
            self.parameters.S_ref = S_ref
            root_chord_input = 2 * S_ref/((1 + taper_ratio) * span)
            tip_chord_input = root_chord_input * taper_ratio
            MAC = (2/3) * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio) * root_chord_input
            self.parameters.MAC = MAC

        x_c_m = self.parameters.thickness_to_chord_loc
        t_o_c = self.parameters.thickness_to_chord

        if t_o_c is None:
            t_o_c = Q_(0.0, 'dimensionless')

        if sweep is None:
            sweep = Q_(0.0, 'rad')
            self.parameters.sweep = sweep

        self.parameters.S_wet = self.surface_area


        
        if dihedral is None:
            dihedral = csdl.Variable(name=f"{self._name}_dihedral", value=0)
            self.parameters.dihedral = dihedral

        if self._orientation == "horizontal":
            wing_le_left_parametric = parametric_geometry[0]
            wing_le_right_parametric = parametric_geometry[1]
            wing_le_center_parametric = parametric_geometry[2]
            wing_te_left_parametric = parametric_geometry[3]
            wing_te_right_parametric = parametric_geometry[4]
            wing_te_center_parametric = parametric_geometry[5]
            wing_qc_center = parametric_geometry[6]
            wing_qc_tip_right = parametric_geometry[7]
            wing_qc_tip_left = parametric_geometry[8]
        else:
            wing_le_base_parametric = parametric_geometry[0]
            wing_le_tip_parametric = parametric_geometry[1]
            wing_le_base_parametric = parametric_geometry[2]
            wing_te_base_parametric = parametric_geometry[3]
            wing_te_tip_parametric = parametric_geometry[4]
            wing_te_mid_parametric = parametric_geometry[5]
            wing_qc_parametric = parametric_geometry[6]
            wing_qc_base_parametric = parametric_geometry[7]
            wing_qc_tip_parametric = parametric_geometry[8]


        if actuate_angle is not None:
            if actuate_axis_location is None:
                axis_location = 0.25
            else:
                axis_location = actuate_axis_location
            
            if self._orientation == "horizontal":
                LE_center = geometry.evaluate(wing_le_center_parametric)
                TE_center = geometry.evaluate(wing_te_center_parametric)
                actuation_center = csdl.linear_combination(
                    LE_center, TE_center, 1, np.array([1 -axis_location]), np.array([axis_location])
                ).flatten()
                var = csdl.Variable(shape=(3, ), value=np.array([0., 1., 0.])) 
            else:
                LE_root = geometry.evaluate(wing_le_base_parametric)
                TE_root = geometry.evaluate(wing_te_base_parametric)
                actuation_center = csdl.linear_combination(
                    LE_root, TE_root, 1, np.array([1 -axis_location]), np.array([axis_location])).flatten()
                var = csdl.Variable(shape=(3, ), value=np.array([0., 0., 1.]))

            axis_origin = actuation_center - var
            axis_vector = actuation_center + var - axis_origin
 
 
            # Rotate the component about the axis
            geometry.rotate(axis_origin=axis_origin, axis_vector=axis_vector / csdl.norm(axis_vector), angles=actuate_angle)
        

        if self._orientation == "horizontal":
            num_coefficients = (3,11,3)
        else:
            num_coefficients = (3,11,3)



        if self._tight_fit_ffd is True:
            ffd_block = lg.construct_ffd_block_around_entities(
                entities=geometry, 
                num_coefficients=(2,2,2), 
                degree=(1,1,1)
            )
        else:
            ffd_block = lg.construct_ffd_block_around_entities(
                entities=geometry, 
                num_coefficients=num_coefficients, 
                degree=(1,3,1)
            )
        self._ffd_block = ffd_block
        




        from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
            VolumeSectionalParameterization,
            VolumeSectionalParameterizationInputs
        )

        if self._orientation == "horizontal":
            principal_parametric_dimension = 1
        else:
            principal_parametric_dimension = 2

        ffd_sectional_parameterization = VolumeSectionalParameterization(
            name="ffd_sectional_parameterization",
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=principal_parametric_dimension,
        )

        space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
        space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

        chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                                coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0., 0., 0.])), name='chord_stretching_b_spline_coefficients')

        wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                    coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([0., 0.])), name='wingspan_stretching_b_spline_coefficients')

        sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                                    coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0.0, 0.0, 0.0])), name='sweep_translation_b_spline_coefficients')

        twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                        coefficients=csdl.Variable(shape=(3,), value=np.array([0, 0., 0])*np.pi/180), name='twist_b_spline_coefficients')

        dihedral_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                        coefficients=csdl.Variable(shape=(3,), value=np.array([0, 0., 0])*np.pi/180), name='dihedral_b_spline_coefficients')

        num_ffd_sections = ffd_sectional_parameterization.num_sections

        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
        chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
            parametric_b_spline_inputs
        )

        if self.parameters.sweep is not None:
            sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
                parametric_b_spline_inputs
            )

        twist_sectional_parameters = twist_b_spline.evaluate(
            parametric_b_spline_inputs
        )

        if self.parameters.dihedral is not None:
            dihedral_sectional_parameters = dihedral_b_spline.evaluate(
                parametric_b_spline_inputs
            )


        sectional_parameters = VolumeSectionalParameterizationInputs()
        if self._orientation == "horizontal":
            sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
            sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
            if self.parameters.sweep is not None:
                sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
            sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)
            if self.parameters.dihedral is not None:
                sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_sectional_parameters)

        else:
            sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
            sectional_parameters.add_sectional_translation(axis=2, translation=wingspan_stretch_sectional_parameters)
            if self.parameters.sweep is not None:
                sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)





        ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False) 
        ffd_coefficients.name = 'ffd_coefficients'


        geometry_coefficients = ffd_block.evaluate(ffd_coefficients, plot=False)
        # print(geometry_coefficients)
        geometry.set_coefficients(geometry_coefficients)
        # wing.plot()




        if self._orientation == "horizontal":
            wingspan = csdl.norm(geometry.evaluate(wing_le_right_parametric) - geometry.evaluate(wing_le_left_parametric))
            root_chord = csdl.norm(geometry.evaluate(wing_te_center_parametric) - geometry.evaluate(wing_le_center_parametric))
            tip_chord_left = csdl.norm(geometry.evaluate(wing_te_left_parametric) - geometry.evaluate(wing_le_left_parametric))
            tip_chord_right = csdl.norm(geometry.evaluate(wing_te_right_parametric) - geometry.evaluate(wing_le_right_parametric))
            spanwise_direction_left = geometry.evaluate(wing_qc_tip_left) - geometry.evaluate(wing_qc_center)
            spanwise_direction_right = geometry.evaluate(wing_qc_tip_right) - geometry.evaluate(wing_qc_center)
            sweep_angle_left = csdl.arcsin(-spanwise_direction_left[0] / csdl.norm(spanwise_direction_left))
            sweep_angle_right = csdl.arcsin(-spanwise_direction_right[0] / csdl.norm(spanwise_direction_right))
            dihedral_angle_left = csdl.arcsin(spanwise_direction_left[2] / csdl.norm(spanwise_direction_left))
            dihedral_angle_right = csdl.arcsin(spanwise_direction_right[2] / csdl.norm(spanwise_direction_right))
        else:
            wingspan = csdl.norm(geometry.evaluate(wing_le_tip_parametric) - geometry.evaluate(wing_le_base_parametric))
            root_chord = csdl.norm(geometry.evaluate(wing_te_base_parametric) - geometry.evaluate(wing_le_base_parametric))
            tip_chord = csdl.norm(geometry.evaluate(wing_te_tip_parametric) - geometry.evaluate(wing_le_tip_parametric))
            spanwise_direction = geometry.evaluate(wing_qc_tip_parametric) - geometry.evaluate(wing_qc_base_parametric)
            sweep_angle = csdl.arcsin(spanwise_direction[0] / csdl.norm(spanwise_direction))





        chord_stretching_b_spline.coefficients.add_name('chord_stretching_b_spline_coefficients')
        wingspan_stretching_b_spline.coefficients.add_name('wingspan_stretching_b_spline_coefficients')
        sweep_translation_b_spline.coefficients.add_name('sweep_translation_b_spline_coefficients')
        twist_b_spline.coefficients.add_name('twist_b_spline_coefficients')
        dihedral_b_spline.coefficients.add_name('dihedral_b_spline_coefficients')


        wingspan_outer_dv = csdl.Variable(shape=(1,), value=self.parameters.span.value)
        root_chord_outer_dv = csdl.Variable(shape=(1,), value=root_chord_input.value)
        tip_chord_outer_dv = csdl.Variable(shape=(1,), value=tip_chord_input.value)
        sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=self.parameters.sweep.value * np.pi / 180)
        dihedral_outer_dv = csdl.Variable(shape=(1,), value=self.parameters.dihedral.value* np.pi / 180)

        if self.parameters.actuate_angle is not None:
            if self._orientation == "horizontal":
                geometry.rotate(axis_origin=self.parameters.actuate_axis_location, axis_vector=np.array([0., 0., 1.]), angles=self.parameters.actuate_angle)
            else:
                geometry.rotate(axis_origin=self.parameters.actuate_axis_location, axis_vector=np.array([0., 1., 0.]), angles=self.parameters.actuate_angle)

        parameterization_solver.add_parameter(chord_stretching_b_spline.coefficients)
        parameterization_solver.add_parameter(wingspan_stretching_b_spline.coefficients)
        parameterization_solver.add_parameter(sweep_translation_b_spline.coefficients)
        parameterization_solver.add_parameter(twist_b_spline.coefficients)
        parameterization_solver.add_parameter(dihedral_b_spline.coefficients)




        if self._orientation == "horizontal":
            ffd_geometric_variables.add_variable(wingspan, wingspan_outer_dv)
            ffd_geometric_variables.add_variable(root_chord, root_chord_outer_dv)
            ffd_geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv)
            ffd_geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv)
            if self.parameters.sweep is not None:
                ffd_geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv)
                ffd_geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv)
            if self.parameters.dihedral is not None:
                ffd_geometric_variables.add_variable(dihedral_angle_left, dihedral_outer_dv)
                ffd_geometric_variables.add_variable(dihedral_angle_right, dihedral_outer_dv) 
        else:
            ffd_geometric_variables.add_variable(wingspan, wingspan_outer_dv)
            ffd_geometric_variables.add_variable(root_chord, root_chord_outer_dv)
            ffd_geometric_variables.add_variable(tip_chord, tip_chord_outer_dv)
            if self.parameters.sweep is not None:
                ffd_geometric_variables.add_variable(sweep_angle, sweep_angle_outer_dv)
       


        # print("Component Name: ", self._name)
        # print("Wingspan: ", wingspan.value)
        # print("Root Chord: ", root_chord.value)
        # if self._orientation == "horizontal":
        #     print("Tip Chord Left: ", tip_chord_left.value)
        #     print("Tip Chord Right: ", tip_chord_right.value)
        #     print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
        #     print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
        #     print("Dihedral Angle Left: ", dihedral_angle_left.value*180/np.pi)
        #     print("Dihedral Angle Right: ", dihedral_angle_right.value*180/np.pi)
        # else:
        #     print("Tip Chord: ", tip_chord.value)
        #     print("Sweep Angle: ", sweep_angle.value*180/np.pi)
        # print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
        # print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
        # print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)
        # print("Twist: ", twist_b_spline.coefficients.value)






        


     
    


    