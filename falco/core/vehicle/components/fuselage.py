from falco.core.vehicle.components.component import Component

from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import lsdo_function_spaces as lfs
from typing import Union
import numpy as np
import csdl_alpha as csdl 
from dataclasses import dataclass
from lsdo_function_spaces import FunctionSet
import lsdo_geo as lg
from falco import ureg, Q_




@dataclass
class FuselageParameters:
    length : Union[ureg.Quantity, csdl.Variable]
    max_width : Union[ureg.Quantity, csdl.Variable]
    max_height : Union[ureg.Quantity, csdl.Variable]
    S_wet : Union[ureg.Quantity, csdl.Variable, None]=Q_(1, "m**2")

@dataclass
class FuselageGeometricQuantities:
    length: csdl.Variable
    width: csdl.Variable
    height: csdl.Variable

class Fuselage(Component):
    """The fuslage component class.
    
    Parameters
    ----------
    - length
    - max_width
    - max_height

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
        length : Union[ureg.Quantity, csdl.Variable],
        max_width : Union[ureg.Quantity, csdl.Variable, None] = None, 
        max_height : Union[ureg.Quantity, csdl.Variable, None] = None,
        S_wet : Union[ureg.Quantity, csdl.Variable, None] = Q_(1, "m**2"),
        geometry : Union[FunctionSet, None] = None,
        skip_ffd: bool = False,
        parameterization_solver = None,
        ffd_geometric_variables = None,
        **kwargs
    ) -> None:
        parameterization_solver = parameterization_solver
        ffd_geometric_variables = ffd_geometric_variables
        kwargs["do_not_remake_ffd_block"] = True
        super().__init__(geometry, **kwargs)
        
        # Do type checking

        def define_checks(self):
            self.add_check('length', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('max_width', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('max_height', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('S_wet', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
    

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

        self._name = f"Fuselage"
        self.geometry = geometry
        self.skip_ffd = skip_ffd
        self.geometry = geometry
        

        # Assign parameters
        self.parameters : FuselageParameters = FuselageParameters(
            length=length,
            max_height=max_height,
            max_width=max_width,
            S_wet=S_wet
        )

        # print(f"Initializing Wing with parameters: {self.parameters.length.value}, {self.parameters.max_width.value}, {self.parameters.max_height.value}")


        self.parameters.S_wet = S_wet






        
        # Automatically make the FFD block upon instantiation 

        num_ffd_sections = 3
        ffd_block = lg.construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))


        # Extract dimensions (height, width, length) from the FFD block
        self._nose_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])))
        self._tail_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])))

        self.nose_point = geometry.evaluate(self._nose_point)
        self.tail_point = geometry.evaluate(self._tail_point)

        self._left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0., 0.5])))
        self._right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.5])))

        self.left_point = geometry.evaluate(self._left_point)
        self.right_point = geometry.evaluate(self._right_point)

        self._top_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 1.])))
        self._bottom_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.])))

        self.top_point = geometry.evaluate(self._top_point)
        self.bottom_point = geometry.evaluate(self._bottom_point)


        fuselage_length_stretch_coefficients = csdl.Variable(name='fuselage_length_stretch_coefficients', value=np.array([0., 0.]))
        fuselage_height_stretch_coefficients = csdl.Variable(name='fuselage_height_stretch_coefficients', value=np.array([0., 0.]))
        fuselage_width_stretch_coefficients = csdl.Variable(name='fuselage_width_stretch_coefficients', value=np.array([0., 0.]))      



        

        linear_b_spline_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=0
        )
        # if plot:
        #     ffd_block_sectional_parameterization.plot()


        # Make B-spline functions for changing geometric quantities
        length_stretch_b_spline = lfs.Function(
            space=linear_b_spline_2_dof_space,
            coefficients=fuselage_length_stretch_coefficients,
            name=f"{self._name}_length_stretch_b_sp_coeffs",
        )

        height_stretch_b_spline = lfs.Function(
            space=linear_b_spline_2_dof_space,
            coefficients=fuselage_height_stretch_coefficients,
            name=f"{self._name}_height_stretch_b_sp_coeffs",
        )

        width_stretch_b_spline = lfs.Function(
            space=linear_b_spline_2_dof_space,
            coefficients=fuselage_width_stretch_coefficients,
            name=f"{self._name}_width_stretch_b_sp_coeffs",
        )


        # evaluate b-splines 
        num_ffd_sections = ffd_block_sectional_parameterization.num_sections
        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))

        length_stretch_sectional_parameters = length_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        height_stretch_sectional_parameters = height_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        width_stretch_sectional_parameters = width_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )

        sectional_parameters = VolumeSectionalParameterizationInputs()
        sectional_parameters.add_sectional_translation(axis=0, translation=length_stretch_sectional_parameters)
        sectional_parameters.add_sectional_stretch(axis=1, stretch=height_stretch_sectional_parameters)
        sectional_parameters.add_sectional_stretch(axis=2, stretch=width_stretch_sectional_parameters)

        ffd_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
        ffd_coefficients.name = 'ffd_coefficients'

        # set the coefficient in the geometry
        geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
                    
        self.geometry.set_coefficients(geometry_coefficients)

        nose = self.geometry.evaluate(self._nose_point)
        tail = self.geometry.evaluate(self._tail_point)

        left = self.geometry.evaluate(self._left_point)
        right = self.geometry.evaluate(self._right_point)

        top = self.geometry.evaluate(self._top_point)
        bottom = self.geometry.evaluate(self._bottom_point)

        fuselage_length = csdl.norm(tail - nose)
        fuselage_width = csdl.norm(right - left)
        fuselage_height = csdl.norm(top - bottom)

        fuselage_geometric_qts = FuselageGeometricQuantities(
            length=fuselage_length,
            width=fuselage_width,
            height=fuselage_height,
        )


        length_stretch_b_spline.coefficients.add_name('fuselage_length_stretch_coefficients')
        height_stretch_b_spline.coefficients.add_name('fuselage_height_stretch_coefficients')
        width_stretch_b_spline.coefficients.add_name('fuselage_width_stretch_coefficients')
        
        length_outer_dv = csdl.Variable(shape=(1,), value=self.parameters.length.value)
        height_outer_dv = csdl.Variable(shape=(1,), value=self.parameters.max_height.value)
        width_outer_dv = csdl.Variable(shape=(1,), value=self.parameters.max_width.value)

        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            if len(function.coefficients.shape) != 2:
                function.coefficients = function.coefficients.reshape((-1, function.coefficients.shape[-1]))
            shape = function.coefficients.shape
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action='j->ij')
            
        
        if self.skip_ffd:
            if parameterization_solver is not None:
                parameterization_solver.add_parameter(rigid_body_translation)
        else:
            if parameterization_solver is not None:
                parameterization_solver.add_parameter(length_stretch_b_spline.coefficients)
                parameterization_solver.add_parameter(height_stretch_b_spline.coefficients)
                parameterization_solver.add_parameter(width_stretch_b_spline.coefficients)

        if self.skip_ffd is False and ffd_geometric_variables is not None:
            if self.parameters.length is not None:
                ffd_geometric_variables.add_variable(fuselage_geometric_qts.length, length_outer_dv)
            if self.parameters.max_height is not None:
                ffd_geometric_variables.add_variable(fuselage_geometric_qts.height, height_outer_dv)
            if self.parameters.max_width is not None:
                ffd_geometric_variables.add_variable(fuselage_geometric_qts.width, width_outer_dv)