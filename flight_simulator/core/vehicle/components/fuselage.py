from flight_simulator.core.vehicle.component import Component

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



@dataclass
class FuselageParameters:
    length : Union[float, int, csdl.Variable]
    max_width : Union[float, int, csdl.Variable]
    max_height : Union[float, int, csdl.Variable]
    S_wet : Union[float, int, csdl.Variable, None]=None

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
        length : Union[int, float, csdl.Variable],
        max_width : Union[int, float, csdl.Variable, None] = None, 
        max_height : Union[int, float, csdl.Variable, None] = None, 
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
        csdl.check_parameter(length, "length", types=(int, float, csdl.Variable))
        csdl.check_parameter(max_width, "max_width", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(max_height, "max_height", types=(int, float, csdl.Variable), allow_none=True)

        self._name = f"Fuselage"
        self.geometry = geometry
        self._skip_ffd = skip_ffd
        self.geometry = geometry
        

        # Assign parameters
        self.parameters : FuselageParameters = FuselageParameters(
            length=length,
            max_height=max_height,
            max_width=max_width,
        )

        # print(f"Initializing Wing with parameters: {self.parameters.length.value}, {self.parameters.max_width.value}, {self.parameters.max_height.value}")

        # compute form factor (according to Raymer) if parameters are provided
        if all(arg is not None for arg in [length, max_height, max_width]):
            if not isinstance(max_height, csdl.Variable):
                max_height = csdl.Variable(shape=(1, ), value=max_height)
            if not isinstance(max_width, csdl.Variable):
                max_width = csdl.Variable(shape=(1, ), value=max_width)
            f = length/csdl.maximum(max_height, max_width)
            FF = 1 + 60/f**3 + f/400
            self.parameters.S_wet = self.quantities.surface_area






        
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

        
        parameterization_solver.add_parameter(length_stretch_b_spline.coefficients)
        parameterization_solver.add_parameter(height_stretch_b_spline.coefficients)
        parameterization_solver.add_parameter(width_stretch_b_spline.coefficients)
    

        if self.parameters.length is None:
            pass
        else:
            ffd_geometric_variables.add_variable(fuselage_geometric_qts.length, length_outer_dv)

        if self.parameters.max_height is None:
            pass
        else:
            ffd_geometric_variables.add_variable(fuselage_geometric_qts.height, height_outer_dv)

        if self.parameters.max_width is None:
            pass
        else:
            ffd_geometric_variables.add_variable(fuselage_geometric_qts.width, width_outer_dv)


        print('Computed Fuselage Length',fuselage_geometric_qts.length.value)
        print('Computed Fuselage Width',fuselage_geometric_qts.width.value)
        print('Computed Fuselage Height',fuselage_geometric_qts.height.value)