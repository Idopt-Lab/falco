from flight_simulator.core.component import Component

from lsdo_geo import construct_ffd_block_around_entities, construct_tight_fit_ffd_block
import lsdo_function_spaces as lfs
from typing import Union, List
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
import lsdo_geo as lg
import time


@dataclass
class WingParameters:
    AR : Union[float, int, csdl.Variable]
    S_ref : Union[float, int, csdl.Variable]
    span : Union[float, int, csdl.Variable]
    sweep : Union[float, int, csdl.Variable]
    incidence : Union[float, int, csdl.Variable]
    taper_ratio : Union[float, int, csdl.Variable]
    dihedral : Union[float, int, csdl.Variable]
    root_twist_delta : Union[int, float, csdl.Variable, None]
    tip_twist_delta : Union[int, float, csdl.Variable, None]
    thickness_to_chord : Union[float, int, csdl.Variable] = 0.15
    thickness_to_chord_loc : float = 0.3
    MAC: Union[float, None] = None
    S_wet : Union[float, int, csdl.Variable, None]=None
    actuate_angle: csdl.Variable = None
    actuate_axis_location: Union[float, int, csdl.Variable, None] = 0.

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
        AR : Union[int, float, csdl.Variable, None] = None, 
        S_ref : Union[int, float, csdl.Variable, None] = None,
        span : Union[int, float, csdl.Variable, None] = None, 
        dihedral : Union[int, float, csdl.Variable, None] = None, 
        sweep : Union[int, float, csdl.Variable, None] = None, 
        taper_ratio : Union[int, float, csdl.Variable, None] = None,
        incidence : Union[int, float, csdl.Variable] = 0, 
        root_twist_delta : Union[int, float, csdl.Variable] = 0,
        tip_twist_delta : Union[int, float, csdl.Variable] = 0,
        thickness_to_chord: float = 0.15,
        thickness_to_chord_loc: float = 0.3,
        actuate_angle: Union[float, int, csdl.Variable, None] = None,
        actuate_axis_location: Union[int, float, csdl.Variable, None] = 0.25,
        geometry : Union[lfs.FunctionSet, None]=None,
        tight_fit_ffd: bool = False,
        skip_ffd: bool = False,
        orientation: str = "horizontal",
        **kwargs
    ) -> None:
        kwargs["do_not_remake_ffd_block"] = True
        super().__init__(geometry=geometry, **kwargs)
    
        # Print statements for debugging
        # print(f"Initializing Wing with parameters:")
        # print(f"AR: {AR}")
        # print(f"S_ref: {S_ref}")
        # print(f"span: {span}")
        # print(f"sweep: {sweep}")
        # print(f"dihedral: {dihedral}")
        # print(f"incidence: {incidence}")
        # print(f"root_twist_delta: {root_twist_delta}")
        # print(f"tip_twist_delta: {tip_twist_delta}")
        # print(f"thickness_to_chord: {thickness_to_chord}")
        # print(f"thickness_to_chord_loc: {thickness_to_chord_loc}")
        # print(f"actuate_angle: {actuate_angle}")
        # print(f"actuate_axis_location: {actuate_axis_location}")
        # print(f"geometry: {geometry}")
        # print(f"tight_fit_ffd: {tight_fit_ffd}")
        # print(f"skip_ffd: {skip_ffd}")
        # print(f"orientation: {orientation}")
        
        # Do type checking 
        csdl.check_parameter(AR, "AR", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(S_ref, "S_ref", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(span, "span", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(dihedral, "dihedral", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(sweep, "sweep", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(incidence, "incidence", types=(int, float, csdl.Variable))
        csdl.check_parameter(taper_ratio, "taper_ratio", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(root_twist_delta, "root_twist_delta", types=(int, float, csdl.Variable))
        csdl.check_parameter(tip_twist_delta, "tip_twist_delta", types=(int, float, csdl.Variable))
        csdl.check_parameter(orientation, "orientation", values=["horizontal", "vertical"])

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


        # Compute MAC (i.e., characteristic length)
        if taper_ratio is None:
            taper_ratio = 1
        if AR is not None and S_ref is not None:
            lam = taper_ratio
            span = (AR * S_ref)**0.5
            self.parameters.span = span
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            # self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC
        elif S_ref is not None and span is not None:
            lam = taper_ratio
            span = self.parameters.span
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            # self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC
        elif span is not None and AR is not None:
            lam = taper_ratio
            S_ref = span**2 / AR
            self.parameters.S_ref = S_ref
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            # self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC

        # Compute form factor according to Raymer 
        # (ignoring Mach number; include in drag build up model)
        x_c_m = self.parameters.thickness_to_chord_loc
        t_o_c = self.parameters.thickness_to_chord

        if t_o_c is None:
            t_o_c = 0.15
        if sweep is None:
            sweep = 0.

        FF = (1 + 0.6 / x_c_m + 100 * (t_o_c) ** 4) * csdl.cos(sweep) ** 0.28
        # self.quantities.drag_parameters.form_factor = FF


        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (lfs.FunctionSet)):
                raise TypeError(f"wing gometry must be of type {lfs.FunctionSet}")
            else:
                # Set the wetted area
                self.parameters.S_wet = self.quantities.surface_area

                # Make the FFD block upon instantiation
                ffd_block = self._make_ffd_block(self.geometry, tight_fit=tight_fit_ffd, degree=(1, 3, 1), num_coefficients=(2, 11, 2))

                # Compute the corner points of the wing 
                if self._orientation == "horizontal":
                    self._LE_left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.8, 0., 0.5])), plot=False, extrema=True)
                    self._LE_mid_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])), plot=False, extrema=True)
                    self._LE_right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.8, 1., 0.5])), plot=False, extrema=True)

                    self._TE_left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0., 0.5])), plot=False, extrema=True)
                    self._TE_mid_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])),  plot=False, extrema=True)
                    self._TE_right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 1.0, 0.5])), plot=False, extrema=True)

                    # print(f"_LE_left_point: {self._LE_left_point}")
                    # print(f"_LE_mid_point: {self._LE_mid_point}")
                    # print(f"_LE_right_point: {self._LE_right_point}")
                    # print(f"_TE_left_point: {self._TE_left_point}")
                    # print(f"_TE_mid_point: {self._TE_mid_point}")
                    # print(f"_TE_right_point: {self._TE_right_point}")


                else:
                    self._LE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.])), direction=np.array([-1., 0., 0.]), plot=False, extrema=False)
                    self._LE_root_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 1.])), direction=np.array([-1., 0., 0.]), plot=False, extrema=False)

                    self._TE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.])), plot=False, extrema=True)
                    self._TE_root_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 1.])), plot=False, extrema=True)

                    self.LE_root = geometry.evaluate(self._LE_root_point)
                    self.TE_root = geometry.evaluate(self._TE_root_point)


                    # print(f"_LE_tip_point: {self._LE_tip_point}")
                    # print(f"_LE_root_point: {self._LE_root_point}")
                    # print(f"_TE_tip_point: {self._TE_tip_point}")
                    # print(f"_TE_root_point: {self._TE_root_point}")

                    # print(self.geometry.evaluate(self._LE_tip_point).value)
                    # print(self.geometry.evaluate(self._LE_root_point).value)
                    # print(self.geometry.evaluate(self._TE_tip_point).value)
                    # print(self.geometry.evaluate(self._TE_root_point).value)
                    # exit()

                self._ffd_block = self._make_ffd_block(self.geometry, tight_fit=False)

                # Print FFD block details for debugging
                # print("FFD block created:")
                # print(f"  num_coefficients: {ffd_block.coefficients.value}")

                # print("time for computing corner points", t6-t5)
            # internal geometry projection info
            self._dependent_geometry_points = [] # {'parametric_points', 'function_space', 'fitting_coords', 'mirror'}
            self._base_geometry = self.geometry.copy()
        
            
        if actuate_angle is not None:
            self.actuate(actuate_angle, actuate_axis_location)
  


    def actuate(self, angle : Union[float, int, csdl.Variable], axis_location : float = 0.25):
        """Actuate (i.e., rotate) the wing about an axis location at or behind the leading edge.
        
        Parameters
        ----------
        angle : float, int, or csdl.Variable
            rotation angle (deg)

        axis_location : float (default is 0.25)
            location of actuation axis with respect to the leading edge;
            0.0 corresponds the leading and 1.0 corresponds to the trailing edge
        """
        wing_geometry = self.geometry
        # check if wing_geometry is not None
        if wing_geometry is None:
            raise ValueError("wing component cannot be actuated since it does not have a geometry (i.e., geometry=None)")

        # Check if if actuation axis is between 0 and 1
        if axis_location < 0.0 or axis_location > 1.0:
            raise ValueError("axis_loaction should be between 0 and 1")
        
        LE_center = wing_geometry.evaluate(self._LE_mid_point)
        TE_center = wing_geometry.evaluate(self._TE_mid_point)

        # Add the user_specified axis location
        actuation_center = csdl.linear_combination(
            LE_center, TE_center, 1, np.array([1 -axis_location]), np.array([axis_location])
        ).flatten()


        var = csdl.Variable(shape=(3, ), value=np.array([0., 1., 0.]))

        # Compute the actuation axis vector
        axis_origin = actuation_center - var
        axis_vector = actuation_center + var - axis_origin


        # Rotate the component about the axis
        wing_geometry.rotate(axis_origin=axis_origin, axis_vector=axis_vector / csdl.norm(axis_vector), angles=angle)


    def _make_ffd_block(self, 
            entities : List[lfs.Function], 
            num_coefficients : tuple=(2, 2, 2), 
            degree: tuple=(1, 1, 1), 
            num_physical_dimensions : int=3,
            tight_fit: bool = True,
        ):
        """
        Call 'construct_ffd_block_around_entities' function. 

        Note that we overwrite the Component class's method to 
        - make a "tight-fit" ffd block instead of a cartesian one
        - to provide higher degree B-splines or more degrees of freedom
        if needed (via num_coefficients)
        """
        if tight_fit:
            ffd_block = construct_tight_fit_ffd_block(name=self._name, entities=entities, 
                                                    num_coefficients=num_coefficients, degree=degree)
        else:
            if self._orientation == "horizontal":
                num_coefficients = (2, 11, 2) # NOTE: hard coding here might be limiting
                degree = (1, 3, 1)
            else:
                degree = (1, 1, 1)
                num_coefficients = (2, 2, 2)
            ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities,
                                                            num_coefficients=num_coefficients, degree=degree)
        
        ffd_block.coefficients.name = f'{self._name}_coefficients'

        # ffd_block.plot()
        # print(f"Creating FFD block for wing:")
        # print(f"  num_coefficients: {num_coefficients}")
        # print(f"  order: {degree}")
        # ffd_block = construct_ffd_block_around_entities(
        #     name=self._name, 
        #     entities=entities,
        #     num_coefficients=num_coefficients,
        #     degree=degree
        # )

        return ffd_block 
    
    def _setup_ffd_block(self, ffd_block, parameterization_solver, plot : bool=False):
        """Set up the wing ffd block."""
        self._linear_b_spline_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
        self._linear_b_spline_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
        if self._orientation == "horizontal":
            principal_parametric_dimension = 1
        else:
            principal_parametric_dimension = 2

        
        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=principal_parametric_dimension,
        )
       
        # if plot:
        #     ffd_block_sectional_parameterization.plot()
        
        # Make B-spline functions for changing geometric quantities
        chord_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_3_dof_space, 
            coefficients=csdl.ImplicitVariable(
                shape=(3, ), 
                value=np.array([-0, 0, 0])
            ),
            name=f"{self._name}_chord_stretch_b_sp_coeffs"
        )

        span_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_2_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(2, ),
                value=np.array([0., 0.]),
            ),
            name=f"{self._name}_span_stretch_b_sp_coeffs",
        )

        if self.parameters.sweep is not None:
            sweep_translation_b_spline = lfs.Function(
                space=self._linear_b_spline_3_dof_space,
                coefficients=csdl.ImplicitVariable(
                    shape=(3, ),
                    value=np.array([0., 0., 0.,]),
                ),
                name=f"{self._name}_sweep_transl_b_sp_coeffs"
            )

        if self.parameters.dihedral is not None:
            dihedral_translation_b_spline = lfs.Function(
                space=self._linear_b_spline_3_dof_space,
                coefficients=csdl.ImplicitVariable(
                    shape=(3, ),
                    value=np.array([0., 0., 0.,]),
                ),
                name=f"{self._name}_dihedral_transl_b_sp_coeffs"
            )

        coefficients=csdl.Variable(
                shape=(3, ),
                value=np.array([0., 0., 0.,]),
        )
        coefficients = coefficients.set(csdl.slice[0], self.parameters.tip_twist_delta)
        coefficients = coefficients.set(csdl.slice[1], self.parameters.root_twist_delta)
        coefficients = coefficients.set(csdl.slice[2], self.parameters.tip_twist_delta)
        twist_b_spline = lfs.Function(
            space=self._linear_b_spline_3_dof_space,
            coefficients=coefficients,
            name=f"{self._name}_twist_b_sp_coeffs"
        )

        if self.parameters.actuate_angle is not None:
            actuate_b_spline = lfs.Function(
                space=self._linear_b_spline_3_dof_space,
                coefficients=csdl.ImplicitVariable(
                    shape=(3,),
                    value=np.array([0, 0, 0]),
                ),
                name=f"{self._name}_actuate_b_sp_coeffs"
            )

        # evaluate b-splines 
        num_ffd_sections = ffd_block_sectional_parameterization.num_sections
        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
        
        chord_stretch_sectional_parameters = chord_stretch_b_spline.evaluate(
            parametric_b_spline_inputs,
        )
        span_stretch_sectional_parameters = span_stretch_b_spline.evaluate(
            parametric_b_spline_inputs,
        )
        if self.parameters.sweep is not None:
            sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
                parametric_b_spline_inputs
            )
        if self.parameters.dihedral is not None:
            dihedral_translation_sectional_parameters = dihedral_translation_b_spline.evaluate(
                parametric_b_spline_inputs
            )
        twist_sectional_parameters = twist_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        if self.parameters.actuate_angle is not None:
            actuate_sectional_parameters = actuate_b_spline.evaluate(
                parametric_b_spline_inputs
            )

        sectional_parameters = VolumeSectionalParameterizationInputs()
        if self._orientation == "horizontal":
            sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
            sectional_parameters.add_sectional_translation(axis=1, translation=span_stretch_sectional_parameters)
            if self.parameters.sweep is not None:
                sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
            if self.parameters.dihedral is not None:
                sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translation_sectional_parameters)
            sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)
            if self.parameters.actuate_angle is not None:
                sectional_parameters.add_sectional_rotation(axis=2, rotation=actuate_sectional_parameters)
        else:
            sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
            sectional_parameters.add_sectional_translation(axis=2, translation=span_stretch_sectional_parameters)
            if self.parameters.sweep is not None:
                sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
            if self.parameters.actuate_angle is not None:
                sectional_parameters.add_sectional_rotation(axis=1, rotation=actuate_sectional_parameters)

        ffd_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False) 

        # set the base coefficients
        geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
        self._base_geometry.set_coefficients(geometry_coefficients)
        
        # re-fit the dependent geometry points
        coeff_flip = np.eye(3)
        coeff_flip[1,1] = -1
        for item in self._dependent_geometry_points:
            fitting_points = self._base_geometry.evaluate(item['parametric_points'])
            coefficients = item['function_space'].fit(fitting_points, item['fitting_coords'])
            geometry_coefficients.append(coefficients)
            if item['mirror']:
                if not len(coefficients.shape) == 2:
                    coefficients = coefficients.reshape((-1, coefficients.shape[-1]))
                geometry_coefficients.append(coefficients @ coeff_flip)



        # set full geometry coefficients
        self.geometry.set_coefficients(geometry_coefficients)

        # Add rigid body translation (without FFD)
        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            # if len(function.coefficients.shape) != 2:
            #     function.coefficients = function.coefficients.reshape((-1, 3))
            # shape = function.coefficients.shape
            # print('Wing Shape: ', shape)
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')
        
        # Add the coefficients of all B-splines to the parameterization solver
        if self.skip_ffd:
            parameterization_solver.add_parameter(rigid_body_translation, cost=10)
        
        else:            
            parameterization_solver.add_parameter(chord_stretch_b_spline.coefficients)
            parameterization_solver.add_parameter(span_stretch_b_spline.coefficients)
            if self.parameters.sweep is not None:
                parameterization_solver.add_parameter(sweep_translation_b_spline.coefficients)
            if self.parameters.dihedral is not None:
                parameterization_solver.add_parameter(dihedral_translation_b_spline.coefficients)
            if self.parameters.actuate_angle is not None:
                parameterization_solver.add_parameter(actuate_b_spline.coefficients)    
            parameterization_solver.add_parameter(rigid_body_translation, cost=10)

        return 

    def _extract_geometric_quantities_from_ffd_block(self) -> WingGeometricQuantities:
        """Extract the following quantities from the FFD block:
            - Span
            - root chord length
            - tip chord lengths
            - sweep/dihedral angles

        Note that this helper function will not work well in all cases (e.g.,
        in cases with high sweep or taper)
        """
        if self.parameters.actuate_angle is not None:
            self.actuate(self.parameters.actuate_angle, self.parameters.actuate_axis_location)

        if self._orientation == "horizontal":
            # Re-evaluate the corner points of the FFD block (plus center)
            # Root
            LE_center = self.geometry.evaluate(self._LE_mid_point)
            TE_center = self.geometry.evaluate(self._TE_mid_point)
            # print(f"LE_center: {LE_center.value}")
            # print(f"TE_center: {TE_center.value}")

            qc_center = 0.75 * LE_center + 0.25 * TE_center

            # Tip
            LE_left = self.geometry.evaluate(self._LE_left_point)
            TE_left = self.geometry.evaluate(self._TE_left_point)
            # print(f"LE_left: {LE_left.value}")
            # print(f"TE_left: {TE_left.value}")

            qc_left = 0.75 * LE_left + 0.25 * TE_left

            # Right side 
            LE_right = self.geometry.evaluate(self._LE_right_point)
            TE_right = self.geometry.evaluate(self._TE_right_point)
            # print(f"LE_right: {LE_right.value}")
            # print(f"TE_right: {TE_right.value}")

            qc_right = 0.75 * LE_right + 0.25 * TE_right

            # Compute span, root/tip chords, sweep, and dihedral
            span = LE_left - LE_right
            center_chord = TE_center - LE_center
            left_tip_chord = TE_left - LE_left
            right_tip_chord = TE_right - LE_right

            qc_spanwise_left = qc_left - qc_center
            qc_spanwise_right = qc_right - qc_center

            sweep_angle_left = csdl.arcsin(qc_spanwise_left[0] / csdl.norm(qc_spanwise_left))
            sweep_angle_right = csdl.arcsin(qc_spanwise_right[0] / csdl.norm(qc_spanwise_right))

            dihedral_angle_left = csdl.arcsin(qc_spanwise_left[2] / csdl.norm(qc_spanwise_left))
            dihedral_angle_right = csdl.arcsin(qc_spanwise_right[2] / csdl.norm(qc_spanwise_right))


            wing_geometric_qts = WingGeometricQuantities(
                span=csdl.norm(span),
                center_chord=csdl.norm(center_chord),
                left_tip_chord=csdl.norm(left_tip_chord),
                right_tip_chord=csdl.norm(right_tip_chord),
                sweep_angle_left=sweep_angle_left,
                sweep_angle_right=sweep_angle_right,
                dihedral_angle_left=dihedral_angle_left,
                dihedral_angle_right=dihedral_angle_right
            )

        else:
            # Re-evaluate the corner points of the FFD block (plus center)
            # Root
            LE_root = self.geometry.evaluate(self._LE_root_point)
            TE_root = self.geometry.evaluate(self._TE_root_point)

            qc_root = 0.75 * LE_root + 0.25 * TE_root

            # Tip 
            LE_tip = self.geometry.evaluate(self._LE_tip_point)
            TE_tip = self.geometry.evaluate(self._TE_tip_point)

            qc_tip = 0.75 * LE_tip + 0.25 * TE_tip

            # Compute span, root/tip chords, sweep, and dihedral
            span = TE_tip - TE_root
            root_chord = TE_root - LE_root
            tip_chord = TE_tip - LE_tip
            qc_tip_root = qc_tip - qc_root
            sweep_angle = csdl.arcsin(qc_tip_root[0] / csdl.norm(qc_tip_root))


            wing_geometric_qts = WingGeometricQuantities(
                span=csdl.norm(span),
                center_chord=csdl.norm(root_chord),
                left_tip_chord=csdl.norm(tip_chord),
                right_tip_chord=None,
                sweep_angle_left=sweep_angle,
                sweep_angle_right=None,
                dihedral_angle_left=None,
                dihedral_angle_right=None,
            )

        return wing_geometric_qts

    def _setup_ffd_parameterization(self, wing_geom_qts: WingGeometricQuantities, ffd_geometric_variables):
        """Set up the wing parameterization."""
        # TODO: set up parameters as constraints

        # Set or compute the values for those quantities
        # AR = b**2/S_ref

        if self.parameters.AR is not None and self.parameters.S_ref is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio
            
            if not isinstance(self.parameters.AR, csdl.Variable):
                self.parameters.AR = csdl.Variable(shape=(1, ), value=self.parameters.AR)

            if not isinstance(self.parameters.S_ref, csdl.Variable):
                self.parameters.S_ref = csdl.Variable(shape=(1, ), value=self.parameters.S_ref)
                
            span_input = (self.parameters.AR * self.parameters.S_ref)**0.5
            root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
            tip_chord_left_input = root_chord_input * taper_ratio 
            tip_chord_right_input = tip_chord_left_input * 1

        elif self.parameters.S_ref is not None and self.parameters.span is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio
        
            AR = self.parameters.span**2 / self.parameters.S_ref
            self.parameters.AR = AR

            if not isinstance(self.parameters.span , csdl.Variable):
                self.parameters.span = csdl.Variable(shape=(1, ), value=self.parameters.span)

            if not isinstance(self.parameters.S_ref , csdl.Variable):
                self.parameters.S_ref = csdl.Variable(shape=(1, ), value=self.parameters.S_ref)

            if not isinstance(self.parameters.AR , csdl.Variable):
                self.parameters.AR = csdl.Variable(shape=(1, ), value=self.parameters.AR)

            span_input = self.parameters.span
            root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
            tip_chord_left_input = root_chord_input * taper_ratio 
            tip_chord_right_input = tip_chord_left_input * 1
        
        elif self.parameters.span is not None and self.parameters.AR is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio

            if not isinstance(self.parameters.AR , csdl.Variable):
                self.parameters.AR = csdl.Variable(shape=(1, ), value=self.parameters.AR)

            if not isinstance(self.parameters.span , csdl.Variable):
                self.parameters.span = csdl.Variable(shape=(1, ), value=self.parameters.span)

            span_input = self.parameters.span
            root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
            tip_chord_left_input = root_chord_input * taper_ratio 
            tip_chord_right_input = tip_chord_left_input * 1

        else:
            raise NotImplementedError

        # Set constraints: user input - geometric qty equivalent
        if self._orientation == "horizontal":
            ffd_geometric_variables.add_variable(wing_geom_qts.span, span_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.center_chord, root_chord_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.left_tip_chord, tip_chord_left_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.right_tip_chord, tip_chord_right_input)
        else:
            ffd_geometric_variables.add_variable(wing_geom_qts.span, span_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.center_chord, root_chord_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.left_tip_chord, tip_chord_left_input)

        if self.parameters.sweep is not None:
            sweep_input = self.parameters.sweep
            ffd_geometric_variables.add_variable(wing_geom_qts.sweep_angle_left, sweep_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.sweep_angle_right, sweep_input)

        if self.parameters.dihedral is not None:
            dihedral_input = self.parameters.dihedral
            ffd_geometric_variables.add_variable(wing_geom_qts.dihedral_angle_left, dihedral_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.dihedral_angle_right, dihedral_input)

        if self.parameters.actuate_angle is not None:
            actuate_angle_input = self.parameters.actuate_angle
            actuate_axis_location_input = self.parameters.actuate_axis_location
            self.actuate(actuate_angle_input, actuate_axis_location_input)

        
        # print(wing_geom_qts.span.value, span_input.value)

        # print(wing_geom_qts.center_chord.value, root_chord_input.value)
        # print(wing_geom_qts.left_tip_chord.value, tip_chord_left_input.value)
        # print(wing_geom_qts.right_tip_chord.value, tip_chord_right_input.value)
        # print(f"Name: {self._name}")
        # print(f"Target AR: {self.parameters.AR.value}")
        # print(f"Target S_ref: {self.parameters.S_ref.value}")
        # print(f"Computed span: {span_input.value}")
        # print(f"Computed root chord: {root_chord_input.value}")





        # print(f"\nWing Geometry Parameters:")
        # print(f"Desired:")
        # print(f"  AR: {self.parameters.AR.value[0]}")
        # print(f"  S_ref: {self.parameters.S_ref.value[0]} m^2") 
        # print(f"  Span: {span_input.value[0]} m")
        # print(f"  Root chord: {root_chord_input.value[0]} m")
        # print(f"\nCurrent:")
        # print(f"  Span: {wing_geom_qts.span.value[0]} m")
        # print(f"  Root chord: {wing_geom_qts.center_chord.value[0]} m")
        # print(f"  Tip chord left: {wing_geom_qts.left_tip_chord.value[0]} m")
        # print(f"  Tip chord right: {wing_geom_qts.right_tip_chord.value[0]} m")
        return




    def apply_incidence(self, incidence: Union[float, int, csdl.Variable]):
        """Apply the incidence angle to the wing geometry.
        
        Parameters
        ----------
        incidence : float, int, or csdl.Variable
            The incidence angle (degrees) to be applied to the wing.
        """
        wing_geometry = self.geometry
        if wing_geometry is None:
            raise ValueError("wing component cannot apply incidence since it does not have a geometry (i.e., geometry=None)")

        # Convert incidence angle to radians
        incidence_rad = np.radians(incidence)

        # Define the rotation axis (y-axis)
        rotation_axis = np.array([0., 1., 0.])

        # Rotate the wing geometry about the y-axis by the incidence angle
        wing_geometry.rotate(axis_origin=np.array([0., 0., 0.]), axis_vector=rotation_axis, angles=incidence_rad)

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot=False):
        """Set up the wing geometry (mainly the FFD)"""
        # Get the ffd block
        wing_ffd_block = self._ffd_block

        # Set up the ffd block
        self._setup_ffd_block(wing_ffd_block, parameterization_solver, plot=plot)

        if self.skip_ffd is False:
            print("DO WING FFD")
            # Get wing geometric quantities (as csdl variable)
            wing_geom_qts = self._extract_geometric_quantities_from_ffd_block()

            # Define the geometric constraints
            self._setup_ffd_parameterization(wing_geom_qts, ffd_geometric_variables)

        return 


        

    def _fit_surface(self, parametric_points:list, fitting_coords:list, function_space:lfs.FunctionSpace, mirror:bool, dependent:bool):
        """Fit a surface to the given parametric points."""
        if dependent:
            self._dependent_geometry_points.append({'parametric_points':parametric_points, 
                                                    'fitting_coords':fitting_coords,
                                                    'function_space':function_space,
                                                    'mirror':mirror})    
        fitting_values = self.geometry.evaluate(parametric_points)
        coefficients = function_space.fit(fitting_values, fitting_coords)
        function = lfs.Function(function_space, coefficients)
        if mirror:
            coeff_flip = np.eye(3)
            coeff_flip[1,1] = -1
            if len(coefficients.shape) != 2:
                coefficients = coefficients.reshape((-1, 3))
            coefficients = coefficients @ coeff_flip
            mirror_function = lfs.Function(function_space, coefficients)
            return function, mirror_function
        return function

    def _add_geometry(self, surf_index, function, name, append=None, geometry=None):
        """Add a function to the geometry object."""
        if geometry is None:
            geometry = self.geometry
        if not isinstance(function, tuple):
            function = (function,)
        if append is None:
            append = ""
        for i, f in enumerate(function):
            geometry.functions[surf_index+i] = f
            if i > 0:
                geometry.function_names[surf_index+i] = name + str(-append)
            else:
                geometry.function_names[surf_index+i] = name + str(append)
        return surf_index + len(function)
    


    