from flight_simulator.core.vehicle.component import Component

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
            sweep = csdl.Variable(name=f"{self._name}_sweep", value=0)
            self.parameters.sweep = sweep


        FF = (1 + 0.6 / x_c_m + 100 * (t_o_c) ** 4) * csdl.cos(sweep) ** 0.28
        # self.quantities.drag_parameters.form_factor = FF
  


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
        
        if self._orientation == "horizontal":
        
            LE_center = wing_geometry.evaluate(self._LE_mid_point)
            TE_center = wing_geometry.evaluate(self._TE_mid_point)

            # Add the user_specified axis location
            actuation_center = csdl.linear_combination(
                LE_center, TE_center, 1, np.array([1 -axis_location]), np.array([axis_location])
            ).flatten()


            var = csdl.Variable(shape=(3, ), value=np.array([0., 1., 0.]))  
        else:
            LE_root = wing_geometry.evaluate(self._LE_root_point)
            TE_root = wing_geometry.evaluate(self._TE_root_point)
                # Add the user_specified axis location
            actuation_center = csdl.linear_combination(
                LE_root, TE_root, 1, np.array([1 -axis_location]), np.array([axis_location])
            ).flatten()

            var = csdl.Variable(shape=(3, ), value=np.array([0., 0., 1.]))
        

        # Compute the actuation axis vector
        axis_origin = actuation_center - var
        axis_vector = actuation_center + var - axis_origin


        # Rotate the component about the axis
        wing_geometry.rotate(axis_origin=axis_origin, axis_vector=axis_vector / csdl.norm(axis_vector), angles=angle)



    

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


            qc_center = 0.75 * LE_center + 0.25 * TE_center

            # Tip
            LE_left = self.geometry.evaluate(self._LE_left_point)
            TE_left = self.geometry.evaluate(self._TE_left_point)


            qc_left = 0.75 * LE_left + 0.25 * TE_left

            # Right side 
            LE_right = self.geometry.evaluate(self._LE_right_point)
            TE_right = self.geometry.evaluate(self._TE_right_point)


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
                right_tip_chord=csdl.norm(tip_chord),
                sweep_angle_left=sweep_angle,
                sweep_angle_right=None,
                dihedral_angle_left=None,
                dihedral_angle_right=None,
            )

        return wing_geometric_qts
  



    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot=False):
        """Set up the wing geometry (mainly the FFD)"""
        geometry=self.geometry
        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (lfs.FunctionSet)):
                raise TypeError(f"wing gometry must be of type {lfs.FunctionSet}")
            else:
                # Set the wetted area
                self.parameters.S_wet = self.quantities.surface_area
                num_ffd_sections = 3
                num_wing_sections = 2
                ffd_block = construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, (num_ffd_sections), 2), degree=(1,1,1))

                # Compute the corner points of the wing 
                if self._orientation == "horizontal":
                    self._LE_left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.8, 0., 0.5])), plot=False, extrema=True)
                    self._LE_mid_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])), plot=False, extrema=True)
                    self._LE_right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.8, 1., 0.5])), plot=False, extrema=True)

                    self._TE_left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0., 0.5])), plot=False, extrema=True)
                    self._TE_mid_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])),  plot=False, extrema=True)
                    self._TE_right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 1.0, 0.5])), plot=False, extrema=True)

                else:
                    self._LE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.])), direction=np.array([-1., 0., 0.]), plot=False, extrema=False)
                    self._LE_root_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 1.])), direction=np.array([-1., 0., 0.]), plot=False, extrema=False)

                    self._TE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.])), plot=False, extrema=True)
                    self._TE_root_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 1.])), plot=False, extrema=True)

                    self.LE_root = geometry.evaluate(self._LE_root_point)
                    self.TE_root = geometry.evaluate(self._TE_root_point)




                ffd_sectional_parameterization = VolumeSectionalParameterization(
                    name="ffd_sectional_parameterization",
                    parameterized_points=ffd_block.coefficients,
                    principal_parametric_dimension=1,
                )
                # ffd_sectional_parameterization.plot()

                space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
                space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

                chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                                        coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0., 0., 0.])), name='chord_stretching_b_spline_coefficients')

                wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                            coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([0., 0.])), name='wingspan_stretching_b_spline_coefficients')

                sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                                            coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0.0, 0.0, 0.0])), name='sweep_translation_b_spline_coefficients')

                dihedral_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                                            coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0.0, 0.0, 0.0])), name='dihedral_translation_b_spline_coefficients')


                twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline_coefficients')

                # endregion

                # region Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver
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

                dihedral_translation_sectional_parameters = dihedral_translation_b_spline.evaluate(
                    parametric_b_spline_inputs
                )

                twist_sectional_parameters = twist_b_spline.evaluate(
                    parametric_b_spline_inputs
                )


                
                sectional_parameters = VolumeSectionalParameterizationInputs()
                if self._orientation == "horizontal":
                    sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
                    sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
                    sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
                    sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translation_sectional_parameters)
                    sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)

                else:
                    sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
                    sectional_parameters.add_sectional_translation(axis=2, translation=wingspan_stretch_sectional_parameters)
                    sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)


                ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False) 
                ffd_coefficients.name = 'ffd_coefficients'


                geometry_coefficients = ffd_block.evaluate(ffd_coefficients, plot=False)
                geometry.set_coefficients(geometry_coefficients)
  

                wing_geom_qts = self._extract_geometric_quantities_from_ffd_block()

                wingspan = wing_geom_qts.span
                root_chord = wing_geom_qts.center_chord
                tip_chord_left = wing_geom_qts.left_tip_chord
                tip_chord_right = wing_geom_qts.right_tip_chord

                spanwise_direction_left = 0.25*wing_geom_qts.left_tip_chord - 0.25*wing_geom_qts.center_chord
                spanwise_direction_right = 0.25*wing_geom_qts.right_tip_chord - 0.25*wing_geom_qts.center_chord
                sweep_angle_left = csdl.arccos(-spanwise_direction_left / csdl.norm(spanwise_direction_left))
                sweep_angle_right = csdl.arccos(spanwise_direction_right / csdl.norm(spanwise_direction_right))


                # Create Newton solver for inner optimization
                chord_stretching_b_spline.coefficients.add_name('chord_stretching_b_spline_coefficients')
                wingspan_stretching_b_spline.coefficients.add_name('wingspan_stretching_b_spline_coefficients')
                sweep_translation_b_spline.coefficients.add_name('sweep_translation_b_spline_coefficients')
                dihedral_translation_b_spline.coefficients.add_name('dihedral_translation_b_spline_coefficients')
                twist_b_spline.coefficients.add_name('twist_b_spline_coefficients')



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
                    tip_chord_right_input = tip_chord_left_input

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
                    tip_chord_right_input = tip_chord_left_input
                
                elif self.parameters.span is not None and self.parameters.AR is not None:
                    if self.parameters.taper_ratio is None:
                        taper_ratio = 1.
                    else:
                        taper_ratio = self.parameters.taper_ratio

                    if not isinstance(self.parameters.AR , csdl.Variable):
                        self.parameters.AR = csdl.Variable(shape=(1, ), value=self.parameters.AR)

                    if not isinstance(self.parameters.span , csdl.Variable):
                        self.parameters.span = csdl.Variable(shape=(1, ), value=self.parameters.span)

                    S_ref = self.parameters.span**2 / self.parameters.AR
                    self.parameters.S_ref = S_ref
                    span_input = self.parameters.span
                    root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
                    tip_chord_left_input = root_chord_input * taper_ratio 
                    tip_chord_right_input = tip_chord_left_input

            

                parameterization_solver.add_parameter(chord_stretching_b_spline.coefficients)
                parameterization_solver.add_parameter(wingspan_stretching_b_spline.coefficients)
                parameterization_solver.add_parameter(sweep_translation_b_spline.coefficients)
                parameterization_solver.add_parameter(dihedral_translation_b_spline.coefficients)
                parameterization_solver.add_parameter(twist_b_spline.coefficients)



            if self._orientation == "horizontal":
                ffd_geometric_variables.add_variable(wingspan, span_input)
                ffd_geometric_variables.add_variable(root_chord, root_chord_input)
                ffd_geometric_variables.add_variable(tip_chord_left, tip_chord_left_input)
                ffd_geometric_variables.add_variable(tip_chord_right, tip_chord_left_input)
            
                if self.parameters.dihedral is not None:
                    dihedral_input = self.parameters.dihedral
                    ffd_geometric_variables.add_variable(wing_geom_qts.dihedral_angle_left, dihedral_input)
                    ffd_geometric_variables.add_variable(wing_geom_qts.dihedral_angle_right, dihedral_input)

            else:
                ffd_geometric_variables.add_variable(wingspan, span_input)
                ffd_geometric_variables.add_variable(root_chord, root_chord_input)
                ffd_geometric_variables.add_variable(tip_chord_left, tip_chord_left_input)

            if self.parameters.sweep is not None:
                ffd_geometric_variables.add_variable(sweep_angle_left, self.parameters.sweep)
                if wing_geom_qts.sweep_angle_right is not None:
                    ffd_geometric_variables.add_variable(sweep_angle_right, self.parameters.sweep)

            print("Wingspan: ", wingspan.value)
            print("Root Chord: ", root_chord.value)
            print("Tip Chord Left: ", tip_chord_left.value)
            print("Tip Chord Right: ", tip_chord_right.value)
            print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
            print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
            print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
            print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
            print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)

        return

     
    


    