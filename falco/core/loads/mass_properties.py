import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from falco import ureg, Q_
from falco.core.loads.loads import Loads
from falco.core.dynamics.axis import Axis
from falco.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from falco.core.loads.forces_moments import Vector, ForcesMoments
from falco.core.dynamics.axis import Axis, ValidOrigins


class MassMI:
    """Represents the mass moment of inertia tensor for a body.

    Attributes
    ----------
    axis : Axis or AxisLsdoGeo
        The axis in which the inertia tensor is defined.
    mass_mi_components : MomentOfInertiaComponents
        The individual components of the inertia tensor.
    inertia_tensor : csdl.Variable
        The 3x3 inertia tensor.
    """

    @dataclass
    class MomentOfInertiaComponents(csdl.VariableGroup):
        """Holds the components of the inertia tensor.

        Attributes
        ----------
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz : csdl.Variable or ureg.Quantity
            Components of the inertia tensor.
        """
        Ixx: csdl.Variable
        Iyy: csdl.Variable
        Izz: csdl.Variable
        Ixy: csdl.Variable
        Ixz: csdl.Variable
        Iyz: csdl.Variable

        def define_checks(self):
            self.add_check('Ixx', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Iyy', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Izz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Ixy', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Ixz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Iyz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

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

    def __init__(
            self,
            axis: Union[Axis, AxisLsdoGeo],
            Ixx: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Iyy: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Izz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Ixy: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Ixz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Iyz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
    ):
        """Initialize the mass moment of inertia tensor.

        Parameters
        ----------
        axis : Axis or AxisLsdoGeo
            The axis in which the inertia tensor is defined.
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz : ureg.Quantity or csdl.Variable, optional
            Components of the inertia tensor.
        """

        self.axis = axis

        self.mass_mi_components = self.MomentOfInertiaComponents(
            Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=Ixy, Ixz=Ixz, Iyz=Iyz
        )

        self.inertia_tensor = csdl.Variable(shape=(3, 3), value=0.)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[0, 0], self.mass_mi_components.Ixx)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[1, 1], self.mass_mi_components.Iyy)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[2, 2], self.mass_mi_components.Izz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[0, 1], -self.mass_mi_components.Ixy)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[1, 0], -self.mass_mi_components.Ixy)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[0, 2], -self.mass_mi_components.Ixz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[2, 0], -self.mass_mi_components.Ixz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[1, 2], -self.mass_mi_components.Iyz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[2, 1], -self.mass_mi_components.Iyz)
        return


class MassProperties:
    """Represents the mass properties of a body, including mass, center of gravity, and inertia tensor.

    Attributes
    ----------
    mass : csdl.Variable
        The mass of the body.
    cg_vector : Vector
        The center of gravity vector.
    inertia_tensor : MassMI
        The mass moment of inertia tensor.
    """
    def __init__(self,
                 cg: Vector, 
                 inertia: MassMI,
                 mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg')):
        """Initialize the mass properties.

        Parameters
        ----------
        cg : Vector
            Center of gravity vector.
        inertia : MassMI
            Mass moment of inertia tensor.
        mass : ureg.Quantity or csdl.Variable, optional
            Mass value (default is 0 kg).

        Raises
        ------
        AssertionError
            If the CG and inertia tensor are not defined in the same axis.
        IOError
            If mass is not a recognized type.
        """

        assert cg.axis.name == inertia.axis.name

        if isinstance(mass, ureg.Quantity):
            value_si = mass.to_base_units()
            self.mass = csdl.Variable(value=value_si.magnitude, shape=(1,), name='mass')
            self.mass.add_tag(tag=str(value_si.units))
        elif isinstance(mass, csdl.Variable):
            self.mass = mass
        else:
            raise IOError

        self.cg_vector = cg
        self.inertia_tensor = inertia

    # @staticmethod
    # def create_default_mass_properties() -> "MassProperties":
    #     default_axis = Axis(name="Default Axis", origin=ValidOrigins.Inertial.value)
    #     default_inertia = MassMI(axis=default_axis)
    #     default_cg = Vector(vector=Q_(np.zeros(3), 'm'), axis=default_axis)
    #     return MassProperties(cg=default_cg, inertia=default_inertia)


class GravityLoads(Loads):
    """Computes gravity-induced forces and moments for a body.

    Attributes
    ----------
    states : object
        The flight dynamics state object.
    controls : object
        The control inputs.
    mass_properties : MassProperties
        The mass properties of the body.
    """

    def __init__(self, fd_state, controls, mass_properties):
        """Initialize the gravity loads object.

        Parameters
        ----------
        fd_state : object
            The flight dynamics state.
        controls : object
            The control inputs.
        mass_properties : MassProperties
            The mass properties of the body.
        """
        self.states = fd_state
        self.controls = controls
        self.mass_properties = mass_properties


    def get_FM_localAxis(self):
        """Compute gravity forces and moments about a reference point.

        Uses the vehicle state and mass properties to estimate the gravity force vector and moment.

        Returns
        -------
        ForcesMoments
            The gravity-induced forces and moments in the local axis.
        """
        # Store the states and mass properties
        load_axis = self.states.axis
        cg = self.mass_properties.cg_vector.vector
        m = self.mass_properties.mass
        
        # Gravity FM
        g=9.81

        th = self.states.states.theta
        ph = self.states.states.phi

        Fxg = -m * g * csdl.sin(th)
        Fyg = m * g * csdl.cos(th) * csdl.sin(ph)
        Fzg = m * g * csdl.cos(th) * csdl.cos(ph)
        forceVec = csdl.concatenate([Fxg, Fyg, Fzg])

        Mgrav = csdl.cross(cg, forceVec)

        F_FD_BodyFixed = Vector(forceVec,axis=load_axis)
        M_FD_BodyFixed = Vector(csdl.concatenate([Mgrav[0],Mgrav[1],Mgrav[2]]),axis=load_axis)

        loads = ForcesMoments(force=F_FD_BodyFixed, moment=M_FD_BodyFixed)
        return loads
    
    

if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    mi = MassMI(axis=inertial_axis)
    cg = Vector(vector=np.array([0, 0, 0])*ureg.meter, axis=inertial_axis)
    mass_properties = MassProperties(cg=cg, inertia=mi)