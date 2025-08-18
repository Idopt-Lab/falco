import jax
import jax.numpy as jnp
from jax import jit
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
    class MomentOfInertiaComponents:
        """Holds the components of the inertia tensor.

        Attributes
        ----------
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz : jnp.ndarray or ureg.Quantity
            Components of the inertia tensor.
        """
        Ixx: jnp.ndarray
        Iyy: jnp.ndarray
        Izz: jnp.ndarray
        Ixy: jnp.ndarray
        Ixz: jnp.ndarray
        Iyz: jnp.ndarray

        def __post_init__(self):
            self._check_parameters()

        def _check_parameters(self):
            params = ['Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz']
            for name in params:
                value = getattr(self, name)
                if not isinstance(value, (jnp.ndarray, np.ndarray, ureg.Quantity)):
                    raise ValueError(f"Variable {name} must be of type jnp.ndarray, np.ndarray, or ureg.Quantity.")
                
                # Convert quantities to JAX arrays
                if isinstance(value, ureg.Quantity):
                    value_si = value.to_base_units()
                    setattr(self, name, jnp.array(value_si.magnitude))
                # Convert numpy arrays to JAX arrays
                elif isinstance(value, np.ndarray):
                    setattr(self, name, jnp.array(value))

    def __init__(
            self,
            axis: Union[Axis, AxisLsdoGeo],
            Ixx=Q_(0, 'kg*(m*m)'),
            Iyy=Q_(0, 'kg*(m*m)'),
            Izz=Q_(0, 'kg*(m*m)'),
            Ixy=Q_(0, 'kg*(m*m)'),
            Ixz=Q_(0, 'kg*(m*m)'),
            Iyz=Q_(0, 'kg*(m*m)'),
    ):
        """Initialize the mass moment of inertia tensor.

        Parameters
        ----------
        axis : Axis or AxisLsdoGeo
            The axis in which the inertia tensor is defined.
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz : ureg.Quantity or jnp.ndarray or np.ndarray, optional
            Components of the inertia tensor.
        """

        self.axis = axis

        self.mass_mi_components = self.MomentOfInertiaComponents(
            Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=Ixy, Ixz=Ixz, Iyz=Iyz
        )

        # Create the inertia tensor
        self.inertia_tensor = self._build_inertia_tensor(
            self.mass_mi_components.Ixx,
            self.mass_mi_components.Iyy,
            self.mass_mi_components.Izz,
            self.mass_mi_components.Ixy,
            self.mass_mi_components.Ixz,
            self.mass_mi_components.Iyz
        )
        return

    @staticmethod
    @jit
    def _build_inertia_tensor(Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
        """Construct the inertia tensor from its components.

        Parameters
        ----------
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz : jnp.ndarray
            Components of the inertia tensor.

        Returns
        -------
        jnp.ndarray
            The 3x3 inertia tensor.
        """
        # Create a zero matrix
        tensor = jnp.zeros((3, 3))
        
        # Update with diagonal elements
        tensor = tensor.at[0, 0].set(Ixx)
        tensor = tensor.at[1, 1].set(Iyy)
        tensor = tensor.at[2, 2].set(Izz)
        
        # Update with off-diagonal elements
        tensor = tensor.at[0, 1].set(-Ixy)
        tensor = tensor.at[1, 0].set(-Ixy)
        tensor = tensor.at[0, 2].set(-Ixz)
        tensor = tensor.at[2, 0].set(-Ixz)
        tensor = tensor.at[1, 2].set(-Iyz)
        tensor = tensor.at[2, 1].set(-Iyz)
        
        return tensor


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
                 mass=Q_(0, 'kg')):
        """Initialize the mass properties.

        Parameters
        ----------
        cg : Vector
            Center of gravity vector.
        inertia : MassMI
            Mass moment of inertia tensor.
        mass : ureg.Quantity or jnp.ndarray or np.ndarray, optional
            Mass value (default is 0 kg).

        Raises
        ------
        AssertionError
            If the CG and inertia tensor are not defined in the same axis.
        ValueError
            If mass is not a recognized type.
        """

        assert cg.axis.name == inertia.axis.name, "CG and inertia tensor must be defined in the same axis"

        if isinstance(mass, ureg.Quantity):
            value_si = mass.to_base_units()
            self.mass = jnp.array(value_si.magnitude)
        elif isinstance(mass, np.ndarray):
            self.mass = jnp.array(mass)
        elif isinstance(mass, jnp.ndarray):
            self.mass = mass
        else:
            raise ValueError("Mass must be a Quantity, numpy array, or JAX array")

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
        
        # Gravity constant
        g = 9.81

        th = self.states.states.theta
        ph = self.states.states.phi

        # Calculate gravity forces
        Fxg = -m * g * jnp.sin(th)
        Fyg = m * g * jnp.cos(th) * jnp.sin(ph)
        Fzg = m * g * jnp.cos(th) * jnp.cos(ph)
        forceVec = jnp.concatenate([Fxg, Fyg, Fzg], axis=0)

        # Calculate gravity moments
        Mgrav = jnp.cross(cg, forceVec)

        # Create force and moment vectors
        F_FD_BodyFixed = Vector(forceVec, axis=load_axis)
        M_FD_BodyFixed = Vector(jnp.concatenate([Mgrav[0],Mgrav[1],Mgrav[2]]),axis=load_axis)

        # Return the forces and moments
        loads = ForcesMoments(force=F_FD_BodyFixed, moment=M_FD_BodyFixed)
        return loads
    
    

if __name__ == "__main__":
    # Configure JAX for high precision
    jax.config.update("jax_enable_x64", True)

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    mi = MassMI(axis=inertial_axis)
    cg = Vector(vector=np.array([0, 0, 0])*ureg.meter, axis=inertial_axis)
    mass_properties = MassProperties(cg=cg, inertia=mi)
    
    # Print statements to display the objects
    print("\n=== Mass Moment of Inertia (mi) ===")
    print(f"Axis: {mi.axis.name}")
    print(f"Ixx: {mi.mass_mi_components.Ixx}")
    print(f"Iyy: {mi.mass_mi_components.Iyy}")
    print(f"Izz: {mi.mass_mi_components.Izz}")
    print(f"Ixy: {mi.mass_mi_components.Ixy}")
    print(f"Ixz: {mi.mass_mi_components.Ixz}")
    print(f"Iyz: {mi.mass_mi_components.Iyz}")
    print("Inertia Tensor:")
    print(mi.inertia_tensor)
    
    print("\n=== Center of Gravity (cg) ===")
    print(f"Axis: {cg.axis.name}")
    print(f"Vector: {cg.vector}")
    
    print("\n=== Mass Properties ===")
    print(f"Mass: {mass_properties.mass}")
    print(f"CG Vector: {mass_properties.cg_vector.vector}")
    print(f"Inertia Tensor:")
    print(mass_properties.inertia_tensor.inertia_tensor)