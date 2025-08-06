from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from jax import jit
from falco import ureg
from typing import Union, Literal
from enum import Enum
from dataclasses import dataclass


@jit
def concatenate_translation_vector(x, y, z):
    """JIT-compiled function to concatenate translation vectors."""
    return jnp.concatenate([x, y, z], axis=0)


@jit
def concatenate_euler_vector(phi, theta, psi):
    """JIT-compiled function to concatenate Euler angle vectors."""
    return jnp.concatenate([phi, theta, psi], axis=0)


@jit
def copy_array(arr):
    """JIT-compiled function to copy JAX arrays."""
    return jnp.copy(arr)


class ValidOrigins(Enum):
    Inertial = "inertial"
    OpenVSP = "openvsp_rotated_to_fd"


def axis_checkers(func):
    def test_origin_value(*args, **kwargs):
        # Check kwargs for origin
        origin_in = kwargs.get('origin')
        if origin_in not in ValidOrigins._value2member_map_:
            print('Axis origin "%s" not permitted' % origin_in)
            raise IOError
            kwargs['origin'] = ValidOrigins.Inertial.value
        func(*args, **kwargs)

    return test_origin_value


class Axis:
    """Represents a coordinate axis with translation and orientation.

    Supports translation from an origin and orientation via Euler angles.
    Used as a reference for expressing positions, velocities, and rotations.

    Attributes
    ----------
    name : str
        Name of the axis.
    origin : str
        Origin identifier (must be a ValidOrigins value).
    translation_from_origin : Axis.translation_from_origin or None
        Translation from the origin.
    translation_from_origin_vector : jnp.ndarray or None
        Translation vector [x, y, z] from the origin.
    translation : jnp.ndarray or None
        Alias for translation_from_origin_vector.
    euler_angles : Axis.euler_angles or None
        Euler angles (phi, theta, psi) for orientation.
    euler_angles_vector : jnp.ndarray or None
        Euler angles as a vector.
    sequence : any
        Euler rotation sequence.
    reference : object or None
        Reference axis or frame.
    """
    @dataclass
    class euler_angles:
        """Euler angles for axis orientation.

        Attributes
        ----------
        phi : jnp.ndarray
            Roll angle.
        theta : jnp.ndarray
            Pitch angle.
        psi : jnp.ndarray
            Yaw angle.
        """
        phi: jnp.ndarray
        theta: jnp.ndarray
        psi: jnp.ndarray

        def __post_init__(self):
            """Process values after initialization."""
            self.phi = self._process_value(self.phi)
            self.theta = self._process_value(self.theta)
            self.psi = self._process_value(self.psi)

        def _process_value(self, value):
            """Convert value to JAX array if needed."""
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                return jnp.array([value_si.magnitude])
            elif isinstance(value, (int, float, np.number)):
                return jnp.array([value])
            elif isinstance(value, (list, tuple, np.ndarray)):
                return jnp.array(value).reshape(-1)
            else:
                # Assume it's already a JAX array, ensure it's 1D
                value_array = jnp.array(value)
                if value_array.ndim == 0:
                    return jnp.array([value_array])
                return value_array

    @dataclass
    class translation_from_origin:
        """Translation from the origin for the axis.

        Attributes
        ----------
        x : jnp.ndarray
            X-coordinate of translation.
        y : jnp.ndarray
            Y-coordinate of translation.
        z : jnp.ndarray
            Z-coordinate of translation.
        """
        x: jnp.ndarray
        y: jnp.ndarray
        z: jnp.ndarray

        def __post_init__(self):
            """Process values after initialization."""
            self.x = self._process_value(self.x)
            self.y = self._process_value(self.y)
            self.z = self._process_value(self.z)

        def _process_value(self, value):
            """Convert value to JAX array if needed."""
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                return jnp.array([value_si.magnitude])
            elif isinstance(value, (int, float, np.number)):
                return jnp.array([value])
            elif isinstance(value, (list, tuple, np.ndarray)):
                return jnp.array(value).reshape(-1)
            else:
                # Assume it's already a JAX array, ensure it's 1D
                value_array = jnp.array(value)
                if value_array.ndim == 0:
                    return jnp.array([value_array])
                return value_array

    @axis_checkers
    def __init__(self, name: str,
                 origin: str,
                 x = None,
                 y = None,
                 z = None,
                 phi = None,
                 theta = None,
                 psi = None,
                 sequence=None,
                 reference=None):
        """Initialize an Axis object.

        Parameters
        ----------
        name : str
            Name of the axis.
        origin : str
            Origin identifier (must be a ValidOrigins value).
        x, y, z : ureg.Quantity or jnp.ndarray or np.ndarray or float or int, optional
            Translation from the origin.
        phi, theta, psi : ureg.Quantity or jnp.ndarray or np.ndarray or float or int, optional
            Euler angles for orientation.
        sequence : any, optional
            Euler rotation sequence.
        reference : object, optional
            Reference axis or frame.
        """

        self.name = name

        if x is not None:
            self.translation_from_origin = self.translation_from_origin(
                x=x, y=y, z=z
            )
            self.translation_from_origin_vector = concatenate_translation_vector(
                self.translation_from_origin.x, 
                self.translation_from_origin.y, 
                self.translation_from_origin.z
            )
            self.translation = self.translation_from_origin_vector
        else:
            self.translation_from_origin = None
            self.translation_from_origin_vector = None

        if phi is not None:
            self.euler_angles = self.euler_angles(phi=phi, theta=theta, psi=psi)
            self.euler_angles_vector = concatenate_euler_vector(
                self.euler_angles.phi, 
                self.euler_angles.theta, 
                self.euler_angles.psi
            )
        else:
            self.euler_angles = None
            self.euler_angles_vector = None

        self.sequence = sequence
        self.reference = reference
        self.origin = origin

    def copy(self, new_name: str = None):
        """Create a copy of the Axis object.

        Parameters
        ----------
        new_name : str, optional
            Name for the new Axis object.

        Returns
        -------
        Axis
            A new Axis object with the same properties as the original.
        """
        if new_name is None:
            self.name = self.name + "_copy"
        else:
            self.name = new_name

        # Copy translation variables if set
        if self.translation_from_origin is not None:
            new_x = copy_array(self.translation_from_origin.x)
            new_y = copy_array(self.translation_from_origin.y)
            new_z = copy_array(self.translation_from_origin.z)
        else:
            new_x = new_y = new_z = None

        # Copy Euler angle variables if set
        if hasattr(self, 'euler_angles') and self.euler_angles is not None:
            new_phi = copy_array(self.euler_angles.phi)
            new_theta = copy_array(self.euler_angles.theta)
            new_psi = copy_array(self.euler_angles.psi)
        else:
            new_phi = new_theta = new_psi = None

        return Axis(
            name=self.name,
            origin=self.origin,
            x=new_x,
            y=new_y,
            z=new_z,
            phi=new_phi,
            theta=new_theta,
            psi=new_psi,
            sequence=self.sequence,
            reference=self.reference
        )

    def jax_copy(self, new_name: str = None):
        """
        Create a deep copy of the current Axis object using JAX arrays.

        This method replicates all the Axis properties, including translation,
        Euler angles, sequence, reference, and origin, producing a new instance
        with the same configuration.

        Returns
        -------
        Axis
            A new Axis instance identical to the original.
        """
        if new_name is None:
            self.name = self.name + "_copy"
        else:
            self.name = new_name

        # Copy translation variables if set
        if self.translation_from_origin is not None:
            new_x = copy_array(self.translation_from_origin.x)
            new_y = copy_array(self.translation_from_origin.y)
            new_z = copy_array(self.translation_from_origin.z)
        else:
            new_x = new_y = new_z = None

        # Copy Euler angle variables if set
        if hasattr(self, 'euler_angles') and self.euler_angles is not None:
            new_phi = copy_array(self.euler_angles.phi)
            new_theta = copy_array(self.euler_angles.theta)
            new_psi = copy_array(self.euler_angles.psi)
        else:
            new_phi = new_theta = new_psi = None

        return Axis(
            name=self.name,
            origin=self.origin,
            x=new_x,
            y=new_y,
            z=new_z,
            phi=new_phi,
            theta=new_theta,
            psi=new_psi,
            sequence=self.sequence,
            reference=self.reference)


if __name__ == "__main__":
    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    axis = Axis(name='Reference Axis',
                x=np.array([10, ]) * ureg.meter,
                y=np.array([0, ]) * ureg.meter,
                z=np.array([0, ]) * ureg.meter,
                phi=np.array([0, ]) * ureg.degree,
                theta=np.array([5, ]) * ureg.degree,
                psi=np.array([0, ]) * ureg.degree,
                reference=inertial_axis,
                origin=ValidOrigins.Inertial.value)

    print('Axis translation: ', axis.translation_from_origin_vector)
    print('Axis angles: ', axis.euler_angles_vector)
    pass