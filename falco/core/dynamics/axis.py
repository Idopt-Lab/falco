import numpy as np
import jax
import jax.numpy as jnp
from falco import ureg
from typing import Union, Literal
from enum import Enum
from dataclasses import dataclass


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
        func(*args, **kwargs)

    return test_origin_value


# JIT-compiled pure functions for vector operations
@jax.jit
def create_translation_vector(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Create translation vector with JIT compilation."""
    return jnp.concatenate([x, y, z], axis=0)


@jax.jit
def create_euler_angles_vector(phi: jax.Array, theta: jax.Array, psi: jax.Array) -> jax.Array:
    """Create Euler angles vector with JIT compilation."""
    return jnp.concatenate([phi, theta, psi], axis=0)


@jax.jit
def reshape_to_vector(arr: jax.Array) -> jax.Array:
    """Reshape array to vector with JIT compilation."""
    return arr.reshape(1)


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
    translation_from_origin_vector : jax.Array or None
        Translation vector [x, y, z] from the origin.
    translation : jax.Array or None
        Alias for translation_from_origin_vector.
    euler_angles : Axis.euler_angles or None
        Euler angles (phi, theta, psi) for orientation.
    euler_angles_vector : jax.Array or None
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
        phi : jax.Array
            Roll angle.
        theta : jax.Array
            Pitch angle.
        psi : jax.Array
            Yaw angle.
        """
        phi: jax.Array
        theta: jax.Array
        psi: jax.Array

    @dataclass
    class translation_from_origin:
        """Translation from the origin for the axis.

        Attributes
        ----------
        x : jax.Array
            X-coordinate of translation.
        y : jax.Array
            Y-coordinate of translation.
        z : jax.Array
            Z-coordinate of translation.
        """
        x: jax.Array
        y: jax.Array
        z: jax.Array

    @axis_checkers
    def __init__(self, name: str,
                 origin: str,
                 x: Union[ureg.Quantity, jax.Array] = None,
                 y: Union[ureg.Quantity, jax.Array] = None,
                 z: Union[ureg.Quantity, jax.Array] = None,
                 phi: Union[ureg.Quantity, jax.Array] = None,
                 theta: Union[ureg.Quantity, jax.Array] = None,
                 psi: Union[ureg.Quantity, jax.Array] = None,
                 sequence=None,
                 reference=None):
        """Initialize an Axis object.

        Parameters
        ----------
        name : str
            Name of the axis.
        origin : str
            Origin identifier (must be a ValidOrigins value).
        x, y, z : ureg.Quantity or jax.Array, optional
            Translation from the origin.
        phi, theta, psi : ureg.Quantity or jax.Array, optional
            Euler angles for orientation.
        sequence : any, optional
            Euler rotation sequence.
        reference : object, optional
            Reference axis or frame.
        """

        self.name = name

        if x is not None:
            # Convert to JAX arrays
            x_val = self._to_jax_array(x)
            y_val = self._to_jax_array(y)
            z_val = self._to_jax_array(z)
            
            self.translation_from_origin = self.translation_from_origin(
                x=x_val, y=y_val, z=z_val
            )
            # Use JIT-compiled function for vector creation
            self.translation_from_origin_vector = create_translation_vector(
                self.translation_from_origin.x, 
                self.translation_from_origin.y, 
                self.translation_from_origin.z
            )
            self.translation = self.translation_from_origin_vector
        else:
            self.translation_from_origin = None
            self.translation_from_origin_vector = None

        if phi is not None:
            # Convert to JAX arrays
            phi_val = self._to_jax_array(phi)
            theta_val = self._to_jax_array(theta)
            psi_val = self._to_jax_array(psi)
            
            self.euler_angles = self.euler_angles(phi=phi_val, theta=theta_val, psi=psi_val)
            # Use JIT-compiled function for vector creation
            self.euler_angles_vector = create_euler_angles_vector(
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

    def _to_jax_array(self, value):
        """Convert value to JAX array."""
        if value is None:
            raise ValueError("Cannot convert None to JAX array")
        if isinstance(value, ureg.Quantity):
            value_si = value.to_base_units()
            return reshape_to_vector(jnp.array(value_si.magnitude))
        elif isinstance(value, jax.Array):
            return reshape_to_vector(value)
        else:
            return reshape_to_vector(jnp.array(value))

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
            new_name = self.name + "_copy"
        else:
            self.name = new_name

        # Copy translation variables if set
        if self.translation_from_origin is not None:
            new_x = self.translation_from_origin.x
            new_y = self.translation_from_origin.y
            new_z = self.translation_from_origin.z
        else:
            new_x = new_y = new_z = None

        # Copy Euler angle variables if set
        if hasattr(self, 'euler_angles') and self.euler_angles is not None:
            new_phi = self.euler_angles.phi
            new_theta = self.euler_angles.theta
            new_psi = self.euler_angles.psi
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
