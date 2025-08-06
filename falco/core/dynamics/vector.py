from falco.core.dynamics.axis import Axis
from falco.core.dynamics.axis_lsdogeo import AxisLsdoGeo
import jax
import jax.numpy as jnp
from falco import ureg
import numpy as np

class Vector:
    """Represents a 3D vector in a specified coordinate axis system.

    Supports initialization from a Pint Quantity (with units) or a JAX array.
    Stores the vector in SI units and associates it with an axis.

    Attributes
    ----------
    vector : jax.Array
        The 3-component vector in SI units.
    axis : Axis or AxisLsdoGeo
        The coordinate system in which the vector is defined.
    magnitude : jax.Array
        The Euclidean norm (magnitude) of the vector.
    units : str
        The units of the vector.
    """
    def __init__(self, vector, axis):
        """
        Initialize a Vector object.

        Parameters
        ----------
        vector : ureg.Quantity or jax.Array
            3-component vector, either as a Pint Quantity (with units) or a JAX array.
        axis : Axis or AxisLsdoGeo
            The coordinate system in which the vector is stored.

        Raises
        ------
        IOError
            If the vector is not a recognized type.
        TypeError
            If the axis is not an instance of Axis or AxisLsdoGeo.
        Exception
            If the axis is not assigned correctly.
        """

        if isinstance(vector, ureg.Quantity):
            vector_si = vector.to_base_units()
            self.vector = jnp.array(vector_si.magnitude)
            self.units = str(vector_si.units)
        elif isinstance(vector, (jnp.ndarray, np.ndarray)):
            self.vector = jnp.array(vector)
            self.units = 'dimensionless'  # Default units for raw arrays
        else:
            raise IOError("Vector must be a ureg.Quantity or a numpy/jax array")

        if not isinstance(axis, (Axis, AxisLsdoGeo)):
            raise TypeError("axis must be an instance of Axis or AxisLSDOGeo")
        self.axis = axis
        self.magnitude = jnp.linalg.norm(self.vector)
        # Ensure axis is assigned correctly
        if not hasattr(self, 'axis'):
            raise Exception("Axis not assigned correctly.")


    def __str__(self):
        """Return a string representation of the vector, including values, units, and axis name.

        Returns
        -------
        str
            String representation of the vector.
        """
        print_string = """Vector: %s \nUnit: %s \nAxis: %s""" % \
                       (np.array_str(np.around(np.array(self.vector), 2)),
                        self.units,
                        self.axis.name)
        return print_string


# JAX JIT compiled functions for vector operations
@jax.jit
def vector_magnitude(vector):
    """JIT-compiled function to calculate vector magnitude."""
    return jnp.linalg.norm(vector)


@jax.jit
def vector_dot_product(vec1, vec2):
    """JIT-compiled function to calculate dot product of two vectors."""
    return jnp.dot(vec1, vec2)


@jax.jit
def vector_cross_product(vec1, vec2):
    """JIT-compiled function to calculate cross product of two vectors."""
    return jnp.cross(vec1, vec2)


@jax.jit
def vector_normalize(vector):
    """JIT-compiled function to normalize a vector."""
    return vector / jnp.linalg.norm(vector)


if __name__ == "__main__":
    # Example usage demonstrating JAX-based Vector class
    from falco.core.dynamics.axis import Axis, ValidOrigins
    test_axis = Axis(name="test_axis", origin=ValidOrigins.Inertial.value)

    
    # Test with JAX array
    print("Testing Vector with JAX array:")
    jax_vector = jnp.array([1.0, 2.0, 3.0])
    vec1 = Vector(jax_vector, test_axis)
    print(vec1)
    print(f"Magnitude: {vec1.magnitude}")
    print()
    
    # Test with numpy array
    print("Testing Vector with NumPy array:")
    np_vector = np.array([4.0, 5.0, 6.0])
    vec2 = Vector(np_vector, test_axis)
    print(vec2)
    print(f"Magnitude: {vec2.magnitude}")
    print()
    
    # Demonstrate JIT-compiled functions
    print("Testing JIT-compiled vector operations:")
    mag1 = vector_magnitude(vec1.vector)
    mag2 = vector_magnitude(vec2.vector)
    print(f"JIT magnitude of vec1: {mag1}")
    print(f"JIT magnitude of vec2: {mag2}")
    
    dot_prod = vector_dot_product(vec1.vector, vec2.vector)
    print(f"Dot product: {dot_prod}")
    
    cross_prod = vector_cross_product(vec1.vector, vec2.vector)
    print(f"Cross product: {cross_prod}")
    
    normalized_vec1 = vector_normalize(vec1.vector)
    print(f"Normalized vec1: {normalized_vec1}")
    print(f"Magnitude of normalized vec1: {vector_magnitude(normalized_vec1)}")
    
    # Test with Pint quantities if ureg is available
    velocity_vector = ureg.Quantity(jnp.array([10.0, 20.0, 30.0]),'m/s')
    vec_with_units = Vector(velocity_vector, test_axis)
    print(f"\nVector with units:")
    print(vec_with_units)

    
    print("\nAll tests completed successfully!")