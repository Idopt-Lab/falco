from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit


class Loads(ABC):
    """Basic class for Loads objects.

    All subclasses require implementation of the get_FM_refPoint method.
    """
    @abstractmethod
    def get_FM_localAxis(self, states, controls, axis):
        """Use vehicle state and control objects to generate an estimate
        of forces and moments about a reference point."""
        pass


# class JaxLoads(Loads):
#     """
#     Loads obtained by calling code that is written in JAX
#     """
#     def __init__(self, states, controls, **kwargs):
#         super().__init__(states=states, controls=controls)
#
#     @abstractmethod
#     def get_FM_localAxis(self):
#         """Use vehicle state and control objects to generate an estimate
#         of forces and moments about a reference point."""
#         state_vector: jax.Array = self.states.state_vector
#         control_vector: jax.Array = self.controls.control_vector
#
#         pass
#
#
# class FunctionalLoads(Loads):
#     """
#     Loads implemented as JAX-compatible functions
#     """
#
#     def __init__(self, states, controls, *args, **kwargs):
#         self.states = states
#         self.controls = controls
#
#     @jit
#     def compute_forces_moments(self, state_vector, control_vector):
#         """
#         Compute forces and moments from state and control vectors.
#         
#         Parameters
#         ----------
#         state_vector : jax.Array
#             Vector of state variables
#         control_vector : jax.Array
#             Vector of control variables
#             
#         Returns
#         -------
#         tuple
#             (forces, moments) as JAX arrays
#         """
#         # Implement in subclasses
#         raise NotImplementedError("Subclasses must implement compute_forces_moments")
#
#     @abstractmethod
#     def get_FM_localAxis(self):
#         """Use vehicle state and control objects to generate an estimate
#         of forces and moments about a reference point."""
#
#         state_vector: jax.Array = self.states.state_vector
#         control_vector: jax.Array = self.controls.control_vector
#         
#         # Call compute_forces_moments to get forces and moments
#         forces, moments = self.compute_forces_moments(state_vector, control_vector)
#         
#         return forces, moments


if __name__ == "__main__":
    # Configure JAX
    jax.config.update("jax_enable_x64", True)
    
    # Example usage could be added here

