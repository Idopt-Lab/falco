from abc import ABC, abstractmethod

import numpy as np

import csdl_alpha as csdl


class Loads(ABC):
    """Basic class for Loads objects.

    All subclasses require implementation of the get_FM_refPoint method.
    """
    @abstractmethod
    def get_FM_localAxis(self):
        """Use vehicle state and control objects to generate an estimate
        of forces and moments about a reference point."""
        pass


class CsdlLoads(Loads):
    """
    Loads obtained by calling code that is written in CSDL
    """
    def __init__(self, states, controls, **kwargs):
        super().__init__(states=states, controls=controls)

    @abstractmethod
    def get_FM_localAxis(self):
        """Use vehicle state and control objects to generate an estimate
        of forces and moments about a reference point."""
        state_vector: csdl.Variable = self.states.state_vector
        control_vector: csdl.Variable = self.controls.control_vector

        pass


class NonCsdlLoads(csdl.CustomExplicitOperation, Loads):
    """
    Loads obtained by calling code that is not CSDL
    """

    def __init__(self, states, controls, *args, **kwargs):
        super().__init__()
        self.states = states
        self.controls = controls

    def evaluate(self):
        # Define the inputs and outputs for the custom operation
        self.declare_input('state_vector', self.states.state_vector)
        self.declare_input('control_vector', self.controls.control_vector)

        # Declare output variables
        forces = self.create_output('forces', (3, ))
        moments = self.create_output('moments', (3, ))

        # Construct output of the model
        output = csdl.VariableGroup()
        output.forces = forces
        output.moments = moments

        return output

    def compute(self, input_vals, output_vals):
        state_vector = input_vals['state_vector']
        control_vector = input_vals['control_vector']

        # Perform the computation to get forces and moments
        forces, moments = self.compute_forces_moments(state_vector, control_vector)

        output_vals['forces'] = forces
        output_vals['moments'] = moments

    @abstractmethod
    def get_FM_localAxis(self):
        """Use vehicle state and control objects to generate an estimate
        of forces and moments about a reference point."""

        state_vector: np.ndarray = self.states.state_vector.value
        control_vector: np.ndarray = self.controls.control_vector.value
        pass

