from abc import ABC, abstractmethod


class Loads(ABC):
    """Basic class for Loads objects.

    All subclasses require implementation of the get_FM_refPoint method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_FM_refPoint(self, states_obj, controls_obj):
        """Use vehicle state and control objects to generate an estimate
        of forces and moments about a reference point."""
        pass


