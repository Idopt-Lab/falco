from lsdo_function_spaces import FunctionSet
from flight_simulator.core.vehicle.component import Component
from lsdo_geo import Geometry

from typing import Union


class Aircraft(Component):
    """Aircraft container component"""
    def __init__(self, geometry: Union[FunctionSet, None] = None, **kwargs) -> None:
        kwargs["do_not_remake_ffd_block"] = True
        super().__init__(geometry, **kwargs)
        self._skip_ffd = False


    def _extract_geometric_quantities_from_ffd_block(self):
        """
        Aircraft specific implementation for extracting geometric quantities.
        For the aircraft container, we don't need to extract quantities since 
        individual components handle their own FFD.
        """
        return {}  # Return empty dict since aircraft container doesn't need geometric quantities

    def _setup_ffd_parameterization(self, geometric_quantities, ffd_geometric_variables):
        """
        Aircraft specific implementation for FFD parameterization.
        For the aircraft container, we don't need to set up FFD parameters since
        individual components handle their own FFD.
        """
        pass