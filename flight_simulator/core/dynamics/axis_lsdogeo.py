import numpy as np
from flight_simulator import ureg
import csdl_alpha as csdl
from typing import Union
import lsdo_geo

from flight_simulator.core.dynamics.axis import Axis

class AxisLsdoGeo(Axis):
    """Represents an axis defined by a parametric location on an LSDO-Geo geometry.

    Inherits from Axis and uses a geometry object to evaluate the translation vector
    at specified parametric coordinates.

    Attributes
    ----------
    geometry : lsdo_geo.Geometry
        The LSDO-Geo geometry object.
    parametric_coords : list
        Parametric coordinates on the geometry.
    translation : np.ndarray
        The evaluated translation vector at the parametric coordinates.
    """
    def __init__(self, name: str, parametric_coords: list,
                 geometry: lsdo_geo.Geometry,
                 origin: str,
                 phi: Union[ureg.Quantity, csdl.Variable] = np.array([0., ]) * ureg.radian,
                 theta: Union[ureg.Quantity, csdl.Variable] = np.array([0., ]) * ureg.radian,
                 psi: Union[ureg.Quantity, csdl.Variable] = np.array([0., ]) * ureg.radian,
                 sequence=np.array([3, 2, 1]), reference=None):
        """Initialize an AxisLsdoGeo object.

        Parameters
        ----------
        name : str
            Name of the axis.
        parametric_coords : list
            Parametric coordinates on the geometry.
        geometry : lsdo_geo.Geometry
            LSDO-Geo geometry object.
        origin : str
            Origin identifier.
        phi : ureg.Quantity or csdl.Variable, optional
            Roll angle (default is 0 rad).
        theta : ureg.Quantity or csdl.Variable, optional
            Pitch angle (default is 0 rad).
        psi : ureg.Quantity or csdl.Variable, optional
            Yaw angle (default is 0 rad).
        sequence : np.ndarray, optional
            Euler rotation sequence (default is [3, 2, 1]).
        reference : object, optional
            Reference axis or frame.
        """
        self.geometry = geometry
        self.parametric_coords = parametric_coords
        self.translation = self.geometry.evaluate(self.parametric_coords)
        

        super().__init__(name=name, x=self.translation[0], y=self.translation[1], z=self.translation[2],
                         phi=phi, theta=theta, psi=psi,
                         origin=origin, sequence=sequence, reference=reference)
        
