import numpy as np
from flight_simulator import ureg, Q_
import csdl_alpha as csdl
from typing import Union, Literal
from enum import Enum
import lsdo_geo

from flight_simulator.utils.axis import validate_origin, Axis



class AxisLsdoGeo(Axis):
    def __init__(self, name: str,
                 geometry: lsdo_geo.Geometry, parametric_coords: list,
                 angles: Union[ureg.Quantity, csdl.Variable],
                 origin: str = 'ref', sequence=np.array([3, 2, 1]), reference=None):
        self.geometry = geometry
        self.parametric_coords = parametric_coords
        translation = geometry.evaluate(parametric_coords)
        super().__init__(name, translation, angles, origin, sequence, reference)

    @property
    def translation(self):
        return self.geometry.evaluate(self.parametric_coords)
