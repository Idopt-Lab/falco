import numpy as np
from flight_simulator import ureg, Q_
import csdl_alpha as csdl
from typing import Union, Literal
from enum import Enum
import lsdo_geo

from flight_simulator.utils.axis import validate_origin, Axis



class AxisLsdoGeo(Axis):
    def __init__(self, name: str, geometry: lsdo_geo.Geometry, parametric_coords: list,
                 angles: Union[ureg.Quantity, csdl.Variable],
                 origin: str = 'ref', sequence=np.array([3, 2, 1]), reference=None):
        self.geometry = geometry
        self.parametric_coords = parametric_coords
        translation = geometry.evaluate(parametric_coords)
        super().__init__(name, translation, angles, origin, sequence, reference)

    @property
    def translation(self):
        return self.geometry.evaluate(self.parametric_coords)

    @translation.setter
    def translation(self, translation_vector):
        if translation_vector is None:
            self._translation = None
        elif isinstance(translation_vector, ureg.Quantity):
            vector_si = translation_vector.to_base_units()
            self._translation.set_value(vector_si.magnitude)
            self._translation.add_tag(str(vector_si.units))
        elif isinstance(translation_vector, csdl.Variable):
            self._translation = translation_vector
        else:
            raise IOError