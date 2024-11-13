import numpy as np
from flight_simulator import ureg
import csdl_alpha as csdl
from typing import Union
import lsdo_geo

from flight_simulator.core.dynamics.axis import Axis



class AxisLsdoGeo(Axis):
    def __init__(self, name: str,
                 geometry: lsdo_geo.Geometry, parametric_coords: list,
                 origin: str,
                 phi: Union[ureg.Quantity, csdl.Variable] = np.array([0., ]) * ureg.radian,
                 theta: Union[ureg.Quantity, csdl.Variable] = np.array([0., ]) * ureg.radian,
                 psi: Union[ureg.Quantity, csdl.Variable] = np.array([0., ]) * ureg.radian,
                 sequence=np.array([3, 2, 1]), reference=None):
        self.geometry = geometry
        self.parametric_coords = parametric_coords
        translation = geometry.evaluate(parametric_coords)
        super().__init__(name=name, translation=translation,
                         phi=phi, theta=theta, psi=psi,
                         origin=origin, sequence=sequence, reference=reference)

    @property
    def translation(self):
        return self.geometry.evaluate(self.parametric_coords)

    @translation.setter
    def translation(self, translation_vector):
        pass
