import csdl_alpha as csdl
from typing import Union
import numpy as np


class MassProperties:
    def __init__(
            self,
            mass: Union[float, int, csdl.Variable, None] = None,
            cg_vector: Union[np.ndarray, csdl.Variable, None] = None,
            inertia_tensor: Union[np.ndarray, csdl.Variable, None] = None,
    ):
        self.mass = mass
        self.cg_vector = cg_vector
        self.inertia_tensor = inertia_tensor