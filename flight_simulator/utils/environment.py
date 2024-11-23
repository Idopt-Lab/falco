# -----------------------------------------------------------
# The classes and methods that will be used to define the atmospheric environment for a simulation.
# As originally conceived, typical properties will include local air properties (temperature, density, etc.) and local
# wind, as defined within various classes.
#
# -----------------------------------------------------------

import numpy as np
from typing import Union
from dataclasses import dataclass
import csdl_alpha as csdl
from flight_simulator import Q_, ureg
from vector import Vector
from flight_simulator.core.dynamics.axis import Axis
@dataclass
class AtmosphericStates(csdl.VariableGroup):
    density : Union[float, int, csdl.Variable] = 1.225
    speed_of_sound : Union[float, int, csdl.Variable] = 343
    temperature : Union[float, int, csdl.Variable] =  288.16
    pressure : Union[float, int, csdl.Variable] = 101325
    dynamic_viscosity : Union[float, int, csdl.Variable] = 1.735e-5


class SimpleAtmosphereModel:
    """Model class for simple atmosphere model."""
    def __init__(self, altitude:Union[ureg.Quantity, csdl.Variable]):
        self.altitude = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.altitude = altitude

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, altitude_input):
        if altitude_input is None:
            raise IOError
        elif isinstance(altitude_input, ureg.Quantity):
            altitude_si = altitude_input.to_base_units()
            self._altitude.set_value(altitude_si.magnitude)
        elif isinstance(altitude_input, csdl.Variable):
            self._altitude = altitude_input
        else:
            raise IOError

    def evaluate(self) -> AtmosphericStates:
        """Evaluate the atmospheric states at a given altitude"""
        h = self.altitude * 1e-3

        # Constants
        L = 6.5  # K/km
        R = 287
        T0 = 288.16
        P0 = 101325
        g0 = 9.81
        mu0 = 1.735e-5
        S1 = 110.4
        gamma = 1.4

        # Temperature
        T = - h * L + T0

        # Pressure
        P = P0 * (T / T0) ** (g0 / (L * 1e-3) / R)

        # Density
        rho = P / R / T
        # self.print_var(rho)

        # Dynamic viscosity (using Sutherland's law)
        mu = mu0 * (T / T0) ** (3 / 2) * (T0 + S1) / (T + S1)

        # speed of sound
        a = (gamma * R * T) ** 0.5

        atmos_states = AtmosphericStates(
            density=rho, speed_of_sound=a, temperature=T, pressure=P,
            dynamic_viscosity=mu
        )

        return atmos_states


class ConstantWind:
    def __init__(self, world_axis: Axis, wind_direction: Union[ureg.Quantity, csdl.Variable],
                 wind_speed: Union[ureg.Quantity, csdl.Variable]):

        self.wind_speed = Vector(wind_speed, world_axis)
        self.wind_direction = Vector(wind_direction, world_axis)
        self.axis = world_axis

    def update_wind(self, wind_direction: Union[ureg.Quantity, csdl.Variable], wind_speed: Union[ureg.Quantity, csdl.Variable]):
        """
        Defines a constant wind that originates from some direction and is blowing at a defined constant speed.

        :param wind_direction: Specified in degrees, with 0 deg. indicating northerly wind, 90 deg. easterly wind
        :param wind_speed: Wind speed, specified in a vector in the inertial reference frame.
        """
        # Wind velocity expressed in North-East-Down Reference Frame
        if isinstance(wind_direction, ureg.Quantity):
            direction = wind_direction.to_base_units()
            self.wind_speed.vector.set_value(direction)
        elif isinstance(wind_direction, Vector):
            self.wind_direction = wind_direction
        else:
            raise IOError
        if isinstance(wind_speed, ureg.Quantity):
            speed = wind_speed.to_base_units()
            self.wind_speed.vector.set_value(speed)
        elif isinstance(wind_speed, Vector):
            self.wind_speed = wind_speed
        else:
            raise IOError


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    atmosphere = SimpleAtmosphereModel(altitude=np.array([40000, ]) * ureg.feet)
    result_atm = atmosphere.evaluate()

    pass