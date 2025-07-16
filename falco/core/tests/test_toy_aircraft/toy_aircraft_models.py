from typing import Union
from typing import List
import numpy as np

import csdl_alpha as csdl

from falco.core.vehicle.controls.vehicle_control_system import (
    VehicleControlSystem, ControlSurface, PropulsiveControl)
from falco.core.loads.loads import Loads
from falco import Q_
from falco.core.dynamics.vector import Vector
from falco.core.loads.forces_moments import ForcesMoments


class ToyAircraftControlSystem(VehicleControlSystem):
    def __init__(self):
        self.elevator = ControlSurface('elevator', lb=-26, ub=28)
        self.aileron = ControlSurface('aileron', lb=-15, ub=20)
        self.rudder = ControlSurface('rudder', lb=-16, ub=16)
        self.engine = PropulsiveControl(name='engine')

        self.u = csdl.concatenate((self.aileron.deflection,
                                   self.elevator.deflection,
                                   self.rudder.deflection,
                                   self.engine.throttle), axis=0)

        super().__init__(pitch_control=[self.elevator],
                         roll_control=[self.aileron],
                         yaw_control=[self.rudder],
                         throttle_control=[self.engine])

    @property
    def control_order(self) -> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']


class ToyAircraftAerodynamics(Loads):

    def __init__(self, S: Q_=Q_(10, 'm**2'), c: Q_=Q_(1, 'm'), b: Q_=Q_(5, 'm')):
        self.S = csdl.Variable(name='S', shape=(1,), value=S.to_base_units())
        self.c = csdl.Variable(name='c', shape=(1,), value=c.to_base_units())
        self.b = csdl.Variable(name='b', shape=(1,), value=b.to_base_units())

    def get_FM_localAxis(self, states, controls, axis):

        Cl_alpha = 2 * np.pi

        AR = self.b**2/self.S
        e = 1
        K = 1 / (np.pi * e * AR)
        CL = Cl_alpha * states.alpha

        CD0 = 0.
        CD = CD0 + K * CL**2

        q = 0.5 * states.rho * states.VTAS**2

        L = q * self.S * CL
        D = q * self.S * CD
        m = q * self.S * self.c * CL

        wind_axis = states.windAxis

        force_vector = Vector(vector=csdl.concatenate((-D,
                                                       csdl.Variable(name='Y', shape=(1,), value=0.),
                                                       -L),
                                                      axis=0), axis=wind_axis)

        moment_vector = Vector(vector=csdl.concatenate((csdl.Variable(name='l', shape=(1,), value=0.),
                                                        m,
                                                        csdl.Variable(name='n', shape=(1,), value=0.)),
                                                       axis=0), axis=wind_axis)
        loads_waxis = ForcesMoments(force=force_vector, moment=moment_vector)

        return loads_waxis


class ToyAircraftPropulsion(Loads):

    def __init__(self, radius:Q_=Q_(1, 'm'), max_thrust:Q_=Q_(1000, 'N')):
        self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        self.max_thrust = csdl.Variable(name='max_thrust', shape=(1,), value=max_thrust.to_base_units())

    def get_FM_localAxis(self, states, controls, axis):
        throttle = controls.u[4]

        # Compute Thrust
        T =  self.radius * throttle * self.max_thrust  # N

        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)
        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads