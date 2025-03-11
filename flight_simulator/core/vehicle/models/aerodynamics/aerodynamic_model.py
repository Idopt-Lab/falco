from flight_simulator.core.vehicle.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt
from flight_simulator import ureg, Q_
from typing import List, Union
from scipy.interpolate import Akima1DInterpolator
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.aircraft_states import AircaftStates
from flight_simulator.core.vehicle.aircraft_control_system import AircraftControlSystem

class LiftModel(csdl.CustomExplicitOperation):
    def __init__(self, AR:Union[ureg.Quantity, csdl.Variable], e:Union[ureg.Quantity, csdl.Variable], CD0:Union[ureg.Quantity, csdl.Variable], 
                 S:Union[ureg.Quantity, csdl.Variable], incidence:Union[ureg.Quantity, csdl.Variable]):
        super().__init__()


        if AR is None:
            self.AR = csdl.Variable(name='AR', shape=(1,), value=15)
        elif isinstance(AR, ureg.Quantity):
            self.AR = csdl.Variable(name='AR', shape=(1,), value=AR.to_base_units())
        else:
            self.AR = AR
        

        if e is None:
            self.e = csdl.Variable(name='e', shape=(1,), value=0.87)
        elif isinstance(e, ureg.Quantity):
            self.e = csdl.Variable(name='e', shape=(1,), value=e.to_base_units())
        else:
            self.e = e

        if CD0 is None:
            self.CD0 = csdl.Variable(name='CD0', shape=(1,), value=0.001)
        elif isinstance(CD0, ureg.Quantity):
            self.CD0 = csdl.Variable(name='CD0', shape=(1,), value=CD0.to_base_units())
        else:
            self.CD0 = CD0

        if S is None:
            self.S = csdl.Variable(name='S', shape=(1,), value=6.22)
        elif isinstance(S, ureg.Quantity):
            self.S = csdl.Variable(name='S', shape=(1,), value=S.to_base_units())
        else:
            self.S = S

        if incidence is None:
            self.incidence = csdl.Variable(name='incidence', shape=(1,), value=2*np.pi/180)
        elif isinstance(incidence, ureg.Quantity):
            self.incidence = csdl.Variable(name='incidence', shape=(1,), value=incidence.to_base_units())
        else:
            self.incidence = incidence
        
    


class AircraftAerodynamics(Loads):

    def __init__(self, states, controls, lift_model:LiftModel):
        super().__init__(states=states, controls=controls)
        self.lift_model = lift_model


    def get_FM_refPoint(self):
            density = self.states.atmospheric_states.density
            velocity = self.states.VTAS
            axis = self.states.axis
            theta = self.states.states.theta
            alpha = theta + self.lift_model.incidence

            CL = 2*np.pi*alpha
            CD = self.lift_model.CD0 + (1/(self.lift_model.e*self.lift_model.AR*np.pi))*CL**2
            L = 0.5*density*velocity**2*self.lift_model.S*CL
            D = 0.5*density*velocity**2*self.lift_model.S*CD

            force_vector = Vector(vector=csdl.concatenate((-D,
                                                        csdl.Variable(shape=(1,), value=0.),
                                                        -L),
                                                        axis=0), axis=axis)

            moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
            loads = ForcesMoments(force=force_vector, moment=moment_vector)
            return loads
    

