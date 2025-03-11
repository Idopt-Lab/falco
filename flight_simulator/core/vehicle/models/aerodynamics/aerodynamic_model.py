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
    def __init__(self, AR, e, CD0, S, incidence=0):
        super().__init__()
        self.AR = AR 
        self.e = e
        self.CD0 = CD0
        self.S = S
        self.incidence = incidence 

    def define(self):
        # Define inputs
        self.add_input('AR', shape=(1,), val=self.AR)
        self.add_input('e', shape=(1,), val=self.e)
        self.add_input('CD0', shape=(1,), val=self.CD0)
        self.add_input('S', shape=(1,), val=self.S)
        self.add_input('incidence', shape=(1,), val=self.incidence)

        # Define outputs
        self.add_outputs('AR', shape=(1,), val=self.AR)
        self.add_outputs('e', shape=(1,), val=self.e)
        self.add_outputs('CD0', shape=(1,), val=self.CD0)
        self.add_outputs('S', shape=(1,), val=self.S)
        self.add_outputs('incidence', shape=(1,), val=self.incidence)

    
    def compute(self, inputs, outputs):
        alpha = inputs['alpha']
        AR = inputs['AR']
        e = inputs['e']
        CD0 = inputs['CD0']
        incidence = inputs['incidence']
        
        # Compute lift and drag coefficients
        outputs['alpha'] = alpha
        outputs['AR'] = AR
        outputs['e'] = e
        outputs['CD0'] = CD0
        outputs['incidence'] = incidence
     


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
    

