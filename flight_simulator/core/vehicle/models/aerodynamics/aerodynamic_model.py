import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt
from flight_simulator import ureg
from typing import Union
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments

# TODO: IMPROVE AERODYNAMIC MODEL TO INCLUDE MORE COMPLEX AERODYNAMIC EFFECTS

class LiftModel:
    def __init__(self, AR:Union[ureg.Quantity, csdl.Variable], e:Union[ureg.Quantity, csdl.Variable], CD0:Union[ureg.Quantity, csdl.Variable], 
                 S:Union[ureg.Quantity, csdl.Variable], incidence:Union[ureg.Quantity, csdl.Variable]):
        super().__init__()


        if AR is None:
            self.AR = csdl.Variable(name='AR', shape=(1,), value=15)
        else:
            self.AR = AR
        

        if e is None:
            self.e = csdl.Variable(name='e', shape=(1,), value=0.87)
        else:
            self.e = e

        if CD0 is None:
            self.CD0 = csdl.Variable(name='CD0', shape=(1,), value=0.001)
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

    # TODO: Improve aerodynamic model to include more complex aerodynamic effects

    def __init__(self, component, lift_model:LiftModel):
        self.lift_model = lift_model
        self.wing_axis = component.mass_properties.cg_vector.axis
        

    def get_FM_refPoint(self, x_bar, u_bar):
            """
            Compute forces and moments about the reference point.

            Parameters
            ----------
            x_bar : csdl.VariableGroup
                Flight-dynamic state (x̄) which should include:
                - density
                - VTAS
                - states.theta
            u_bar : csdl.Variable or csdl.VariableGroup
                Control input (ū) [currently not used in the aerodynamics calculation]

            Returns
            -------
            loads : ForcesMoments
                Computed forces and moments about the reference point.
            """

            density = x_bar.atmospheric_states.density
            velocity = x_bar.VTAS
            theta = x_bar.state_vector.theta
            alpha = theta + self.lift_model.incidence

            CL = 2*np.pi*alpha
            CD = self.lift_model.CD0 + (1 / (self.lift_model.e * self.lift_model.AR * np.pi)) * CL**2  
            L = 0.5 * density * velocity**2 * self.lift_model.S * CL
            D = 0.5 * density * velocity**2 * self.lift_model.S * CD

            aero_force = csdl.Variable(shape=(3,), value=0.)
            aero_force = aero_force.set(csdl.slice[0], -D)
            aero_force = aero_force.set(csdl.slice[2], -L)
            force_vector = Vector(vector=aero_force, axis=self.wing_axis)

            moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=self.wing_axis)
            loads = ForcesMoments(force=force_vector, moment=moment_vector)
            return loads

