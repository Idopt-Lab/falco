import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins



class GravityLoads(Loads):

    def __init__(self, load_axis, fd_state, controls, component):
        super().__init__(states=fd_state, controls=controls)

        # Store the states and mass properties
        self.states = fd_state
        self.load_axis = load_axis
        self.cg = component.quantities.mass_properties.cg_vector
        self.mass = component.quantities.mass_properties.mass

        if self.mass is None:
            self.mass = csdl.Variable(name='mass', shape=(1,), value=1630) 
        elif isinstance(self.mass, ureg.Quantity):
            self.mass = csdl.Variable(name='mass', shape=(1,), value=self.mass.to_base_units().magnitude)
        else:
            self.mass = self.mass




    def get_FM_refPoint(self, x_bar, u_bar):
        """Use vehicle state and control objects to generate an estimate
        of gravity forces and moments about a reference point."""
        # Gravity FM
        g=9.81
        cg_vals = self.cg.vector.value 
        x = cg_vals[0]
        y = cg_vals[1]
        z = cg_vals[2]
        
        Rbc = np.array([x,y,z])


        m = self.mass
        th = self.states.states.theta
        ph = self.states.states.phi

        Fxg = -m * g * csdl.sin(th)
        Fyg = m * g * csdl.cos(th) * csdl.sin(ph)
        Fzg = m * g * csdl.cos(th) * csdl.cos(ph)
        forceVec = csdl.concatenate([Fxg, Fyg, Fzg])

        Rbsksym = np.array([[0, -Rbc[2], Rbc[1]],
                            [Rbc[2], 0, -Rbc[0]],
                            [-Rbc[1], Rbc[0], 0]])
        
        Mgrav = np.dot(Rbsksym, np.array([Fxg, Fyg, Fzg]))



        F_FD_BodyFixed = Vector(forceVec,axis=self.load_axis)
        M_FD_BodyFixed = Vector(csdl.concatenate([Mgrav[0],Mgrav[1],Mgrav[2]]),axis=self.load_axis)

        loads = ForcesMoments(force=F_FD_BodyFixed, moment=M_FD_BodyFixed)
        return loads
