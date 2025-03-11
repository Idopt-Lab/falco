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



class PropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        # Obtained with JavaProp
        J_data = np.array(
            [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.76, 0.77, 0.78,
             0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94])
        Ct_data = np.array(
            [0.102122, 0.11097, 0.107621, 0.105191, 0.102446, 0.09947, 0.096775, 0.094706, 0.092341, 0.088912, 0.083878,
             0.076336, 0.066669, 0.056342, 0.045688, 0.034716, 0.032492, 0.030253, 0.028001, 0.025735, 0.023453,
             0.021159, 0.018852, 0.016529, 0.014194, 0.011843, 0.009479, 0.0071, 0.004686, 0.002278, -0.0002, -0.002638,
             -0.005145, -0.007641, -0.010188])
        self.ct = Akima1DInterpolator(J_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)

    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, advance_ratio: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('advance_ratio', advance_ratio)

        # declare output variables
        ct = self.create_output('ct', advance_ratio.shape)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.ct = ct

        return outputs

    def compute(self, input_vals, output_vals):
        advance_ratio = input_vals['advance_ratio']
        output_vals['ct'] = self.ct(advance_ratio)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        advance_ratio = input_vals['advance_ratio']
        derivatives['ct', 'advance_ratio'] = np.diag(self.ct_derivative(advance_ratio))


class AircraftPropulsion(Loads):

    def __init__(self, states, controls, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:PropCurve):
        super().__init__(states=states, controls=controls)
        self.prop_curve = prop_curve

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=1.2192/2) # prop diameter in ft is 4 ft = 1.2192 m
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius

class AircraftPropulsion(Loads):

    def __init__(self, states, controls, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:PropCurve):
        super().__init__(states=states, controls=controls)
        self.prop_curve = prop_curve

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=1.2192/2) # prop diameter in ft is 4 ft = 1.2192 m
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius


    def get_FM_refPoint(self):
        throttle = self.controls.u[6]
        density = self.states.atmospheric_states.density
        velocity = self.states.VTAS
        axis = self.states.axis

        # Compute RPM
        min_RPM = 1000
        max_RPM = 2500
        rpm = min_RPM + (max_RPM - min_RPM) * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s

        # Compute advance ratio
        J = (np.pi * velocity) / (omega_RAD * self.radius)  # non-dimensional

        # Compute Ct
        ct = self.prop_curve.evaluate(advance_ratio=J).ct

        # Compute Thrust
        T =  (2 / np.pi) ** 2 * density * (omega_RAD * self.radius) ** 2 * ct  # N

        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads


    