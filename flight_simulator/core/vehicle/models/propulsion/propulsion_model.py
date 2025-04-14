import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt
from flight_simulator import ureg
from typing import Union
from scipy.interpolate import Akima1DInterpolator
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments


# TODO: Account for Torque

class HLPropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        V_inf_data = np.array(
            [0,61.3,87.6,101.6,113.8,131.3,157.6,175])
        RPM_data = np.array(
        [0,3545,4661,4702,4379,3962,3428,3451])
        self.rpm = Akima1DInterpolator(V_inf_data, RPM_data, method="akima")
        self.rpm_derivative = Akima1DInterpolator.derivative(self.rpm)
        self.min_RPM = min(RPM_data)
        self.max_RPM = max(RPM_data)
        # Obtained Mod-IV Propeller Data from CFD database
        J_data = np.array(
            [0,0.5490,0.5966,0.6860,0.8250,1.0521,1.4595,1.6098])
        Ct_data = np.array(
            [0,0.3125,0.3058,0.2848,0.2473,0.1788,0.0366,-0.0198])
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
    

    def evaluate_rpm(self, velocity: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('velocity', velocity)

        # declare output variables
        rpm = self.create_output('rpm', velocity.shape)

        if hasattr(velocity, 'value') and (velocity.value is not None):
            rpm.set_value(self.rpm(velocity.value))

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.rpm = rpm

        return outputs
    
    def compute(self, input_vals, output_vals):
        advance_ratio = input_vals['advance_ratio']
        output_vals['ct'] = self.ct(advance_ratio)

    def compute_rpm(self, input_vals, output_vals):
        velocity = input_vals['velocity']
        output_vals['rpm'] = self.rpm(velocity)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        advance_ratio = input_vals['advance_ratio']
        derivatives['ct', 'advance_ratio'] = np.diag(self.ct_derivative(advance_ratio))

    def compute_rpm_derivatives(self, input_vals, outputs_vals, derivatives):
        velocity = input_vals['velocity']
        derivatives['rpm', 'velocity'] = np.diag(self.rpm_derivative(velocity))



class CruisePropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        # Obtained Mod-III Propeller Data from CFD database
        V_inf_data = np.array(
            [0,18.75,75,112.5,150,187.5,225,243.75,262.5,266.67,300])
        RPM_data = np.array(
        [0,2250,2250,2250,2250,2250,2250,2250,2250,2000,2000])
        self.rpm = Akima1DInterpolator(V_inf_data, RPM_data, method="akima")
        self.rpm_derivative = Akima1DInterpolator.derivative(self.rpm)
        self.min_RPM = min(RPM_data)
        self.max_RPM = max(RPM_data)
        J_data = np.array(
            [0,0.1,0.4,0.6,0.8,1.0,1.2,1.3,1.4,1.6,1.8])
        Ct_data = np.array(
            [0,0.1831,0.1673,0.1422,0.1003,0.0479,-0.0085,-0.0366,-0.0057,0.0030,-0.0504])
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
    

    def evaluate_rpm(self, velocity: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('velocity', velocity)

        # declare output variables
        rpm = self.create_output('rpm', velocity.shape)
    
        if hasattr(velocity, 'value') and (velocity.value is not None):
            rpm.set_value(self.rpm(velocity.value))

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.rpm = rpm

        return outputs
    
    def compute(self, input_vals, output_vals):
        advance_ratio = input_vals['advance_ratio']
        output_vals['ct'] = self.ct(advance_ratio)

    def compute_rpm(self, input_vals, output_vals):
        velocity = input_vals['velocity']
        output_vals['rpm'] = self.rpm(velocity)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        advance_ratio = input_vals['advance_ratio']
        derivatives['ct', 'advance_ratio'] = np.diag(self.ct_derivative(advance_ratio))

    def compute_rpm_derivatives(self, input_vals, outputs_vals, derivatives):
        velocity = input_vals['velocity']
        derivatives['rpm', 'velocity'] = np.diag(self.rpm_derivative(velocity))



class AircraftPropulsion(Loads):

    def __init__(self, component, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:Union[HLPropCurve, CruisePropCurve], **kwargs):
        self.prop_curve = prop_curve
        self.prop_axis = component.quantities.mass_properties.cg_vector.axis

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=1.89/2) 
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius



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
            Control input (ū) which should include:
            - throttle

        Returns
        -------
        loads : ForcesMoments
            Computed forces and moments about the reference point.
        """

        throttle = u_bar.u[6]
        density = x_bar.atmospheric_states.density * 0.00194032  # kg/m^3 to slugs/ft^3
        velocity = x_bar.VTAS * 3.281  # m/s to ft/s

        # Compute RPM
        rpm_curve = type(self.prop_curve)() 
        rpm = rpm_curve.evaluate_rpm(velocity=velocity).rpm * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s


        # Compute advance ratio
        J = (np.pi * velocity) / (omega_RAD * self.radius)  # non-dimensional

        # Compute Ct
        ct_curve = type(self.prop_curve)()
        ct = ct_curve.evaluate(advance_ratio=J).ct

        # Compute Thrust
        T = ct * density * (rpm/60)**2 * ((self.radius*2)**4) * 4.44822 # lbf to N
    
        

        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=self.prop_axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=self.prop_axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads    

class PropulsionLoad(Loads):
    def __init__(self, component, omega, radius:Union[ureg.Quantity, csdl.Variable], ct, **kwargs):
        self.omega = omega
        self.ct = ct
        self.prop_axis = component.quantities.mass_properties.cg_vector.axis

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=1.89/2) 
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius

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
            Control input (ū) which should include:
            - throttle

        Returns
        -------
        loads : ForcesMoments
            Computed forces and moments about the reference point.
        """
        throttle = u_bar.u[6]
        density = x_bar.atmospheric_states.density * 0.00194032  # kg/m^3 to slugs/ft^3
        velocity = x_bar.VTAS * 3.281  # m/s to ft/s

        omega_RAD = (self.omega(throttle) * 2 * np.pi)/60.0

        J = (np.pi * velocity) / (omega_RAD * self.radius)
        C_t = self.ct(J)

        T = (2 / np.pi) ** 2 * density * \
            (omega_RAD * self.radius) ** 2 * C_t
        
        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=self.prop_axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=self.prop_axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads