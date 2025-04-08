from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
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
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.vehicle.controls.aircraft_control_system import AircraftControlSystem

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

    def __init__(self, prop_axis, fd_state, controls, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:Union[HLPropCurve, CruisePropCurve], **kwargs):
        super().__init__(states=fd_state, controls=controls)
        self.prop_curve = prop_curve
        self.prop_axis = prop_axis

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


    def plot_propulsion(self, 
                   J_range=None,
                   velocity_range=(20, 100),
                   throttle_range=(0, 1),
                   ref_velocities=[67],
                   ref_throttles=[1.],
                   num_points=100,
                   rpm_ranges=[(1000, 2500)],
                   radius_values = None,
                   figsize=(12, 12),
                   labels=None,
                   colors=None,
                   styles=None,
                   save_path=None,
                   title=None):
            """
            Plot comprehensive propulsion system characteristics with multiple configurations.
            
            Parameters:
            -----------
            J_range : tuple
                Range of advance ratios (min, max)
            velocity_range : tuple
                Range of velocities in mph (min, max)
            throttle_range : tuple
                Range of throttle settings (min, max)
            ref_velocities : list
                List of reference velocities in mph for throttle sweeps
            ref_throttles : list
                List of reference throttle settings for velocity sweeps
            num_points : int
                Number of points for plotting
            rpm_ranges : list of tuples
                List of RPM ranges (min, max) for different configurations
            figsize : tuple
                Figure size in inches (width, height)
            labels : list, optional
                Labels for each configuration
            colors : list, optional
                Colors for each configuration
            styles : list, optional
                Line styles for each configuration
            save_path : str, optional
                Path to save the figure
            """
            # Set default styles if not provided
        
            if radius_values is None:
                radius_values = [self.radius.value]

            if J_range is None:
                min_RPM = self.prop_curve.min_RPM
                max_RPM = self.prop_curve.max_RPM

                # Convert velocity_range from mph to ft/s
                v_fps_min = velocity_range[0] * 1.46667
                v_fps_max = velocity_range[1] * 1.46667

                min_radius = min(radius_values)
                max_radius = max(radius_values)

                # Use the highest RPM (max_RPM) with largest radius to get the lowest J,
                # and the lowest RPM (min_RPM) with smallest radius for the highest J.
                omega_max = (max_RPM * 2 * np.pi) / 60.0
                omega_min = (min_RPM * 2 * np.pi) / 60.0

                computed_min_J = (np.pi * v_fps_min) / (omega_max * max_radius)
                computed_max_J = (np.pi * v_fps_max) / (omega_min * min_radius)
                J_range = (computed_min_J, computed_max_J)


            n_configs = max(len(ref_velocities), len(ref_throttles), len(rpm_ranges), len(radius_values))
            if labels is None:
                labels = [f'R={r:.2f} ft' for r in radius_values]
            if colors is None:
                colors = plt.cm.viridis(np.linspace(0, 1, n_configs))
            if styles is None:
                styles = ['-'] * n_configs

            # Create subplots
            fig = plt.figure(figsize=figsize)

            if title:
                fig.suptitle(f'{title} Propulsion Characteristics', fontsize=16, y=0.95)

            gs = plt.GridSpec(3, 2, figure=fig, hspace=0.5)
            
            # Create axes
            ax_ct = fig.add_subplot(gs[0, :])
            ax_thrust_v = fig.add_subplot(gs[1, 0])
            ax_advance = fig.add_subplot(gs[1, 1])
            ax_thrust_t = fig.add_subplot(gs[2, 0])
            ax_rpm = fig.add_subplot(gs[2, 1])

            # Plot thrust coefficient curve
            J = np.linspace(J_range[0], J_range[1], num_points)
            Ct = self.prop_curve.ct(J)
            ax_ct.plot(J, Ct, '-k', label='Thrust Coefficient')
            ax_ct.set_xlabel('Advance Ratio (J)')
            ax_ct.set_ylabel('Thrust Coefficient (Ct)')
            ax_ct.set_title('Propeller Thrust Characteristics')
            ax_ct.grid(True)
            ax_ct.legend()

            velocities = np.linspace(velocity_range[0], velocity_range[1], num_points)
            throttles = np.linspace(throttle_range[0], throttle_range[1], num_points)
            density = self.states.atmospheric_states.density.value * 0.00194032
            min_RPM = self.prop_curve.min_RPM
            max_RPM = self.prop_curve.max_RPM
            # Plot multiple configurations
            for i in range(n_configs):
                ref_throttle = ref_throttles[min(i, len(ref_throttles)-1)]
                ref_velocity = ref_velocities[min(i, len(ref_velocities)-1)]
                current_radius = radius_values[min(i, len(radius_values)-1)]
                label = labels[i]
                color = colors[i]
                style = styles[i]

                # Velocity sweep calculations
                thrust_vs_velocity = []
                advance_ratios = []
                for v in velocities:
                    v_fps = v * 1.46667  # mph to ft/s
                    rpm = min_RPM + (max_RPM - min_RPM) * ref_throttle
                    omega = (rpm * 2 * np.pi) / 60.0
                    J = (np.pi * v_fps) / (omega * current_radius)
                    ct = self.prop_curve.ct(J)
                    T = ct * density * (rpm/60)**2 * ((current_radius*2)**4) * 4.44822  # lbf to N

                    
                    thrust_vs_velocity.append(T)
                    advance_ratios.append(J)

                # Throttle sweep calculations
                thrust_vs_throttle = []
                rpms = []
                for t in throttles:
                    ref_velocity_fps = ref_velocity * 1.46667  # mph to ft/s
                    rpm = min_RPM + (max_RPM - min_RPM) * t
                    omega = (rpm * 2 * np.pi) / 60.0
                    J = (np.pi * ref_velocity_fps) / (omega * current_radius)
                    ct = self.prop_curve.ct(J)
                    T = ct * density * (rpm/60)**2 * ((current_radius*2)**4) * 4.44822  # lbf to N
                    thrust_vs_throttle.append(T)
                    rpms.append(rpm)

                # Plot results for this configuration
                ax_thrust_v.plot(velocities, thrust_vs_velocity, color=color, ls=style, 
                                label=f'{label} τ={ref_throttle:.1f}')
                ax_advance.plot(velocities, advance_ratios, color=color, ls=style, 
                            label=f'{label} τ={ref_throttle:.1f}')
                ax_thrust_t.plot(throttles, thrust_vs_throttle, color=color, ls=style, 
                                label=f'{label} (V={ref_velocity}mph)')
                ax_rpm.plot(throttles, rpms, color=color, ls=style, 
                        label=f'{label} ({min_RPM}-{max_RPM}RPM)')

            # Set labels and grid for all plots
            for ax in [ax_thrust_v, ax_advance, ax_thrust_t, ax_rpm]:
                ax.grid(True)
                ax.legend()

            ax_thrust_v.set_xlabel('Velocity (mph)')
            ax_thrust_v.set_ylabel('Thrust (N)')
            ax_thrust_v.set_title('Thrust vs Velocity')

            ax_advance.set_xlabel('Velocity (mph)')
            ax_advance.set_ylabel('Advance Ratio')
            ax_advance.set_title('Advance Ratio vs Velocity')

            ax_thrust_t.set_xlabel('Throttle Setting')
            ax_thrust_t.set_ylabel('Thrust (N)')
            ax_thrust_t.set_title('Thrust vs Throttle')

            ax_rpm.set_xlabel('Throttle Setting')
            ax_rpm.set_ylabel('RPM')
            ax_rpm.set_title('RPM vs Throttle')

            # plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            return fig, (ax_ct, ax_thrust_v, ax_advance, ax_thrust_t, ax_rpm)