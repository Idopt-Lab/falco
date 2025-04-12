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
        self.wing_axis = component.quantities.mass_properties.cg_vector.axis
        

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
            theta = x_bar.states.theta
            alpha = theta + self.lift_model.incidence + u_bar.elevator.deflection

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

    

    def plot_aerodynamics(self, 
                          velocity_range=(20, 100), 
                          alpha_range=(-5, 15), 
                          num_points=50, 
                          AR_values=None, 
                          e_values=None):
        """
        Plot aerodynamic coefficients and forces for different configurations
        
        Parameters:
        -----------
        velocity_range : tuple
            Range of velocities in mph
        alpha_range : tuple
            Range of angles of attack in degrees
        num_points : int
            Number of points to plot
        AR_values : list
            List of aspect ratios to plot
        e_values : list
            List of efficiency factors to plot
        """
        velocities = np.linspace(velocity_range[0], velocity_range[1], num_points)
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points) * np.pi/180

        # Create 4x2 subplot layout
        fig = plt.figure(figsize=(10, 8))
        gs = plt.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Coefficient plots
        ax_cl = fig.add_subplot(gs[0, 0])
        ax_cd = fig.add_subplot(gs[0, 1])
        ax_polar = fig.add_subplot(gs[1, :])  # Drag polar takes full width
        
        # Force plots
        ax_lift_v = fig.add_subplot(gs[2, 0])
        ax_drag_v = fig.add_subplot(gs[2, 1])
        ax_lift_a = fig.add_subplot(gs[3, 0])
        ax_drag_a = fig.add_subplot(gs[3, 1])

        # Default values if none provided
        if AR_values is None:
            AR_values = [self.lift_model.AR.value]
        if e_values is None:
            e_values = [self.lift_model.e.value]

        colors = plt.cm.viridis(np.linspace(0, 1, len(AR_values) * len(e_values)))
        color_idx = 0

        for AR in AR_values:
            for e in e_values:
                # Calculate coefficients
                CLs = 2 * np.pi * alphas + 0.4 # Add 0.4 for lift polar offset compared to CFD data (guesstimate)
                CDs = self.lift_model.CD0.value + (1/(e * AR * np.pi)) * CLs**2 + 0.04 # Add 0.04 for drag polar offset compared to CFD data (guesstimate)
                
                # Velocity sweep calculations
                lift_forces = []
                drag_forces = []
                ref_alpha = float(self.lift_model.incidence.value)  # 2 degrees reference angle
                
                for v in velocities:
                    density = self.states.atmospheric_states.density.value
                    CL = 2 * np.pi * ref_alpha+0.2
                    CD = self.lift_model.CD0.value + (1/(e * AR * np.pi)) * CL**2
                    
                    L = 0.5 * density * v**2 * self.lift_model.S.value * CL
                    D = 0.5 * density * v**2 * self.lift_model.S.value * CD
                    
                    lift_forces.append(L)
                    drag_forces.append(D)

                # Alpha sweep calculations
                ref_velocity = self.states.VTAS.value 
                alpha_lifts = []
                alpha_drags = []
                
                for alpha in alphas:
                    density = self.states.atmospheric_states.density.value
                    CL = 2 * np.pi * alpha
                    CD = self.lift_model.CD0.value + (1/(e * AR * np.pi)) * CL**2
                    
                    L = 0.5 * density * ref_velocity**2 * self.lift_model.S.value * CL
                    D = 0.5 * density * ref_velocity**2 * self.lift_model.S.value * CD
                    
                    alpha_lifts.append(L)
                    alpha_drags.append(D)

                label = f'AR={AR:.1f}, e={e:.2f}'
                
                # Plot all curves for this configuration
                ax_cl.plot(np.degrees(alphas), CLs, '-', color=colors[color_idx], label=label)
                ax_cd.plot(np.degrees(alphas), CDs, '-', color=colors[color_idx], label=label)
                ax_polar.plot(CDs, CLs, '-', color=colors[color_idx], label=label)
                ax_lift_v.plot(velocities, np.array(lift_forces) / np.array(drag_forces), '-', color=colors[color_idx], label=label)
                ax_drag_v.plot(velocities, drag_forces, '-', color=colors[color_idx], label=label)
                ax_lift_a.plot(np.degrees(alphas), np.array(alpha_lifts)/np.array(alpha_drags), '-', color=colors[color_idx], label=label)
                ax_drag_a.plot(np.degrees(alphas), alpha_drags, '-', color=colors[color_idx], label=label)
                
                # Add alpha annotations on drag polar
                alpha_markers = [-5, 0, 5, 10, 15]
                for alpha_marker in alpha_markers:
                    idx = np.abs(np.degrees(alphas) - alpha_marker).argmin()
                    ax_polar.plot(CDs[idx], CLs[idx], 'o', color=colors[color_idx], markersize=5)
                    ax_polar.annotate(f'{alpha_marker}°', 
                                    (CDs[idx], CLs[idx]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=6)
                
                color_idx += 1
        plt.rcParams.update({'font.size': 8})
        # Configure all axes
        ax_cl.set(xlabel='Angle of Attack (degrees)', ylabel='Lift Coefficient',
                title='Lift Coefficient vs Alpha')
        ax_cd.set(xlabel='Angle of Attack (degrees)', ylabel='Drag Coefficient',
                title='Drag Coefficient vs Alpha')
        ax_polar.set(xlabel='Drag Coefficient (CD)', ylabel='Lift Coefficient (CL)',
                    title='Drag Polar')
        
        ax_lift_v.set(xlabel='Velocity (mph)', ylabel='L/D Ratio',
                    title=f'Lift/Drag vs Velocity (α = {np.degrees(ref_alpha):.1f}°)')
        ax_drag_v.set(xlabel='Velocity (mph)', ylabel='Force (N)',
                    title=f'Drag Force vs Velocity (α = {np.degrees(ref_alpha):.1f}°)')
        
        ax_lift_a.set(xlabel='Angle of Attack (degrees)', ylabel='L/D Ratio',
                    title=f'Lift/Drag vs Alpha (V = {ref_velocity} mph)')
        ax_drag_a.set(xlabel='Angle of Attack (degrees)', ylabel='Force (N)',
                    title=f'Drag Force vs Alpha (V = {ref_velocity} mph)')

        # Add grid and legend to all subplots
        for ax in [ax_cl, ax_cd, ax_polar, ax_lift_v, ax_drag_v, ax_lift_a, ax_drag_a]:
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=6)
            ax.tick_params(labelsize=8)

        # plt.tight_layout()
        return fig, (ax_cl, ax_cd, ax_polar, ax_lift_v, ax_drag_v, ax_lift_a, ax_drag_a)