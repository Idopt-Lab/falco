from flight_simulator.core.vehicle.condition import Condition
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.utils.euler_rotations import build_rotation_matrix
from typing import Union, List, Tuple
import csdl_alpha as csdl
from dataclasses import dataclass
import numpy as np
import warnings
import NRLMSIS2
from flight_simulator import ureg, Q_



@dataclass
class AircraftStateQuantities(csdl.VariableGroup):
    ac_states : AircraftStates = None
    ac_states_atmos : NRLMSIS2.Atmosphere = None
    inertial_forces = None
    inertial_moments = None



@dataclass
class HoverParameters(csdl.VariableGroup):
    altitude : csdl.Variable
    time : csdl.Variable

    def define_checks(self):
        self.add_check('altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('time', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

    def _check_parameters(self, name, value):
        if self._metadata[name]['type'] is not None:
            if type(value) not in self._metadata[name]['type']:
                raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")

        if self._metadata[name]['variablize']:
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                value.add_tag(tag=str(value_si.units))

        if self._metadata[name]['shape'] is not None:
            if value.shape != self._metadata[name]['shape']:
                raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value


@dataclass
class ClimbParameters(csdl.VariableGroup):
    initial_altitude : csdl.Variable
    final_altitude : csdl.Variable
    pitch_angle : csdl.Variable
    climb_gradient : csdl.Variable
    rate_of_climb : csdl.Variable
    speed : csdl.Variable
    mach_number : csdl.Variable
    flight_path_angle : csdl.Variable
    time : csdl.Variable

    def define_checks(self):
        self.add_check('initial_altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('final_altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('pitch_angle', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('climb_gradient', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('rate_of_climb', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('speed', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('mach_number', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('flight_path_angle', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('time', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)


    def _check_parameters(self, name, value):
        if self._metadata[name]['type'] is not None:
            if type(value) not in self._metadata[name]['type']:
                raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")

        if self._metadata[name]['variablize']:
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                value.add_tag(tag=str(value_si.units))

        if self._metadata[name]['shape'] is not None:
            if value.shape != self._metadata[name]['shape']:
                raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value

@dataclass
class CruiseParameters(csdl.VariableGroup):
    altitude : csdl.Variable
    speed : csdl.Variable
    mach_number : csdl.Variable
    pitch_angle : csdl.Variable
    range : csdl.Variable
    time : csdl.Variable


    def define_checks(self):
            self.add_check('altitude', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('speed', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('mach_number', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('range', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('time', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)



    def _check_parameters(self, name, value):
        if self._metadata[name]['type'] is not None:
            if type(value) not in self._metadata[name]['type']:
                raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")

        if self._metadata[name]['variablize']:
            if isinstance(value, ureg.Quantity):
                value_si = value.to_base_units()
                value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                value.add_tag(tag=str(value_si.units))

        if self._metadata[name]['shape'] is not None:
            if value.shape != self._metadata[name]['shape']:
                raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value


class AircraftCondition(Condition):
    """General aircraft condition."""
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo], 
                 u: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 v: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 w: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 p: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),
                 q: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),
                 r: Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad/s'),
                 component = None,
                controls = None) -> None:
        self.axis = fd_axis
        self.controls = controls
        self.component = component
        self.quantities : AircraftStateQuantities = AircraftStateQuantities()
        self.quantities.ac_states = AircraftStates(
            axis=fd_axis,
            u=u, v=v, w=w,
            p=p, q=q, r=r,
        )
          
        self.quantities.ac_states_atmos = self.quantities.ac_states.atm.evaluate(self.quantities.ac_states.axis.translation_from_origin.z)

        if self.component is not None:
            self.assemble_forces_moments()

    def assemble_forces_moments(self,print_output: bool = False):
        total_forces = csdl.Variable(value=0., shape=(3,))
        total_moments = csdl.Variable(value=0., shape=(3,))
        if hasattr(self.component, "comps") and self.component.comps:
            for sub_comp in self.component.comps.values():
                forces, moments = sub_comp.compute_total_loads(fd_state=self.quantities.ac_states,
                                                                controls=self.controls,
                                                                fd_axis=self.axis)
                total_forces += forces
                total_moments += moments
        else:
            forces, moments = self.component.compute_total_loads(fd_state=self.quantities.ac_states,
                                                                controls=self.controls,
                                                                    fd_axis=self.axis)
            total_forces += forces
            total_moments += moments
        if print_output:
            print('Aircraft Condition:', self.__class__.__name__, 'in the Flight Dynamics Axis:')
            print('u:', self.quantities.ac_states.states.u.value, 'm/s')
            print('v:', self.quantities.ac_states.states.v.value, 'm/s')
            print('w:', self.quantities.ac_states.states.w.value, 'm/s')
            print('p:', self.quantities.ac_states.states.p.value, 'rad/s')
            print('q:', self.quantities.ac_states.states.q.value,  'rad/s')
            print('r:', self.quantities.ac_states.states.r.value, 'rad/s')
            print('Altitude:', self.quantities.ac_states.axis.translation_from_origin.z.value, 'm')
            print('Atmospheric Density:', self.quantities.ac_states_atmos.density.value, 'kg/m^3')
            print("Total Forces: ", total_forces.value, 'N')
            print("Total Moments: ", total_moments.value, 'N*m')
        return total_forces, total_moments
    
    


        
class CruiseCondition(AircraftCondition):
    """Cruise condition: intended for steady analyses.
    
    Parameters
    ----------
    - speed : int | float | np.ndarray | csdl.Variable

    - mach_number : Union[float, int, csdl.Variable]

    - time : int | float | np.ndarray | csdl.Variable

    - pitch_angle : int | float | np.ndarray | csdl.Variable

    - range : int | float | np.ndarray | csdl.Variable
    
    - altitude : int | float | np.ndarray | csdl.Variable

    Note: cannot specify all parameters at once 
    (e.g., cannot specify speed and mach_number at the same time)
    """

    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo], 
                 altitude : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm'),
                 range : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm'),
                 pitch_angle : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad'),
                 speed : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 mach_number : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'dimensionless'),
                 time : Union[ureg.Quantity, csdl.Variable]=Q_(0, 's'),
                 component = None,
                 controls = None
                 ):



        self.parameters : CruiseParameters = CruiseParameters(
            altitude=altitude,
            speed=speed,
            range=range,
            pitch_angle=pitch_angle,
            mach_number=mach_number,
            time=time,
        )
        self.quantities = AircraftStateQuantities()


        self._atmos_model = NRLMSIS2.Atmosphere()  
        self.axis = fd_axis
        self.component = component
        self.controls = controls


        # Compute ac states
        self._setup_condition()


    def _setup_condition(self):
            # Different combinations of conflicting attributes
            conflicting_attributes_1 = ["speed", "mach_number"]
            conflicting_attributes_2 = ["speed", "time", "range"]
            conflicting_attributes_3 = ["mach_number", "time", "range"]

            # Check for conflicting attributes:
            
            if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_1):
                raise Exception("Cannot specify 'mach_number' and 'speed' at the same time")
            if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_2):
                raise Exception("Cannot specify 'speed', 'time', and 'range' at the same time")
            if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_3):
                raise Exception("Cannot specify 'mach_number', 'time', and 'range' at the same time")
        
            x = y = v = phi = psi = p = q = r = csdl.Variable(value=0.)

            # set z to altitude and evaluate atmosphere model
            z = self.parameters.altitude
            self.axis.translation_from_origin.z = z
            atmos_states = self._atmos_model.evaluate(z)



            # set theta to pitch_angle
            theta = self.parameters.pitch_angle
    
            # Compute or set speed, mach, range, and time
            mach_number = self.parameters.mach_number
            speed = self.parameters.speed
            time = self.parameters.time
            range = self.parameters.range

            if mach_number.value != 0 and range.value != 0:
                V = atmos_states.speed_of_sound * mach_number
                self.parameters.speed = V
                time = range / V
                self.parameters.time = time
            elif mach_number.value != 0 and time.value != 0:
                V = atmos_states.speed_of_sound * mach_number
                self.parameters.speed = V
                range = V * time
                self.parameters.range = range
            elif speed.value != 0 and range.value != 0:
                V = speed
                mach_number = V / atmos_states.speed_of_sound
                self.parameters.mach_number = mach_number
                time = range / V
                self.parameters.time = time
            elif speed.value != 0 and time.value != 0:
                V = speed
                mach_number = V / atmos_states.speed_of_sound
                self.parameters.mach_number = mach_number
                range = V * time 
                self.parameters.range = range
            else:
                raise NotImplementedError

            theta = csdl.Variable(name='theta', value=theta.magnitude, shape=(1,))
            u = V * csdl.cos(theta)
            w = V * csdl.sin(theta)
    
            self.quantities.ac_states = AircraftStates(
                u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis)
            self.quantities.ac_states_atmos = atmos_states
    



class ClimbCondition(AircraftCondition):
    """Climb condition (intended for steady analyses)
    

    """

    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo], 
                 initial_altitude : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm'), 
                 final_altitude : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm'),
                 pitch_angle : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad'),
                 fligth_path_angle : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'rad'),
                 speed : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 mach_number : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'dimensionless'),
                 time : Union[ureg.Quantity, csdl.Variable]=Q_(0, 's'),
                 climb_gradient : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 rate_of_climb : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm/s'),
                 component = None,
                 controls = None
                 ) -> None:
        
        
        self.parameters : ClimbParameters = ClimbParameters(
            initial_altitude=initial_altitude,
            final_altitude=final_altitude,
            pitch_angle=pitch_angle,
            speed=speed,
            mach_number=mach_number,
            flight_path_angle=fligth_path_angle,
            time=time,
            rate_of_climb=rate_of_climb,
            climb_gradient=climb_gradient,
        )
        self.quantities = AircraftStateQuantities()
        self._atmos_model = NRLMSIS2.Atmosphere()  
        self.axis = fd_axis
        self.component = component
        self.controls = controls
        self._setup_condition()



    def _setup_condition(self):
            # Different combinations of conflicting attributes
            conflicting_attributes_1 = ["speed", "mach_number"]
            conflicting_attributes_2 = ["speed", "time"]
            conflicting_attributes_3 = ["mach_number", "time"]

            # Check for conflicting attributes:
            if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_1):
                raise Exception("Cannot specify 'mach_number' and 'speed' at the same time")
            if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_2):
                raise Exception("Cannot specify 'speed', 'time', and 'range' at the same time")
            if all(getattr(self.parameters, attr).value != 0 for attr in conflicting_attributes_3):
                raise Exception("Cannot specify 'mach_number', 'time', and 'range' at the same time")

            # zero aircraft states in climb
            v = p = q = r = csdl.Variable(value=0.)

            # Non-zero aircraft states in climb
            # Compute mean altitude
            hi = self.parameters.initial_altitude
            hf = self.parameters.final_altitude
            h_mean = z = 0.5 * (hi + hf)

            # Get pitch agle and flight path angle 
            theta = self.parameters.pitch_angle
            gamma = self.parameters.flight_path_angle

            # Compute atmospheric states for mean altitude
            self.axis.translation_from_origin.z = h_mean

            atmos_states = self._atmos_model.evaluate(altitude=h_mean)

            # Compute speed from mach number or time or set V = speed
            mach_number = self.parameters.mach_number
            speed = self.parameters.speed
            time = self.parameters.time
            if mach_number.value !=0:
                V = mach_number * atmos_states.speed_of_sound
                self.parameters.speed = V
            elif speed.value !=0:
                V = speed
                mach_number = V / atmos_states.speed_of_sound
                self.parameters.mach_number = mach_number
            else:
                w = (hf - hi) / time
                u = w / csdl.tan(gamma)
                V = (u**2 + w**2)**0.5
                mach_number = V / atmos_states.speed_of_sound
                self.parameters.mach_number = mach_number
                self.parameters.speed = V

            # Compute time spent in climb if time is None
            if time.value !=0:
                h = ((hf - hi)**2)**0.5 # avoid getting a negative time 
                d = h / csdl.tan(gamma)
                time = (d**2 + h**2)**0.5 / V
                self.parameters.time = time

            # Compute a.o.a and compute u, w knowing the speed
            alfa = theta - gamma
            u = V * csdl.cos(alfa)
            w = V * csdl.sin(alfa)

            # Set aircraft and atmospheric states
            self.quantities.ac_states = AircraftStates(
                u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis)
            
            self.quantities.ac_states_atmos = atmos_states



class HoverCondition(AircraftCondition):
    """Hover condition (intended for steady analyses)
    
    Parameters
    ----------
    - altitude : Union[float, int, csdl.Variable]

    - time : Union[float, int, csdl.Variable]
    """
    
    def __init__(self,
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 altitude : Union[ureg.Quantity, csdl.Variable]=Q_(0, 'm'),
                 time : Union[ureg.Quantity, csdl.Variable]=Q_(0, 's'),
                 component = None,
                 controls = None
                 ) -> None:

        self.parameters : HoverParameters = HoverParameters(
            altitude=altitude,
            time=time,
        )
        self.quantities = AircraftStateQuantities()
        self._atmos_model = NRLMSIS2.Atmosphere()  
        self.axis = fd_axis
        self.component = component
        self.controls = controls
        

        # Compute ac states
        self._setup_condition()




    def _setup_condition(self):
        hover_parameters = self.parameters
        
        # All aircraft states except z will be zero
        u = v = w = p = q = r = csdl.Variable(value=0.)
        z = hover_parameters.altitude
        self.axis.translation_from_origin.z = z
        
        # Evaluate atmospheric states
        atmos_states = self._atmos_model.evaluate(z)

        # Set aircraft and atmospheric states
        self.quantities.ac_states = AircraftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, axis=self.axis
        )

        self.quantities.ac_states_atmos = atmos_states