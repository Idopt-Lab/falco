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
from flight_simulator.core.loads.mass_properties import MassProperties

@dataclass
class MPS_Parameters(csdl.VariableGroup):
    wing_AR: csdl.Variable
    wing_area: csdl.Variable
    fuselage_length: csdl.Variable
    battery_mass: csdl.Variable
    cruise_speed: csdl.Variable

    def define_checks(self):
        self.add_check('wing_AR', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('wing_area', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('fuselage_length', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('battery_mass', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('cruise_speed', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        
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
class Empennage_MPS_Parameters(csdl.VariableGroup):
    h_tail_area: csdl.Variable
    v_tail_area: csdl.Variable

    def define_checks(self):
        self.add_check('h_tail_area', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
        self.add_check('v_tail_area', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
    
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

class BoomMPSParameters(MPS_Parameters):
    """Compute the mass properties of the booms.

    Parameters
    ----------
    wing_area : Union[csdl.Variable, float, int]
        _description_
    wing_AR : Union[csdl.Variable, float, int]
        _description_
    fuselage_length : Union[csdl.Variable, float, int]
        _description_
    battery_mass : Union[csdl.Variable, float, int]
        _description_
    cruise_speed : Union[csdl.Variable, float, int]
        _description_
    fd_axis : Flight Dynamics Axis
        the reference w.r.t. which the mass properties are computed

    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """   
    def __init__(self, 
                 wing_AR: Union[csdl.Variable, float, int],
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 wing_area: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm^2'),
                 fuselage_length: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 battery_mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg'),
                 cruise_speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 component = None) -> MassProperties:
        self.component = component
        self.fd_axis = fd_axis
        self.parameters: MPS_Parameters = MPS_Parameters(
            wing_AR=wing_AR,
            wing_area=wing_area,
            fuselage_length=fuselage_length,
            battery_mass=battery_mass,
            cruise_speed=cruise_speed
        )
        
    def compute_boom_mps(self):
        cg_list = []
        it_list = []
        for boom_reg in booms_reg:
            cg_vec = csdl.Variable(shape=(3, ), value=0.)
            i_mat = csdl.Variable(shape=(3, 3), value=0.)
            for name, coeffs in boom_reg.items():
                qty = evaluate_regression(
                    self.parameters.wing_area, 
                    self.parameters.wing_AR, 
                    self.parameters.fuselage_length,
                    self.parameters.battery_mass, 
                    self.parameters.cruise_speed, 
                    coeffs
                )
                if 'cg_X' in name:
                    cg_vec =  cg_vec.set(csdl.slice[0], qty)
                elif 'cg_Y' in name:
                    cg_vec =  cg_vec.set(csdl.slice[1], qty)
                elif 'cg_Z' in name:
                    cg_vec =  cg_vec.set(csdl.slice[2], qty)
                elif 'Ixx' in name:
                    i_mat = i_mat.set(csdl.slice[0, 0], qty)
                elif 'Iyy' in name:
                    i_mat = i_mat.set(csdl.slice[1, 1], qty)
                elif 'Izz' in name:
                    i_mat = i_mat.set(csdl.slice[2, 2], qty)
            cg_list.append(cg_vec)
            it_list.append(i_mat)

        mass_coeffs = boom_mass_coeffs
        total_boom_mass = evaluate_regression(
            self.parameters.wing_area, self.parameters.wing_AR, self.parameters.fuselage_length,
            self.parameters.battery_mass, self.parameters.cruise_speed, mass_coeffs,
        )

        mass_per_boom_pair = total_boom_mass / 4 # left/right + inner/outer
        total_boom_cg = csdl.Variable(shape=(3, ), value=0.)

        # compute total boom cg
        for cg in cg_list:
            total_boom_cg = total_boom_cg + cg * mass_per_boom_pair

        total_boom_cg = total_boom_cg / total_boom_mass

        # zero out cg-y and flip x,z depending on reference frame
        if self.fd_axis == self.fd_axis:
            total_boom_cg = total_boom_cg * np.array([-1, 0, -1])

        else:
            total_boom_cg = total_boom_cg * np.array([1, 0, 1])

        # compute total boom inertia tensor (about total cg)
        total_boom_I = csdl.Variable(shape=(3, 3), value=0.)
        
        # parallel axis theorem (parallel axis is total boom cg)
        x =  total_boom_cg[0]
        y =  total_boom_cg[1]
        z =  total_boom_cg[2]
        
        transl_mat = csdl.Variable(shape=(3, 3), value=0.)
        transl_mat = transl_mat.set(csdl.slice[0, 0], y**2 + z**2)
        transl_mat = transl_mat.set(csdl.slice[0, 1], -x * y)
        transl_mat = transl_mat.set(csdl.slice[0, 2], -x * z)
        transl_mat = transl_mat.set(csdl.slice[1, 0], -y * x)
        transl_mat = transl_mat.set(csdl.slice[1, 1], x**2 + z**2)
        transl_mat = transl_mat.set(csdl.slice[1, 2], -y * z)
        transl_mat = transl_mat.set(csdl.slice[2, 0], -z * x)
        transl_mat = transl_mat.set(csdl.slice[2, 1], -z * y)
        transl_mat = transl_mat.set(csdl.slice[2, 2], x**2 + y**2)
        transl_mat = mass_per_boom_pair * transl_mat

        for it in it_list:
            it_boom_cg = it + transl_mat
            total_boom_I = total_boom_I + it_boom_cg

        # assemble boom mps data class
        boom_mps = MassProperties(
            mass=total_boom_mass,
            cg_vector=total_boom_cg,
            inertia_tensor=total_boom_I,
        )

        return boom_mps

class EmpennageMPSParameters(Empennage_MPS_Parameters):
    """Compute the mass properties of the empennage.

    Parameters
    ----------
    h_tail_area : Union[csdl.Variable, float, int]
        _description_
    v_tail_area : Union[csdl.Variable, float, int]
        _description_
    fd_axis : Flight Dynamics Axis
        the reference w.r.t. which the mass properties are computed

    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """
    def __init__(self, 
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 h_tail_area: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm^2'),
                 v_tail_area: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm^2'),
                 component = None) -> MassProperties:
        self.component = component
        self.fd_axis = fd_axis
        self.parameters: Empennage_MPS_Parameters = Empennage_MPS_Parameters(
            h_tail_area=h_tail_area,
            v_tail_area=v_tail_area
        )


    def compute_empennage_mps(self):
        cg_list = []
        it_list = []
        for reg in empennage_reg:
            cg_vec = csdl.Variable(shape=(3, ), value=0.)
            i_mat = csdl.Variable(shape=(3, 3), value=0.)
            for name, coeffs in reg.items():
                qty = evaluate_empennage_regression(
                    self.parameters.h_tail_area, 
                    self.parameters.v_tail_area, 
                    coeffs
                )
                if 'cg_X' in name:
                    cg_vec =  cg_vec.set(csdl.slice[0], qty)
                elif 'cg_Y' in name:
                    cg_vec =  cg_vec.set(csdl.slice[1], qty)
                elif 'cg_Z' in name:
                    cg_vec =  cg_vec.set(csdl.slice[2], qty)
                elif 'Ixx' in name:
                    i_mat = i_mat.set(csdl.slice[0, 0], qty)
                elif 'Iyy' in name:
                    i_mat = i_mat.set(csdl.slice[1, 1], qty)
                elif 'Izz' in name:
                    i_mat = i_mat.set(csdl.slice[2, 2], qty)
                elif 'Ixz' in name:
                    i_mat = i_mat.set(csdl.slice[0, 2], qty)
                    i_mat = i_mat.set(csdl.slice[2, 0], qty)
                elif 'Iyz' in name:
                    i_mat = i_mat.set(csdl.slice[1, 2], qty)
                    i_mat = i_mat.set(csdl.slice[2, 1], qty)

            cg_list.append(cg_vec)
            it_list.append(i_mat)

        mass_coeffs = empennage_mass_coeff
        total_empennage_mass = evaluate_empennage_regression(
            htail_area=self.parameters.h_tail_area, 
            vtail_area=self.parameters.v_tail_area, 
            coeffs=mass_coeffs
        )

        total_empennage_cg = csdl.Variable(shape=(3, ), value=0.)

        # compute total boom cg
        for i, cg in enumerate(cg_list):
            if i == 0:
                # weigh h tail a more
                total_empennage_cg = total_empennage_cg + cg * 0.65 * total_empennage_mass
            else:
                # weigh v tail a less
                total_empennage_cg = total_empennage_cg + cg * 0.35 * total_empennage_mass

        total_empennage_cg = total_empennage_cg / total_empennage_mass

        # zero out cg-y and flip x,z depending on reference frame
        if self.fd_axis == self.fd_axis:
            total_empennage_cg = total_empennage_cg * np.array([-1, 0, -1])

        else:
            total_empennage_cg = total_empennage_cg * np.array([1, 0, 1])

        # compute total empennage inertia tensor (about total cg)
        total_empennage_I = csdl.Variable(shape=(3, 3), value=0.)
        
        # parallel axis theorem (parallel axis is total empennage cg)
        x =  total_empennage_cg[0]
        y =  total_empennage_cg[1]
        z =  total_empennage_cg[2]
        
        transl_mat = csdl.Variable(shape=(3, 3), value=0.)
        transl_mat = transl_mat.set(csdl.slice[0, 0], y**2 + z**2)
        transl_mat = transl_mat.set(csdl.slice[0, 1], -x * y)
        transl_mat = transl_mat.set(csdl.slice[0, 2], -x * z)
        transl_mat = transl_mat.set(csdl.slice[1, 0], -y * x)
        transl_mat = transl_mat.set(csdl.slice[1, 1], x**2 + z**2)
        transl_mat = transl_mat.set(csdl.slice[1, 2], -y * z)
        transl_mat = transl_mat.set(csdl.slice[2, 0], -z * x)
        transl_mat = transl_mat.set(csdl.slice[2, 1], -z * y)
        transl_mat = transl_mat.set(csdl.slice[2, 2], x**2 + y**2)
        transl_mat = (total_empennage_mass / 2) * transl_mat

        for it in it_list:
            it_empennage_cg = it + transl_mat
            total_empennage_I = total_empennage_I + it_empennage_cg

        empennage_mps = MassProperties(
            mass=total_empennage_mass,
            cg_vector=total_empennage_cg,
            inertia_tensor=total_empennage_I,
        )

        return empennage_mps

class WingMPSParameters(MPS_Parameters):
    """Compute the mass properties of the wing

    Parameters
    ----------
    wing_area : Union[csdl.Variable, float, int]
        _description_
    wing_AR : Union[csdl.Variable, float, int]
        _description_
    fuselage_length : Union[csdl.Variable, float, int]
        _description_
    battery_mass : Union[csdl.Variable, float, int]
        _description_
    cruise_speed : Union[csdl.Variable, float, int]
        _description_
    fd_axis : Flight Dynamics Axis
        the reference w.r.t. which the mass properties are computed
    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """
    def __init__(self, 
                 wing_AR: Union[csdl.Variable, float, int],
                 fd_axis: Union[Axis, AxisLsdoGeo],
                 wing_area: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm^2'),
                 fuselage_length: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                 battery_mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg'),
                 cruise_speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                 component = None) -> MassProperties:
        self.component = component
        self.fd_axis = fd_axis
        self.parameters: MPS_Parameters = MPS_Parameters(
            wing_AR=wing_AR,
            wing_area=wing_area,
            fuselage_length=fuselage_length,
            battery_mass=battery_mass,
            cruise_speed=cruise_speed
        )

    def compute_wing_mps(self):

        if not isinstance(self.parameters.wing_area, csdl.Variable):
            self.parameters.wing_area = csdl.Variable(shape=(1, ), value=self.parameters.wing_area)

        if not isinstance(self.parameters.wing_AR, csdl.Variable):
            self.parameters.wing_AR = csdl.Variable(shape=(1, ), value=self.parameters.wing_AR)

        if not isinstance(self.parameters.fuselage_length, csdl.Variable):
            self.parameters.fuselage_length = csdl.Variable(shape=(1, ), value=self.parameters.fuselage_length)

        if not isinstance(self.parameters.battery_mass, csdl.Variable):
            self.parameters.battery_mass = csdl.Variable(shape=(1, ), value=self.parameters.battery_mass)

        if not isinstance(self.parameters.cruise_speed, csdl.Variable):
            self.parameters.cruise_speed = csdl.Variable(shape=(1, ), value=self.parameters.cruise_speed)

        cg_vec = csdl.Variable(shape=(3, ), value=0.)
        i_mat = csdl.Variable(shape=(3, 3), value=0.)

        for name, coeffs in wing_reg.items():
            qty = evaluate_regression(
                self.parameters.wing_area, 
                self.parameters.wing_AR, 
                self.parameters.fuselage_length,
                self.parameters.battery_mass, 
                self.parameters.cruise_speed, 
                coeffs
            )
            if "mass" in name:
                m = qty
            elif 'cg_X' in name:
                cg_vec =  cg_vec.set(csdl.slice[0], qty)
            elif 'cg_Z' in name:
                cg_vec =  cg_vec.set(csdl.slice[2], qty)
            elif 'Ixx' in name:
                i_mat = i_mat.set(csdl.slice[0, 0], qty)
            elif 'Iyy' in name:
                i_mat = i_mat.set(csdl.slice[1, 1], qty)
            elif 'Izz' in name:
                i_mat = i_mat.set(csdl.slice[2, 2], qty)
            elif 'Ixz' in name:
                i_mat = i_mat.set(csdl.slice[0, 2], qty)
                i_mat = i_mat.set(csdl.slice[2, 0], qty)

        if self.fd_axis == self.fd_axis:
            cg_vec = cg_vec * np.array([-0.9, 0, -1])

        wing_mps = MassProperties(
            mass=m, cg_vector=cg_vec, inertia_tensor=i_mat
        )

        return wing_mps

class FuselageMPSParameters(MPS_Parameters):
    """Compute the mass properties of the fuselage of NASA's 
    lift-plus-cruise air taxi.

    Parameters
    ----------
    wing_area : Union[csdl.Variable, float, int]
        _description_
    wing_AR : Union[csdl.Variable, float, int]
        _description_
    fuselage_length : Union[csdl.Variable, float, int]
        _description_
    battery_mass : Union[csdl.Variable, float, int]
        _description_
    cruise_speed : Union[csdl.Variable, float, int]
        _description_
    fd_axis : Flight Dynamics Axis
        the reference w.r.t. which the mass properties are computed
    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """
    def __init__(self, 
                wing_AR: Union[csdl.Variable, float, int],
                fd_axis: Union[Axis, AxisLsdoGeo],
                wing_area: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm^2'),
                fuselage_length: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm'),
                battery_mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg'),
                cruise_speed: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'm/s'),
                component = None) -> MassProperties:
        self.component = component
        self.fd_axis = fd_axis
        self.parameters: MPS_Parameters = MPS_Parameters(
            wing_AR=wing_AR,
            wing_area=wing_area,
            fuselage_length=fuselage_length,
            battery_mass=battery_mass,
            cruise_speed=cruise_speed
        )


    def compute_fuselage_mps(self):
        cg_vec = csdl.Variable(shape=(3, ), value=0.)
        i_mat = csdl.Variable(shape=(3, 3), value=0.)

        for name, coeffs in fuselage_reg.items():
            qty = evaluate_regression(
                self.parameters.wing_area, 
                self.parameters.wing_AR, 
                self.parameters.fuselage_length,
                self.parameters.battery_mass, 
                self.parameters.cruise_speed, 
                coeffs
            )
            if "mass" in name:
                m = qty
            elif 'cg_X' in name:
                cg_vec =  cg_vec.set(csdl.slice[0], qty)
            elif 'cg_Z' in name:
                cg_vec =  cg_vec.set(csdl.slice[2], qty)
            elif 'Ixx' in name:
                i_mat = i_mat.set(csdl.slice[0, 0], qty)
            elif 'Iyy' in name:
                i_mat = i_mat.set(csdl.slice[1, 1], qty)
            elif 'Izz' in name:
                i_mat = i_mat.set(csdl.slice[2, 2], qty)
            elif 'Ixz' in name:
                i_mat = i_mat.set(csdl.slice[0, 2], qty)
                i_mat = i_mat.set(csdl.slice[2, 0], qty)

        if self.fd_axis == self.fd_axis:
            cg_vec = cg_vec * np.array([-0.88, 0, -1])

        fuselage_mps = MassProperties(
            mass=m, cg_vector=cg_vec, inertia_tensor=i_mat
        )

        return fuselage_mps


def evaluate_regression(wing_area, wing_AR, fuselage_length, battery_mass, cruise_speed, coeffs):
    qty = coeffs[0] * wing_area + coeffs[1] * wing_AR + coeffs[2] * fuselage_length \
          + coeffs[3] * battery_mass + coeffs[4] * cruise_speed + coeffs[5]
    
    return qty

def evaluate_empennage_regression(htail_area, vtail_area, coeffs):
    qty = coeffs[0] * htail_area + coeffs[1] * vtail_area + coeffs[2]
    
    return qty