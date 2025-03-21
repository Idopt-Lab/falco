import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_


class StructuralWeights:
    def __init__(self, states, cg, mass, design_weight, atmospheric_states:csdl.VariableGroup=None)->None:

        self.states = states
        self.cg = cg
        self.mass = mass
        self.design_gross_weight = design_weight
        self.velocity = self.states.VTAS

        if atmospheric_states is None:
            raise Exception("Atmospheric Conditions/states are not provided.")
        else:
            self.atmospheric_states = atmospheric_states
            self.dynamic_pressure = 0.5 * self.atmospheric_states.density * self.velocity**2


    def evaluate_wing_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        batt_weight : Union[float, int, csdl.Variable],
        AR : Union[float, int, csdl.Variable],
        sweep : Union[float, int, csdl.Variable],
        taper_ratio : Union[float, int, csdl.Variable],
        thickness_to_chord : Union[float, int, csdl.Variable]=0.12,
        nz : Union[float, int, csdl.Variable] = 3.75,
    ):
        

        if isinstance(S_ref, ureg.Quantity):
            self.S_ref = csdl.Variable(name='S_ref', shape=(1,), value=S_ref.to_base_units().magnitude)
        else:
            self.S_ref = S_ref

        if isinstance(batt_weight, ureg.Quantity):
            self.batt_weight = csdl.Variable(name='batt_weight', shape=(1,), value=batt_weight.to_base_units().magnitude)
        else:
            self.batt_weight = batt_weight

        if isinstance(sweep, ureg.Quantity):
            self.sweep = csdl.Variable(name='sweep', shape=(1,), value=sweep.to_base_units().magnitude)
        else:
            self.sweep = sweep
        


        W_wing = 0.036 * self.S_ref**0.758 * self.batt_weight**0.035 * (AR/csdl.cos(self.sweep)**2)**0.6 * self.dynamic_pressure**0.006 \
        * taper_ratio**0.04 * (100 * thickness_to_chord / csdl.cos(self.sweep))**-0.3 * (nz * self.design_gross_weight)**0.49 

        return W_wing
    

    def evaluate_fuselage_weight(
        self,
        Nult : Union[float, int, csdl.Variable] = 3.75,
        fuselage_length : Union[float, int, csdl.Variable] = None,
        fuselage_height : Union[float, int, csdl.Variable] = None,
        fuselage_width : Union[float, int, csdl.Variable] = None,
    ):
        
        if isinstance(fuselage_length, ureg.Quantity):
            self.fuselage_length = csdl.Variable(name='fuselage_length', shape=(1,), value=fuselage_length.to_base_units().magnitude)
        else:
            self.fuselage_length = fuselage_length
        
        if isinstance(fuselage_height, ureg.Quantity):
            self.fuselage_height = csdl.Variable(name='fuselage_height', shape=(1,), value=fuselage_height.to_base_units().magnitude)
        else:
            self.fuselage_height = fuselage_height

        if isinstance(fuselage_width, ureg.Quantity):
            self.fuselage_width = csdl.Variable(name='fuselage_width', shape=(1,), value=fuselage_width.to_base_units().magnitude)
        else:
            self.fuselage_width = fuselage_width

        if fuselage_length is not None and fuselage_height is not None and fuselage_width is not None:
            xl = self.fuselage_length
            
            average_fuselage_diameter = (self.fuselage_height + self.fuselage_width) / 2

            d_av = average_fuselage_diameter

            self.S_wet = 3.14159 * (xl / d_av - 1.7) * d_av**2

        else:
            raise Exception("Insufficient inputs defined")

        W_fuse = 0.052 * self.S_wet**1.086 * (Nult * self.design_gross_weight) ** 0.177 * self.dynamic_pressure**0.241 

        return W_fuse 
    



    def evaluate_horizontal_tail_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        Nult : Union[float, int, csdl.Variable] = 4.5,
    ):
        
        if isinstance(S_ref, ureg.Quantity):
            self.S_ref = csdl.Variable(name='S_ref', shape=(1,), value=S_ref.to_base_units().magnitude)
        else:
            self.S_ref = S_ref


        W_h_tail = 0.016 * self.S_ref**0.873 * (Nult * self.design_gross_weight)**0.414 * self.dynamic_pressure**0.122

        return W_h_tail
    
    def evaluate_vertical_tail_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        AR : Union[float, int, csdl.Variable],
        sweep_c4 : Union[float, int, csdl.Variable],
        thickness_to_chord : Union[float, int, csdl.Variable] = 0.12,
        Nult : Union[float, int, csdl.Variable] = 4.5,
        hht : Union[float, int, csdl.Variable] = 0.,
    ):
        

        if isinstance(S_ref, ureg.Quantity):
            self.S_ref = csdl.Variable(name='S_ref', shape=(1,), value=S_ref.to_base_units().magnitude)
        else:
            self.S_ref = S_ref

        if isinstance(sweep_c4, ureg.Quantity):
            self.sweep_c4 = csdl.Variable(name='sweep_c4', shape=(1,), value=sweep_c4.to_base_units().magnitude)
        else:
            self.sweep_c4 = sweep_c4


        W_v_tail = 0.073 * (1.0 + 0.2 * hht) * (Nult * self.design_gross_weight)**0.376 * self.dynamic_pressure**0.122 * \
            self.S_ref**0.873 * (AR / csdl.cos(self.sweep_c4)**2)**0.357 / ((100 * thickness_to_chord) / csdl.cos(self.sweep_c4))**0.49
        
        return W_v_tail
        
    