import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_


class WeightsModel:
    def __init__(self, 
                 design_weight : Union[float, int, csdl.Variable],
                 dynamic_pressure : Union[float, int, csdl.Variable],
                 )->None:
        

        if isinstance(design_weight, ureg.Quantity):
            self.design_gross_weight = csdl.Variable(name='design_weight', shape=(1,), value=design_weight.to_base_units().magnitude)
        else:
            self.design_gross_weight = design_weight

        if isinstance(dynamic_pressure, ureg.Quantity):
            self.dynamic_pressure = csdl.Variable(name='dynamic_pressure', shape=(1,), value=dynamic_pressure.to_base_units().magnitude)
        else:
            self.dynamic_pressure = dynamic_pressure
    

    def evaluate_wing_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        AR : Union[float, int, csdl.Variable],
        sweep : Union[float, int, csdl.Variable],
        taper_ratio : Union[float, int, csdl.Variable] = 1.,
        thickness_to_chord : Union[float, int, csdl.Variable]=0.12,
        nz : Union[float, int, csdl.Variable] = 3.75,
        fuel_weight : Union[float, int, csdl.Variable] = 0,
    ):
        

        if isinstance(S_ref, ureg.Quantity):
            S_ref = csdl.Variable(name='S_ref', shape=(1,), value=S_ref.to_base_units().magnitude)
        else:
            S_ref = S_ref
        
        if isinstance(fuel_weight, ureg.Quantity):
            fuel_weight = csdl.Variable(name='fuel_weight', shape=(1,), value=fuel_weight.to_base_units().magnitude)
        else:
            fuel_weight = fuel_weight

        if isinstance(sweep, ureg.Quantity):
            sweep = csdl.Variable(name='sweep', shape=(1,), value=sweep.to_base_units().magnitude)
        else:
            sweep = sweep
        


        W_wing = 0.036 * S_ref**0.758 * fuel_weight**0.035 * (AR/csdl.cos(sweep)**2)**0.6 * self.dynamic_pressure**0.006 \
        * taper_ratio**0.04 * (100 * thickness_to_chord / csdl.cos(sweep))**-0.3 * (nz * self.design_gross_weight)**0.49 

        return W_wing
    

    def evaluate_fuselage_weight(
        self,
        Nult : Union[float, int, csdl.Variable] = 3.75,
        fuselage_length : Union[float, int, csdl.Variable] = None,
        fuselage_height : Union[float, int, csdl.Variable] = None,
        fuselage_width : Union[float, int, csdl.Variable] = None,
    ):
        
        if isinstance(fuselage_length, ureg.Quantity):
            fuselage_length = csdl.Variable(name='fuselage_length', shape=(1,), value=fuselage_length.to_base_units().magnitude)
        else:
            fuselage_length = fuselage_length
        
        if isinstance(fuselage_height, ureg.Quantity):
            fuselage_height = csdl.Variable(name='fuselage_height', shape=(1,), value=fuselage_height.to_base_units().magnitude)
        else:
            fuselage_height = fuselage_height

        if isinstance(fuselage_width, ureg.Quantity):
            fuselage_width = csdl.Variable(name='fuselage_width', shape=(1,), value=fuselage_width.to_base_units().magnitude)
        else:
            fuselage_width = fuselage_width

        if fuselage_length is not None and fuselage_height is not None and fuselage_width is not None:
            xl = fuselage_length
            
            average_fuselage_diameter = (fuselage_height + fuselage_width) / 2

            d_av = average_fuselage_diameter

            S_wet = 3.14159 * (xl / d_av - 1.7) * d_av**2

        else:
            raise Exception("Insufficient inputs defined")

        W_fuse = 0.052 * S_wet**1.086 * (Nult * self.design_gross_weight) ** 0.177 * self.dynamic_pressure**0.241 

        return W_fuse 
    



    def evaluate_horizontal_tail_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        Nult : Union[float, int, csdl.Variable] = 4.5,
    ):
        
        if isinstance(S_ref, ureg.Quantity):
            S_ref = csdl.Variable(name='S_ref', shape=(1,), value=S_ref.to_base_units().magnitude)
        else:
            S_ref = S_ref


        W_h_tail = 0.016 * S_ref**0.873 * (Nult * self.design_gross_weight)**0.414 * self.dynamic_pressure**0.122

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
            S_ref = csdl.Variable(name='S_ref', shape=(1,), value=S_ref.to_base_units().magnitude)
        else:
            S_ref = S_ref

        if isinstance(sweep_c4, ureg.Quantity):
            sweep_c4 = csdl.Variable(name='sweep_c4', shape=(1,), value=sweep_c4.to_base_units().magnitude)
        else:
            sweep_c4 = sweep_c4
            sweep_c4.value = np.deg2rad(sweep_c4.value)

        W_v_tail = 0.073 * (1.0 + 0.2 * hht) * (Nult * self.design_gross_weight)**0.376 * self.dynamic_pressure**0.122 * \
            S_ref**0.873 * (AR / csdl.cos(sweep_c4)**2)**0.357 / ((100 * thickness_to_chord) / csdl.cos(sweep_c4))**0.49
        
        return W_v_tail
    


class WeightsSolverModel:
    def evaluate(self, gross_weight_guess : csdl.ImplicitVariable, *component_weights):

        if isinstance(gross_weight_guess, ureg.Quantity):
            gross_weight_guess = csdl.Variable(name='gross_weight_guess', shape=(1,), value=gross_weight_guess.to_base_units().magnitude)
        else:
            gross_weight_guess = gross_weight_guess
    
        gross_weight = csdl.Variable(shape=(1, ), value=0)

        for weight in component_weights:
            gross_weight =  gross_weight +  weight

        weight_residual = gross_weight_guess - gross_weight

        solver = csdl.nonlinear_solvers.Newton()
        solver.add_state(gross_weight_guess, weight_residual)
        solver.run()
        
    