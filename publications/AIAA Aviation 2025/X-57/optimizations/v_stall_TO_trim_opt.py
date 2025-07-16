import time
import csdl_alpha as csdl
import numpy as np
from falco.core.vehicle.conditions import aircraft_conditions
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP
from falco import REPO_ROOT_FOLDER, Q_
import sys
import pandas as pd


x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))
from x57_geometry import get_geometry, get_geometry_related_axis
from x57_component import build_aircraft_component
from x57_mp import add_mp_to_components
from x57_control_system import X57ControlSystem, Blower
from x57_solvers import X57Aerodynamics, X57Propulsion, HLPropCurve, CruisePropCurve

debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
recorder.start()

geometry_data = get_geometry()
axis_dict = get_geometry_related_axis(geometry_data)
aircraft_component = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)

add_mp_to_components(aircraft_component, geometry_data, axis_dict)

aircraft_component.mass_properties = aircraft_component.compute_total_mass_properties()
print(repr(aircraft_component))

hlb = Blower()

x57_controls = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                hl_engine_count=12,cm_engine_count=2, high_lift_blower_component=hlb)
x57_controls.update_high_lift_control(flap_flag=True, blower_flag=True)


takeoff = aircraft_conditions.ClimbCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    initial_altitude=Q_(1, 'm'),
    final_altitude=Q_(2438.4, 'm'),
    mach_number=Q_(0.15, 'dimensionless')) 





x57_controls.elevator.deflection.set_as_design_variable(lower=np.deg2rad(-30), upper=np.deg2rad(30))
x57_controls.flap_left.deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(-10))
x57_controls.flap_right.deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(-10))
takeoff.parameters.pitch_angle.set_as_design_variable(lower=np.deg2rad(-5), upper=np.deg2rad(10)) 
takeoff.parameters.mach_number.set_as_design_variable(lower=0.05, upper=0.3) 

flap_diff = (x57_controls.flap_right.deflection - x57_controls.flap_left.deflection)
flap_diff.name = f'Flap Diff{x57_controls.flap_left.deflection.name} - {x57_controls.flap_right.deflection.name}'
flap_diff.set_as_constraint(equals=0)  



for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    hl_throt_diff.set_as_constraint(equals=0)  


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    cm_throttle_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_throttle_diff.name = f'CM throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_throttle_diff.set_as_constraint(equals=0)  


x57_controls.update_controls(x57_controls.u())    

x57_aerodynamics = X57Aerodynamics(component=aircraft_component)
aircraft_component.comps['Wing'].load_solvers.append(x57_aerodynamics)
aero_results = x57_aerodynamics.get_FM_localAxis(states=takeoff.ac_states, controls=x57_controls, axis=axis_dict['fd_axis'])
CL = aero_results['CL']
CD = aero_results['CD']


HL_radius_x57 = Q_(1.89/2, 'ft') # HL propeller radius in ft
cruise_radius_x57 = Q_(5/2, 'ft') # cruise propeller radius in ft


hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

for i, hl_motor in enumerate(hl_motors):
    hl_prop = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(),engine_index=i,RPMmin=1500, RPMmax=5400)
    hl_motor.load_solvers.append(hl_prop)
    results = hl_prop.get_torque_power(states=takeoff.ac_states, controls=x57_controls)
    hl_engine_torque = results['torque']
    hl_engine_torque.name = f'HL_Engine_{i}_Torque'
    hl_engine_torque.set_as_constraint(lower=0.0, upper=20.5, scaler=1e-2) # values from High-Lift Propeller Operating Conditions v2 paper

cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

for i, cruise_motor in enumerate(cruise_motors):
    engine_index = len(hl_motors) + i
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index, RPMmin=1700, RPMmax=2700)
    cruise_motor.load_solvers.append(cruise_prop)
    results = cruise_prop.get_torque_power(states=takeoff.ac_states, controls=x57_controls)
    cm_engine_torque = results['torque']
    cm_engine_torque.name = f'Cruise_Engine_{i}_Torque'
    cm_engine_torque.set_as_constraint(lower=0.0, upper=225, scaler=1e-3) # values from x57_DiTTo_manuscript paper



tf, tm = aircraft_component.compute_total_loads(fd_state=takeoff.ac_states,controls=x57_controls)

takeoff_r, takeoff_x = takeoff.evaluate_eom(component=aircraft_component, forces=tf, moments=tm)
h_dot = takeoff_r[11]



Drag = x57_aerodynamics.get_FM_localAxis(states=takeoff.ac_states, controls=x57_controls, axis=axis_dict['wing_axis'])['loads'].F.vector[0]
Lift = x57_aerodynamics.get_FM_localAxis(states=takeoff.ac_states, controls=x57_controls, axis=axis_dict['wing_axis'])['loads'].F.vector[2]
Moment = x57_aerodynamics.get_FM_localAxis(states=takeoff.ac_states, controls=x57_controls, axis=axis_dict['wing_axis'])['loads'].M.vector[1]
ThrustR = tf[0] - csdl.absolute(Drag)

Lift_scaling = 1/csdl.absolute(Lift)
Drag_scaling = 1/csdl.absolute(Drag)
Moment_scaling = 1/csdl.absolute(Moment)


res1 = (tf[0]) * Drag_scaling
res1.name = 'Fx Residual' 
res1.set_as_constraint(equals=0.0) 

res2 = (tf[2]) * Lift_scaling
res2.name = 'Fz Residual'
# res2.set_as_constraint(equals=0.0)

res3 = (tm[1]) * Moment_scaling
res3.name = 'My Residual'
res3.set_as_constraint(equals=0.0) 


V_stall = takeoff_r[9]
velocity_residual = takeoff_r[9] * 1e-1
velocity_residual.set_as_objective()







sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[takeoff.parameters.initial_altitude],
    derivatives_kwargs= {
        "concatenate_ofs" : True})


altitudes = np.array([2500]) * 0.308 # in m, this is the altitude from the X57 CFD data

sim_results = {
    'VTAS': [],
    'Initial Altitude': [],
    'Final Altitude': [],
    'Drag': [],
    'Lift': [],
    'CL': [],
    'CD': [],
    'V_stall': [],
}




for i, alt in enumerate(altitudes):

    
    sim[takeoff.parameters.initial_altitude] = alt
    recorder.execute()


    sim.check_optimization_derivatives()
    t1 = time.time()
    prob = CSDLAlphaProblem(problem_name='v_stall_TO_trim_opt', simulator=sim)
    optimizer = IPOPT(problem=prob)
    optimizer.solve()
    optimizer.print_results()
    t2 = time.time()
    print('Time to solve Optimization:', t2-t1)
    recorder.execute()



    sim_results['VTAS'].append(takeoff_r[9].value)
    sim_results['Initial Altitude'].append(takeoff.parameters.initial_altitude.value[0])
    sim_results['Final Altitude'].append(takeoff.parameters.final_altitude.value[0])
    sim_results['Drag'].append(Drag.value[0])
    sim_results['Lift'].append(Lift.value[0])
    sim_results['CL'].append(CL.value[0])
    sim_results['CD'].append(CD.value[0])
    sim_results['V_stall'].append(V_stall.value[0])


    dv_save_dict = {}
    constraints_save_dict = {}
    obj_save_dict = {}

    dv_dict = recorder.design_variables
    constraint_dict = recorder.constraints
    obj_dict = recorder.objectives

    for dv in dv_dict.keys():
        dv_save_dict[dv.name] = dv.value

    for c in constraint_dict.keys():
        constraints_save_dict[c.name] = c.value

    for obj in obj_dict.keys():
        obj_save_dict[obj.name] = obj.value

    print("=====Aircraft States=====")
    print("Aircraft Conditions")
    print(takeoff)
    print("Aircraft Mach Number")
    print(takeoff.ac_states.Mach.value)
    print("Time to Reach Final Altitude (min)")
    print((takeoff.parameters.initial_altitude.value / (takeoff.ac_states.VTAS.value / (CL.value / CD.value)))/60)
    print("Throttle")
    for engine in x57_controls.engines:
        print(engine.throttle.value)
    print("High Lift Engine Torque (N*m)")
    for engine in x57_controls.hl_engines:  
        print(hl_engine_torque.value)  
    print("Cruise Engine Torque (N*m)")
    for engine in x57_controls.cm_engines:
        print(cm_engine_torque.value)
    print("Elevator Deflection (deg)")
    print(x57_controls.pitch_control['Elevator'].deflection.value * 180 / np.pi)
    print("Rudder Deflection (deg)")
    print(x57_controls.yaw_control['Rudder'].deflection.value * 180 / np.pi)
    print("Left Aileron Deflection (deg)")
    print(x57_controls.roll_control['Left Aileron'].deflection.value * 180 / np.pi)
    print("Right Aileron Deflection (deg)")
    print(x57_controls.roll_control['Right Aileron'].deflection.value * 180 / np.pi)
    print("Left Flap Deflection (deg)")
    print(x57_controls.high_lift_control['Left Flap'].deflection.value * 180 / np.pi)
    print("Right Flap Deflection (deg)")
    print(x57_controls.high_lift_control['Right Flap'].deflection.value * 180 / np.pi)
    print("Trim Tab Deflection (deg)")
    print(x57_controls.pitch_control['Trim Tab'].deflection.value * 180 / np.pi)
    print("Pitch Angle (deg)")
    print(takeoff.parameters.pitch_angle.value * 180 / np.pi)
    print('Angle of Attack (deg)')
    print(takeoff.ac_states.alpha.value * 180 / np.pi)
    print("Flight Path Angle (deg)")
    print(takeoff.ac_states.gamma.value * 180 / np.pi)
    print('CL')
    print(CL.value)
    print('CD')
    print(CD.value)
    print("V_stall: (m/s) | (KTAS)")
    print(V_stall.value[0], V_stall.value[0] * 1.944)

    
    print("TF[0]", tf[0].value)
    print("TF[1]", tf[1].value)
    print("TF[2]", tf[2].value)
    print("TM[0]", tm[0].value)
    print("TM[1]", tm[1].value)
    print("TM[2]", tm[2].value)



    








# Convert the dictionary to a DataFrame
results_df = pd.DataFrame(sim_results)



import os
outputs_folder_path = REPO_ROOT_FOLDER / 'AIAA Aviation 2025' / 'X-57'
outputs_dir = os.path.join(outputs_folder_path, 'optimizations/results')
os.makedirs(outputs_dir, exist_ok=True)
results_df.to_csv(os.path.join(outputs_dir, 'v_stall_TO_trim_opt.csv'), index=False)






recorder.stop()
