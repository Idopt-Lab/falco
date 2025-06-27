import time
import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.vehicle.conditions import aircraft_conditions
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP
from flight_simulator import REPO_ROOT_FOLDER, Q_
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
x57_controls.update_high_lift_control(flap_flag=False, blower_flag=False)

glide = aircraft_conditions.ClimbCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    initial_altitude=Q_(100, 'm'),
    final_altitude=Q_(1, 'm'),  # Altitude from X57 CFD data
    speed=Q_(50, 'm/s'),
    flight_path_angle=Q_(0.5, 'deg'))  # Flight path angle from X57 CFD data


x57_controls.elevator.deflection.set_as_design_variable(lower=np.deg2rad(-30), upper=np.deg2rad(30))
glide.parameters.pitch_angle.set_as_design_variable(lower=np.deg2rad(-5), upper=np.deg2rad(10)) # in radians, this is the pitch angle range for the optimization
glide.parameters.flight_path_angle.set_as_design_variable(lower=np.deg2rad(-5), upper=np.deg2rad(10)) # in radians, this is the pitch angle range for the optimization



for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1e-12)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1e-12)
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
aero_results = x57_aerodynamics.get_FM_localAxis(states=glide.ac_states, controls=x57_controls, axis=axis_dict['fd_axis'])
CL = aero_results['CL']
CD = aero_results['CD']


HL_radius_x57 = Q_(1.89/2, 'ft') # HL propeller radius in ft
cruise_radius_x57 = Q_(5/2, 'ft') # cruise propeller radius in ft


hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

for i, hl_motor in enumerate(hl_motors):
    hl_prop = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(),engine_index=i,RPMmin=1500, RPMmax=5400)
    hl_motor.load_solvers.append(hl_prop)
    results = hl_prop.get_torque_power(states=glide.ac_states, controls=x57_controls)
    hl_engine_torque = results['torque']
    hl_engine_torque.name = f'HL_Engine_{i}_Torque'
    hl_engine_torque.set_as_constraint(upper=1e-6)  # setting the torque to 0 for the glide condition

cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

for i, cruise_motor in enumerate(cruise_motors):
    engine_index = len(hl_motors) + i
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index, RPMmin=1700, RPMmax=2700)
    cruise_motor.load_solvers.append(cruise_prop)
    results = cruise_prop.get_torque_power(states=glide.ac_states, controls=x57_controls)
    cm_engine_torque = results['torque']
    cm_engine_torque.name = f'Cruise_Engine_{i}_Torque'
    cm_engine_torque.set_as_constraint(upper=1e-6)  # setting the torque to 0 for the glide condition 



tf, tm = aircraft_component.compute_total_loads(fd_state=glide.ac_states,controls=x57_controls)

cruise_r, cruise_x = glide.evaluate_eom(component=aircraft_component, forces=tf, moments=tm)
h_dot = cruise_r[11]
h_dot.name = 'dzdt'



Drag = - x57_aerodynamics.get_FM_localAxis(states=glide.ac_states, controls=x57_controls, axis=axis_dict['fd_axis'])['loads'].F.vector[0]
Lift = - x57_aerodynamics.get_FM_localAxis(states=glide.ac_states, controls=x57_controls, axis=axis_dict['fd_axis'])['loads'].F.vector[2]
Moment = x57_aerodynamics.get_FM_localAxis(states=glide.ac_states, controls=x57_controls, axis=axis_dict['fd_axis'])['loads'].M.vector[1]
ThrustR = Drag

Lift_scaling = 1/csdl.absolute(Lift)
Drag_scaling = 1/csdl.absolute(Drag)
Moment_scaling = 1/csdl.absolute(Moment)


res1 = tf[0] * Drag_scaling
res1.name = 'Fx Force'
res1.set_as_constraint(lower=-1e-6, upper=1e-6)  # setting a small value to ensure the thrust is close to zero

res2 = tf[2] * Lift_scaling
res2.name = 'Fz Force'
res2.set_as_constraint(lower=-1e-6, upper=1e-6)

res4 = tm[1] * Moment_scaling
res4.name = 'My Moment'
res4.set_as_constraint(lower=-1e-6, upper=1e-6)


max_LD = csdl.absolute(1 / (csdl.tan((glide.ac_states.gamma)))) # L/D ratio is 1/tan(gamma) where gamma is the flight path angle
negative_max_LD = -max_LD * 1e-1  # scaling to make it a minimization problem
negative_max_LD.name = 'Negative Max L/D Ratio'
negative_max_LD.set_as_objective()






sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[glide.parameters.speed, glide.parameters.initial_altitude],
    additional_outputs=[h_dot],
    derivatives_kwargs= {
        "concatenate_ofs" : True})





initial_altitudes = np.array([8000]) * 0.3048 # altitudes in meters
speeds = np.array([150]) / 1.944 # KTAS to m/s

results_dict = {
    'VTAS': [],
    'Drag': [],
    'Lift': [],
    'L/D Ratio': [],
    'Propeller Efficiency': [],
    'J Value': [],
    'Sink Rate': [],
    'CL': [],
    'CD': [],
    'dSink/dVTAS': [],
    'Max L/D Ratio': [],
}

for j, initial_altitude in enumerate(initial_altitudes):
    for i, speed in enumerate(speeds):


        sim[glide.parameters.speed] = speed
        sim[glide.parameters.initial_altitude] = initial_altitude

        recorder.execute()
        sim.check_optimization_derivatives()
        derivatives = sim.compute_totals()
        dsinkdVTAS = derivatives[h_dot, glide.parameters.speed]
        t1 = time.time()
        prob = CSDLAlphaProblem(problem_name='trim_glide_perf_opt', simulator=sim)
        optimizer = IPOPT(problem=prob)
        optimizer.solve()
        optimizer.print_results()
        t2 = time.time()
        print('Time to solve Optimization:', t2-t1)
        recorder.execute()
        print("max L/D Ratio:", max_LD.value)

        
        results = cruise_prop.get_torque_power(states=glide.ac_states, controls=x57_controls)
        prop_efficiency = results['eta']
        results_dict['Propeller Efficiency'].append(results['eta'].value[0])
        results_dict['VTAS'].append(glide.ac_states.VTAS.value[0])
        results_dict['Drag'].append(Drag.value[0])
        results_dict['Lift'].append(Lift.value[0])
        results_dict['L/D Ratio'].append(Lift.value[0] / Drag.value[0])
        results_dict['Max L/D Ratio'].append(max_LD.value[0])
        results_dict['Sink Rate'].append(h_dot.value[0])
        results_dict['CL'].append(CL.value[0])
        results_dict['CD'].append(CD.value[0])
        results_dict['dSink/dVTAS'].append(dsinkdVTAS)
        results_dict['J Value'].append(results['J'].value[0])


        dv_save_dict = {}
        constraints_save_dict = {}
        obj_save_dict = {}

        dv_dict = recorder.design_variables
        constraint_dict = recorder.constraints
        obj_dict = recorder.objectives

        for dv in dv_dict.keys():
            dv_save_dict[dv.name] = dv.value
            # print("Design Variable", dv.name, dv.value)

        for c in constraint_dict.keys():
            constraints_save_dict[c.name] = c.value
            # print("Constraint", c.name, c.value)

        for obj in obj_dict.keys():
            obj_save_dict[obj.name] = obj.value
            # print("Objective", obj.name, obj.value)

        print("=====Aircraft States=====")
        print("Aircraft Conditions")
        print(glide)
        print("Aircraft Mach Number")
        print(glide.ac_states.Mach.value)
        print("Time to Reach Final Altitude (min)")
        print((glide.parameters.initial_altitude.value / (glide.ac_states.VTAS.value / (CL.value / CD.value)))/60)
        print("Sink Rate (m/s)")
        print(glide.ac_states.VTAS.value / (CL.value / CD.value))
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
        print(glide.parameters.pitch_angle.value * 180 / np.pi)
        print('Angle of Attack (deg)')
        print(glide.ac_states.alpha.value * 180 / np.pi)





        print("TF[0]", tf[0].value)
        print("TF[1]", tf[1].value)
        print("TF[2]", tf[2].value)
        print("TM[0]", tm[0].value)
        print("TM[1]", tm[1].value)
        print("TM[2]", tm[2].value)




# Convert the dictionary to a DataFrame
results_df = pd.DataFrame(results_dict)

import os
outputs_folder_path = REPO_ROOT_FOLDER / 'AIAA Aviation 2025' / 'X-57'
outputs_dir = os.path.join(outputs_folder_path, 'optimizations/results')
os.makedirs(outputs_dir, exist_ok=True)
results_df.to_csv(os.path.join(outputs_dir, 'glide_perf_trim_sim_results.csv'), index=False)



recorder.stop()
