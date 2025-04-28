import time
import csdl_alpha as csdl
import numpy as np
from flight_simulator import REPO_ROOT_FOLDER, Q_
from flight_simulator.core.vehicle.conditions import aircraft_conditions
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP
import matplotlib.pyplot as plt
import sys




recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
recorder.start()

x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))

from x57_vehicle_models import X57ControlSystem, fd_axis, build_aircraft, X57Propulsion, X57Aerodynamics

Aircraft = build_aircraft(do_geo_param=False)
x57_controls = X57ControlSystem(hl_engine_count=12,cm_engine_count=2,symmetrical=True)
x57_controls.elevator.deflection = Aircraft.comps['Elevator'].parameters.actuate_angle
x57_controls.rudder.deflection = Aircraft.comps['Rudder'].parameters.actuate_angle
x57_controls.aileron.deflection = Aircraft.comps['Left Aileron'].parameters.actuate_angle
x57_controls.flap.deflection = Aircraft.comps['Left Flap'].parameters.actuate_angle
x57_controls.trim_tab.deflection = Aircraft.comps['Trim Tab'].parameters.actuate_angle



hl_propulsions = []
cruise_propulsions = []

if hasattr(Aircraft, 'load_solvers'):
    for solver in Aircraft.load_solvers:
        if isinstance(solver, X57Aerodynamics):
            aero_loads=solver


for comp in Aircraft.comps.values():
    if hasattr(comp, 'load_solvers'):
        for solver in comp.load_solvers:
            if isinstance(solver, X57Propulsion):
                if "HL Motor" in comp._name:
                    hl_propulsions.append(solver)
                elif "Cruise Motor" in comp._name:
                    cruise_propulsions.append(solver)







takeoff = aircraft_conditions.ClimbCondition(
    fd_axis=fd_axis,
    controls=x57_controls,
    initial_altitude=Q_(0, 'm'),
    final_altitude=Q_(1000, 'm'),
    speed=Q_(50, 'mph'),
    pitch_angle=Q_(10, 'deg'),
    flight_path_angle=Q_(10, 'deg'))

cruise = aircraft_conditions.CruiseCondition(
    fd_axis=fd_axis,
    controls=x57_controls,
    altitude=Q_(1000, 'm'),
    range=Q_(70, 'km'),
    speed=Q_(100, 'mph'),
    pitch_angle=Q_(0, 'deg'))

land = aircraft_conditions.ClimbCondition(
    fd_axis=fd_axis,
    controls=x57_controls,
    initial_altitude=Q_(1000, 'm'),
    final_altitude=Q_(0, 'm'),
    speed=Q_(50, 'mph'),
    pitch_angle=Q_(0, 'deg'),
    flight_path_angle=Q_(0, 'deg'))

# x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
# x57_controls.rudder.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
# x57_controls.aileron.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
# x57_controls.flap.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
# x57_controls.trim_tab.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))




for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL Throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    hl_throt_diff.set_as_constraint(lower=-1e-6, upper=1e-6)


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    cm_thrott_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_thrott_diff.name = f'CM Throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_thrott_diff.set_as_constraint(lower=-1e-6, upper=1e-6)
                        


J = cruise.evaluate_trim_res(component=Aircraft)
J.name = 'J'
J.set_as_constraint(lower=-1e-6, upper=1e-6, scaler=1e-4)

mission_phases = [takeoff, cruise, land]

currentThrust = []
currentPwr = []
MissionThrust = []
MissionDrag = []
MissionPwr = []
PhaseThrust = []
PhaseDrag = []
PhasePwr = []

def add_linking_constraints(phases):
    # Combine constraints into a csdl model
    for i in range(len(phases)-1):

        # for hl in hl_propulsions:
        #     propload = hl.get_FM_localAxis(states=phases[i].ac_states, controls=x57_controls)
        #     prop_pwr = hl.get_power(states=phases[i].ac_states, controls=x57_controls)
        #     propload_new = propload.rotate_to_axis(fd_axis)
        #     thrust = propload_new.F.vector[0]
        #     currentThrust.append(thrust)
        #     currentPwr.append(prop_pwr)

        # for cr in cruise_propulsions:
        #     propload = cr.get_FM_localAxis(states=phases[i].ac_states, controls=x57_controls)
        #     prop_pwr = cr.get_power(states=phases[i].ac_states, controls=x57_controls)
        #     propload_new = propload.rotate_to_axis(fd_axis)
        #     thrust = propload_new.F.vector[0]
        #     currentThrust.append(thrust)
        #     currentPwr.append(prop_pwr)

        # PhaseThrust.append(np.sum(currentThrust))

        # PhasePwr.append(np.sum(currentPwr))



        loads = aero_loads.get_FM_localAxis(states=phases[i].ac_states, controls=x57_controls)
        loads_new = loads.rotate_to_axis(fd_axis)
        drag = csdl.absolute(loads_new.F.vector[0])

        PhaseDrag.append(drag)


        current_state = phases[i]
        next_phase = phases[i+1]


        if hasattr(current_state.parameters, 'final_altitude'):
            alt1 = current_state.parameters.final_altitude
        elif hasattr(current_state.parameters, 'altitude'):
            alt1 = current_state.parameters.altitude

        

        if hasattr(next_phase.parameters, 'initial_altitude'):
            alt2 = next_phase.parameters.initial_altitude
        elif hasattr(next_phase.parameters, 'altitude'):
            alt2 = next_phase.parameters.altitude


        
        alt_var = csdl.absolute(alt1-alt2)
        print('Altitude Linking Constraint', i, alt_var.value)
        alt_var.name = f'Altitude Linking Constraint {i}'
        alt_var.set_as_constraint(lower=-1e-6, upper=1e-6)





add_linking_constraints(mission_phases)
MissionDrag = csdl.sum(*PhaseDrag)
MissionDrag.name = 'Mission Drag'
MissionDrag.set_as_objective()
        


sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    derivatives_kwargs= {
        "concatenate_ofs" : True
    })




sim.check_optimization_derivatives()
t1 = time.time()
prob = CSDLAlphaProblem(problem_name='combined_mission_optimization', simulator=sim)
optimizer = IPOPT(problem=prob)
optimizer.solve()
optimizer.print_results()
t2 = time.time()
print('Time to solve Optimization:', t2-t1)
recorder.execute()

dv_save_dict = {}
constraints_save_dict = {}
obj_save_dict = {}

dv_dict = recorder.design_variables
constraint_dict = recorder.constraints
obj_dict = recorder.objectives

for dv in dv_dict.keys():
    dv_save_dict[dv.name] = dv.value
    print("Design Variable", dv.name, dv.value)

for c in constraint_dict.keys():
    constraints_save_dict[c.name] = c.value
    print("Constraint", c.name, c.value)

for obj in obj_dict.keys():
    obj_save_dict[obj.name] = obj.value
    print("Objective", obj.name, obj.value)




recorder.stop()
