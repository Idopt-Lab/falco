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

from x57_vehicle_models import X57ControlSystem, fd_axis, build_aircraft, X57Propulsion, X57Aerodynamics, HL_motor_axes, cruise_motor_axes, wing_axis

Aircraft = build_aircraft(do_geo_param=False)
x57_controls = X57ControlSystem(hl_engine_count=12,cm_engine_count=2,symmetrical=True)
x57_controls.elevator.deflection = Aircraft.comps['Elevator'].parameters.actuate_angle
x57_controls.rudder.deflection = Aircraft.comps['Rudder'].parameters.actuate_angle
x57_controls.aileron.deflection = Aircraft.comps['Left Aileron'].parameters.actuate_angle
x57_controls.flap.deflection = Aircraft.comps['Left Flap'].parameters.actuate_angle
x57_controls.trim_tab.deflection = Aircraft.comps['Trim Tab'].parameters.actuate_angle



cruise = aircraft_conditions.CruiseCondition(
    fd_axis=fd_axis,
    controls=x57_controls,
    altitude=Q_(1000, 'm'),
    range=Q_(70, 'km'),
    speed=Q_(100, 'mph'),
    pitch_angle=Q_(0, 'deg'))
print(cruise)

x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
x57_controls.rudder.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
x57_controls.aileron.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
x57_controls.flap.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
x57_controls.trim_tab.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
# cruise.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10))
# cruise.parameters.altitude.set_as_design_variable(lower=1, upper=2000)
# cruise.parameters.speed.set_as_design_variable(lower=0.1, upper=200)



for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=0.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=0.0)
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL Throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    hl_throt_diff.set_as_constraint(equals=0)


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    cm_thrott_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_thrott_diff.name = f'CM Throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_thrott_diff.set_as_constraint(equals=0)
                        
tf, tm = Aircraft.compute_total_loads(fd_state=cruise.ac_states,
                                            controls=cruise.controls)
print('Total Forces:', tf.value)
print('Total Moments:', tm.value)

J = cruise.evaluate_trim_res(component=Aircraft)
J.name = 'J'
J.set_as_constraint(equals=0, scaler=1e-2)


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


currentThrust = []
currentPwr = []
for i, hl in enumerate(hl_propulsions):
    propload = hl.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=HL_motor_axes[i])
    prop_pwr_hl = hl.get_power(states=cruise.ac_states, controls=x57_controls)
    propload_new = propload.rotate_to_axis(fd_axis)
    thrust = propload_new.F.vector[0]
    currentThrust.append(thrust)
    currentPwr.append(prop_pwr_hl)

for i, cr in enumerate(cruise_propulsions):
    propload = cr.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=cruise_motor_axes[i])
    prop_pwr_cm = cr.get_power(states=cruise.ac_states, controls=x57_controls)
    propload_new = propload.rotate_to_axis(fd_axis)
    thrust = propload_new.F.vector[0]
    currentThrust.append(thrust)
    currentPwr.append(prop_pwr_cm)

TotalThrust = np.sum(currentThrust)
TotalThrust.name = 'Total Thrust'
TotalPwr = np.sum(currentPwr)
TotalPwr.name = 'Total Power'



loads = aero_loads.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=wing_axis)
loads_new = loads.rotate_to_axis(fd_axis)
drag = csdl.absolute(loads_new.F.vector[0])
ThrustRequired = drag
ThrustRequired.name = 'Thrust Required'
ThrustRequired.set_as_objective(scaler=1e-2)
        


sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[cruise.parameters.speed],
    additional_outputs=[TotalThrust, ThrustRequired,TotalPwr, cruise.ac_states.VTAS],
    derivatives_kwargs= {
        "concatenate_ofs" : True
    })

TRequireds = []
TAvails = []
PAvails = []
PReqs = []
speeds = np.arange(10, 30, 10)
vtas_list = []

for i, speed in enumerate(speeds):
    sim[cruise.parameters.speed] = speed


    sim.check_optimization_derivatives()
    t1 = time.time()
    prob = CSDLAlphaProblem(problem_name='trim_optimization', simulator=sim)
    optimizer = IPOPT(problem=prob)
    optimizer.solve()
    optimizer.print_results()
    t2 = time.time()
    print('Time to solve Optimization:', t2-t1)
    # print('Total code run time:', t2-t0)
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

    TRequireds.append(obj.value[0])
    TAvails.append(TotalThrust.value[0])
    PAvails.append(TotalPwr.value[0])
    PReqs.append((obj.value[0] * cruise.ac_states.VTAS.value[0])*1e-3) # convert to kW
    vtas_list.append(cruise.ac_states.VTAS.value[0])

plt.figure()
plt.plot(vtas_list, TRequireds, marker='s', linestyle='--', label='Required Thrust')
plt.plot(vtas_list, TAvails, marker='o', linestyle='-', label='Available Thrust')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Thrust (N)')
plt.title('Thrust vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()


plt.figure()
plt.plot(vtas_list, PReqs, marker='s', linestyle='--', label='Required Power')
plt.plot(vtas_list, PAvails, marker='o', linestyle='-', label='Available Power')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Power (kW)')
plt.title('Power vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()



recorder.stop()
