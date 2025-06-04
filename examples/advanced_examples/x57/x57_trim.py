import time
import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.vehicle.conditions import aircraft_conditions
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP
from flight_simulator import REPO_ROOT_FOLDER, Q_
import sys

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

cruise = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(8000, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(76.8909, 'm/s'),
    pitch_angle=Q_(0, 'deg'))



x57_controls.elevator.deflection.set_as_design_variable(lower=np.deg2rad(x57_controls.elevator.lower_bound), upper=np.deg2rad(x57_controls.elevator.upper_bound),scaler=1e2)
cruise.parameters.pitch_angle.set_as_design_variable(lower=(-10)*np.pi/180, upper=10*np.pi/180, scaler=1e2)





for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.value = 0
    right_engine.throttle.value= 0
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    # hl_throttle_diff.set_as_constraint(equals=0)  


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.7, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.7, upper=1.0)
    cm_throttle_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_throttle_diff.name = f'CM throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_throttle_diff.set_as_constraint(equals=0)  


x57_aerodynamics = X57Aerodynamics(component=aircraft_component)
aircraft_component.comps['Wing'].load_solvers.append(x57_aerodynamics)

HL_radius_x57 = csdl.Variable(name="high_lift_motor_radius",shape=(1,), value=1.89/2) # HL propeller radius in ft
cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft


hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

# for i, hl_motor in enumerate(hl_motors):
#     hl_prop = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(),engine_index=i)
#     hl_motor.load_solvers.append(hl_prop)


cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

for i, cruise_motor in enumerate(cruise_motors):
    engine_index = len(hl_motors) + i
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index, RPMmin=1700, RPMmax=2700)
    cruise_motor.load_solvers.append(cruise_prop)

x57_controls.update_controls(x57_controls.u)

x57_controls.update_high_lift_control(flap_flag=False, blower_flag=False)

tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states,controls=x57_controls)


Lift_scaling = 1 / (aircraft_component.mass_properties.mass * 9.81)
Drag_scaling = Lift_scaling / 10
Moment_scaling = Lift_scaling * 10


FM = csdl.concatenate((Drag_scaling * tf[0], tf[1], Lift_scaling * tf[2], Moment_scaling * tm[0], Moment_scaling * tm[1], Moment_scaling * tm[2]), axis=0)
residual = csdl.absolute(csdl.norm(FM, ord=2))
residual.name = 'FM Minimization'
# residual.set_as_constraint()
residual.set_as_objective()


Total_torque_req, Total_power_req = aircraft_component.compute_total_torque_total_power(fd_state=cruise.ac_states,controls=x57_controls)
# print("Total Torque Required:", Total_torque_req.value)
# print("Total Power Required:", Total_power_req.value)
Total_power_req.name = 'Total Power Required'
# Total_power_req.set_as_objective()

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[cruise.parameters.speed],
    additional_outputs=[cruise.ac_states.VTAS, Total_power_req],
    derivatives_kwargs= {
        "concatenate_ofs" : True})




TRequireds = []
TAvails = []
PAvails = []
PReqs = []
# speeds = np.linspace(10, 76.8909, 10) # in m/s
speeds = [76.8909]  # in m/s, this is the cruise speed from the X57 CFD data
Drags = []
Lifts = []
LD_ratios = []
LD_ratios_no_opt = []   # Baseline/unoptimized L/D ratios
ThrustR_no_opt = []
PowerReq_no_opt = []
vtas_list = []
eta_list = []
Jval_list = []

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
    recorder.execute()

    Drag =  -tf[0]
    ThrustR = Drag
    Lift = -tf[2] 
    results = cruise_prop.get_torque_power(states=cruise.ac_states, controls=x57_controls)
    prop_efficiency = results['eta'] 
    eta_list.append(prop_efficiency.value[0])
    Jval = results['J']
    Jval_list.append(Jval.value[0])


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
    print(cruise)
    print("Throttle")
    for engine in x57_controls.engines:
        print(engine.throttle.value)    
    print("Elevator Deflection (deg)")
    print(x57_controls.pitch_control['Elevator'].deflection.value)
    print("Pitch Angle (deg)")
    print(cruise.parameters.pitch_angle.value * 180 / np.pi)
    print("Residual Magnitude: (FM Minimization)")
    print(FM.value)
    print("Residual Norm")
    print(csdl.norm(FM, ord=2).value)


    vtas_list.append(cruise.ac_states.VTAS.value[0])
    TRequireds.append(ThrustR.value[0])
    PReqs.append(Total_power_req.value[0] * 1e-3)  # Convert to kW
    Drags.append(Drag.value[0])
    Lifts.append(Lift.value[0])
    LD_ratios.append(Lift.value[0] / Drag.value[0])

    print("TF[0]", tf[0].value)
    print("TF[1]", tf[1].value)
    print("TF[2]", tf[2].value)
    print("TM[0]", tm[0].value)
    print("TM[1]", tm[1].value)
    print("TM[2]", tm[2].value)

import matplotlib.pyplot as plt


plt.figure()
plt.plot(Jval_list, eta_list, marker='s', linestyle='--', label='Propeller Efficiency')
plt.xlabel('Advaced Ratio (J)')
plt.ylabel('Propeller Efficiency')
plt.title('Propeller Efficiency vs. Advaced Ratio (J)')
plt.grid(True)
plt.legend()
plt.show()



plt.figure()
plt.plot(vtas_list, TRequireds, marker='s', linestyle='--', label='Required Thrust')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Thrust (N)')
plt.title('Thrust vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(vtas_list, PReqs, marker='s', linestyle='--', label='Required Power')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Power (kW)')
plt.title('Power vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()


plt.figure()
plt.plot(vtas_list, LD_ratios, marker='s', linestyle='--', label='L/D Ratio')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylim(4,16)
plt.xlim(40, 100)
plt.ylabel('L/D Ratio')
plt.title('L/D Ratio vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()



recorder.stop()
