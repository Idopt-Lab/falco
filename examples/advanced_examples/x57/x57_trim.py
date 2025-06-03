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
from x57_control_system import X57ControlSystem
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

x57_controls = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                hl_engine_count=12,cm_engine_count=2)

cruise = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(8000, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(76.8909, 'm/s'),
    pitch_angle=Q_(0, 'deg'))



x57_controls.elevator.deflection.set_as_design_variable(lower=x57_controls.elevator.lower_bound*np.pi/180, upper=x57_controls.elevator.upper_bound*np.pi/180,scaler=1e2)
# x57_controls.rudder.deflection.set_as_design_variable(lower=x57_controls.rudder.lower_bound*np.pi/180, upper=x57_controls.rudder.upper_bound*np.pi/180,scaler=100)
# x57_controls.aileron_left.deflection.set_as_design_variable(lower=x57_controls.aileron_left.lower_bound*np.pi/180, upper=x57_controls.aileron_left.upper_bound*np.pi/180,scaler=100)
# x57_controls.aileron_right.deflection.set_as_design_variable(lower=x57_controls.aileron_right.lower_bound*np.pi/180, upper=x57_controls.aileron_right.upper_bound*np.pi/180,scaler=100)
# x57_controls.flap_left.deflection.set_as_design_variable(lower=x57_controls.flap_left.lower_bound*np.pi/180, upper=x57_controls.flap_left.upper_bound*np.pi/180,scaler=100)
# x57_controls.flap_right.deflection.set_as_design_variable(lower=x57_controls.flap_right.lower_bound*np.pi/180, upper=x57_controls.flap_right.upper_bound*np.pi/180,scaler=100)
# x57_controls.trim_tab.deflection.set_as_design_variable(lower=x57_controls.trim_tab.lower_bound*np.pi/180, upper=x57_controls.trim_tab.upper_bound*np.pi/180,scaler=100)
cruise.parameters.pitch_angle.set_as_design_variable(lower=(-10)*np.pi/180, upper=10*np.pi/180, scaler=1e2)

flap_diff = (x57_controls.flap_right.deflection - x57_controls.flap_left.deflection)
flap_diff.name = f'Flap Diff{x57_controls.flap_left.deflection.name} - {x57_controls.flap_right.deflection.name}'
# flap_diff.set_as_constraint(equals=0) 

aileron_diff = (x57_controls.aileron_right.deflection - x57_controls.aileron_left.deflection)
aileron_diff.name = f'Aileron Diff{x57_controls.aileron_left.deflection.name} - {x57_controls.aileron_right.deflection.name}'
# aileron_diff.set_as_constraint(equals=0)  


# cruise.parameters.altitude.set_as_design_variable(lower=1, upper=2000)
# cruise.parameters.speed.set_as_design_variable(lower=0.1, upper=200)


for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.rpm.value = 0
    right_engine.rpm.value= 0
    # left_engine.rpm.set_as_design_variable(lower=left_engine.lower_bound, upper=left_engine.upper_bound,scaler=1e-4)
    # right_engine.rpm.set_as_design_variable(lower=right_engine.lower_bound, upper=right_engine.upper_bound,scaler=1e-4)
    hl_rpm_diff = (right_engine.rpm - left_engine.rpm) # setting all engines to the same rpm setting, because of symmetry
    hl_rpm_diff.name = f'HL rpm Diff{left_engine.rpm.name} - {right_engine.rpm.name}'
    # hl_rpm_diff.set_as_constraint(equals=0)  


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.rpm.set_as_design_variable(lower=left_engine.lower_bound, upper=left_engine.upper_bound,scaler=1e-4)
    right_engine.rpm.set_as_design_variable(lower=right_engine.lower_bound, upper=right_engine.upper_bound,scaler=1e-4)
    cm_rpm_diff = (right_engine.rpm - left_engine.rpm) # setting all engines to the same rpm setting, because of symmetry
    cm_rpm_diff.name = f'CM rpm Diff{left_engine.rpm.name} - {right_engine.rpm.name}'
    cm_rpm_diff.set_as_constraint(equals=0)  


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
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index)
    cruise_motor.load_solvers.append(cruise_prop)



tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states,controls=x57_controls)


Lift_scaling = 1 / (aircraft_component.mass_properties.mass * 9.81)
Drag_scaling = Lift_scaling / 10
Moment_scaling = Lift_scaling * 10

# res1 = tf[0]
# res1.name = 'Fx=0'
# res1.set_as_constraint(equals=0,scaler=Drag_scaling.value)

res2 = tf[2]
res2.name = 'Fz=0'
res2.set_as_constraint(equals=0,scaler=Lift_scaling.value)    

res3 = tm[1]
res3.name = 'My=0'
res3.set_as_constraint(equals=0,scaler=Moment_scaling.value)



Total_torque_req, Total_power_req = aircraft_component.compute_total_torque_total_power(fd_state=cruise.ac_states,controls=x57_controls)
# print("Total Torque Required:", Total_torque_req.value)
# print("Total Power Required:", Total_power_req.value)
Total_power_req = Total_power_req * (Lift_scaling / 10)
Total_power_req.name = 'Total Power Required'
Total_power_req.set_as_objective()

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[cruise.parameters.speed],
    additional_outputs=[cruise.ac_states.VTAS, tf[0], tf[1], tf[2], tm[0], tm[1], tm[2], Total_power_req],
    derivatives_kwargs= {
        "concatenate_ofs" : True})




TRequireds = []
TAvails = []
PAvails = []
PReqs = []
# speeds = np.linspace(10, 130, 20) # in m/s
speeds = [76.8909]  # in m/s, this is the cruise speed from the X57 CFD data
Drags = []
Lifts = []
LD_ratios = []
LD_ratios_no_opt = []   # Baseline/unoptimized L/D ratios
ThrustR_no_opt = []
PowerReq_no_opt = []
vtas_list = []

for i, speed in enumerate(speeds):

    sim[cruise.parameters.speed] = speed

    recorder.execute()

    baseline_Drag = -tf[0].value[0]
    baseline_Lift = tf[2].value[0]
    LD_no_opt = baseline_Lift / baseline_Drag
    ThrustR_no_opt.append(baseline_Drag)
    PowerReq_no_opt.append(Total_power_req.value[0])  # in kW
    LD_ratios_no_opt.append(LD_no_opt)

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



    CLmax = 3.85

    # Vs = csdl.sqrt((2* aircraft_component.mass_properties.mass * 9.81) / (cruise.ac_states.atmospheric_states.density * aircraft_component.comps["Wing"].parameters.S_ref * CLmax)) # in m/s

    # V_H = cruise.ac_states.VTAS * 

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
    print("RPMs")
    for engine in x57_controls.engines:
        print(engine.rpm.value)    
    print("Elevator Deflection (deg)")
    print(x57_controls.elevator.deflection.value * 180 / np.pi)
    print("Rudder Deflection (deg)")
    print(x57_controls.rudder.deflection.value * 180 / np.pi)
    print("Left Aileron Deflection (deg)")
    print(x57_controls.aileron_left.deflection.value * 180 / np.pi)
    print("Right Aileron Deflection (deg)")
    print(x57_controls.aileron_right.deflection.value * 180 / np.pi)
    print("Left Flap Deflection (deg)")
    print(x57_controls.flap_left.deflection.value * 180 / np.pi)
    print("Right Flap Deflection (deg)")
    print(x57_controls.flap_right.deflection.value * 180 / np.pi)
    print("Trim Tab Deflection (deg)")
    print(x57_controls.trim_tab.deflection.value * 180 / np.pi)
    print("Pitch Angle (deg)")
    print(cruise.parameters.pitch_angle.value * 180 / np.pi)
    print("Residual Magnitude: (Power Required Minimization)")
    print(Total_power_req.value)
    print("Residual Norm")
    print(csdl.norm(Total_power_req, ord=2).value)


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
plt.plot(vtas_list, TRequireds, marker='s', linestyle='--', label='Required Thrust')
plt.plot(vtas_list, ThrustR_no_opt, marker='o', linestyle='-', label='Thrust (No Optimization)')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Thrust (N)')
plt.title('Thrust vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(vtas_list, PReqs, marker='s', linestyle='--', label='Required Power')
plt.plot(vtas_list, PowerReq_no_opt, marker='o', linestyle='-', label='Power (No Optimization)')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylabel('Power (kW)')
plt.title('Power vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()


plt.figure()
plt.plot(vtas_list, LD_ratios, marker='s', linestyle='--', label='L/D Ratio')
plt.plot(vtas_list, LD_ratios_no_opt, marker='o', linestyle='-', label='L/D Ratio (No Optimization)')
plt.xlabel('True Airspeed (VTAS) [m/s]')
plt.ylim(4,16)
plt.xlim(40, 100)
plt.ylabel('L/D Ratio')
plt.title('L/D Ratio vs. VTAS')
plt.grid(True)
plt.legend()
plt.show()



recorder.stop()
