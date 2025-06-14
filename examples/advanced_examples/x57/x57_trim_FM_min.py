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

cruise = aircraft_conditions.CruiseCondition(
    fd_axis=axis_dict['fd_axis'],
    controls=x57_controls,
    altitude=Q_(8000, 'ft'),
    range=Q_(160, 'km'),
    speed=Q_(76.8909, 'm/s'),
    pitch_angle=Q_(0, 'deg'))



x57_controls.elevator.deflection.set_as_design_variable(lower=np.deg2rad(-50), upper=np.deg2rad(50),scaler=1e2)
cruise.parameters.pitch_angle.set_as_design_variable(lower=np.deg2rad(-3), upper=np.deg2rad(3), scaler=1e2)
# cruise.parameters.speed.set_as_design_variable(lower=10, upper=76.8909, scaler=1e-2)



for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1e-6)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1e-6)
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    hl_throt_diff.set_as_constraint(equals=0)  


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.3, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.3, upper=1.0)
    cm_throttle_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_throttle_diff.name = f'CM throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_throttle_diff.set_as_constraint(equals=0)  


x57_aerodynamics = X57Aerodynamics(component=aircraft_component)
aircraft_component.comps['Wing'].load_solvers.append(x57_aerodynamics)
aero_results = x57_aerodynamics.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=axis_dict['fd_axis'])
CL = aero_results['CL']
CD = aero_results['CD']
CM = aero_results['CM']

HL_radius_x57 = Q_(1.89/2, 'ft') # HL propeller radius in ft
cruise_radius_x57 = Q_(5/2, 'ft') # cruise propeller radius in ft


hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

for i, hl_motor in enumerate(hl_motors):
    hl_prop = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(),engine_index=i,RPMmin=1500, RPMmax=5400)
    hl_motor.load_solvers.append(hl_prop)
    hl_results = hl_prop.get_torque_power(states=cruise.ac_states, controls=x57_controls)
    hl_engine_torque = hl_results['torque']
    hl_engine_torque.name = f'HL_Engine_{i}_Torque'
    hl_engine_torque.set_as_constraint(lower=0.0, upper=20.5, scaler=1/20.5) # values from High-Lift Propeller Operating Conditions v2 paper


cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

for i, cruise_motor in enumerate(cruise_motors):
    engine_index = len(hl_motors) + i
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index, RPMmin=1700, RPMmax=2700)
    cruise_motor.load_solvers.append(cruise_prop)
    cm_results = cruise_prop.get_torque_power(states=cruise.ac_states, controls=x57_controls)
    cm_engine_torque = cm_results['torque']
    cm_engine_torque.name = f'Cruise_Engine_{i}_Torque'
    cm_engine_torque.set_as_constraint(lower=0.0, upper=225, scaler=1/225) # values from x57_DiTTo_manuscript paper

# x57_controls.update_controls(x57_controls.u)

tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states,controls=x57_controls)

Drag = - x57_aerodynamics.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=axis_dict['wing_axis'])['loads'].F.vector[0]
Lift = - x57_aerodynamics.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=axis_dict['wing_axis'])['loads'].F.vector[2]
Moment = x57_aerodynamics.get_FM_localAxis(states=cruise.ac_states, controls=x57_controls, axis=axis_dict['wing_axis'])['loads'].M.vector[1]


Lift_scaling = 1/(aircraft_component.mass_properties.mass * 9.81)
Drag_scaling = 1/csdl.absolute(Drag)
Moment_scaling = 1/csdl.absolute(Moment)


res1 = tf[0] * Drag_scaling 
res1.name = 'Fx Force'
res1.set_as_constraint(lower=-1e-6,upper=1e-6) 

res2 = tf[2] * Lift_scaling
res2.name = 'Fz Force'
res2.set_as_constraint(lower=-1e-6,upper=1e-6)

res4 = tm[1] * Moment_scaling
res4.name = 'My Moment'
res4.set_as_constraint(lower=-1e-6,upper=1e-6)

FM = csdl.concatenate((Drag_scaling * tf[0], Lift_scaling * tf[2], Moment_scaling * tm[1]), axis=0)
FM_res = csdl.absolute(csdl.norm(FM, ord=2))
FM_res.name = 'FM Minimization Objective'
FM_res.set_as_objective()  # Scale the objective for better optimization performance


Total_torque_avail, Total_power_avail = aircraft_component.compute_total_torque_total_power(fd_state=cruise.ac_states,controls=x57_controls)
# print("Total Torque Available:", Total_torque_avail.value)
# print("Total Power Available:", Total_power_avail.value)
Total_power_avail.name = 'Total Power Available'



sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[cruise.parameters.speed, cruise.parameters.altitude],
    additional_outputs=[cruise.ac_states.VTAS],
    derivatives_kwargs= {
        "concatenate_ofs" : True})


TRequireds = []
TAvails = []
PAvails = []
PReqs = []
# speeds = np.linspace(10, 76.8909, 5) # in m/s
speeds = np.array([150]) / 1.944  # Convert knots to m/s
altitudes = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])*0.3048
Drags = []
Lifts = []
LD_ratios = []
vtas_list = []
eta_list = []
Jval_list = []
elevator_deflections = []

results = {
    'VTAS': [],
    'Drag': [],
    'Lift': [],
    'Moment': [],
    'CL': [],
    'CD': [],
    'CM': [],
    'Elevator Deflection': [],
}


for j, alt in enumerate(altitudes):

    for i, speed in enumerate(speeds):

        sim[cruise.parameters.speed] = speed
        sim[cruise.parameters.altitude] = alt
        recorder.execute()


        # sim.check_optimization_derivatives()
        t1 = time.time()
        prob = CSDLAlphaProblem(problem_name='trim_FM_min_opt', simulator=sim)
        optimizer = IPOPT(problem=prob)
        optimizer.solve()
        optimizer.print_results()
        t2 = time.time()
        print('Time to solve Optimization:', t2-t1)
        recorder.execute()

        results['VTAS'].append(cruise.ac_states.VTAS.value[0])
        results['Drag'].append(Drag.value[0])
        results['Lift'].append(Lift.value[0])
        results['Moment'].append(Moment.value[0])
        results['CL'].append(CL.value[0])
        results['CD'].append(CD.value[0])
        results['CM'].append(CM.value[0])
        results['Elevator Deflection'].append(x57_controls.elevator.deflection.value[0] * 180 / np.pi)


        prop_results = cruise_prop.get_torque_power(states=cruise.ac_states, controls=x57_controls)
        prop_efficiency = prop_results['eta'] 
        eta_list.append(prop_efficiency.value[0])
        Jval = prop_results['J']
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
        # print(cruise)
        # print("Mach Number")
        # print(cruise.ac_states.VTAS.value[0]/ cruise.ac_states.atmospheric_states.speed_of_sound.value[0])
        # print("KTAS (knots)")
        # print(cruise.ac_states.VTAS.value[0] * 1.944)  # Convert m/s to knots
        print("Throttle")
        for engine in x57_controls.engines:
            print(engine.throttle.value)
        # print("High Lift Engine Torque (N*m)")
        # for engine in x57_controls.hl_engines:  
        #     print(hl_engine_torque.value)  
        # print("Cruise Engine Torque (N*m)")
        # for engine in x57_controls.cm_engines:
        #     print(cm_engine_torque.value)
        # print("Total Torque Available (N*m)")
        # print(Total_torque_avail.value)    
        print("Elevator Deflection (deg)")
        print(x57_controls.pitch_control['Elevator'].deflection.value * 180 / np.pi)
        print("Pitch Angle (deg)")
        print(cruise.ac_states.states.theta.value * 180 / np.pi)
        print('Flight Path Angle (deg)')
        print(cruise.ac_states.gamma.value * 180 / np.pi)
        print('Angle of Attack (deg)')
        print(cruise.ac_states.alpha.value * 180 / np.pi)
        print("Residual Magnitude: (FM Minimization)")
        print(FM.value)
        print("Residual Norm")
        print(csdl.norm(FM, ord=2).value)


        vtas_list.append(cruise.ac_states.VTAS.value[0])
        TRequireds.append(Drag.value[0])
        PReqs.append(Drag.value[0] * cruise.ac_states.VTAS.value[0] * 1e-3)  # Convert to kW
        PAvails.append(Total_power_avail.value[0] * 1e-3)  # Convert to kW
        Drags.append(Drag.value[0])
        Lifts.append(Lift.value[0])
        LD_ratios.append(Lift.value[0] / Drag.value[0])


        print("TF[0]", tf[0].value)
        print("TF[1]", tf[1].value)
        print("TF[2]", tf[2].value)
        print("TM[0]", tm[0].value)
        print("TM[1]", tm[1].value)
        print("TM[2]", tm[2].value)

# Convert the dictionary to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
with open('trim_FM_min.csv', 'w') as f:
    results_df.to_csv(f, index=False)





recorder.stop()
