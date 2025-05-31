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
    speed=Q_(45, 'm/s'),
    pitch_angle=Q_(0, 'deg'))



x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
x57_controls.rudder.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
x57_controls.aileron_left.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
x57_controls.aileron_right.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
x57_controls.flap_left.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
x57_controls.flap_right.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
x57_controls.trim_tab.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)
cruise.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10),scaler=10)

flap_diff = (x57_controls.flap_right.deflection - x57_controls.flap_left.deflection)
flap_diff.name = f'Flap Diff{x57_controls.flap_left.deflection.name} - {x57_controls.flap_right.deflection.name}'
flap_diff.set_as_constraint(equals=0,scaler=1e-3)  # Allow a small difference due to numerical precision

aileron_diff = (x57_controls.aileron_right.deflection - x57_controls.aileron_left.deflection)
aileron_diff.name = f'Aileron Diff{x57_controls.aileron_left.deflection.name} - {x57_controls.aileron_right.deflection.name}'
aileron_diff.set_as_constraint(equals=0,scaler=1e-3)  # Allow a small difference due to numerical precision
# cruise.parameters.altitude.set_as_design_variable(lower=1, upper=2000)
# cruise.parameters.speed.set_as_design_variable(lower=0.1, upper=200)


for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0, upper=1e-6)
    right_engine.throttle.set_as_design_variable(lower=0, upper=1e-6)
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL Throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    hl_throt_diff.set_as_constraint(equals=0,scaler=1e-3)  # Allow a small difference due to numerical precision


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    right_engine.throttle.set_as_design_variable(lower=0.0, upper=1.0)
    cm_thrott_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_thrott_diff.name = f'CM Throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_thrott_diff.set_as_constraint(equals=0, scaler=1e-3)  # Allow a small difference due to numerical precision


x57_aerodynamics = X57Aerodynamics(component=aircraft_component)
aircraft_component.comps['Wing'].load_solvers.append(x57_aerodynamics)

HL_radius_x57 = csdl.Variable(name="high_lift_motor_radius",shape=(1,), value=1.89/2) # HL propeller radius in ft
cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft


hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

for i, hl_motor in enumerate(hl_motors):
    hl_prop = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(),engine_index=i)
    hl_motor.load_solvers.append(hl_prop)


cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

for i, cruise_motor in enumerate(cruise_motors):
    engine_index = len(hl_motors) + i
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index)
    cruise_motor.load_solvers.append(cruise_prop)



tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states,controls=x57_controls)


M_aircraft = aircraft_component.compute_total_mass_properties()

Lift_scaling = 1/(M_aircraft.mass.value*9.81)
Drag_scaling = Lift_scaling * 10
Moment_scaling = Lift_scaling / 10

FM = csdl.concatenate((Drag_scaling * tf[0], tf[1], Lift_scaling * tf[2], Moment_scaling * tm[0], Moment_scaling * tm[1], Moment_scaling * tm[2]), axis=0)
residual = csdl.absolute(csdl.norm(FM, ord=2))
residual.name = 'Trim Residual'
residual.set_as_objective()






sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[cruise.parameters.speed],
    additional_outputs=[cruise.ac_states.VTAS, tf[0], tf[1], tf[2], tm[0], tm[1], tm[2], residual],
    derivatives_kwargs= {
        "concatenate_ofs" : True})




TRequireds = []
TAvails = []
PAvails = []
PReqs = []
# speeds = np.arange(1e-6, 100, 10) # in m/s
speeds = [76.8909]  # in m/s, this is the cruise speed from the X57 CFD data
Drags = []
Lifts = []
LD_ratios = []
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
    recorder.execute()
    dv_save_dict = {}
    constraints_save_dict = {}
    obj_save_dict = {}

    dv_dict = recorder.design_variables
    constraint_dict = recorder.constraints
    obj_dict = recorder.objectives

    print("=====Aircraft States=====")

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
