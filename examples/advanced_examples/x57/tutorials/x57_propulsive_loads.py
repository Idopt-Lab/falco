import time
import csdl_alpha as csdl
import numpy as np
from falco.core.vehicle.conditions import aircraft_conditions
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP
from falco import REPO_ROOT_FOLDER, Q_
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
aircraft_component.mass_properties.mass.value = 0
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



for left_engine, right_engine in zip(x57_controls.hl_engines_left, x57_controls.hl_engines_right):
    left_engine.throttle.value = 0.0
    right_engine.throttle.value = 0.0
    hl_throt_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    hl_throt_diff.name = f'HL throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    hl_throt_diff.set_as_constraint(equals=0)  


for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
    left_engine.throttle.value = 1.0
    right_engine.throttle.value=1.0
    cm_throttle_diff = (right_engine.throttle - left_engine.throttle) # setting all engines to the same throttle setting, because of symmetry
    cm_throttle_diff.name = f'CM throttle Diff{left_engine.throttle.name} - {right_engine.throttle.name}'
    cm_throttle_diff.set_as_constraint(equals=0) 


HL_radius_x57 = Q_(1.89/2, 'ft') # HL propeller radius in ft
cruise_radius_x57 = Q_(5/2, 'ft') # cruise propeller radius in ft


hl_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('HL Motor')]

for i, hl_motor in enumerate(hl_motors):
    hl_prop = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(),engine_index=i, RPMmin=1500, RPMmax=5400)
    hl_motor.load_solvers.append(hl_prop)


cruise_motors = [comp for comp in aircraft_component.comps['Wing'].comps.values() if comp._name.startswith('Cruise Motor')]

for i, cruise_motor in enumerate(cruise_motors):
    engine_index = len(hl_motors) + i
    cruise_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=engine_index, RPMmin=2250, RPMmax=2250)
    cruise_motor.load_solvers.append(cruise_prop)



tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states,controls=x57_controls)


sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    additional_inputs=[cruise.parameters.speed],
    derivatives_kwargs= {
        "concatenate_ofs" : True})



speeds = np.array([225]) *0.3048 # in m/s, this is the cruise speed from the X57 CFD data


for i, speed in enumerate(speeds):

    sim[cruise.parameters.speed] = speed


    recorder.execute()

    print("=====Aircraft States=====")
    print("Aircraft Conditions")
    print(cruise)
    print("Throttles")
    for engine in x57_controls.engines:
        print(engine.throttle.value)    
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


    print("TF[0]", tf[0].value)
    print("TF[1]", tf[1].value)
    print("TF[2]", tf[2].value)
    print("TM[0]", tm[0].value)
    print("TM[1]", tm[1].value)
    print("TM[2]", tm[2].value)









recorder.stop()
