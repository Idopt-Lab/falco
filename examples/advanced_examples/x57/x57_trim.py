import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
from flight_simulator import REPO_ROOT_FOLDER, Q_
from flight_simulator.core.vehicle.conditions import aircraft_conditions
from modopt import CSDLAlphaProblem, SLSQP, IPOPT, SNOPT, PySLSQP
import matplotlib.pyplot as plt

debug = False
recorder = csdl.Recorder(inline=True, expand_ops=True, debug=debug)
recorder.start()

import sys
x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))

from x57_vehicle_models import x57_controls, fd_axis, Aircraft

do_trim_optimization1 = False
do_cruise = True
do_trim_optimization2 = True
do_handling_qualities_optimization = False





if do_trim_optimization1 is True and do_cruise is True:    

        for engine in x57_controls.engines:
            engine.throttle.set_as_design_variable(lower=0.5, upper=1, scaler=10)

        x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(5), upper=np.deg2rad(5), scaler=10)

        cruiseCondition = aircraft_conditions.CruiseCondition(
            fd_axis=fd_axis,
            controls=x57_controls,
            altitude=Q_(1, 'ft'),
            range=Q_(70, 'km'),
            speed=Q_(100, 'mph'),
            pitch_angle=Q_(0, 'deg'))
        print(cruiseCondition)

        cruiseCondition.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(25), upper=np.deg2rad(25), scaler=10)


        total_forces_cruise, total_moments_cruise = cruiseCondition.assemble_forces_moments(component=Aircraft)
        print("Total Forces", total_forces_cruise.value)
        print("Total Moments", total_moments_cruise.value)

        total_forces_cruise[0].set_as_constraint(equals=0) 
        total_forces_cruise[1].set_as_constraint(equals=0) 

        (csdl.absolute(total_forces_cruise[2])).set_as_objective()  # Minimize vertical force (Fz)


if do_trim_optimization2 is True and do_cruise is True:    

        for engine in x57_controls.engines:
            engine.throttle.set_as_design_variable(lower=0.4, upper=1.0, scaler=10)
            

        x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(5), upper=np.deg2rad(5), scaler=10)

        cruiseCondition = aircraft_conditions.CruiseCondition(
            fd_axis=fd_axis,
            controls=x57_controls,
            altitude=Q_(1, 'ft'),
            range=Q_(70, 'km'),
            speed=Q_(100, 'mph'),
            pitch_angle=Q_(0, 'deg'))
        print(cruiseCondition)

        cruiseCondition.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(25), upper=np.deg2rad(25), scaler=10)


        total_forces_cruise, total_moments_cruise = cruiseCondition.assemble_forces_moments(component=Aircraft)
        print("Total Forces", total_forces_cruise.value)
        print("Total Moments", total_moments_cruise.value)

        total_forces_cruise[1].set_as_constraint(equals=0)
        total_forces_cruise[2].set_as_constraint(equals=0)
        for engine in x57_controls.engines[1:]:
            (engine.throttle - x57_controls.engines[0].throttle).set_as_constraint(equals=0)
    


        drag = 900
        (csdl.absolute(total_forces_cruise[0]-drag)).set_as_objective()  # Minimize vertical force (Fx)

# if do_trim_optimization2 is True and do_cruise is True:    
    
#     for engine in x57_controls.engines:
#         engine.throttle.set_as_design_variable(lower=0.5, upper=1, scaler=10)

#     # x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10), scaler=1)

#     cruiseCondition = aircraft_conditions.CruiseCondition(
#         fd_axis=fd_axis,
#         controls=x57_controls,
#         altitude=Q_(1, 'ft'),
#         range=Q_(70, 'km'),
#         speed=Q_(70, 'mph'),
#         pitch_angle=Q_(0, 'deg'))
#     print(cruiseCondition)

#     cruiseCondition.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(25), upper=np.deg2rad(25), scaler=1)

#     J = cruiseCondition.evaluate_trim_res(component=Aircraft)
#     print("J Scalar Initial:", J.value)
#     J.name = 'J'
#     J.set_as_objective()  # Minimize J (norm(state vector dot 1-6))

# do_handling_qualities_optimization = False
# if do_handling_qualities_optimization is True and do_cruise is True:
#     for engine in x57_controls.engines:
#         engine.throttle.set_as_design_variable(lower=0.5, upper=1, scaler=10)

#     # x57_controls.elevator.deflection.set_as_design_variable(lower=-np.deg2rad(10), upper=np.deg2rad(10), scaler=1)

#     cruiseCondition = aircraft_conditions.CruiseCondition(
#         fd_axis=fd_axis,
#         controls=x57_controls,
#         altitude=Q_(1, 'ft'),
#         range=Q_(70, 'km'),
#         speed=Q_(70, 'mph'),
#         pitch_angle=Q_(0, 'deg'))
#     print(cruiseCondition)

#     cruiseCondition.parameters.pitch_angle.set_as_design_variable(lower=-np.deg2rad(25), upper=np.deg2rad(25), scaler=1)

#     stab_metrics = cruiseCondition.eval_linear_stability(component=Aircraft)

#     stab_metrics.time_2_double_phugoid.set_as_constraint(upper=0.5)
#     J = cruiseCondition.evaluate_trim_res(component=Aircraft)
#     print("J Scalar Initial:", J.value)
#     J.name = 'J'
#     J.set_as_objective()  # Minimize J (norm(state vector dot 1-6))


if debug is True:
    pass
else:
    recorder.inline = False

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    gpu=False,
    derivatives_kwargs= {
        "concatenate_ofs" : True
    })


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






recorder.stop()