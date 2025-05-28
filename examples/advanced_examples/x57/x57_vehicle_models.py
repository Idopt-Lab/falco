import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
from flight_simulator import REPO_ROOT_FOLDER, Q_, ureg
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.vehicle.components.wing import Wing as WingComp
from flight_simulator.core.vehicle.components.fuselage import Fuselage as FuseComp
from flight_simulator.core.vehicle.components.aircraft import Aircraft as AircraftComp
from flight_simulator.core.vehicle.components.rotor import Rotor as RotorComp
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
from typing import List
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
from typing import Union
from scipy.interpolate import Akima1DInterpolator
from flight_simulator.core.loads.loads import Loads
import os
import scipy.io as sio



import sys
from flight_simulator import REPO_ROOT_FOLDER
x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))



debug = False



## AXIS/AXISLSDOGEO CREATION



def create_axes():
    from x57_geometry import get_geometry

    geo = get_geometry()

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    # OpenVSP Model Axis
    openvsp_axis = Axis(
        name='OpenVSP Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        origin=ValidOrigins.OpenVSP.value
    )


    wing_axis = AxisLsdoGeo(
        name='Wing Axis',
        geometry=geo['wing'],
        parametric_coords=geo['wing_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    left_flap_axis = AxisLsdoGeo(
        name='Left Flap Axis',
        geometry=geo['flapL'],
        parametric_coords=geo['left_flap_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_flap_axis = AxisLsdoGeo(
        name='Right Flap Axis',
        geometry=geo['flapR'],
        parametric_coords=geo['right_flap_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    left_aileron_axis = AxisLsdoGeo(
        name='Left Aileron Axis',
        geometry=geo['aileronL'],
        parametric_coords=geo['left_aileron_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_aileron_axis = AxisLsdoGeo(
        name='Right Aileron Axis',
        geometry=geo['aileronR'],
        parametric_coords=geo['right_aileron_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Tail Region Axis

    ht_tail_axis = AxisLsdoGeo(
        name='Horizontal Tail Axis',
        geometry=geo['h_tail'],
        parametric_coords=geo['ht_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    trimTab_axis = AxisLsdoGeo(
        name='Trim Tab Axis',
        geometry=geo['trimTab'],
        parametric_coords=geo['trimTab_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=ht_tail_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    vt_tail_axis = AxisLsdoGeo(
        name='Vertical Tail Axis',
        geometry=geo['vertTail'],
        parametric_coords=geo['vt_le_mid_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    rudder_axis = AxisLsdoGeo(
        name= 'Rudder Axis',
        geometry=geo['rudder'],
        parametric_coords=geo['rudder_le_mid_parametric'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Distributed Propulsion Motors Axes

    M1_axis = AxisLsdoGeo(
        name= 'Motor 1 Axis',
        geometry=geo['spinner1'],
        parametric_coords=geo['M1_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    M2_axis = AxisLsdoGeo(
        name= 'Motor 2 Axis',
        geometry=geo['spinner2'],
        parametric_coords=geo['M2_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    M3_axis = AxisLsdoGeo(
        name= 'Motor 3 Axis',
        geometry=geo['spinner3'],
        parametric_coords=geo['M3_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M4_axis = AxisLsdoGeo(
        name= 'Motor 4 Axis',
        geometry=geo['spinner4'],
        parametric_coords=geo['M4_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M5_axis = AxisLsdoGeo(
        name= 'Motor 5 Axis',
        geometry=geo['spinner5'],
        parametric_coords=geo['M5_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M6_axis = AxisLsdoGeo(
        name= 'Motor 6 Axis',
        geometry=geo['spinner6'],
        parametric_coords=geo['M6_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M7_axis = AxisLsdoGeo(
        name= 'Motor 7 Axis',
        geometry=geo['spinner7'],
        parametric_coords=geo['M7_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M8_axis = AxisLsdoGeo(
        name= 'Motor 8 Axis',
        geometry=geo['spinner8'],
        parametric_coords=geo['M8_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M9_axis = AxisLsdoGeo(
        name= 'Motor 9 Axis',
        geometry=geo['spinner9'],
        parametric_coords=geo['M9_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M10_axis = AxisLsdoGeo(
        name= 'Motor 10 Axis',
        geometry=geo['spinner10'],
        parametric_coords=geo['M10_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M11_axis = AxisLsdoGeo(
        name= 'Motor 11 Axis',
        geometry=geo['spinner11'],
        parametric_coords=geo['M11_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M12_axis = AxisLsdoGeo(
        name= 'Motor 12 Axis',
        geometry=geo['spinner12'],
        parametric_coords=geo['M12_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    HL_motor_axes = [M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis]
    # Cruise Motor Region


    cruise_motor1_axis = AxisLsdoGeo(
        name= 'Cruise Motor 1 Axis',
        geometry=geo['cruise_spinner1'],
        parametric_coords=geo['cruise_motor1_tip_parametric'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    cruise_motor2_axis = AxisLsdoGeo(
        name= 'Cruise Motor 2 Axis',
        geometry=geo['cruise_spinner2'],
        parametric_coords=geo['cruise_motor2_tip_parametric'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    cruise_motor_axes = [cruise_motor1_axis, cruise_motor2_axis]


    fd_axis = Axis(
        name='Flight Dynamics Body Fixed Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    wind_axis = Axis(
        name='Wind Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )


    aircraft_axis = Axis(
        name='Aircraft Inertial Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        origin=ValidOrigins.Inertial.value,
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis)

    return openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geo, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis, rudder_axis, aircraft_axis, geo

openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geo, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis, rudder_axis, aircraft_axis, _ = create_axes()

## Aircraft Component Creation

def build_aircraft(do_geo_param: bool = False):
    geometry = geo['geometry']

    parameterization_solver = ParameterizationSolver()
    ffd_geometric_variables = GeometricVariables()


    Aircraft = AircraftComp(geometry=geometry, compute_surface_area_flag=False, 
                            parameterization_solver=parameterization_solver,
                            ffd_geometric_variables=ffd_geometric_variables)



    Fuselage = FuseComp(
        length=csdl.Variable(name="fuselage_length", shape=(1, ), value=8.2242552),
        max_height=csdl.Variable(name="fuselage_max_height", shape=(1, ), value=1.09),
        max_width=csdl.Variable(name="fuselage_max_width", shape=(1, ), value=1.24070602),
        geometry=geo['fuselage'], skip_ffd=False, 
        parameterization_solver=parameterization_solver,
        ffd_geometric_variables=ffd_geometric_variables)


    Aircraft.add_subcomponent(Fuselage)


    wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=15)
    wing_span = csdl.Variable(name="wing_span", shape=(1, ), value=9.6)
    wing_sweep = csdl.Variable(name="wing_sweep", shape=(1, ), value=0)
    wing_dihedral = csdl.Variable(name="wing_dihedral", shape=(1, ), value=0)

    Wing = WingComp(AR=wing_AR,
                    span=wing_span,
                    sweep=wing_sweep,
                    dihedral=wing_dihedral,
                    geometry=geo['wingALL'],
                    parametric_geometry=geo['wing_parametric_geometry'],
                    tight_fit_ffd=False, 
                    orientation='horizontal', 
                    name='Wing', parameterization_solver=parameterization_solver, 
                    ffd_geometric_variables=ffd_geometric_variables
                    )


    Aircraft.add_subcomponent(Wing)
    wing_qc_fuse_connection = geometry.evaluate(geo['wing_qc_center_parametric']) - geometry.evaluate(geo['fuselage_wing_qc_center_parametric'])
    parameterization_solver.add_variable(computed_value=wing_qc_fuse_connection, desired_value=wing_qc_fuse_connection.value)

    LeftAileron = Component(name='Left Aileron')
    LeftAileron.parameters.actuate_angle = csdl.Variable(name="left_aileron_actuate_angle", shape=(1,), value=np.deg2rad(15))
    Aircraft.add_subcomponent(LeftAileron)


    RightAileron = Component(name='Right Aileron')
    RightAileron.parameters.actuate_angle = csdl.Variable(name="right_aileron_actuate_angle", shape=(1,), value=np.deg2rad(15))
    Aircraft.add_subcomponent(RightAileron)


    LeftFlap = Component(name='Left Flap')
    LeftFlap.parameters.actuate_angle = csdl.Variable(name="left_flap_actuate_angle", shape=(1,), value=np.deg2rad(10))
    Aircraft.add_subcomponent(LeftFlap)

    RightFlap = Component(name='Right Flap')
    RightFlap.parameters.actuate_angle = csdl.Variable(name="right_flap_actuate_angle", shape=(1,), value=np.deg2rad(10))
    Aircraft.add_subcomponent(RightFlap)


    HorTailArea = geo['ht_span']*geo['ht_chord']
    htAR = geo['ht_span']**2/HorTailArea
    HorTail_AR = csdl.Variable(name="HT_AR", shape=(1, ), value=4)
    HT_span = csdl.Variable(name="HT_span", shape=(1, ), value=3.14986972)
    HT_sweep = csdl.Variable(name="HT_sweep", shape=(1, ), value=0)

    HorTail = WingComp(AR=HorTail_AR, span=HT_span, sweep=HT_sweep,
                    geometry=geo['htALL'], parametric_geometry=geo['ht_parametric_geometry'],
                    tight_fit_ffd=False, skip_ffd=False,
                    name='Elevator', orientation='horizontal', 
                    parameterization_solver=parameterization_solver,
                    ffd_geometric_variables=ffd_geometric_variables)
    Aircraft.add_subcomponent(HorTail)

    HorTail.parameters.actuate_angle = csdl.Variable(name="elevator_actuate_angle", shape=(1,), value=np.deg2rad(0))

    TrimTab = Component(name='Trim Tab')
    TrimTab.parameters.actuate_angle = csdl.Variable(name="Trim Tab Actuate Angle", shape=(1,), value=np.deg2rad(0))
    Aircraft.add_subcomponent(TrimTab)

    tail_moment_arm_computed = csdl.norm(geometry.evaluate(geo['ht_qc_center_parametric']) - geometry.evaluate(geo['wing_qc_center_parametric']))
    h_tail_fuselage_connection = geometry.evaluate(geo['ht_te_center_parametric']) - geometry.evaluate(geo['fuselage_tail_te_center_parametric'])
    # parameterization_solver.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm_computed.value)
    parameterization_solver.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)



    vt_AR= csdl.Variable(name="VT_AR", shape=(1, ), value=1.998)
    VT_span = csdl.Variable(name="VT_span", shape=(1, ), value=1.6191965383361169)
    VT_sweep = csdl.Variable(name="VT_sweep", shape=(1, ), value=-30)


    VertTail = WingComp(AR=vt_AR, span=VT_span, sweep=VT_sweep,
                        geometry=geo['vtALL'], parametric_geometry=geo['vt_parametric_geometry'],
                        tight_fit_ffd=False, 
                        name='Vertical Tail', orientation='vertical',
                        parameterization_solver=parameterization_solver,
                        ffd_geometric_variables=ffd_geometric_variables)
    Aircraft.add_subcomponent(VertTail)

    Rudder = Component(name='Rudder')
    Rudder.parameters.actuate_angle = csdl.Variable(name="Rudder Actuate Angle", shape=(1,), value=np.deg2rad(0))
    Aircraft.add_subcomponent(Rudder)

    vtail_fuselage_connection = geometry.evaluate(geo['fuselage_rear_pts_parametric']) - geometry.evaluate(geo['vt_qc_base_parametric'])
    parameterization_solver.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

    lift_rotors = []
    for i in range(1, 13):
        HL_motor = RotorComp(name=f'HL Motor {i}',radius=geo['MotorDisks'][i-1])
        lift_rotors.append(HL_motor)
        Aircraft.add_subcomponent(HL_motor)

    cruise_motors = []
    for i in range(1, 3):
        cruise_motor = RotorComp(name=f'Cruise Motor {i}',radius=geo['cruise_motors_base'][i-1])
        cruise_motors.append(cruise_motor)
        Aircraft.add_subcomponent(cruise_motor)

    Battery = Component(name='Battery')
    Aircraft.add_subcomponent(Battery)

    LandingGear = Component(name='Landing Gear')
    Aircraft.add_subcomponent(LandingGear)

    Pilots = Component(name='Pilots')
    Aircraft.add_subcomponent(Pilots)

    Miscellaneous = Component(name='Miscellaneous')
    Aircraft.add_subcomponent(Miscellaneous)


    HL_radius_x57 = csdl.Variable(name="high_lift_motor_radius",shape=(1,), value=1.89/2) # HL propeller radius in ft
    cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft

    e_x57 = csdl.Variable(name="wing_e",shape=(1,), value=0.87) # Oswald efficiency factor
    CD0_x57 = csdl.Variable(name="wing_CD0",shape=(1,), value=0.001) # Zero-lift drag coefficient
    Wing.parameters.actuate_angle = csdl.Variable(name="wing_incidence",shape=(1,), value=np.deg2rad(2)) # Wing incidence angle in radians

    Wing.mass_properties = MassProperties(mass=Q_(152.88, 'kg'), cg=Vector(vector=Q_(geo['wing_le_center'].value, 'm'), axis=wing_axis), inertia=MassMI(axis=wing_axis))
    LeftAileron.mass_properties = MassProperties(mass=Q_(0.1, 'kg'), cg=Vector(vector=Q_(geo['left_aileron_le_center'].value, 'm'), axis=left_aileron_axis), inertia=MassMI(axis=left_aileron_axis))
    RightAileron.mass_properties = MassProperties(mass=Q_(0.1, 'kg'), cg=Vector(vector=Q_(geo['right_aileron_le_center'].value, 'm'), axis=right_aileron_axis), inertia=MassMI(axis=right_aileron_axis))

    LeftFlap.mass_properties = MassProperties(mass=Q_(0.1, 'kg'), cg=Vector(vector=Q_(geo['left_flap_le_center'].value, 'm'), axis=left_flap_axis), inertia=MassMI(axis=left_flap_axis))
    RightFlap.mass_properties = MassProperties(mass=Q_(0.1, 'kg'), cg=Vector(vector=Q_(geo['right_flap_le_center'].value, 'm'), axis=right_flap_axis), inertia=MassMI(axis=right_flap_axis))

    Fuselage.mass_properties = MassProperties(mass=Q_(235.87, 'kg'), cg=Vector(vector=Q_(geo['fuselage_wing_qc'].value + np.array([0,0,2.6]), 'm'), axis=wing_axis), inertia=MassMI(axis=wing_axis))
    HorTail.mass_properties = MassProperties(mass=Q_(27.3/2, 'kg'), cg=Vector(vector=Q_(geo['ht_le_center'].value, 'm'), axis=ht_tail_axis), inertia=MassMI(axis=ht_tail_axis))
    TrimTab.mass_properties = MassProperties(mass=Q_(0.1, 'kg'), cg=Vector(vector=Q_(geo['trimTab_le_center'].value, 'm'), axis=trimTab_axis), inertia=MassMI(axis=trimTab_axis))

    VertTail.mass_properties = MassProperties(mass=Q_(27.3/2, 'kg'), cg=Vector(vector=Q_(geo['vt_le_mid'].value, 'm'), axis=vt_tail_axis), inertia=MassMI(axis=vt_tail_axis))
    Rudder.mass_properties = MassProperties(mass=Q_(0.1, 'kg'), cg=Vector(vector=Q_(geo['rudder_le_mid'].value, 'm'), axis=rudder_axis), inertia=MassMI(axis=rudder_axis))
    
    Battery.mass_properties = MassProperties(mass=Q_(390.08, 'kg'), cg=Vector(vector=Q_(geo['fuselage_wing_qc'].value + np.array([0.1,0,2.6]), 'm'), axis=wing_axis), inertia=MassMI(axis=wing_axis))
    LandingGear.mass_properties = MassProperties(mass=Q_(61.15, 'kg'), cg=Vector(vector=Q_(geo['fuselage_wing_qc'].value + np.array([0,0,2.6]), 'm'), axis=wing_axis), inertia=MassMI(axis=wing_axis))
    Pilots.mass_properties = MassProperties(mass=Q_(170, 'kg'), cg=Vector(vector=Q_(geo['fuselage_wing_qc'].value + np.array([1,0,2]), 'm'), axis=wing_axis), inertia=MassMI(axis=wing_axis))
    Miscellaneous.mass_properties = MassProperties(mass=Q_(135.7, 'kg'), cg=Vector(vector=Q_(geo['fuselage_wing_qc'].value + np.array([1,0,2.6]), 'm'), axis=wing_axis), inertia=MassMI(axis=wing_axis))

    for i, HL_motor in enumerate(lift_rotors):
        HL_motor.mass_properties = MassProperties(mass=Q_(81.65/12, 'kg'), cg=Vector(vector=Q_(geo['MotorDisks'][i].value, 'm'), axis=HL_motor_axes[i]), inertia=MassMI(axis=HL_motor_axes[i]))
        HL_motor_propulsion = X57Propulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(), engine_index=i)
        HL_motor.load_solvers.append(HL_motor_propulsion)


    for i, cruise_motor in enumerate(cruise_motors):
        engine_index = len(lift_rotors) + i
        cruise_motor.mass_properties = MassProperties(mass=Q_(106.14/2, 'kg'), cg=Vector(vector=Q_(geo['cruise_motors_base'][i].value, 'm'), axis=cruise_motor_axes[i]), inertia=MassMI(axis=cruise_motor_axes[i]))
        cruise_motor_propulsion = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(), engine_index=engine_index)
        cruise_motor.load_solvers.append(cruise_motor_propulsion)


    Aircraft.mass_properties = MassProperties(mass=Q_(0, 'kg'), cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=aircraft_axis), inertia=MassMI(axis=aircraft_axis))
    Aircraft.mass_properties = Aircraft.compute_total_mass_properties()
    Aircraft_Aerodynamics = X57Aerodynamics(component=Aircraft)
    Aircraft.load_solvers.append(Aircraft_Aerodynamics)
    print(repr(Aircraft))

    if do_geo_param is True:
        parameterization_solver.evaluate(ffd_geometric_variables)
        geometry.plot(camera=dict(pos=(12, 15, -12),  # Camera position 
                                focal_point=(-Fuselage.parameters.length.value/2, 0, 0),  # Point camera looks at
                                viewup=(0, 0, -1)),    # Camera up direction
                                title= f'X-57 Maxwell Aircraft Geometry\nWing Span: {Wing.parameters.span.value[0]:.2f} m\nWing AR: {Wing.parameters.AR.value[0]:.2f}\nWing Area S: {Wing.parameters.S_ref.value[0]:.2f} m^2\nWing Sweep: {Wing.parameters.sweep.value[0]:.2f} deg',
                                #  title=f'X-57 Maxwell Aircraft Geometry\nFuselage Length: {Fuselage.parameters.length.value[0]:.2f} m\nFuselage Height: {Fuselage.parameters.max_height.value[0]:.2f} m\nFuselage Width: {Fuselage.parameters.max_width.value[0]:.2f} m',
                                screenshot= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57'/ 'images' / f'x_57_{Wing.parameters.span.value[0]}_AR_{Wing.parameters.AR.value[0]}_S_ref_{Wing.parameters.S_ref.value[0]}_sweep_{Wing.parameters.sweep.value[0]}.png')

    return Aircraft


    


class X57ControlSystem(VehicleControlSystem):

    def __init__(self, elevator_component, trim_tab_component, aileron_right_component, aileron_left_component, flap_left_component, flap_right_component, 
                 rudder_component, hl_engine_count: int, cm_engine_count: int, symmetrical: bool = True)-> None:
        self.symmetrical = symmetrical


        self.elevator = ControlSurface('elevator', lb=-26, ub=28, component=elevator_component)
        self.trim_tab = ControlSurface('trim_tab', lb=-15, ub=20, component=trim_tab_component)
        if not symmetrical:
            self.aileron_left = ControlSurface('aileron_left', lb=-15, ub=20, component=aileron_left_component)
            self.aileron_right = ControlSurface('aileron_right', lb=-15, ub=20, component=aileron_right_component)
            self.flap_left = ControlSurface('flap_left', lb=-15, ub=20, component=flap_left_component)
            self.flap_right = ControlSurface('flap_right', lb=-15, ub=20, component=flap_right_component)
        else:
            self.aileron = ControlSurface('aileron', lb=-15, ub=20, component=aileron_right_component)
            self.flap = ControlSurface('flap', lb=-15, ub=20, component=flap_right_component)
        self.rudder = ControlSurface('rudder', lb=-16, ub=16, component=rudder_component)

        self.hl_engines = self._init_hl_engines(hl_engine_count)
        midpoint_hl = len(self.hl_engines) // 2
        self.hl_engines_left = self.hl_engines[:midpoint_hl]
        self.hl_engines_right = self.hl_engines[midpoint_hl:]
        self.cm_engines = self._init_cm_engines(cm_engine_count)
        midpoint_cm = len(self.cm_engines) // 2
        self.cm_engines_left = self.cm_engines[:midpoint_cm]
        self.cm_engines_right = self.cm_engines[midpoint_cm:]
        self.engines = self.hl_engines + self.cm_engines

        if symmetrical:
            control=(
                self.aileron.deflection,
                -self.aileron.deflection,
                self.flap.deflection,
                -self.flap.deflection,
                self.elevator.deflection,
                self.trim_tab.deflection,
                self.rudder.deflection
            )
            # Use all engine throttle values for control vector
            engine_controls = tuple(engine.throttle for engine in self.engines)
            self.u = csdl.concatenate(control + engine_controls, axis=0)
        else:
            control=(
                self.left_aileron.deflection,
                self.right_aileron.deflection,
                self.left_flap.deflection,
                self.right_flap.deflection,
                self.elevator.deflection,
                self.trim_tab.deflection,
                self.rudder.deflection
            )
            # Use all engine throttle values for control vector
            engine_controls = tuple(engine.throttle for engine in self.engines)
            self.u = csdl.concatenate(control + engine_controls, axis=0)           



        if symmetrical:
            super().__init__(
                pitch_control=[self.elevator, self.trim_tab],
                roll_control=[self.aileron],
                yaw_control=[self.rudder],
                throttle_control=self.engines
            )
        else:
            super().__init__(
                pitch_control=[self.elevator, self.trim_tab],
                roll_control=[self.left_aileron, self.right_aileron],
                yaw_control=[self.rudder],
                throttle_control=self.engines
            )


    

    def _init_hl_engines(self, count: int) -> list:
        """Initialize high-lift engines."""
        return [PropulsiveControl(name=f'HL_Motor{i+1}', throttle=1.0) for i in range(count)]

    def _init_cm_engines(self, count: int) -> list:
        """Initialize cruise engines."""
        return [PropulsiveControl(name=f'Cruise_Motor{i+1}', throttle=1.0) for i in range(count)]
    
        
        
    
    @property
    def control_order(self)-> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']


    @property
    def min_values(self):
        return np.array([
            self.left_aileron.min_value,
            self.right_aileron.min_value,
            self.left_flap.min_value,
            self.right_flap.min_value,
            self.elevator.min_value,
            self.rudder.min_value
        ] + [engine.min_value for engine in self.engines])
    

    @property
    def max_values(self):
        return np.array([
            self.left_aileron.max_value,
            self.right_aileron.max_value,
            self.left_flap.max_value,
            self.right_flap.max_value,
            self.elevator.max_value,
            self.rudder.max_value
        ] + [engine.max_value for engine in self.engines])


## Aerodynamic Forces - from Modification IV
    
    


class X57Aerodynamics(Loads):

    # TODO: Improve aerodynamic model to include more complex aerodynamic effects

    def __init__(self, component):

        self.AR_wing = component.comps['Wing'].parameters.AR.value
        self.i_wing = component.comps['Wing'].parameters.actuate_angle
        self.Sref_wing = component.comps['Wing'].parameters.S_ref
        self.span_wing = component.comps['Wing'].parameters.span
        self.bref_wing = self.span_wing/2
        self.taper_wing = component.comps['Wing'].parameters.taper_ratio
        self.cref_wing = 2 * self.Sref_wing/((1 + self.taper_wing) * self.span_wing)

        self.Sref_stab = component.comps['Elevator'].parameters.S_ref
        self.span_stab = component.comps['Elevator'].parameters.span
        self.bref_stab = self.span_stab/2
        self.taper_stab = component.comps['Elevator'].parameters.taper_ratio
        self.cref_stab = 2 * self.Sref_stab/((1 + self.taper_stab) * self.span_stab)


        self.Sref_VT = component.comps['Vertical Tail'].parameters.S_ref
        self.span_VT = component.comps['Vertical Tail'].parameters.span
        self.bref_VT = self.span_VT/2
        self.taper_VT = component.comps['Vertical Tail'].parameters.taper_ratio
        self.cref_VT = 2 * self.Sref_VT/((1 + self.taper_VT) * self.span_VT)

        self.HT_axis = component.comps['Elevator'].mass_properties.cg_vector
        self.VT_axis = component.comps['Vertical Tail'].mass_properties.cg_vector
        self.Wing_axis = component.comps['Wing'].mass_properties.cg_vector

        self.wind_axis = wind_axis

        
        package_dir = os.path.dirname(os.path.abspath(__file__))
        thefile = os.path.join(package_dir, 'X57_aeroDer.mat')
        self.aeroDer = sio.loadmat(thefile)


    def __C1_CD_tot(self, alpha):
        # C1 = wing + tip nacelle. Fig 24a
        CL_tot = self.__C1_CL_tot(alpha)
        CD = 0.1033 * CL_tot ** 2 - 0.1302 * CL_tot + 0.0584
        return CD

    def __C2_CD_tot(self, alpha):
        # add HLN to C1. Fig 24a
        CL_tot = self.__C2_CL_tot(alpha)
        CD = 0.1059 * CL_tot ** 2 - 0.1049 * CL_tot + 0.0491
        return CD

    def __C8_CD_tot(self, alpha):
        # add stab + trim tab to C2. Fig 24a
        # Warning: low R2 fit
        CL_tot = self.__C8_CL_tot(alpha)
        CD = 0.0754 * CL_tot ** 2 - 0.0687 * CL_tot + 0.0419
        return CD

    def __C11_noblow_CD_tot(self, alpha):
        # Fig 16c
        CL_tot = self.__C11_noblow_CL_tot(alpha)
        CD = 0.0579 * CL_tot ** 2 - 0.1283 * CL_tot + 0.1661
        return CD

    def __C11_blow_CD_tot(self, alpha):
        # Fig 16c
        CL_tot = self.__C11_blow_CL_tot(alpha)
        CD = 0.0461 * CL_tot ** 2 - 0.1294 * CL_tot + 0.2942
        return CD

    def __C12_CD_tot(self, alpha):
        # add fus+Vtail to C8. Fig 24a
        # Warning: low R2 fit for commented out
        # second eq has better R2 but excludes last 5 points that go haywire on Fig 24a
        CL_tot = self.__C12_CL_tot(alpha)
        #CD = 0.2082 * CL_tot ** 2 - 0.3955 * CL_tot + 0.2263
        CD = 0.0514 * CL_tot**2 - 0.029 * CL_tot + 0.04
        return CD

    def __C1_CL_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24e
        alpha = alpha * (180 / np.pi)
        # CL = -0.0021 * alpha ** 2 + 0.0939 * alpha + 0.8036
        CL = 0.0633 * alpha + 0.8055
        return CL

    def __C2_CL_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24e
        alpha = alpha * (180 / np.pi)
        # CL = -0.002*alpha**2 + 0.0963*alpha + 0.6728
        CL = 0.0721 * alpha + 0.6633
        return CL

    def __C8_CL_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24e
        alpha = alpha * (180 / np.pi)
        # CL =  -0.0005*alpha**2 + 0.0746*alpha + 0.6899
        CL = 0.0674 * alpha + 0.6946
        return CL

    def __C11_noblow_CL_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition. Fig 15
        alpha = alpha * (180 / np.pi)
        # CL = -0.0034*alpha**2 + 0.1109*alpha + 1.7157
        CL = 0.075 * alpha + 1.7047
        return CL

    def __C11_blow_CL_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition. Fig 15
        alpha = alpha * (180 / np.pi)
        # CL = -0.0046*alpha**2 + 0.1774*alpha + 2.5755
        CL = 0.1153 * alpha + 2.6807
        return CL

    def __C12_CL_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24e
        alpha = alpha * (180 / np.pi)
        # CL =  -0.004*alpha**2 + 0.1343*alpha + 0.6455
        CL = 0.082 * alpha + 0.7155
        return CL

    def __C1_Cm_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24c
        CL_tot = self.__C1_CL_tot(alpha)
        Cm = 0.053 * CL_tot ** 2 - 0.0604 * CL_tot - 0.1675
        return Cm

    def __C2_Cm_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24c
        CL_tot = self.__C2_CL_tot(alpha)
        Cm = 0.0665 * CL_tot ** 2 - 0.085 * CL_tot - 0.1411
        return Cm

    def __C8_Cm_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24c
        CL_tot = self.__C8_CL_tot(alpha)
        Cm = 0.0467 * CL_tot ** 2 - 0.0452 * CL_tot - 0.1754
        return Cm

    def __C11_noblow_Cm_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition. Fig 16e
        alpha = alpha * (180 / np.pi)
        # Cm = 0.0005*alpha**2 - 0.0008*alpha -0.4063
        Cm = 0.0062 * alpha - 0.407
        return Cm

    def __C11_blow_Cm_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition. Fig 16e
        alpha = alpha * (180 / np.pi)
        # Cm = 0.0006*alpha**2 - 0.0024*alpha - 0.7404
        Cm = 0.0064 * alpha - 0.7585
        return Cm

    def __C12_Cm_tot(self, alpha):
        # Excludes stabilator contribution. Fig 24c
        CL_tot = self.__C12_CL_tot(alpha)
        Cm = 0.0811 * CL_tot ** 2 + 0.4284 * CL_tot - 0.5138
        return Cm

    # above are private. do not use outside aerodynamics class
    # begin component breakdown. use outside Aerod class

    def blow_CL_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition
        CL = self.__C11_blow_CL_tot(alpha) - self.__C11_noblow_CL_tot(alpha)
        return CL

    def blow_CD_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition
        CD = self.__C11_blow_CD_tot(alpha) - self.__C11_noblow_CD_tot(alpha)
        return CD

    def blow_Cm_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition
        Cm = self.__C11_blow_Cm_tot(alpha) - self.__C11_noblow_Cm_tot(alpha)
        return Cm

    def flap_CL_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition
        CL = self.__C11_noblow_CL_tot(alpha) - self.__C8_CL_tot(alpha)
        return CL

    def flap_CD_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition
        CD = self.__C11_noblow_CD_tot(alpha) - self.__C8_CD_tot(alpha)
        return CD

    def flap_Cm_tot(self, alpha):
        # Excludes stabilator contribution, and is at TO condition
        Cm = self.__C11_noblow_Cm_tot(alpha) - self.__C8_Cm_tot(alpha)
        return Cm

    def Wing_tipNacelle_CD_tot(self, alpha):
        #
        CD = self.__C1_CD_tot(alpha)
        return CD

    def Wing_tipNacelle_CL_tot(self, alpha):
        #
        CL_tot = self.__C1_CL_tot(alpha)
        return CL_tot

    def Wing_tipNacelle_Cm_tot(self, alpha):
        #
        Cm_tot = self.__C1_Cm_tot(alpha)
        return Cm_tot

    def HLN_CD_tot(self, alpha):
        #
        CD = self.__C2_CD_tot(alpha) - self.__C1_CD_tot(alpha)
        return CD

    def HLN_CL_tot(self, alpha):
        #
        CL_tot = self.__C2_CL_tot(alpha) - self.__C1_CL_tot(alpha)
        return CL_tot

    def HLN_Cm_tot(self, alpha):
        #
        Cm = self.__C2_Cm_tot(alpha) - self.__C1_Cm_tot(alpha)
        return Cm

    def Fus_Vtail_CD_tot(self, alpha):
        #
        CD = self.__C12_CD_tot(alpha) - self.__C8_CD_tot(alpha)
        return CD

    def Fus_Vtail_CL_tot(self, alpha):
        #
        CL_tot = self.__C12_CL_tot(alpha) - self.__C8_CL_tot(alpha)
        return CL_tot

    def Fus_Vtail_Cm_tot(self, alpha):
        #
        Cm = self.__C12_Cm_tot(alpha) - self.__C8_Cm_tot(alpha)
        return Cm

    def Stab_CL_tot(self, stab_alpha, trimtab):
        # stab alpha given on pg 11 of computational component buildup of X57
        # -0.2245 for -10deg trimtab, -0.0941 for -5deg trimtab, approx slope 0.02/deg trimtab
        stab_alpha = stab_alpha * (180 / np.pi)
        trimtab = trimtab * (180 / np.pi)
        CL_tot = 0.065558 * stab_alpha + 0.02 * trimtab
        return CL_tot

    def Stab_CD_tot(self, stab_alpha, trimtab):
        # Fig. 7d. almost same for all tab values
        CL = self.Stab_CL_tot(stab_alpha, trimtab)
        CD = 0.0871 * CL ** 2 + 0.0005 * CL + 0.0086
        return CD

    def Stab_Cm_tot(self, stab_alpha, trimtab):
        # Fig. 7d. almost same for all tab values
        CL = self.Stab_CL_tot(stab_alpha, trimtab)
        trimtab = trimtab * (180 / np.pi)
        Cm = 0.028 * CL - 0.0056 * trimtab  # approx avg of 6 eqs in Fig 6e and fig 7e
        return Cm

    def Stab_downwash(self, alpha, AR, flap, blow_num):
        # stab downwash given on pg 12 of computational component buildup of X57
        # CL_tot is aircraft CL excluding stabilator contribution
        # does not include flap contribution
        m = 0.65  # C8
        b = 0.33  # C8
        offset = 0.00  # C8
        CL_tot = self.__C8_CL_tot(alpha)
        downwash_C8 = (180 / 3.14) * (2 * (m * CL_tot + b) / (3.14 * AR)) + offset

        m = 1  # C11 no blow
        b = 0  # C11 no blow
        offset = -1.6  # C11 no blow
        CL_tot = self.__C11_noblow_CL_tot(alpha)
        downwash_C11_noblow = (180 / 3.14) * (2 * (m * CL_tot + b) / (3.14 * AR)) + offset
        # downwash_flap_noblow = downwash_C11_noblow - downwash_C8

        m = 1  # C11 blow
        b = 0  # C11 blow
        offset = -2.7  # C11 blow
        CL_tot = self.__C11_blow_CL_tot(alpha)
        downwash_C11_blow = (180 / 3.14) * (2 * (m * CL_tot + b) / (3.14 * AR)) + offset
        downwash_blow = downwash_C11_blow - downwash_C11_noblow

        m = 1.5  # C12, 1.0 for C11 (with flap no fus Vtail)
        b = -0.76  # C12, 0.0 for C11
        offset = 0.00  # C12, -1.6 for C11 noblow, -2.7 for C11 blow
        CL_tot = self.__C12_CL_tot(alpha)
        downwash_C12 = (180 / 3.14) * (2 * (m * CL_tot + b) / (3.14 * AR)) + offset
        downwash_fusVtail = downwash_C12 - downwash_C8

        if flap:
            if blow_num:
                downwash = (downwash_C11_blow + downwash_fusVtail) * (np.pi/180)
            else:
                downwash = downwash_C11_noblow + downwash_fusVtail * (np.pi/180)
        else:
            downwash = downwash_C12 * (np.pi/180)
        return downwash

    def stab_alpha(self, alpha, i_w, i_stab, AR, flap, blow_num):
        stabilator_alpha = alpha + i_w + i_stab - self.Stab_downwash(alpha, AR, flap, blow_num)
        return stabilator_alpha  # in rad

    """ this function below is based on semi-span results and gives non-zero rolling moment at zero aileron deflection
    def aileron_roll(self, alpha, aileron):
        alpha = alpha.to('deg').magnitude
        aileron = aileron.to('deg').magnitude
        c_roll_0 = 0.0012* alpha**2 - 0.0532*alpha - 0.3168
        c_roll_p10 = 0.0014* alpha**2 - 0.053 * alpha - 0.4022
        c_roll_m10 = 0.001*alpha**2 - 0.0547*alpha - 0.2208
        # fit least squares line for aileron deflection
        y = np.array([c_roll_m10, c_roll_0, c_roll_p10])
        x = np.array([-10, 0, 10])
        coeff = np.polyfit(x, y, 1)
        c_l = coeff[0]*aileron + coeff[1]
        return c_l
    """

    def AC_CL(self, alpha, i_w, i_stab, AR, flap, blow_num, trimtab):
        # assumining all other parameters as functions of cref_wing
        stabi_alpha = self.stab_alpha(alpha, i_w, i_stab, AR, flap, blow_num)
        CL = flap * self.flap_CL_tot(alpha) + blow_num / 12 * self.blow_CL_tot(alpha) + \
             self.Wing_tipNacelle_CL_tot(alpha) + self.HLN_CL_tot(alpha) + \
             self.Fus_Vtail_CL_tot(alpha) + \
             self.Stab_CL_tot(stabi_alpha, trimtab) * self.Sref_stab / self.Sref_wing
        return CL

    def AC_CD(self, alpha, i_w, i_stab, AR, flap, blow_num, trimtab):
        # assuming all other parameters as functions of cref_wing
        stabi_alpha = self.stab_alpha(alpha, i_w, i_stab, AR, flap, blow_num)
        CD = flap * self.flap_CD_tot(alpha) + blow_num / 12 * self.blow_CD_tot(alpha) + \
             self.Wing_tipNacelle_CD_tot(alpha) + self.HLN_CD_tot(alpha) + \
             self.Fus_Vtail_CD_tot(alpha) + \
             self.Stab_CD_tot(stabi_alpha, trimtab) * self.Sref_stab / self.Sref_wing
        return CD

    def AC_CM(self, alpha, i_w, i_stab, AR, flap, blow_num, trimtab):
        # assuming all other parameters as functions of cref_wing
        # need aircraft geometry data to compute moment arms
        # assume all lift generated at wing and stabilator c/4
        # pos vector from wing c/4 to HT c/4

        r = csdl.Variable(name='r', shape=(3,), value=0)
        r1 = self.HT_axis.vector[0] + self.cref_stab / 4 - (self.Wing_axis.vector[0] + self.cref_wing / 4)
        r2 = 0
        r3 = self.HT_axis.vector[2] - self.Wing_axis.vector[2]
        r.set(csdl.slice[0], r1)
        r.set(csdl.slice[1], r2)
        r.set(csdl.slice[2], r3)

        
        stabi_alpha = self.stab_alpha(alpha, i_w, i_stab, AR, flap, blow_num)
        
        # Build f_hat using csdl operations so that csdl.cross works as expected
        f1 = -csdl.sin(stabi_alpha)
        f2 = 0
        f3 = csdl.cos(stabi_alpha)

        f_hat = csdl.Variable(name='f_hat', shape=(3,), value=0)
        f_hat.set(csdl.slice[0], f1)
        f_hat.set(csdl.slice[1], f2)
        f_hat.set(csdl.slice[2], f3)

        
        CL_stabi = self.Stab_CL_tot(stabi_alpha, trimtab)
        CM_stabi_wingcby4 = csdl.cross(r, f_hat) * (CL_stabi * self.Sref_stab) / (self.Sref_wing * self.cref_wing)
        
        Cm = (flap * self.flap_Cm_tot(alpha) +
            blow_num / 12 * self.blow_Cm_tot(alpha) +
            self.Wing_tipNacelle_Cm_tot(alpha) +
            self.HLN_Cm_tot(alpha) +
            self.Fus_Vtail_Cm_tot(alpha) +
            self.Stab_Cm_tot(stabi_alpha, trimtab) * self.Sref_stab * self.cref_stab / (self.Sref_wing * self.cref_wing) +
            CM_stabi_wingcby4[1])
    

        return Cm
    
         

    def get_FM_localAxis(self, states, controls, axis):
            """
            Compute forces and moments about the reference point.

            Parameters
            ----------
            x_bar : csdl.VariableGroup
                Flight-dynamic state (x̄) which should include:
                - density
                - VTAS
                - states.theta
            u_bar : csdl.Variable or csdl.VariableGroup
                Control input (ū) [currently not used in the aerodynamics calculation]

            Returns
            -------
            loads : ForcesMoments
                Computed forces and moments about the reference point.
            """
            density = states.atmospheric_states.density
            velocity = states.VTAS
            theta = states.states.theta
            p = states.states.p
            q = states.states.q
            r = states.states.r
            beta = states.beta
            alpha = theta + self.i_wing

            dstab = controls.elevator.deflection

            if hasattr(controls, 'left_flap'):
                dflap = controls.left_flap.deflection
                daileron = controls.left_aileron.deflection
            else:
                dflap = controls.flap.deflection
                daileron = controls.aileron.deflection

            dtrim = controls.trim_tab.deflection
            drudder = controls.rudder.deflection


            blow_num = 0
            for engine in controls.hl_engines:
                blow_num += 1


            CL = self.AC_CL(alpha=alpha, i_w=self.i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            CD = self.AC_CD(alpha=alpha, i_w=self.i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            CM = self.AC_CM(alpha=alpha, i_w=self.i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)

            L = 0.5 * density * velocity**2 * self.Sref_wing * CL
            D = 0.5 * density * velocity**2 * self.Sref_wing * CD
            M = 0.5 * density * velocity**2 * self.Sref_wing * CM * self.cref_wing


            phat = p * self.bref_wing / (2 * velocity)
            qhat = q * self.bref_wing / (2 * velocity)
            rhat = r * self.bref_wing / (2 * velocity)


            Cl = self.aeroDer['Clda'][0][0] * daileron + \
             self.aeroDer['Cldr'][0][0] * drudder + \
             self.aeroDer['Clp'][0][0] * phat + \
             self.aeroDer['Clr'][0][0] * rhat + \
             self.aeroDer['Clbeta'][0][0] * beta

            L_roll = 0.5 * density * velocity**2 * self.Sref_wing * self.bref_wing * Cl  # net rolling moment

            Cn = self.aeroDer['Cnda'][0][0] * daileron + \
                self.aeroDer['Cndr'][0][0] * drudder + \
                self.aeroDer['Cnp'][0][0] * phat + \
                self.aeroDer['Cnr'][0][0] * rhat + \
                self.aeroDer['Cnbeta'][0][0] * beta

            N_yaw = 0.5 * density * velocity**2 * self.Sref_wing * self.bref_wing * Cn  # net yawing moment

            CY = self.aeroDer['CYda'][0][0] * daileron + \
                self.aeroDer['CYdr'][0][0] * drudder + \
                self.aeroDer['CYp'][0][0] * phat + \
                self.aeroDer['CYr'][0][0] * rhat + \
                self.aeroDer['CYbeta'][0][0] * beta

            Y_sideforce = 0.5 * density * velocity**2 * self.Sref_wing * CY  # net sideforce

            aero_force = csdl.Variable(shape=(3,), value=0.)
            aero_force = aero_force.set(csdl.slice[0], -D)
            aero_force = aero_force.set(csdl.slice[1], Y_sideforce)
            aero_force = aero_force.set(csdl.slice[2], -L)
            force_vector = Vector(vector=aero_force, axis=axis)

            aero_moment = csdl.Variable(shape=(3,), value=0.)
            aero_moment = aero_moment.set(csdl.slice[0], L_roll)
            aero_moment = aero_moment.set(csdl.slice[1], M)
            aero_moment = aero_moment.set(csdl.slice[2], N_yaw)
            moment_vector = Vector(vector=aero_moment, axis=axis)
            loads = ForcesMoments(force=force_vector, moment=moment_vector)
            return loads
    
## WORK IN PROGRESS - HINGE MOMENTS
    def get_hinge_moments(self, states, controls):
        """
        Compute hinge moments for control surfaces.

        Parameters
        ----------
        states : csdl.VariableGroup
            Flight-dynamic state (x̄) which should include:
            - density
            - VTAS
            - states.theta
        controls : csdl.VariableGroup
            Control input (ū) which should include:
            - deflections of control surfaces

        Returns
        -------
        hinge_moments : HM
            Computed hinge moments for control surfaces.
        """
        density = states.atmospheric_states.density
        velocity = states.VTAS
        
        Cme_de  = self.aeroDer['Cme_de'][0][0]
        Cma_da  = self.aeroDer['Cma_da'][0][0]
        Cmr_dr  = self.aeroDer['Cmr_dr'][0][0]

        S_e = self.Sref_stab
        c_e = self.cref_stab
        S_a = self.Sref_wing
        c_a = self.cref_wing
        S_r = self.Sref_VT
        c_r = self.cref_VT


        delta_e = controls.elevator.deflection
        delta_tt = controls.trim_tab.deflection


        if hasattr(controls, 'left_aileron'):
            delta_aL = controls.left_aileron.deflection
            delta_aR = controls.right_aileron.deflection
            delta_fL = controls.left_flap.deflection
            delta_fR = controls.right_flap.deflection

        else:
            delta_a = controls.aileron.deflection
            delta_f = controls.flap.deflection
        
        
        delta_r = controls.rudder.deflection

        Ch_f = CH0_f + (CH_alpha_f * alpha_f) + (Ch_delta_f * delta_f) + (Ch_delta_tt * delta_tt) # Flap Hinge Moment Coefficient
        Ch_a = Ch0_a + (Ch_alpha_a * alpha) + (Ch_delta_a * delta_a) + (Ch_delta_tt * delta_tt) # Aileron Hinge Moment Coefficient
        Ch_e = Ch0_e + (Ch_alpha_e * alpha_HT) + (Ch_delta_e * delta_e) + (Ch_delta_tt * delta_tt) # Elevator Hinge Moment Coefficient
        Ch_r = Ch0_r + (Ch_beta_r * beta) + (Ch_delta_r * delta_r) + (Ch_delta_tt * delta_tt) # Rudder Hinge Moment Coefficient

        Hf = 0.5 * density * velocity**2 * S_a * c_a * Ch_f
        Ha = 0.5 * density * velocity**2 * S_a * c_a * Ch_a
        He = 0.5 * density * velocity**2 * S_e * c_e * Ch_e
        Hr = 0.5 * density * velocity**2 * S_r * c_r * Ch_r

        return {
            'elevator': He,
            'aileron':  Ha,
            'rudder':   Hr,
            'flap':     Hf
        }
        


class HLPropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        V_inf_data = np.array(
            [0,61.3,87.6,101.6,113.8,131.3,157.6,175])
        RPM_data = np.array(
        [0,3545,4661,4702,4379,3962,3428,3451])
        self.rpm = Akima1DInterpolator(V_inf_data, RPM_data, method="akima")
        self.rpm_derivative = Akima1DInterpolator.derivative(self.rpm)
        self.min_RPM = min(RPM_data)
        self.max_RPM = max(RPM_data)
        # Obtained Mod-IV Propeller Data from CFD database
        J_data = np.array(
            [0,0.5490,0.5966,0.6860,0.8250,1.0521,1.4595,1.6098])
        Ct_data = np.array(
            [0,0.3125,0.3058,0.2848,0.2473,0.1788,0.0366,-0.0198])
        self.ct = Akima1DInterpolator(V_inf_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)

        Cp_data = np.array([0,0.3134,0.3152,0.3075,0.2874,0.2367,0.0809,0.0018])
        self.cp = Akima1DInterpolator(V_inf_data, Cp_data, method="akima")
        self.cp_derivative = Akima1DInterpolator.derivative(self.cp)

        Torque_data = np.array([0,9.3,16.1,16.0,13.0,8.7,2.2,0.0])
        self.torque = Akima1DInterpolator(V_inf_data, Torque_data, method="akima")
        self.torque_derivative = Akima1DInterpolator.derivative(self.torque)

        # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, velocity: csdl.Variable=None):
        outputs = csdl.VariableGroup()

        if velocity is not None:
            self.declare_input('velocity', velocity)
            cp = self.create_output('cp', velocity.shape)
            rpm = self.create_output('rpm', velocity.shape)
            torque = self.create_output('torque', velocity.shape)
            ct = self.create_output('ct', velocity.shape)

            outputs.ct = ct
            outputs.cp = cp
            outputs.rpm = rpm
            outputs.torque = torque

        return outputs
            
    def compute(self, input_vals, output_vals):

        if 'velocity' in input_vals:
            velocity = input_vals['velocity']
            output_vals['ct'] = self.ct(velocity)
            output_vals['rpm'] = self.rpm(velocity)
            output_vals['cp'] = self.cp(velocity)
            output_vals['torque'] = self.torque(velocity)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        velocity = input_vals['velocity']

        derivatives['ct', 'velocity'] = np.diag(self.ct_derivative(velocity))
        derivatives['rpm', 'velocity'] = np.diag(self.rpm_derivative(velocity))
        derivatives['cp', 'velocity'] = np.diag(self.cp_derivative(velocity))
        derivatives['torque', 'velocity'] = np.diag(self.torque_derivative(velocity))



class CruisePropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        # Obtained Mod-III Propeller Data from CFD database
        V_inf_data = np.array(
            [0,18.75,75,112.5,150,187.5,225,243.75,262.5,266.67,300])
        RPM_data = np.array(
        [0,2250,2250,2250,2250,2250,2250,2250,2250,2000,2000])
        self.rpm = Akima1DInterpolator(V_inf_data, RPM_data, method="akima")
        self.rpm_derivative = Akima1DInterpolator.derivative(self.rpm)
        self.min_RPM = min(RPM_data)
        self.max_RPM = max(RPM_data)
        J_data = np.array(
            [0,0.1,0.4,0.6,0.8,1.0,1.2,1.3,1.4,1.6,1.8])
        Ct_data = np.array(
            [0,0.1831,0.1673,0.1422,0.1003,0.0479,-0.0085,-0.0366,-0.0057,0.0030,-0.0504])
        self.ct = Akima1DInterpolator(V_inf_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)
        Cp_data = np.array([0,0.1155,0.1219,0.1195,0.0979,0.0563,-0.0021,-0.0365,0.0003,0.0134,-0.0756])
        self.cp = Akima1DInterpolator(V_inf_data, Cp_data, method="akima")
        self.cp_derivative = Akima1DInterpolator.derivative(self.cp)
        Torque_data = np.array([0,178.38,188.26,184.47,151.25,73.56,-2.79,-47.67,0.35,11.10,-62.45])
        self.torque = Akima1DInterpolator(V_inf_data, Torque_data, method="akima")
        self.torque_derivative = Akima1DInterpolator.derivative(self.torque)

    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, velocity: csdl.Variable=None):
        outputs = csdl.VariableGroup()

        if velocity is not None:
            self.declare_input('velocity', velocity)
            cp = self.create_output('cp', velocity.shape)
            rpm = self.create_output('rpm', velocity.shape)
            torque = self.create_output('torque', velocity.shape)
            ct = self.create_output('ct', velocity.shape)

            outputs.ct = ct
            outputs.cp = cp
            outputs.rpm = rpm
            outputs.torque = torque

        return outputs
            
    def compute(self, input_vals, output_vals):

        if 'velocity' in input_vals:
            velocity = input_vals['velocity']
            output_vals['ct'] = self.ct(velocity)
            output_vals['rpm'] = self.rpm(velocity)
            output_vals['cp'] = self.cp(velocity)
            output_vals['torque'] = self.torque(velocity)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        velocity = input_vals['velocity']

        derivatives['ct', 'velocity'] = np.diag(self.ct_derivative(velocity))
        derivatives['rpm', 'velocity'] = np.diag(self.rpm_derivative(velocity))
        derivatives['cp', 'velocity'] = np.diag(self.cp_derivative(velocity))
        derivatives['torque', 'velocity'] = np.diag(self.torque_derivative(velocity))


class X57Propulsion(Loads):

    def __init__(self, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:Union[HLPropCurve, CruisePropCurve], engine_index: int = 0, **kwargs):
        self.prop_curve = prop_curve
        self.engine_index = engine_index  # Save the index for later lookup

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=1.89/2) 
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius



    def get_FM_localAxis(self, states, controls, axis):
        """
        This is for the propeller models in the HLPropCurve and CruisePropCurve classes.
        The propeller model is based on the propeller data from the Mod-III and Mod-IV propeller data from the CFD database.
        Compute forces and moments about the reference point.

        Parameters
        ----------
        x_bar : csdl.VariableGroup
            Flight-dynamic state (x̄) which should include:
            - density
            - VTAS
            - states.theta
        u_bar : csdl.Variable or csdl.VariableGroup
            Control input (ū) which should include:
            - throttle

        Returns
        -------
        loads : ForcesMoments
            Computed forces and moments about the reference point.
        """
        throttle = controls.u[7+self.engine_index]  # Get the throttle for the specific engine
        density = states.atmospheric_states.density * 0.00194032  # kg/m^3 to slugs/ft^3
        velocity = states.VTAS * 3.281  # m/s to ft/s

        # Compute RPM
        rpm_curve = type(self.prop_curve)() 
        rpm = rpm_curve.evaluate(velocity=velocity).rpm * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s

        # Compute advance ratio
        J = (np.pi * velocity) / (omega_RAD * self.radius)   # non-dimensional

        # Compute Ct
        ct_curve = type(self.prop_curve)()
        ct = ct_curve.evaluate(velocity=velocity).ct

        # Compute Thrust
        T_raw = (ct * density * (rpm/60)**2 * ((self.radius*2)**4) * 4.44822) # lbf to N
        
        T_raw.value = np.nan_to_num(T_raw.value, nan=1e-6)  # Replace NaN with 1e-6 to prevent division by zero

        T = csdl.Variable(shape=(1,), value=T_raw.value)  
    
        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads

   

    def get_torque_power(self, states, controls):
        """
        Compute power required for the propulsion system.

        Parameters
        ----------
        torque : 
            Power required for the propulsion system (P) which should include:
            - power
        
        rpm :
            Rotational speed of the propeller (RPM) which should include:
            - rpm
        
        x_bar : csdl.VariableGroup
            Flight-dynamic state (x̄) which should include:
            - density

        Returns
        -------
        torque : csdl.VariableGroup
            Computed torque required for the propulsion system.
        """
        throttle = controls.u[7+self.engine_index]  # Get the throttle for the specific engine
        density = states.atmospheric_states.density * 0.00194032
        velocity = states.VTAS * 3.281  # m/s to ft/s

        # Compute RPM
        rpm_curve = type(self.prop_curve)() 
        rpm = rpm_curve.evaluate(velocity=velocity).rpm * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s

        torque_curve = type(self.prop_curve)()
        torque = torque_curve.evaluate(velocity=velocity).torque

        power_raw = (torque * (rpm/60) * 2 * np.pi) * 1.35582 # Convert to Watts (lbf-ft/s to W)

        # power_raw = (torque * (rpm/60) * 2 * np.pi)/737.56 # Convert to horsepower (lbf-ft/s to hp)


        power_raw.value = np.nan_to_num(power_raw.value, nan=1e-6)  # Replace NaN with 1e-6 to prevent division by zero

        power = csdl.Variable(shape=(1,), value=power_raw.value)  

        return torque, power
        

# if __name__ == "__main__":
#     recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
#     recorder.start()
#     openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geo, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis, rudder_axis, _ = create_axes()
#     aircraft = build_aircraft()
#     recorder.stop()


