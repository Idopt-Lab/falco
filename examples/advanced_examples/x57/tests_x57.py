from pathlib import Path
from unittest import TestCase
from flight_simulator import REPO_ROOT_FOLDER, Q_
import sys
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.vehicle.conditions import aircraft_conditions

x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))
from x57_geometry import get_geometry, get_geometry_related_axis
from x57_component import build_aircraft_component
from x57_mp import add_mp_to_components
from x57_control_system import X57ControlSystem
from x57_solvers import X57Aerodynamics, X57Propulsion, HLPropCurve, CruisePropCurve




def get_geo():
    geometry_data = get_geometry()
    axis_dict = get_geometry_related_axis(geometry_data)
    aircraft_component = build_aircraft_component(geo_dict=geometry_data, do_geo_param=False)
    add_mp_to_components(aircraft_component, geometry_data, axis_dict)
    aircraft_component.mass_properties = aircraft_component.compute_total_mass_properties()
    return geometry_data, axis_dict, aircraft_component

def get_control_system(aircraft_component, hl_engines: int = 0, cm_engines: int = 0):
    x57_controls = X57ControlSystem(elevator_component=aircraft_component.comps['Elevator'],
                                    rudder_component=aircraft_component.comps['Vertical Tail'].comps['Rudder'],
                                    aileron_left_component=aircraft_component.comps['Wing'].comps['Left Aileron'],
                                    aileron_right_component=aircraft_component.comps['Wing'].comps['Right Aileron'],
                                    trim_tab_component=aircraft_component.comps['Elevator'].comps['Trim Tab'],
                                    flap_left_component=aircraft_component.comps['Wing'].comps['Left Flap'],
                                    flap_right_component=aircraft_component.comps['Wing'].comps['Right Flap'],
                                    hl_engine_count=hl_engines,cm_engine_count=cm_engines)
    return x57_controls




class TestCase(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

# region Mass properties tests

    def testMP_0_alpha(self):

        geometry_data, axis_dict, aircraft_component = get_geo()
        x57_controls = get_control_system(aircraft_component, cm_engines=2)
              
        cruise = aircraft_conditions.CruiseCondition(
            fd_axis=axis_dict['fd_axis'],
            controls=x57_controls,
            altitude=Q_(2500, 'ft'),
            range=Q_(100, 'km'),
            speed=Q_(5.715, 'm/s'),
            pitch_angle=Q_(0, 'deg'))

        tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        np.testing.assert_almost_equal(tf.value,
                                       desired=np.array([0, 0, 13355]), decimal=0)
        np.testing.assert_almost_equal(tm.value,
                                       desired=np.array([0, 49170, 0.]), decimal=0)
        

    def testMP_2_alpha(self):

        geometry_data, axis_dict, aircraft_component = get_geo()
        x57_controls = get_control_system(aircraft_component, cm_engines=2)
              
        cruise = aircraft_conditions.CruiseCondition(
            fd_axis=axis_dict['fd_axis'],
            controls=x57_controls,
            altitude=Q_(2500, 'ft'),
            range=Q_(100, 'km'),
            speed=Q_(5.715, 'm/s'),
            pitch_angle=Q_(2, 'deg'))

        tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        np.testing.assert_almost_equal(tf.value,
                                       desired=np.array([-466, 0., 13346]), decimal=0)
        np.testing.assert_almost_equal(tm.value,
                                       desired=np.array([0, 49371, 0]), decimal=0)


# endregion

# region Prop tests

    def testPropFM_bothMotors(self):

        geometry_data, axis_dict, aircraft_component = get_geo()
        x57_controls = get_control_system(aircraft_component, cm_engines=2)
    

        cruise = aircraft_conditions.CruiseCondition(
            fd_axis=axis_dict['fd_axis'],
            controls=x57_controls,
            altitude=Q_(8000, 'ft'),
            range=Q_(100, 'km'),
            speed=Q_(57.15, 'm/s'),
            pitch_angle=Q_(0, 'deg'))
        
        tf0, tm0 = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)

        
        for left_engine, right_engine in zip(x57_controls.cm_engines_left, x57_controls.cm_engines_right):
            left_engine.rpm.value = 2250
            right_engine.rpm.value = 2250

        cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft

        cruise_motor1_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=0)
        cruise_motor2_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=1)
        aircraft_component.comps['Wing'].comps['Cruise Motor 1'].load_solvers.append(cruise_motor1_prop)
        aircraft_component.comps['Wing'].comps['Cruise Motor 2'].load_solvers.append(cruise_motor2_prop)           


        tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        tf = tf - tf0
        tm = tm - tm0


        np.testing.assert_almost_equal(tf.value,
                                       desired=np.array([1428, 0, 0]), decimal=0) # desired value comes from X57 CFD data
        np.testing.assert_almost_equal(tm.value,
                                       desired=np.array([0, -3170, 0]), decimal=0)
        

    def testPropFM_leftMotor(self):

        geometry_data, axis_dict, aircraft_component = get_geo()
        x57_controls = get_control_system(aircraft_component, cm_engines=2)
    
        cruise = aircraft_conditions.CruiseCondition(
            fd_axis=axis_dict['fd_axis'],
            controls=x57_controls,
            altitude=Q_(8000, 'ft'),
            range=Q_(100, 'km'),
            speed=Q_(57.15, 'm/s'),
            pitch_angle=Q_(0, 'deg'))
    
        tf0, tm0 = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)

        for left_engine in x57_controls.cm_engines_left:
            left_engine.rpm.value = 2250

        cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft

        cruise_motor1_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=0)
        aircraft_component.comps['Wing'].comps['Cruise Motor 1'].load_solvers.append(cruise_motor1_prop)


        tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        tf = tf - tf0
        tm = tm - tm0
        np.testing.assert_almost_equal(tf.value,
                                       desired=np.array([1428, 0, 0])/2, decimal=0) # desired value comes from X57 CFD data
        np.testing.assert_almost_equal(tm.value,
                                       desired=np.array([0.0, -1585, -3349]), decimal=0)
        

    def testPropFM_rightMotor(self):

        geometry_data, axis_dict, aircraft_component = get_geo()
        x57_controls = get_control_system(aircraft_component, cm_engines=2)
    
        cruise = aircraft_conditions.CruiseCondition(
            fd_axis=axis_dict['fd_axis'],
            controls=x57_controls,
            altitude=Q_(8000, 'ft'),
            range=Q_(100, 'km'),
            speed=Q_(57.15, 'm/s'),
            pitch_angle=Q_(0, 'deg'))
    
        tf0, tm0 = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        
        for right_engine in x57_controls.cm_engines_right:
            right_engine.rpm.value = 2250

        cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft

        cruise_motor2_prop = X57Propulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(),engine_index=1)
        aircraft_component.comps['Wing'].comps['Cruise Motor 2'].load_solvers.append(cruise_motor2_prop)

        tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        tf = tf - tf0
        tm = tm - tm0
        np.testing.assert_almost_equal(tf.value,
                                       desired=np.array([1428, 0, 0])/2, decimal=0) # desired value comes from X57 CFD data
        np.testing.assert_almost_equal(tm.value,
                                       desired=np.array([0.0, -1585, 3349]), decimal=0)
# endregion

# region Aero tests
# # Aero alone test
# # - Aero produces loads at wind axis -> wing axis -> FD body-fixed axis
# # - L/D when cruise motors are off < L/D when cruise motors are on
# # - CL when HLP are off << CL when HLP are on

    def testAeroFM(self):
        geometry_data, axis_dict, aircraft_component = get_geo()
        x57_controls = get_control_system(aircraft_component, hl_engines=12, cm_engines=2)

        cruise = aircraft_conditions.CruiseCondition(
            fd_axis=axis_dict['fd_axis'],
            controls=x57_controls,
            altitude=Q_(2500, 'ft'),
            range=Q_(100, 'km'),
            speed=Q_(5.715, 'm/s'),
            pitch_angle=Q_(0, 'deg'))
    
        
        tf0, tm0 = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)

        x57_aerodynamics = X57Aerodynamics(component=aircraft_component)
        aircraft_component.comps['Wing'].load_solvers.append(x57_aerodynamics)


        tf, tm = aircraft_component.compute_total_loads(fd_state=cruise.ac_states, controls=x57_controls)
        tf = tf - tf0
        tm = tm - tm0

        np.testing.assert_almost_equal(tf.value,
                                       desired=np.array([-25, 0, -211]), decimal=0)
        np.testing.assert_almost_equal(tm.value,
                                        desired=np.array([0, -767, 0]), decimal=0)
    
        

# endregion















# todo: after Aviation paper
# - Test parameterization solver when aircraft components are defined in a heirarchy