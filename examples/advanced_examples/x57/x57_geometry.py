import time
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
from flight_simulator.utils.import_geometry import import_geometry
from flight_simulator import REPO_ROOT_FOLDER, Q_




t0 = time.time()
lfs.num_workers = 1

debug = False



in2m=0.0254
ft2m = 0.3048

def get_geometry():

    geometry = import_geometry(
        file_name="x57.stp",
        file_path= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57',
        refit=False,
        scale=in2m,
        rotate_to_body_fixed_frame=True
    )


    wing = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4'], name='wing')
    aileronR = geometry.declare_component(function_search_names=['Rt_Aileron'], name='aileronR')
    aileronL = geometry.declare_component(function_search_names=['Lt_Aileron'], name='aileronL')
    flapL = geometry.declare_component(function_search_names=['Flap, 1'], name='left_flap')
    flapR = geometry.declare_component(function_search_names=['Flap, 0'], name='right_flap')
    h_tail = geometry.declare_component(function_search_names=['HorzStab'], name='h_tail')
    trimTab = geometry.declare_component(function_search_names=['TrimTab'], name='trimTab')
    htALL = geometry.declare_component(function_search_names=['HorzStab', 'TrimTab'], name='CompleteHT')
    vertTail = geometry.declare_component(function_search_names=['VertTail'], name='vertTail')
    rudder = geometry.declare_component(function_search_names=['Rudder'], name='rudder')
    vtALL = geometry.declare_component(function_search_names=['VertTail', 'Rudder'], name='CompleteVT')
    fuselage = geometry.declare_component(function_search_names=['Fuselage'], name='fuselage')
    gear_pod = geometry.declare_component(function_search_names=['GearPod'], name='gear_pod')

    pylon1 = geometry.declare_component(function_search_names=['Pylon_12, 1'], name='pylon1')
    pylon2 = geometry.declare_component(function_search_names=['Pylon_11, 1'], name='pylon2')
    pylon3 = geometry.declare_component(function_search_names=['Pylon_10, 1'], name='pylon3')
    pylon4 = geometry.declare_component(function_search_names=['Pylon_09, 1'], name='pylon4')
    pylon5 = geometry.declare_component(function_search_names=['Pylon_08, 1'], name='pylon5')
    pylon6 = geometry.declare_component(function_search_names=['Pylon_07, 1'], name='pylon6')
    pylon7 = geometry.declare_component(function_search_names=['Pylon_07, 0'], name='pylon7')
    pylon8 = geometry.declare_component(function_search_names=['Pylon_08, 0'], name='pylon8')
    pylon9 = geometry.declare_component(function_search_names=['Pylon_09, 0'], name='pylon9')
    pylon10 = geometry.declare_component(function_search_names=['Pylon_10, 0'], name='pylon10')
    pylon11 = geometry.declare_component(function_search_names=['Pylon_11, 0'], name='pylon11')
    pylon12 = geometry.declare_component(function_search_names=['Pylon_12, 0'], name='pylon12')

    nacelle7 = geometry.declare_component(function_search_names=['HLNacelle_7_Tail'], name='nacelle7')
    nacelle8 = geometry.declare_component(function_search_names=['HLNacelle_8_Tail'], name='nacelle8')
    nacelle9 = geometry.declare_component(function_search_names=['HLNacelle_9_Tail'], name='nacelle9')
    nacelle10 = geometry.declare_component(function_search_names=['HLNacelle_10_Tail'], name='nacelle10')
    nacelle11 = geometry.declare_component(function_search_names=['HLNacelle_11_Tail'], name='nacelle11')
    nacelle12 = geometry.declare_component(function_search_names=['HLNacelle_12_Tail'], name='nacelle12')

    spinner1 = geometry.declare_component(function_search_names=['HL_Spinner12, 1'], name='spinner1')
    spinner2 = geometry.declare_component(function_search_names=['HL_Spinner11, 1'], name='spinner2')
    spinner3 = geometry.declare_component(function_search_names=['HL_Spinner10, 1'], name='spinner3')
    spinner4 = geometry.declare_component(function_search_names=['HL_Spinner9, 1'], name='spinner4')
    spinner5 = geometry.declare_component(function_search_names=['HL_Spinner8, 1'], name='spinner5')
    spinner6 = geometry.declare_component(function_search_names=['HL_Spinner7, 1'], name='spinner6')
    spinner7 = geometry.declare_component(function_search_names=['HL_Spinner7, 0'], name='spinner7')
    spinner8 = geometry.declare_component(function_search_names=['HL_Spinner8, 0'], name='spinner8')
    spinner9 = geometry.declare_component(function_search_names=['HL_Spinner9, 0'], name='spinner9')
    spinner10 = geometry.declare_component(function_search_names=['HL_Spinner10, 0'], name='spinner10')
    spinner11 = geometry.declare_component(function_search_names=['HL_Spinner11, 0'], name='spinner11')
    spinner12 = geometry.declare_component(function_search_names=['HL_Spinner12, 0'], name='spinner12')

    prop = geometry.declare_component(function_search_names=['HL-Prop'], name='prop')
    motor = geometry.declare_component(function_search_names=['HL_Motor'], name='motor')
    motor_interface = geometry.declare_component(function_search_names=['HL_Motor_Controller_Interface'], name='motor_interface')

    cruise_spinner1 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 0'], name='cruise_spinner1')
    cruise_spinner2 =  geometry.declare_component(function_search_names=['CruiseNacelle-Spinner, 1'], name='cruise_spinner2')

    wingALL = geometry.declare_component(function_search_names=['Wing_Sec1','Wing_Sec2','Wing_Sec3','Wing_Sec4','Rt_Aileron','Lt_Aileron','Flap, 0','Flap, 1',
                                                                'Pylon_12, 1','Pylon_11, 1','Pylon_10, 1','Pylon_09, 1','Pylon_08, 1','Pylon_07, 1',
                                                                'Pylon_12, 0','Pylon_11, 0','Pylon_10, 0','Pylon_09, 0','Pylon_08, 0','Pylon_07, 0',
                                                                'HL_Spinner12, 1','HL_Spinner11, 1','HL_Spinner10, 1','HL_Spinner9, 1','HL_Spinner8, 1','HL_Spinner7, 1',
                                                                'HL_Spinner12, 0','HL_Spinner11, 0','HL_Spinner10, 0','HL_Spinner9, 0','HL_Spinner8, 0','HL_Spinner7, 0',
                                                                'CruiseNacelle-Spinner, 0','CruiseNacelle-Spinner, 1', 'Flap_Cover_7','Flap_Cover_9','Flap_Cover_11'], name='CompleteWing')

                                                        


    cruise_motor =  geometry.declare_component(function_search_names=['CruiseNacelle-Motor'], name='cruise_motor')
    cruise_nacelle =  geometry.declare_component(function_search_names=['CruiseNacelle-Tail'], name='cruise_nacelle')
    cruise_prop = geometry.declare_component(function_search_names=['Cruise-Prop'], name='cruise_prop')

    M1_components = [pylon7, nacelle7, spinner1, prop, motor, motor_interface]
    M2_components = [pylon8, nacelle8, spinner2, prop, motor, motor_interface]
    M3_components = [pylon9, nacelle9, spinner3, prop, motor, motor_interface]
    M4_components = [pylon10, nacelle10, spinner4, prop, motor, motor_interface]
    M5_components = [pylon11, nacelle11, spinner5, prop, motor, motor_interface]
    M6_components = [pylon12, nacelle12, spinner6, prop, motor, motor_interface]
    M7_components = [pylon7, nacelle7, spinner7, prop, motor, motor_interface]
    M8_components = [pylon8, nacelle8, spinner8, prop, motor, motor_interface]
    M9_components = [pylon9, nacelle9, spinner9, prop, motor, motor_interface]
    M10_components = [pylon10, nacelle10, spinner10, prop, motor, motor_interface]
    M11_components = [pylon11, nacelle11, spinner11, prop, motor, motor_interface]
    M12_components = [pylon12, nacelle12, spinner12, prop, motor, motor_interface]
    CM1_components = [cruise_nacelle, cruise_spinner1, cruise_prop, cruise_motor]
    CM2_components = [cruise_nacelle, cruise_spinner2, cruise_prop, cruise_motor]
    total_HL_motor_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components]
    total_prop_sys_components = [M1_components, M2_components, M3_components, M4_components, M5_components, M6_components, M7_components, M8_components, M9_components, M10_components, M11_components, M12_components, CM1_components, CM2_components]
    # geometry.plot()


    # Wing Region Info
    wing_le_left_guess = np.array([-12.356, -16, -7.5])*ft2m
    wing_le_left_parametric = wing.project(wing_le_left_guess, plot=False)
    wing_le_left = geometry.evaluate(wing_le_left_parametric)

    wing_le_right_guess = np.array([-12.356, 16, -7.5])*ft2m
    wing_le_right_parametric = wing.project(wing_le_right_guess, plot=False)
    wing_le_right = geometry.evaluate(wing_le_right_parametric)

    wing_le_center_guess = np.array([-12.356, 0., -7.5])*ft2m
    wing_le_center_parametric = wing.project(wing_le_center_guess, plot=False)
    wing_le_center = geometry.evaluate(wing_le_center_parametric)

    wing_te_left_guess = np.array([-14.25, -16, -7.5])*ft2m
    wing_te_left_parametric = wing.project(wing_te_left_guess, plot=False)
    wing_te_left = geometry.evaluate(wing_te_left_parametric)

    wing_te_right_guess = np.array([-14.25, 16, -7.5])*ft2m
    wing_te_right_parametric = wing.project(wing_te_right_guess, plot=False)
    wing_te_right = geometry.evaluate(wing_te_right_parametric)

    wing_te_center_guess = np.array([-14.25, 0., -7.5])*ft2m
    wing_te_center_parametric = wing.project(wing_te_center_guess, plot=False)
    wing_te_center = geometry.evaluate(wing_te_center_parametric)


    wing_te_aileron_left_parametric = wing.project(np.array([-13.85, -13.4, -7.5])*ft2m, plot=False)
    wing_te_aileron_left = geometry.evaluate(wing_te_aileron_left_parametric)

    wing_te_aileron_right_parametric = wing.project(np.array([-13.85, 13.4, -7.5])*ft2m, plot=False)
    wing_te_aileron_right = geometry.evaluate(wing_te_aileron_right_parametric)

    wing_te_flap_left_parametric = wing.project(np.array([-13.85, -6.05, -7.5])*ft2m, plot=False)
    wing_te_flap_left = geometry.evaluate(wing_te_aileron_left_parametric)

    wing_te_flap_right_parametric = wing.project(np.array([-13.85, 6.05, -7.5])*ft2m, plot=False)
    wing_te_flap_right = geometry.evaluate(wing_te_flap_right_parametric)


    wing_qc_center_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -7.5])*ft2m, plot=False)
    wing_qc_center = geometry.evaluate(wing_qc_center_parametric)
    wing_qc_tip_right_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), 16., -7.5])*ft2m, plot=False)
    wing_qc_tip_left_parametric = geometry.project(np.array([-12.356+(0.25*(-14.25+12.356)), -16., -7.5])*ft2m, plot=False)


    wing_parametric_geometry = [
        wing_le_left_parametric,
        wing_le_right_parametric,
        wing_le_center_parametric,
        wing_te_left_parametric,
        wing_te_right_parametric,
        wing_te_center_parametric,
        wing_qc_center_parametric,
        wing_qc_tip_right_parametric,
        wing_qc_tip_left_parametric
    ]

    wingspan = np.linalg.norm(wing_le_left.value - wing_le_right.value)

    ## ADD CONTROL SURFACE INFO PROJECTIONS HERE

    left_aileron_le_left_guess = np.array([-13.85, -15.5, -7.5])*ft2m
    left_aileron_le_left_parametric = aileronL.project(left_aileron_le_left_guess, plot=False)
    left_aileron_le_left = geometry.evaluate(left_aileron_le_left_parametric)

    left_aileron_le_right_guess = np.array([-13.85, -11.4, -7.5])*ft2m
    left_aileron_le_right_parametric = aileronL.project(left_aileron_le_right_guess, plot=False)
    left_aileron_le_right = geometry.evaluate(left_aileron_le_right_parametric)

    left_aileron_le_center_guess = np.array([-13.85, -13.4, -7.5])*ft2m
    left_aileron_le_center_parametric = aileronL.project(left_aileron_le_center_guess, plot=False)
    left_aileron_le_center = geometry.evaluate(left_aileron_le_center_parametric)

    left_aileron_le_center_on_wing_te_guess = left_aileron_le_center_guess
    left_aileron_le_center_on_wing_te_parametric = wing.project(left_aileron_le_center_on_wing_te_guess, plot=False)

    left_aileron_te_center_parametric = aileronL.project(np.array([-14.25, -13.4, -7.5])*ft2m, plot=False)
    left_aileron_te_center = geometry.evaluate(left_aileron_te_center_parametric)

    left_aileron_te_left_parametric = aileronL.project(np.array([-14.25, -15.5, -7.5])*ft2m, plot=False)
    left_aileron_te_left = geometry.evaluate(left_aileron_te_left_parametric)

    left_aileron_te_right_parametric = aileronL.project(np.array([-14.25, -11.4, -7.5])*ft2m, plot=False)
    left_aileron_te_right = geometry.evaluate(left_aileron_te_right_parametric)

    left_aileron_qc_center_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -13.4, -7.5])*ft2m, plot=False)
    left_aileron_qc = geometry.evaluate(left_aileron_qc_center_parametric)
    left_aileron_qc_tip_right_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.4, -7.5])*ft2m, plot=False)
    left_aileron_qc_tip_left_parametric = aileronL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -15.15, -7.5])*ft2m, plot=False)

    left_aileron_span = np.linalg.norm(left_aileron_le_left.value - left_aileron_le_right.value)
    left_aileron_chord = np.linalg.norm(left_aileron_le_center[0].value - left_aileron_te_center[0].value)

    left_aileron_parametric_geometry = [
        left_aileron_le_left_parametric,
        left_aileron_le_right_parametric,
        left_aileron_le_center_parametric,
        left_aileron_te_left_parametric,
        left_aileron_te_right_parametric,
        left_aileron_te_center_parametric,
        left_aileron_qc_center_parametric,
        left_aileron_qc_tip_right_parametric,
        left_aileron_qc_tip_left_parametric
    ]

    right_aileron_le_left_guess = np.array([-13.85, 11.4, -7.5])*ft2m
    right_aileron_le_left_parametric = aileronR.project(right_aileron_le_left_guess, plot=False)
    right_aileron_le_left = geometry.evaluate(right_aileron_le_left_parametric)

    right_aileron_le_right_guess = np.array([-13.85, 15.5, -7.5])*ft2m
    right_aileron_le_right_parametric = aileronR.project(right_aileron_le_right_guess, plot=False)
    right_aileron_le_right = geometry.evaluate(right_aileron_le_right_parametric)

    right_aileron_le_center_guess = np.array([-13.85, 13.4, -7.5])*ft2m
    right_aileron_le_center_parametric = aileronR.project(right_aileron_le_center_guess, plot=False)
    right_aileron_le_center = geometry.evaluate(right_aileron_le_center_parametric)

    right_aileron_le_center_on_wing_te_guess = right_aileron_le_center_guess
    right_aileron_le_center_on_wing_te_parametric = wing.project(right_aileron_le_center_on_wing_te_guess, plot=False)

    right_aileron_te_center_parametric = aileronR.project(np.array([-14.25, 13.4, -7.5])*ft2m, plot=False)
    right_aileron_te_center = geometry.evaluate(right_aileron_te_center_parametric)

    right_aileron_te_left_parametric = aileronR.project(np.array([-14.25, 11.4, -7.5])*ft2m, plot=False)
    right_aileron_te_left = geometry.evaluate(right_aileron_te_left_parametric)

    right_aileron_te_right_parametric = aileronR.project(np.array([-14.25, 15.5, -7.5])*ft2m, plot=False)
    right_aileron_te_right = geometry.evaluate(right_aileron_te_right_parametric)

    right_aileron_qc_center_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 13.12, -7.5])*ft2m, plot=False)
    right_aileron_qc = geometry.evaluate(right_aileron_qc_center_parametric)
    right_aileron_qc_tip_right_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 11.4, -7.5])*ft2m, plot=False)
    right_aileron_qc_tip_left_parametric = aileronR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 15.15, -7.5])*ft2m, plot=False)
                                                            
    right_aileron_parametric_geometry = [
        right_aileron_le_left_parametric,
        right_aileron_le_right_parametric,
        right_aileron_le_center_parametric,
        right_aileron_te_left_parametric,
        right_aileron_te_right_parametric,
        right_aileron_te_center_parametric,
        right_aileron_qc_center_parametric,
        right_aileron_qc_tip_right_parametric,
        right_aileron_qc_tip_left_parametric
    ]


    right_aileron_span = np.linalg.norm(right_aileron_le_left.value - right_aileron_le_right.value)
    right_aileron_chord = np.linalg.norm(right_aileron_le_center[0].value - right_aileron_te_center[0].value)

    left_flap_le_left_guess = np.array([-13.85, -11.5, -7.5])*ft2m
    left_flap_le_left_parametric = flapL.project(left_flap_le_left_guess, plot=False)
    left_flap_le_left = geometry.evaluate(left_flap_le_left_parametric)

    left_flap_le_right_guess = np.array([-13.85, -0.6, -7.5])*ft2m
    left_flap_le_right_parametric = flapL.project(left_flap_le_right_guess, plot=False)
    left_flap_le_right = geometry.evaluate(left_flap_le_right_parametric)

    left_flap_le_center_guess = np.array([-13.85, -6.05, -7.5])*ft2m
    left_flap_le_center_parametric = flapL.project(left_flap_le_center_guess, plot=False)
    left_flap_le_center = geometry.evaluate(left_flap_le_center_parametric)

    left_flap_le_center_on_wing_te_guess = left_flap_le_center_guess
    left_flap_le_center_on_wing_te_parametric = wing.project(left_flap_le_center_on_wing_te_guess, plot=False)

    left_flap_te_center_parametric = flapL.project(np.array([-14.25, -6.05, -5.5])*ft2m, plot=False)
    left_flap_te_center = geometry.evaluate(left_flap_te_center_parametric)

    left_flap_te_left_parametric = flapL.project(np.array([-14.25, -11.5, -5.5])*ft2m, plot=False)
    left_flap_te_left = geometry.evaluate(left_flap_te_left_parametric)

    left_flap_te_right_parametric = flapL.project(np.array([-14.25, -0.6, -5.5])*ft2m, plot=False)
    left_flap_te_right = geometry.evaluate(left_flap_te_right_parametric)

    left_flap_qc_center_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -6.56, -5.5])*ft2m, plot=False)
    left_flap_qc = geometry.evaluate(left_flap_qc_center_parametric)
    left_flap_qc_tip_right_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -0.6, -5.5])*ft2m, plot=False)
    left_flap_qc_tip_left_parametric = flapL.project(np.array([-13.85+(0.25*(-14.25+13.85)), -11.48, -5.5])*ft2m, plot=False)
                                                    
    left_flap_parametric_geometry = [
        left_flap_le_left_parametric,
        left_flap_le_right_parametric,
        left_flap_le_center_parametric,
        left_flap_te_left_parametric,
        left_flap_te_right_parametric,
        left_flap_te_center_parametric,
        left_flap_qc_center_parametric,
        left_flap_qc_tip_right_parametric,
        left_flap_qc_tip_left_parametric
    ]

    left_flap_span = np.linalg.norm(left_flap_le_left.value - left_flap_le_right.value)
    left_flap_chord = np.linalg.norm(left_flap_le_center[0].value - left_flap_te_center[0].value)


    right_flap_le_left_guess = np.array([-13.85, 0.6, -7.5])*ft2m
    right_flap_le_left_parametric = flapR.project(right_flap_le_left_guess, plot=False)
    right_flap_le_left = geometry.evaluate(right_flap_le_left_parametric)

    right_flap_le_right_guess = np.array([-13.85, 11.5, -7.5])*ft2m
    right_flap_le_right_parametric = flapR.project(right_flap_le_right_guess, plot=False)
    right_flap_le_right = geometry.evaluate(right_flap_le_right_parametric)

    right_flap_le_center_guess = np.array([-13.85, 6.05, -7.5])*ft2m
    right_flap_le_center_parametric = flapR.project(right_flap_le_center_guess, plot=False)
    right_flap_le_center = geometry.evaluate(right_flap_le_center_parametric)

    right_flap_le_center_on_wing_te_guess = right_flap_le_center_guess
    right_flap_le_center_on_wing_te_parametric = wing.project(right_flap_le_center_on_wing_te_guess, plot=False)

    right_flap_te_center_parametric = flapR.project(np.array([-14.25, 6.05, -7.5])*ft2m, plot=False)
    right_flap_te_center = geometry.evaluate(right_flap_te_center_parametric)

    right_flap_te_left_parametric = flapR.project(np.array([-14.25, 0.6, -7.5])*ft2m, plot=False)
    right_flap_te_left = geometry.evaluate(right_flap_te_left_parametric)

    right_flap_te_right_parametric = flapR.project(np.array([-14.25, 11.5, -7.5])*ft2m, plot=False)
    right_flap_te_right = geometry.evaluate(right_flap_te_right_parametric)

    right_flap_qc_center_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 6.56, -7.5])*ft2m, plot=False)
    right_flap_qc = geometry.evaluate(right_flap_qc_center_parametric)
    right_flap_qc_tip_right_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 11.48, -7.5])*ft2m, plot=False)
    right_flap_qc_tip_left_parametric = flapR.project(np.array([-13.85+(0.25*(-14.25+13.85)), 1.5, -7.5])*ft2m, plot=False)

    right_flap_parametric_geometry = [
        right_flap_le_left_parametric,
        right_flap_le_right_parametric,
        right_flap_le_center_parametric,
        right_flap_te_left_parametric,
        right_flap_te_right_parametric,
        right_flap_te_center_parametric,
        right_flap_qc_center_parametric,
        right_flap_qc_tip_right_parametric,
        right_flap_qc_tip_left_parametric
    ]

    right_flap_span = np.linalg.norm(right_flap_le_left.value - right_flap_le_right.value)
    right_flap_chord = np.linalg.norm(right_flap_le_center[0].value - right_flap_te_center[0].value)

    # HT Region Info
    ht_le_left_parametric = h_tail.project(np.array([-26.5, -5.25, -5.5])*ft2m, plot=False)
    ht_le_left = geometry.evaluate(ht_le_left_parametric)

    ht_le_center_parametric = h_tail.project(np.array([-27, 0., -5.5])*ft2m, plot=False)
    ht_le_center = geometry.evaluate(ht_le_center_parametric)

    ht_le_right_parametric = h_tail.project(np.array([-26.5, 5.25, -5.5])*ft2m, plot=False)
    ht_le_right = geometry.evaluate(ht_le_right_parametric)

    ht_te_left_parametric = h_tail.project(np.array([-30, -5.25, -5.5])*ft2m, plot=False)
    ht_te_left = geometry.evaluate(ht_te_left_parametric)

    ht_te_center_guess = np.array([-30, 0., -5.5])*ft2m
    ht_te_center_parametric = h_tail.project(ht_te_center_guess, plot=False)
    ht_te_center = geometry.evaluate(h_tail.project(ht_te_center_guess, plot=False))

    ht_te_right_parametric = h_tail.project(np.array([-30, 5.25, -5.5])*ft2m, plot=False)
    ht_te_right = geometry.evaluate(ht_te_right_parametric)

    ht_qc_center_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False)
    ht_qc = geometry.evaluate(ht_qc_center_parametric)
    ht_qc_tip_right_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), 5.25, -5.5])*ft2m, plot=False)
    ht_qc_tip_left_parametric = h_tail.project(np.array([-27 + (0.25*(-30+27)), -5.25, -5.5])*ft2m, plot=False)
                                                
    ht_qc = geometry.evaluate(h_tail.project(np.array([-27 + (0.25*(-30+27)), 0., -5.5])*ft2m, plot=False))

    ht_parametric_geometry = [
        ht_le_left_parametric,
        ht_le_right_parametric,
        ht_le_center_parametric,
        ht_te_left_parametric,
        ht_te_right_parametric,
        ht_te_center_parametric,
        ht_qc_center_parametric,
        ht_qc_tip_right_parametric,
        ht_qc_tip_left_parametric
    ]
    ht_span = np.linalg.norm(ht_le_left.value - ht_le_right.value)
    ht_chord = np.linalg.norm(ht_le_center[0].value - ht_te_center[0].value)

    trimTab_le_left_parametric = trimTab.project(np.array([-29.4, -3.3, -5.5])*ft2m, plot=False)
    trimTab_le_right_parametric = trimTab.project(np.array([-29.4, 3.3, -5.5])*ft2m, plot=False)
    trimTab_le_left = geometry.evaluate(trimTab_le_left_parametric)
    trimTab_le_center_parametric = trimTab.project(np.array([-29.4, 0, -5.5])*ft2m, plot=False)
    trimTab_le_center = geometry.evaluate(trimTab_le_center_parametric)
    trimTab_le_right = geometry.evaluate(trimTab_le_right_parametric)

    trimTab_te_left_parametric = trimTab.project(np.array([-30, -3.3, -5.5])*ft2m, plot=False)
    trimTab_te_center_parametric = trimTab.project(np.array([-30, 0, -5.5])*ft2m, plot=False)
    trimTab_te_right_parametric = trimTab.project(np.array([-30, 3.3, -5.5])*ft2m, plot=False)

    trimTab_te_center = geometry.evaluate(trimTab.project(np.array([-30, 0, -5.5])*ft2m, plot=False))

    trimTab_qc_center_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), 0, -5.5])*ft2m, plot=False)
    trimTab_qc_tip_right_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), 3.3, -5.5])*ft2m, plot=False)
    trimTab_qc_tip_left_parametric = trimTab.project(np.array([-29.4+(0.25*(-30+29.4)), -3.3, -5.5])*ft2m, plot=False)                                                  


    trimTab_parametric_geometry = [
        trimTab_le_left_parametric,
        trimTab_le_right_parametric,
        trimTab_le_center_parametric,
        trimTab_te_left_parametric,
        trimTab_te_right_parametric,
        trimTab_te_center_parametric,
        trimTab_qc_center_parametric,
        trimTab_qc_tip_right_parametric,
        trimTab_qc_tip_left_parametric
    ]


    trim_tab_span = np.linalg.norm(trimTab_le_left.value - trimTab_le_right.value)

    trimTab_chord = np.linalg.norm(trimTab_le_center[0].value - trimTab_te_center[0].value)

    # VT Region Info
    vt_le_base_parametric = vertTail.project(np.array([-23, 0, -5.5])*ft2m, plot=False)
    vt_le_base = geometry.evaluate(vt_le_base_parametric)
    vt_le_mid_parametric = vertTail.project(np.array([-26, 0., -8])*ft2m, plot=False)
    vt_le_mid = geometry.evaluate(vt_le_mid_parametric)
    vt_le_tip_parametric = vertTail.project(np.array([-28.7, 0, -11])*ft2m, plot=False)
    vt_le_tip = geometry.evaluate(vt_le_tip_parametric)

    vt_te_base_parametric = vertTail.project(np.array([-27.75, 0, -5.5])*ft2m, plot=False)
    vt_te_base = geometry.evaluate(vt_te_base_parametric)
    vt_te_mid_parametric = vertTail.project(np.array([-28.7, 0., -8])*ft2m, plot=False)
    vt_te_mid= geometry.evaluate(vt_te_mid_parametric)
    vt_te_tip_parametric = vertTail.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False)
    vt_te_tip = geometry.evaluate(vt_te_tip_parametric)
    vt_qc = geometry.evaluate(vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m, plot=False))
    vt_qc_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)
    vt_qc_base_parametric = vertTail.project(np.array([-23 + (0.25*(-28.7+23)), 0., -5.5])*ft2m)
    vt_qc_tip_parametric = vertTail.project(np.array([-27.33 + (0.25*(-28.7+23)), 0., -10.85])*ft2m,plot=False)


    vt_span = np.linalg.norm(vt_te_base.value - vt_te_tip.value)

    vt_chord = np.linalg.norm(vt_le_mid.value - vt_te_mid.value)

    rudder_le_base_parametric = rudder.project(np.array([-23, 0, -5.5])*ft2m, plot=False)
    rudder_le_base = geometry.evaluate(rudder_le_base_parametric)
    rudder_le_mid_parametric = rudder.project(np.array([-28.7, 0., -8.5])*ft2m, plot=False)
    rudder_le_mid = geometry.evaluate(rudder_le_mid_parametric)
    rudder_le_tip = geometry.evaluate(rudder.project(np.array([-29.75, 0, -10.6])*ft2m, plot=False))

    rudder_te_base_parametric = rudder.project(np.array([-29.5, 0, -5.5])*ft2m, plot=False)
    rudder_te_mid_parametric = rudder.project(np.array([-30., 0., -8.5])*ft2m, plot=False)
    rudder_te_tip_parametric = rudder.project(np.array([-30.4, 0, -11.])*ft2m, plot=False)


    rudder_span = np.linalg.norm(rudder_le_base.value - rudder_le_tip.value)
    rudder_chord = np.linalg.norm(rudder_le_mid.value - rudder_le_tip.value)

    vt_parametric_geometry = [
        vt_le_base_parametric,
        vt_le_tip_parametric,
        vt_le_base_parametric,
        rudder_te_base_parametric,
        rudder_te_tip_parametric,
        rudder_te_mid_parametric,
        vt_qc_parametric,
        vt_qc_base_parametric,
        vt_qc_tip_parametric
    ]

    # Fuselage Region Info
    fuselage_wing_le_center_parametric = fuselage.project(np.array([-12.356, 0., -7.5])*ft2m, plot=False)
    fuselage_wing_qc_center_parametric = fuselage.project(np.array([-12.356+(0.25*(-14.25+12.356)), 0., -7.5])*ft2m, plot=False)
    fuselage_wing_qc = geometry.evaluate(fuselage_wing_qc_center_parametric)
    fuselage_wing_te_center_parametric = fuselage.project(np.array([-14.25, 0., -7.5])*ft2m, plot=False)
    fuselage_wing_te_center = geometry.evaluate(fuselage_wing_te_center_parametric)
    fuselage_tail_qc = geometry.evaluate(fuselage.project(np.array([-27 + (0.25*(-30+27)), 0., -7.5])*ft2m, plot=False))
    fuselage_tail_te_center_parametric = fuselage.project(np.array([-30, 0., -7.5])*ft2m, plot=False)
    fuselage_tail_te_center = geometry.evaluate(fuselage_tail_te_center_parametric)

    # Propeller Region Info
    M12_disk_pt =  np.array([-12.5, 14, -7.355])*ft2m
    M1_disk_pt = np.array([-12.5, -14, -7.355])*ft2m
    M11_disk_pt =  np.array([-12.35, 12, -7.355])*ft2m
    M2_disk_pt = np.array([-12.35, -12, -7.355])*ft2m
    M10_disk_pt = np.array([-12.2, 10, -7.659])*ft2m
    M3_disk_pt = np.array([-12.2, -10, -7.659])*ft2m
    M9_disk_pt = np.array([-12, 8, -7.659])*ft2m
    M4_disk_pt = np.array([-12, -8, -7.659])*ft2m
    M8_disk_pt = np.array([-11.8, 6, -7.659])*ft2m
    M5_disk_pt = np.array([-11.8, -6, -7.659])*ft2m
    M7_disk_pt = np.array([-11.6, 4, -7.659])*ft2m
    M6_disk_pt = np.array([-11.6, -4, -7.659])*ft2m

    M1_disk_on_wing = spinner1.project(M1_disk_pt, plot=False)
    M1_disk = geometry.evaluate(M1_disk_on_wing)

    M2_disk_on_wing = spinner2.project(M2_disk_pt, plot=False)
    M2_disk = geometry.evaluate(M2_disk_on_wing)

    M3_disk_on_wing = spinner3.project(M3_disk_pt, plot=False)
    M3_disk = geometry.evaluate(M3_disk_on_wing)

    M4_disk_on_wing = spinner4.project(M4_disk_pt, plot=False)
    M4_disk = geometry.evaluate(M4_disk_on_wing)

    M5_disk_on_wing = spinner5.project(M5_disk_pt, plot=False)
    M5_disk = geometry.evaluate(M5_disk_on_wing)

    M6_disk_on_wing = spinner6.project(M6_disk_pt, plot=False)
    M6_disk = geometry.evaluate(M6_disk_on_wing)

    M7_disk_on_wing = spinner7.project(M7_disk_pt, plot=False)
    M7_disk = geometry.evaluate(M7_disk_on_wing)

    M8_disk_on_wing = spinner8.project(M8_disk_pt, plot=False)
    M8_disk = geometry.evaluate(M8_disk_on_wing)

    M9_disk_on_wing = spinner9.project(M9_disk_pt, plot=False)
    M9_disk = geometry.evaluate(M9_disk_on_wing)

    M10_disk_on_wing = spinner10.project(M10_disk_pt, plot=False)
    M10_disk = geometry.evaluate(M10_disk_on_wing)

    M11_disk_on_wing = spinner11.project(M11_disk_pt, plot=False)
    M11_disk = geometry.evaluate(M11_disk_on_wing)

    M12_disk_on_wing = spinner12.project(M12_disk_pt, plot=False)
    M12_disk = geometry.evaluate(M12_disk_on_wing)

    MotorDisks = [M1_disk, M2_disk, M3_disk, M4_disk, M5_disk, M6_disk, M7_disk, M8_disk, M9_disk, M10_disk, M11_disk, M12_disk]


    M1_pylon_pt = np.array([-12.5, -14, -7.355])*ft2m 
    M2_pylon_pt = np.array([-12.35, -12, -7.355])*ft2m
    M3_pylon_pt = np.array([-12.2, -10, -7.659])*ft2m
    M4_pylon_pt = np.array([-12, -8, -7.659])*ft2m
    M5_pylon_pt = np.array([-11.8, -6, -7.659])*ft2m
    M6_pylon_pt = np.array([-11.6, -4, -7.659])*ft2m
    M7_pylon_pt = np.array([-11.6, 4, -7.659])*ft2m
    M8_pylon_pt = np.array([-11.8, 6, -7.659])*ft2m
    M9_pylon_pt = np.array([-12, 8, -7.659])*ft2m
    M10_pylon_pt = np.array([-12.2, 10, -7.659])*ft2m
    M11_pylon_pt = np.array([-12.35, 12, -7.355])*ft2m
    M12_pylon_pt = np.array([-12.5, 14, -7.355])*ft2m

    M1_pylon_on_wing = pylon1.project(M1_pylon_pt, plot=False)
    M2_pylon_on_wing = pylon2.project(M2_pylon_pt, plot=False)
    M3_pylon_on_wing = pylon3.project(M3_pylon_pt, plot=False)
    M4_pylon_on_wing = pylon4.project(M4_pylon_pt, plot=False)
    M5_pylon_on_wing = pylon5.project(M5_pylon_pt, plot=False)
    M6_pylon_on_wing = pylon6.project(M6_pylon_pt, plot=False)
    M7_pylon_on_wing = pylon7.project(M7_pylon_pt, plot=False)
    M8_pylon_on_wing = pylon8.project(M8_pylon_pt, plot=False)
    M9_pylon_on_wing = pylon9.project(M9_pylon_pt, plot=False)
    M10_pylon_on_wing = pylon10.project(M10_pylon_pt, plot=False)
    M11_pylon_on_wing = pylon11.project(M11_pylon_pt, plot=False)
    M12_pylon_on_wing = pylon12.project(M12_pylon_pt, plot=False)




    fuselage_nose_guess = np.array([-1.75, 0, -4])*ft2m
    fuselage_rear_guess = np.array([-29.5, 0, -5.5])*ft2m
    fuselage_nose_pts_parametric = fuselage.project(fuselage_nose_guess, grid_search_density_parameter=20, plot=False)
    fuselage_nose = geometry.evaluate(fuselage_nose_pts_parametric)
    fuselage_rear_pts_parametric = fuselage.project(fuselage_rear_guess, plot=False)
    fuselage_rear = geometry.evaluate(fuselage_rear_pts_parametric)

    # For Cruise Motor 1 Hub Region
    cruise_motor1_tip_guess = np.array([-13, -15.83, -5.5])*ft2m
    cruise_motor1_tip_parametric = cruise_spinner1.project(cruise_motor1_tip_guess, plot=False)
    cruise_motor1_tip = geometry.evaluate(cruise_motor1_tip_parametric)

    cruise_motor1_base_guess = cruise_motor1_tip + np.array([-1.67, 0, 0])*ft2m
    cruise_motor1_base_parametric = cruise_spinner1.project(cruise_motor1_base_guess, plot=False)
    cruise_motor1_base= geometry.evaluate(cruise_motor1_base_parametric)

    # For Cruise Motor 2 Hub Region
    cruise_motor2_tip_guess = np.array([-13, 15.83, -5.5])*ft2m
    cruise_motor2_tip_parametric = cruise_spinner2.project(cruise_motor2_tip_guess, plot=False)
    cruise_motor2_tip = geometry.evaluate(cruise_motor1_tip_parametric)

    cruise_motor2_base_guess = cruise_motor2_tip + np.array([-1.67, 0, 0])*ft2m
    cruise_motor2_base_parametric = cruise_spinner2.project(cruise_motor2_base_guess, plot=False)
    cruise_motor2_base= geometry.evaluate(cruise_motor2_base_parametric)

    wing_on_cruise_motor1_parametric = cruise_spinner2.project(wing_le_left_guess, plot=False)
    wing_on_cruise_motor2_parametric = cruise_spinner1.project(wing_le_right_guess, plot=False)

    cruise_motors_base = [cruise_motor1_base, cruise_motor2_base]
    return locals()




