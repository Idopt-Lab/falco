import numpy as np
import csdl_alpha as csdl
import ozone as oz
import sys


B777_folder_path = REPO_ROOT_FOLDER / 'examples' / 'basic_examples' / 'b777'
sys.path.append(str(B777_folder_path))

from b777 import B777Aerodynamics, B777SMTPropulsion, B777SMTEngineCurve

def ozone_ode_function(ozone_vars:oz.ODEVars, options):
    # Pre-processing
    u = ozone_vars.states['u']
    v = ozone_vars.states['v']
    w = ozone_vars.states['w']
    p = ozone_vars.states['p']
    q = ozone_vars.states['q']
    r = ozone_vars.states['r']
    phi = ozone_vars.states['phi']
    theta = ozone_vars.states['theta']
    psi = ozone_vars.states['psi']
    x = ozone_vars.states['x']
    y = ozone_vars.states['y']
    z = ozone_vars.states['z']
    

    # Dynamic parameters are inputs
    control_elevator = ozone_vars.dynamic_parameters['control_elevator']
    control_aileron_left = ozone_vars.dynamic_parameters['control_aileron_left']
    control_aileron_right = ozone_vars.dynamic_parameters['control_aileron_right']
    control_rudder = ozone_vars.dynamic_parameters['control_rudder']
    control_throttle_left = ozone_vars.dynamic_parameters['control_throttle_left']
    control_throttle_right = ozone_vars.dynamic_parameters['control_throttle_right']
    alpha = ozone_vars.dynamic_parameters['control_alpha']
    beta = ozone_vars.dynamic_parameters['control_beta']


