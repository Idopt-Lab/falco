import csdl_alpha as csdl
import ozone as oz
import numpy as np
from falco import Loads, Vector, ForcesMoments
from falco import Q_, REPO_ROOT_FOLDER
import scipy.io as sio
from pathlib import Path
from scipy.interpolate import Akima1DInterpolator, RectBivariateSpline, RegularGridInterpolator, RBFInterpolator, LinearNDInterpolator

# Load Data files
folder_path = Path(__file__).parent.parent
dragpolar_data = sio.loadmat(folder_path / 'DragPolar.mat')
aerocoeffs_data = sio.loadmat(folder_path / 'AeroDerivatives.mat')

ft2m = 0.3048  # Conversion factor from feet to meters
# Wing / Aero Data
class B777Aerodynamics(Loads):
    """
    Aerodynamics computes aerodynamic forces and moments using flight state parameters, control inputs, and aerodynamic coefficients. It encapsulates the conversion of aerodynamic inputs into local axis forces and moments acting on the aircraft.
    """

    def __init__(self, CDML_interp, CDHM_interp, aero_coefficients, S, b, c, FCDSUB: float=1., FCDI: float=1., FCDO: float=1.):
        self.S = S
        self.b = b
        self.c = c
        # Drag polar
        self.CDML_interp = CDML_interp
        self.CDHM_interp = CDHM_interp
        self.FCDSUB = FCDSUB
        self.FCDI = FCDI
        self.FCDO = FCDO
        # Aerodynamic coefficients
        self.aero_coeffs = aero_coefficients

    def get_FM_localAxis(self, states, controls, axis):

        density = states.atmospheric_states.density
        velocity = states.VTAS
        mach=states.Mach
        AOA = states.alpha
        beta = states.beta

        p = states.states.p
        q = states.states.q
        r = states.states.r
        theta = states.states.theta
        h = -states.states.z

        da = controls.roll_control[0].deflection
        dr = controls.yaw_control[0].deflection
        de = controls.pitch_control[0].deflection

        phat = p * self.b / (2 * velocity)
        qhat = q * self.c / (2 * velocity)
        rhat = r * self.b / (2 * velocity)

        kPG = csdl.sqrt(1 - mach**2)  # Prandtl-Glauert(beta)

        cdml_curve = type(self.CDML_interp)()
        cdhm_curve = type(self.CDHM_interp)()
        
        CL = (self.aero_coeffs['CL0'][0][0] +
              self.aero_coeffs['CLa'][0][0] * AOA +
              self.aero_coeffs['CLb'][0][0] * beta +
              self.aero_coeffs['CLp'][0][0] * phat +
              self.aero_coeffs['CLq'][0][0] * qhat +
              self.aero_coeffs['CLr'][0][0] * rhat +
              self.aero_coeffs['CLda'][0][0] * da +
              self.aero_coeffs['CLde'][0][0] * de +
              self.aero_coeffs['CLdr'][0][0] * dr)/kPG

        CD = self.FCDSUB * (self.FCDI * cdml_curve.evaluate(mach=mach, CL=CL).cdml + self.FCDO * cdhm_curve.evaluate(h, mach).cdhm) + \
             self.aero_coeffs['CDda'][0][0] * da + \
             self.aero_coeffs['CDde'][0][0] * de + \
             self.aero_coeffs['CDdr'][0][0] * dr
        

        CY = (self.aero_coeffs['CYa'][0][0] * AOA +
              self.aero_coeffs['CYb'][0][0] * beta +
              self.aero_coeffs['CYp'][0][0] * phat +
              self.aero_coeffs['CYq'][0][0] * qhat +
              self.aero_coeffs['CYr'][0][0] * rhat +
              self.aero_coeffs['CYda'][0][0] * da +
              self.aero_coeffs['CYde'][0][0] * de +
              self.aero_coeffs['CYdr'][0][0] * dr)/kPG

        Cm = (self.aero_coeffs['Cm0'][0][0] +
              self.aero_coeffs['Cma'][0][0] * AOA +
              self.aero_coeffs['Cmb'][0][0] * beta +
              self.aero_coeffs['Cmp'][0][0] * phat +
              self.aero_coeffs['Cmq'][0][0] * qhat +
              self.aero_coeffs['Cmr'][0][0] * rhat +
              self.aero_coeffs['Cmda'][0][0] * da +
              self.aero_coeffs['Cmde'][0][0] * de +
              self.aero_coeffs['Cmdr'][0][0] * dr)/kPG

        Cn = (self.aero_coeffs['Cna'][0][0] * AOA +
              self.aero_coeffs['Cnb'][0][0] * beta +
              self.aero_coeffs['Cnp'][0][0] * phat +
              self.aero_coeffs['Cnq'][0][0] * qhat +
              self.aero_coeffs['Cnr'][0][0] * rhat +
              self.aero_coeffs['Cnda'][0][0] * da +
              self.aero_coeffs['Cnde'][0][0] * de +
              self.aero_coeffs['Cndr'][0][0] * dr)/kPG

        Cl = (self.aero_coeffs['Cla'][0][0] * AOA +
              self.aero_coeffs['Clb'][0][0] * beta +
              self.aero_coeffs['Clp'][0][0] * phat +
              self.aero_coeffs['Clq'][0][0] * qhat +
              self.aero_coeffs['Clr'][0][0] * rhat +
              self.aero_coeffs['Clda'][0][0] * da +
              self.aero_coeffs['Clde'][0][0] * de +
              self.aero_coeffs['Cldr'][0][0] * dr)/kPG
        
        qBar = 1/2 * density * velocity**2
        L = qBar * self.S * CL
        D = qBar * self.S * CD
        Y = qBar * self.S * CY
        l = qBar * self.S * self.b * Cl
        m = qBar * self.S * self.c * Cm
        n = qBar * self.S * self.b * Cn

        force_vector = Vector(vector=csdl.concatenate((-D, -Y, -L),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.concatenate((l,m,n),
                                                      axis=0), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return {'loads': loads, 'CL': CL, 'CD': CD, 'CY': CY, 'Cm': Cm, 'Cn': Cn, 'Cl': Cl, 'L': L, 'D': D, 'Y': Y, 'l': l, 'm': m, 'n': n,}

b = 199.14 * ft2m # Wing span in m
S = 4927.3 * ft2m**2  # Wing area in m^2
c= 4927.3/199.14*ft2m

class CDML(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()
        # Taken from JPROP
        dragpolar_CL_data = dragpolar_data['CL']
        dragpolar_M_data = dragpolar_data['M']
        dragpolar_CDML_data = dragpolar_data['CDML']
        self.cdml = RectBivariateSpline(dragpolar_M_data,dragpolar_CL_data,dragpolar_CDML_data, kx=3, ky=3)

    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, mach: csdl.Variable,CL: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('CL', CL)
        self.declare_input('mach', mach)

        # declare output variables
        cdml = self.create_output('cdml', shape=(1, ))

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.cdml = cdml

        return outputs

    def compute(self, input_vals, output_vals):
        CL = input_vals['CL']
        mach = input_vals['mach']
        output_vals['cdml'] = self.cdml.ev(mach, CL)


    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        CL = input_vals['CL']
        mach = input_vals['mach']
        derivatives['cdml', 'mach'] = np.diag(self.cdml.ev(mach, CL, dx=1))
        derivatives['cdml', 'CL'] = np.diag(self.cdml.ev(mach, CL, dy=1))

class CDHM(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        h_m_data = dragpolar_data['h_m']
        M_data = dragpolar_data['M']
        cdhm_data = dragpolar_data['CDHM']
        self.cdhm = RectBivariateSpline(h_m_data, M_data, cdhm_data, kx=3, ky=3)

    def evaluate(self, M: csdl.Variable, h_m: csdl.Variable):
        self.declare_input('M', M)
        self.declare_input('h_m', h_m)
        cdhm = self.create_output('cdhm', shape=(1,))
        outputs = csdl.VariableGroup()
        outputs.cdhm = cdhm
        return outputs

    def compute(self, input_vals, output_vals):
        M = input_vals['M']
        h_m = input_vals['h_m']
        output_vals['cdhm'] = self.cdhm.ev(h_m, M)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        M = input_vals['M']
        h_m = input_vals['h_m']
        derivatives['cdhm', 'h_m'] = np.diag(self.cdhm.ev(h_m, M, dx=1))
        derivatives['cdhm', 'M'] = np.diag(self.cdhm.ev(h_m, M, dy=1))

CDML_interp = CDML()
CDHM_interp = CDHM()

# def oz_aero(ozone_vars:oz.ODEVars, options):