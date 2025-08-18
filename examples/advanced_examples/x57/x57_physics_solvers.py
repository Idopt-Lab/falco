import csdl_alpha as csdl
import numpy as np
from typing import Union
from scipy.interpolate import Akima1DInterpolator
from falco.core.loads.loads import Loads
import os
import scipy.io as sio
from falco.core.loads.forces_moments import Vector, ForcesMoments
from falco import REPO_ROOT_FOLDER, Q_, ureg
import sys
import os
import re
import subprocess

x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))
from x57_geometry import get_geometry, get_geometry_related_axis, get_airfoil_mesh, export_airfoil_coords
from x57_component import build_aircraft_component
from x57_mp import add_mp_to_components
from x57_control_system import X57ControlSystem, Blower
from x57_solvers import X57Aerodynamics, X57Propulsion, HLPropCurve, CruisePropCurve



class X57Aerodynamics(Loads):

    # TODO: Improve aerodynamic model to include more complex aerodynamic effects
    def __init__(self, component):

        self.component = component
        self.AR_wing = component.comps['Wing'].parameters.AR.value
        self.i_wing = component.comps['Wing'].parameters.actuate_angle
        self.Sref_wing = component.comps['Wing'].parameters.S_ref
        self.span_wing = component.comps['Wing'].parameters.span
        self.bref_wing = self.span_wing
        self.taper_wing = component.comps['Wing'].parameters.taper_ratio
        self.cref_wing = 2 * self.Sref_wing/((1 + self.taper_wing) * self.span_wing)

        self.Sref_stab = component.comps['Elevator'].parameters.S_ref
        self.span_stab = component.comps['Elevator'].parameters.span
        self.bref_stab = self.span_stab
        self.taper_stab = component.comps['Elevator'].parameters.taper_ratio
        self.cref_stab = 2 * self.Sref_stab/((1 + self.taper_stab) * self.span_stab)


        self.Sref_VT = component.comps['Vertical Tail'].parameters.S_ref
        self.span_VT = component.comps['Vertical Tail'].parameters.span
        self.bref_VT = self.span_VT
        self.taper_VT = component.comps['Vertical Tail'].parameters.taper_ratio
        self.cref_VT = 2 * self.Sref_VT/((1 + self.taper_VT) * self.span_VT)

        self.HT_axis = component.comps['Elevator'].mass_properties.cg_vector.vector 
        self.VT_axis = component.comps['Vertical Tail'].mass_properties.cg_vector.vector 
        self.Wing_axis = component.comps['Wing'].mass_properties.cg_vector.vector 



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
                downwash = (downwash_C11_noblow + downwash_fusVtail) * (np.pi/180)
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
        alpha = alpha * (np.pi / 180)  # convert to radians
        i_w = i_w * (np.pi / 180)  # convert to radians
        i_stab = i_stab * (np.pi / 180)  # convert to radians
        trimtab = trimtab * (np.pi / 180)  # convert to radians

        stabi_alpha = self.stab_alpha(alpha, i_w, i_stab, AR, flap, blow_num)
        CL = flap * self.flap_CL_tot(alpha) + blow_num / 12 * self.blow_CL_tot(alpha) + \
             self.Wing_tipNacelle_CL_tot(alpha) + self.HLN_CL_tot(alpha) + \
             self.Fus_Vtail_CL_tot(alpha) + \
             self.Stab_CL_tot(stabi_alpha, trimtab) * self.Sref_stab / self.Sref_wing
        return CL

    def AC_CD(self, alpha, i_w, i_stab, AR, flap, blow_num, trimtab):
        # assuming all other parameters as functions of cref_wing
        alpha = alpha * (np.pi / 180)  # convert to radians
        i_w = i_w * (np.pi / 180)  # convert to radians
        i_stab = i_stab * (np.pi / 180)  # convert to radians
        trimtab = trimtab * (np.pi / 180)  # convert to radians
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

        alpha = alpha * (np.pi / 180)  # convert to radians
        i_w = i_w * (np.pi / 180)  # convert to radians
        i_stab = i_stab * (np.pi / 180)  # convert to radians
        trimtab = trimtab * (np.pi / 180)  # convert to radians

        r = csdl.Variable(name='r', shape=(3,), value=0)
        r1 = self.HT_axis[0] + self.cref_stab / 4 - self.Wing_axis[0] + self.cref_wing / 4
        r2 = 0
        r3 = self.HT_axis[2] - self.Wing_axis[2]
        r = csdl.concatenate([r1, r2, r3], axis=0)

        stabi_alpha = self.stab_alpha(alpha, i_w, i_stab, AR, flap, blow_num)
        
        # Build f_hat using csdl operations so that csdl.cross works as expected
        f1 = -csdl.sin(stabi_alpha)
        f2 = 0
        f3 = csdl.cos(stabi_alpha)
        f_hat = csdl.concatenate([f1, f2, f3], axis=0)

        
        CL_stabi = self.Stab_CL_tot(stabi_alpha, trimtab)
        CM_stabi_wingcby4 = csdl.cross(r, f_hat) * (CL_stabi * self.Sref_stab) / (self.Sref_wing * self.cref_wing)
        
        Cm = (flap * self.flap_Cm_tot(alpha) + \
            blow_num / 12 * self.blow_Cm_tot(alpha) + \
            self.Wing_tipNacelle_Cm_tot(alpha) + \
            self.HLN_Cm_tot(alpha) + \
            self.Fus_Vtail_Cm_tot(alpha) + \
            self.Stab_Cm_tot(stabi_alpha, trimtab) * self.Sref_stab * self.cref_stab / (self.Sref_wing * self.cref_wing) + \
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
            u = controls.u()
            density = states.atmospheric_states.density
            velocity = states.VTAS
            theta = states.states.theta
            p = states.states.p
            q = states.states.q
            r = states.states.r
            beta = states.beta
            i_wing = self.i_wing * 180/np.pi
            AOA = states.alpha * 180/np.pi
            dstab = controls.pitch_control['Elevator'].deflection
            dflap = controls.high_lift_control['Left Flap'].flag or controls.high_lift_control['Right Flap'].flag 
            blow = controls.high_lift_control['Blower'].flag    
            daileron = controls.roll_control['Left Aileron'].deflection            
            dtrim = controls.pitch_control['Trim Tab'].deflection
            drudder = controls.yaw_control['Rudder'].deflection

            # blowing affect if HL engines are active
            blow_num = 0
            if blow:
                for engine in controls.hl_engines:
                    if engine.throttle.value != 0:
                        blow_num += 1


            CL = self.AC_CL(alpha=AOA, i_w=i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            CD = self.AC_CD(alpha=AOA, i_w=i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            CM = self.AC_CM(alpha=AOA, i_w=i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
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

            L_roll = 0.5 * density * velocity**2 * self.Sref_wing * self.bref_wing * Cl # net rolling moment

            Cn = self.aeroDer['Cnda'][0][0] * daileron + \
                self.aeroDer['Cndr'][0][0] * drudder + \
                self.aeroDer['Cnp'][0][0] * phat + \
                self.aeroDer['Cnr'][0][0] * rhat + \
                self.aeroDer['Cnbeta'][0][0] * beta

            N_yaw = 0.5 * density * velocity**2 * self.Sref_wing * self.bref_wing * Cn # net yawing moment

            CY = self.aeroDer['CYda'][0][0] * daileron + \
                self.aeroDer['CYdr'][0][0] * drudder + \
                self.aeroDer['CYp'][0][0] * phat + \
                self.aeroDer['CYr'][0][0] * rhat + \
                self.aeroDer['CYbeta'][0][0] * beta

            Y_sideforce = 0.5 * density * velocity**2 * self.Sref_wing * CY # net sideforce

            wind_axis = states.windAxis

            aero_force = csdl.Variable(shape=(3,), value=0.)
            aero_force = aero_force.set(csdl.slice[0], -D)
            aero_force = aero_force.set(csdl.slice[1], Y_sideforce)
            aero_force = aero_force.set(csdl.slice[2], -L)
            force_vector = Vector(vector=aero_force, axis=wind_axis)

            aero_moment = csdl.Variable(shape=(3,), value=0.)
            aero_moment = aero_moment.set(csdl.slice[0], L_roll)
            aero_moment = aero_moment.set(csdl.slice[1], M)
            aero_moment = aero_moment.set(csdl.slice[2], N_yaw)
            moment_vector = Vector(vector=aero_moment, axis=wind_axis)
            loads_wind_axis = ForcesMoments(force=force_vector, moment=moment_vector)
            return {
            'loads': loads_wind_axis,
            'CL':  CL,
            'CD':   CD,
            'CM':     CM
        }
        


    def xfoil(self, coord='NACA0012', alpha=0, Re=1e6, Mach=0.2, *extra_cmds):
        """
        Run XFoil and return the results.
        Parameters:
            coord: NACA string, filename, or numpy array of coordinates
            alpha: float or array-like, angle(s) of attack
            Re: Reynolds number
            Mach: Mach number
            extra_cmds: extra XFoil commands as strings
        Returns:
            pol: dict with polar coefficients
            foil: dict with surface data (if requested)
        """
        # Setup
        wd = os.path.dirname(os.path.abspath(__file__))
        fname = 'xfoil'
        file_coord = os.path.join(wd, fname + '.foil')
        Nalpha = np.size(alpha)
        alpha = np.atleast_1d(alpha)
        only_polar = False

        # Save coordinates
        if isinstance(coord, str):
            naca_match = re.match(r'^NACA *[0-9]{4,5}$', coord, re.IGNORECASE)
            if naca_match:
                foil_name = coord
            else:
                file_coord = coord
        else:
            # Write foil ordinate file
            if os.path.exists(file_coord):
                os.remove(file_coord)
            with open(file_coord, 'w') as f:
                f.write(fname + '\n')
                for x, y in np.array(coord):
                    f.write(f'{x:9.5f}   {y:9.5f}\n')

        # Write xfoil command file
        inp_path = os.path.join(wd, fname + '.inp')
        with open(inp_path, 'w') as f:
            if isinstance(coord, str):
                naca_match = re.match(r'^NACA *[0-9]{4,5}$', coord, re.IGNORECASE)
                if naca_match:
                    f.write(f'naca {coord[4:].strip()}\n')
                else:
                    f.write(f'load {file_coord}\n')
            else:
                f.write(f'load {file_coord}\n')
            # Extra XFoil commands
            for txt in extra_cmds:
                txt = re.sub(r'[ \\\/]+', '\n', txt)
                f.write(f'{txt}\n\n\n')
            f.write('\n\noper\n')
            f.write(f're {Re}\n')
            f.write(f'mach {Mach}\n')
            if Re > 0:
                f.write('visc\n')
            f.write('pacc\n\n\n')
            file_dump = []
            file_cpwr = []
            for a in alpha:
                dump_name = f'{fname}_a{a:06.3f}_dump.dat'
                cpwr_name = f'{fname}_a{a:06.3f}_cpwr.dat'
                file_dump.append(dump_name)
                file_cpwr.append(cpwr_name)
                f.write(f'alfa {a}\n')
                f.write(f'dump {dump_name}\n')
                f.write(f'cpwr {cpwr_name}\n')
            file_pwrt = f'{fname}_pwrt.dat'
            f.write(f'pwrt\n{file_pwrt}\n')
            f.write('plis\n')
            f.write('\nquit\n')

        # Run XFoil
        cmd = f'cd "{wd}" && xfoil.exe < xfoil.inp > xfoil.out'
        status = subprocess.call(cmd, shell=True)
        if status != 0:
            raise RuntimeError(f'XFoil execution failed! {cmd}')

        # Parse polar file
        pol = {}
        pwrt_path = os.path.join(wd, file_pwrt)
        with open(pwrt_path, 'r') as f:
            lines = f.readlines()
            # Find "Calculated polar for:" line
            for i, line in enumerate(lines):
                if 'Calculated polar for:' in line:
                    pol['name'] = line.split(':')[1].strip()
                    break
            # Find header for data
            for j in range(i, len(lines)):
                if re.match(r'\s*alpha', lines[j]):
                    data_start = j + 1
                    break
            data = np.genfromtxt(lines[data_start:], invalid_raise=False)
            pol['alpha'] = data[:, 0]
            pol['CL'] = data[:, 1]
            pol['CD'] = data[:, 2]
            pol['CDp'] = data[:, 3]
            pol['Cm'] = data[:, 4]
            pol['Top_xtr'] = data[:, 5]
            pol['Bot_xtr'] = data[:, 6]
        os.remove(pwrt_path)

        # Parse foil data if requested
        foil = None
        if not only_polar:
            foil = {'alpha': np.zeros(Nalpha)}
            for ii in range(Nalpha):
                dump_path = os.path.join(wd, file_dump[ii])
                cpwr_path = os.path.join(wd, file_cpwr[ii])
                # Dump file
                with open(dump_path, 'r') as f:
                    dump_data = np.genfromtxt(f, skip_header=1)
                os.remove(dump_path)
                if ii == 0:
                    Npanel = dump_data.shape[0]
                    for key in ['s', 'x', 'y', 'UeVinf', 'Dstar', 'Theta', 'Cf', 'H']:
                        foil[key] = np.zeros((Npanel, Nalpha))
                foil['s'][:, ii] = dump_data[:, 0]
                foil['x'][:, ii] = dump_data[:, 1]
                foil['y'][:, ii] = dump_data[:, 2]
                foil['UeVinf'][:, ii] = dump_data[:, 3]
                foil['Dstar'][:, ii] = dump_data[:, 4]
                foil['Theta'][:, ii] = dump_data[:, 5]
                foil['Cf'][:, ii] = dump_data[:, 6]
                foil['H'][:, ii] = dump_data[:, 7]
                foil['alpha'][ii] = alpha[ii]
                # Cp file
                with open(cpwr_path, 'r') as f:
                    cp_lines = f.readlines()[3:]
                    cp_data = np.genfromtxt(cp_lines)
                os.remove(cpwr_path)
                if ii == 0:
                    NCp = cp_data.shape[0]
                    foil['xcp'] = cp_data[:, 0]
                    foil['cp'] = np.zeros((NCp, Nalpha))
                foil['cp'][:, ii] = cp_data[:, 2]
        # Clean up input/output files
        os.remove(inp_path)
        if isinstance(coord, np.ndarray) and os.path.exists(file_coord):
            os.remove(file_coord)
        return pol, foil  


    def write_section(self, f, AF, is_naca, Xle, Yle, Zle, chord, ainc, Nspan, Sspace, is_control=False, control_params=None):
        f.write('#-------------------------------------------------------------\n')
        f.write('SECTION\n')
        f.write('#Xle\tYle\tZle\tChord\tAinc\tNspanwise\tSspace\n')
        f.write(f'{Xle:.3f}    {Yle:.3f}    {Zle:.3f}    {chord:.3f}    {ainc:.3f}    {Nspan:.3f}    {Sspace:.3f}\n')
        if is_naca:
            f.write(f'NACA\n{AF}\n\n')
        else:
            f.write(f'AFILE\n{AF}\n\n')
        f.write('#Cname    Cgain    Xhinge    HingeVec    SgnDup\n')
        if is_control and control_params:
            # If control_params is a list, write each as a CONTROL block
            if isinstance(control_params, list):
                for cp in control_params:
                    f.write('CONTROL\n')
                    f.write(f"{cp['Cname']}    {cp['Cgain']:.2f}    {cp['Xhinge']:.3f}    "
                            f"{cp['HingeVec'][0]:.2f} {cp['HingeVec'][1]:.2f} {cp['HingeVec'][2]:.2f}    "
                            f"{cp['SgnDup']:.2f}\n\n")
            else:
                f.write('CONTROL\n')
                f.write(f"{control_params['Cname']}    {control_params['Cgain']:.2f}    {control_params['Xhinge']:.3f}    "
                        f"{control_params['HingeVec'][0]:.2f} {control_params['HingeVec'][1]:.2f} {control_params['HingeVec'][2]:.2f}    "
                        f"{control_params['SgnDup']:.2f}\n\n")
        f.write('CLAF\n1.0\n\n')



    def jvl_write(self, filename, states, controls, airfoil_files=None):

        wd = os.path.dirname(os.path.abspath(__file__))
        avl_path = os.path.join(wd, f"{filename}")
        if os.path.exists(avl_path):
            os.remove(avl_path)

        # Extract symmetry and reference params from airplane.weights
        IYsym = 0
        IZsym = 0
        Zsym = 0

        cg = self.component.mass_properties.cg_vector.vector
        Xref = cg.value[0]
        Yref = cg.value[1]
        Zref = cg.value[2]

        S_ref = self.Sref_wing
        b_ref = self.bref_wing
        c_ref = self.cref_wing
        wing_incidence = self.i_wing
        h_jet = 0.1

        # Breaking the wing into 3 sections: root, mid, and tip
        # The root section the root going out to the first spanwise point, where it is just the wing, without any control surfaces.
        # The mid section is the first spanwise point out from the root to the second spanwise point, where it has a flap control surface.
        # The tip section is the second spanwise point out from the root to the tip, where it has an aileron control surface.

        # Nchord = number of chordwise horseshoe vortices placed on the surface
        # Cspace = chordwise vortex spacing parameter
        # Nspan  = number of spanwise horseshoe vortices placed on the surface
        # Sspace = spanwise vortex spacing parameter


        # Vortex Lattice Spacing Distributions
        # ------------------------------------

        # Discretization of the geometry into vortex lattice panels
        # is controlled by the spacing parameters described earlier:
        # Sspace, Cspace, Bspace

        # These must fall in the range  -3.0 ... +3.0 , and they
        # determine the spanwise and lengthwise horseshoe vortex 
        # or body line node distributions as follows:

        # parameter                              spacing
        # ---------                              -------

        #     3.0        equal         |   |   |   |   |   |   |   |   |

        #     2.0        sine          || |  |   |    |    |     |     |

        #     1.0        cosine        ||  |    |      |      |    |  ||

        #     0.0        equal         |   |   |   |   |   |   |   |   |

        # -1.0        cosine        ||  |    |      |      |    |  ||

        # -2.0       -sine          |     |     |    |    |   |  | ||

        # -3.0        equal         |   |   |   |   |   |   |   |   |

        # Sspace (spanwise)  :    first section        ==>       last section
        # Cspace (chordwise) :    leading edge         ==>       trailing edge
        # Bspace (lengthwise):    frontmost point      ==>       rearmost point

        # Control Parameters
        # ------------------------------------

        # CONTROL                              ! (keyword)
        # elevator  1.0  0.6   0. 1. 0.   1.0  ! name, gain,  Xhinge,  XYZhvec,  SgnDup


        # The CONTROL keyword declares that a hinge deflection at this section
        # is to be governed by one or more control variables.  An arbitrary number 
        # of control variables can be used, limited only by the array limit NDMAX.

        # The data line quantities are...

        # name     name of control variable
        # gain     control deflection gain, units:  degrees deflection / control variable
        # Xhinge   x/c location of hinge.  
        #         If positive, control surface extent is Xhinge..1  (TE surface)
        #         If negative, control surface extent is 0..-Xhinge (LE surface)
        # XYZhvec  vector giving hinge axis about which surface rotates 
        #         + deflection is + rotation about hinge vector by righthand rule
        #         Specifying XYZhvec = 0. 0. 0. puts the hinge vector along the hinge
        # SgnDup   sign of deflection for duplicated surface
        #         An elevator would have SgnDup = +1
        #         An aileron  would have SgnDup = -1
    

        # Jet Parameters
        # ------------------------------------

        with open(avl_path, 'w') as f:
            f.write("x57\n")  # or any simple name, matching your config
            f.write("#Mach\n")
            f.write(f"{states.ac_states.Mach.value[0]:.1f}\n")
            f.write("#IYsym    IZsym    Zsym\n")
            f.write(f"{IYsym}    {IZsym}    {Zsym}\n")
            f.write("#Sref    Cref    Bref\n")
            f.write(f"{S_ref.value[0]:.3f}    {c_ref.value[0]:.3f}    {b_ref.value[0]:.3f}\n")
            f.write("#Xref    Yref    Zref\n")  
            f.write(f"{Xref:.3f}    {Yref:.3f}    {Zref:.3f}\n")
            f.write("#\n")
            f.write("#\n\n")
            f.write("#====================================================================\n")
            f.write("SURFACE\nInner Wing\n")
            f.write("#Nchordwise    Cspace    Nspanwise    Sspace\n")
            f.write("12    1.0    4    1.0\n")
            f.write("#\n")
            f.write("COMPONENT\n1\n")
            f.write("#\n")
            f.write("YDUPLICATE\n0.0\n")
            f.write("#\n")
            f.write(f"ANGLE\n{wing_incidence.value[0] * (180/np.pi):.3f}\n")
            f.write("\nJETPARAM\n#hdisk    fh    djet0   djet1    djet3\n")
            f.write(f"0.45    1.0    0.0   0.0    0.0\n")

            # section 1: root chord point
            self.write_section(
                f,
                AF=airfoil_files[3],
                is_naca=False,
                Xle=0.0,
                Yle=0.0,
                Zle=0.0,
                chord=c_ref.value[0],
                ainc=0.0,
                Nspan=2,
                Sspace=1.0,
                is_control=False,
                control_params=None
            )
            # section 2: first spanwise point out from root
            self.write_section(
                f,
                AF=airfoil_files[4],
                is_naca=False,
                Xle=0.0,
                Yle=0.6,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=2,
                Sspace=1.0,
                is_control=False,
                control_params=None
            )
            f.write("#====================================================================\n")
            f.write("SURFACE\nMid Wing\n")
            f.write("#Nchordwise    Cspace    Nspanwise    Sspace\n")
            f.write("12    1.0    4    1.0\n")
            f.write("#\n")
            f.write("COMPONENT\n1\n")
            f.write("#\n")
            f.write("YDUPLICATE\n0.0\n")
            f.write("#\n")
            f.write(f"ANGLE\n{wing_incidence.value[0] * (180/np.pi):.3f}\n")
            f.write("\nJETPARAM\n#hdisk    fh    djet0   djet1    djet3\n")
            f.write(f"0.45    1.0    0.0   0.0    0.0\n")

            # section 3: second spanwise point out from root
            self.write_section(
                f,
                AF=airfoil_files[4],
                is_naca=False,
                Xle=0.0,
                Yle=0.6,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=2,
                Sspace=1.0,
                is_control=True,
                control_params={
                                'Cname': 'Flap',
                                'Cgain': 1.0,
                                'Xhinge': 0.75,
                                'HingeVec': [0.0, 0.0, 0.0],
                                'SgnDup': 1.0}
            )
            # section 4: third spanwise point out from root
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=0.0,
                Yle=1.6+1.6,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=2,
                Sspace=1.0,
                is_control=True,
                control_params={
                                'Cname': 'Flap',
                                'Cgain': 1.0,
                                'Xhinge': 0.75,
                                'HingeVec': [0.0, 0.0, 0.0],
                                'SgnDup': 1.0}
            )

            f.write("#====================================================================\n")
            f.write("SURFACE\nOuter Wing\n")
            f.write("#Nchordwise    Cspace    Nspanwise    Sspace\n")
            f.write("12    1.0    4    1.0\n")
            f.write("#\n")
            f.write("COMPONENT\n1\n")
            f.write("#\n")
            f.write("YDUPLICATE\n0.0\n")
            f.write("#\n")
            f.write(f"ANGLE\n{wing_incidence.value[0] * (180/np.pi):.3f}\n")
            f.write("\nJETPARAM\n#hdisk    fh    djet0   djet1    djet3\n")
            f.write(f"0.45    1.0    0.0   0.0    0.0\n")

            # section 5: fourth spanwise point out from root
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=0.0,
                Yle=1.6+1.6,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=2,
                Sspace=1.0,
                is_control=True,
                control_params=[
                    {
                        'Cname': 'Aileron',
                        'Cgain': -1.0,
                        'Xhinge': 0.75,
                        'HingeVec': [0.0, 0.0, 0.0],
                        'SgnDup': -1.0
                    }
                ]
            )
            # section 6: fifth spanwise point out from root
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=0.0,
                Yle=1.6+1.6+1.6,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=2,
                Sspace=1.0,
                is_control=True,
                control_params=[
                    {
                        'Cname': 'Aileron',
                        'Cgain': -1.0,
                        'Xhinge': 0.75,
                        'HingeVec': [0.0, 0.0, 0.0],
                        'SgnDup': -1.0
                    }
                ]
            )

            f.write("#====================================================================\n")
            f.write("SURFACE\nHorizontal Tail\n")
            f.write("#Nchordwise    Cspace    Nspanwise    Sspace\n")
            f.write("6    1.0    8    1.0\n")
            f.write("#\n")
            f.write("COMPONENT\n2\n")
            f.write("#\n")
            f.write("YDUPLICATE\n0.0\n")
            f.write("#\n")
            f.write(f"ANGLE\n0\n")

# need to add correct parameters for the horizontal tail
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=15.0,
                Yle=0.0,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=0.0,
                Sspace=0.0,
                is_control=True,
                control_params=[
                    {
                        'Cname': 'Elevator',
                        'Cgain': 1.0,
                        'Xhinge': 0.6,
                        'HingeVec': [0.0, 0.0, 0.0],
                        'SgnDup': 1.0
                    }
                ]
            )

# need to add correct parameters for the horizontal tail
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=15.0,
                Yle=2.5,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=0.0,
                Sspace=0.0,
                is_control=True,
                control_params=[
                    {
                        'Cname': 'Elevator',
                        'Cgain': 1.0,
                        'Xhinge': 0.6,
                        'HingeVec': [0.0, 0.0, 0.0],
                        'SgnDup': 1.0
                    }
                ]
            )
            f.write("#====================================================================\n")
            f.write("SURFACE\nVertical Tail\n")
            f.write("#Nchordwise    Cspace    Nspanwise    Sspace\n")
            f.write("6    1.0    8    1.0\n")
            f.write("#\n")
            f.write("COMPONENT\n2\n")
            f.write("#\n")
            f.write("YDUPLICATE\n0.0\n")
            f.write("#\n")
            f.write(f"ANGLE\n0\n")

# need to add correct parameters for the vertical tail
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=15.0,
                Yle=0.0,
                Zle=0.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=0.0,
                Sspace=0.0,
                is_control=True,
                control_params=[
                    {
                        'Cname': 'Rudder',
                        'Cgain': 1.0,
                        'Xhinge': 0.75,
                        'HingeVec': [0.0, 0.0, 0.0],
                        'SgnDup': 1.0
                    }
                ]
            )

# need to add correct parameters for the horizontal tail
            self.write_section(
                f,
                AF=airfoil_files[5],
                is_naca=False,
                Xle=15.25,
                Yle=0.0,
                Zle=4.0,
                chord=c_ref.value[0] * self.taper_wing.value[0],
                ainc=0.0,
                Nspan=0.0,
                Sspace=0.0,
                is_control=True,
                control_params=[
                    {
                        'Cname': 'Rudder',
                        'Cgain': 1.0,
                        'Xhinge': 0.75,
                        'HingeVec': [0.0, 0.0, 0.0],
                        'SgnDup': 1.0
                    }
                ]
            )

        print(f"AVL file written to {avl_path}")
        return avl_path




def jvl_run(config, alphaCL, alphaCLFlag, flap, cjet, i, j, k):
    """
    Run JVL with basic inputs.
    Make sure xxx.avl and xxx.mass files are in the same folder.
    """
    # Delete old input command files if they exist
    for fname in ['jvlcom.in', 'jvlcom.out', 'jvlcom.txt']:
        if os.path.exists(fname):
            os.remove(fname)

    # Create strings for input/output file names and input commands
    avlin = f'LOAD {config}.avl'
    # massin = f'MASS {config}.mass'
    # Ensure 'JVL results' directory exists in the working directory
    wd = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(wd, 'JVL results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fileout = os.path.join('JVL results', f'{config}_{i}_{j}_{k}.txt')
    if alphaCLFlag == 'alpha':
        Ain = f'A A {alphaCL}'
    elif alphaCLFlag == 'CL':
        Ain = f'A C {alphaCL}'
    else:
        raise ValueError("alphaCLFlag must be 'alpha' or 'CL'")


    # Flap Deflection and Jet Control inputs
    D1in = f'D1 D1 {flap}'
    J1in = f'J1 J1 {cjet}'

    # Geometry output file
    # Input commands
    commands = [
        avlin,
        # massin,
        'OPER',
        Ain,
        D1in,
        J1in,
        'O',
        'P',
        # 'T T T T T T',
        'T T F F',
        'H',
        'D',
        '',
        'X',
        'W',
        fileout,
        '',
        'Q'
    ]

    # Find jvl.exe in the same directory as this script
    wd = os.path.dirname(os.path.abspath(__file__))
    inp_path = os.path.join(wd, 'jvlcom.in')
    out_path = os.path.join(wd, 'jvlcom.out')

    # Write input command file in wd
    with open(inp_path, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')


    jvl_path = os.path.join(wd, 'jvl.exe')
    # If not found, fallback to PATH
    if not os.path.exists(jvl_path):
        jvl_path = 'jvl.exe'
    # Use subprocess with explicit stdin/stdout in wd, and capture output/errors
    with open(inp_path, 'r') as fin:
        proc = subprocess.Popen(jvl_path, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd)
        stdout, stderr = proc.communicate()
        status = proc.returncode
    cmd = f'{jvl_path} < {inp_path} > {out_path}'
    print(f"JVL return code: {status}")
    print(f"JVL stdout:\n{stdout.decode(errors='ignore')}")
    print(f"JVL stderr:\n{stderr.decode(errors='ignore')}")
    if status != 0:
        raise RuntimeError(f'JVL execution failed! {cmd}')

    # Delete input command files
    for fname in ['jvlcom.in', 'jvlcom.out', 'jvlcom.txt']:
        fpath = os.path.join(wd, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    return fileout


def sweep_jvl_outputs(config, alpha, flap, cjet):
    alphal = len(alpha)
    flapl = len(flap)
    cjetl = len(cjet)
    # Initialize output arrays
    CJtot   = np.zeros((alphal, flapl, cjetl))
    CXtot   = np.zeros((alphal, flapl, cjetl))
    CYtot   = np.zeros((alphal, flapl, cjetl))
    CZtot   = np.zeros((alphal, flapl, cjetl))
    CLtot   = np.zeros((alphal, flapl, cjetl))
    CDtot   = np.zeros((alphal, flapl, cjetl))
    CLcir   = np.zeros((alphal, flapl, cjetl))
    CLjet   = np.zeros((alphal, flapl, cjetl))
    CDind   = np.zeros((alphal, flapl, cjetl))
    CDjet   = np.zeros((alphal, flapl, cjetl))
    CDvis   = np.zeros((alphal, flapl, cjetl))

    import os
    wd = os.path.dirname(os.path.abspath(__file__))
    for i in range(alphal):
        for j in range(flapl):
            for k in range(cjetl):
                try:
                    fileout = jvl_run(config, alpha[i], 'alpha', flap[j], cjet[k], i+1, j+1, k+1)
                    fileout_path = os.path.join(wd, fileout)
                    with open(fileout_path, 'r') as f:
                        # Read all lines and search for relevant variables
                        lines = f.readlines()
                    # Helper to extract value after '='
                    def extract_value(line, var):
                        idx = line.find(var)
                        if idx == -1:
                            return None
                        eq_idx = line.find('=', idx)
                        if eq_idx == -1:
                            return None
                        # Find the value after '=' and before next space
                        val_str = line[eq_idx+1:].split()[0]
                        if '*' in val_str:
                            return float('nan')
                        try:
                            return float(val_str)
                        except Exception:
                            return float('nan')
                    # Map variable names to arrays
                    var_map = {
                        'CJtot': ('CJtot', 'CJtot ='),
                        'CXtot': ('CXtot', 'CXtot ='),
                        'CYtot': ('CYtot', 'CYtot ='),
                        'CZtot': ('CZtot', 'CZtot ='),
                        'CLtot': ('CLtot', 'CLtot ='),
                        'CDtot': ('CDtot', 'CDtot ='),
                        'CLcir': ('CLcir', 'CLcir ='),
                        'CLjet': ('CLjet', 'CLjet ='),
                        'CDind': ('CDind', 'CDind ='),
                        'CDjet': ('CDjet', 'CDjet ='),
                        'CDvis': ('CDvis', 'CDvis ='),
                    }
                    # For each variable, search for the line and extract value
                    values = {}
                    for var, (arr_name, search_str) in var_map.items():
                        found = False
                        for line in lines:
                            if search_str in line:
                                val = extract_value(line, search_str)
                                values[arr_name] = val
                                found = True
                                break
                        if not found:
                            values[arr_name] = float('nan')
                    CJtot[i, j, k] = values['CJtot']
                    CXtot[i, j, k] = values['CXtot']
                    CYtot[i, j, k] = values['CYtot']
                    CZtot[i, j, k] = values['CZtot']
                    CLtot[i, j, k] = values['CLtot']
                    CDtot[i, j, k] = values['CDtot']
                    CLcir[i, j, k] = values['CLcir']
                    CLjet[i, j, k] = values['CLjet']
                    CDind[i, j, k] = values['CDind']
                    CDjet[i, j, k] = values['CDjet']
                    CDvis[i, j, k] = values['CDvis']
                except Exception as e:
                    print(f"Failed for alpha={alpha[i]}, flap={flap[j]}, cjet={cjet[k]}: {e}")

    # Save results to CSV
    import csv
    csv_path = os.path.join(wd, f'{config}_jvl_sweep_results.csv')
    header = ['alpha', 'flap', 'cjet', 'CJtot', 'CXtot', 'CYtot', 'CZtot', 'CLtot', 'CDtot', 'CLcir', 'CLjet', 'CDind', 'CDjet', 'CDvis']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i in range(alphal):
            for j in range(flapl):
                for k in range(cjetl):
                    row = [
                        alpha[i], flap[j], cjet[k],
                        CJtot[i, j, k], CXtot[i, j, k], CYtot[i, j, k], CZtot[i, j, k],
                        CLtot[i, j, k], CDtot[i, j, k], CLcir[i, j, k], CLjet[i, j, k],
                        CDind[i, j, k], CDjet[i, j, k], CDvis[i, j, k]
                    ]
                    writer.writerow(row)
    print(f"JVL sweep results saved to {csv_path}")
    return {
        'CJtot': CJtot, 'CXtot': CXtot, 'CYtot': CYtot, 'CZtot': CZtot,
        'CLtot': CLtot, 'CDtot': CDtot, 'CLcir': CLcir, 'CLjet': CLjet,
        'CDind': CDind, 'CDjet': CDjet, 'CDvis': CDvis
    }




if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True, expand_ops=True, debug=False)
    recorder.start()
    from falco.core.vehicle.conditions import aircraft_conditions

    geo_dict = get_geometry()
    axis_dict = None
    if hasattr(geo_dict, 'keys') and 'geometry' in geo_dict:
        from x57_geometry import get_geometry_related_axis
        axis_dict = get_geometry_related_axis(geo_dict)
    aircraft = build_aircraft_component(geo_dict, do_geo_param=False)
    add_mp_to_components(aircraft, geo_dict, axis_dict)
    aircraft.mass_properties = aircraft.compute_total_mass_properties()

    config = 'x57'
    # Example usage: create AVL file using X57Aerodynamics method
    aero = X57Aerodynamics(aircraft)

    hlb = Blower()

    x57_controls = X57ControlSystem(elevator_component=aircraft.comps['Elevator'],
                                    rudder_component=aircraft.comps['Vertical Tail'].comps['Rudder'],
                                    aileron_left_component=aircraft.comps['Wing'].comps['Left Aileron'],
                                    aileron_right_component=aircraft.comps['Wing'].comps['Right Aileron'],
                                    trim_tab_component=aircraft.comps['Elevator'].comps['Trim Tab'],
                                    flap_left_component=aircraft.comps['Wing'].comps['Left Flap'],
                                    flap_right_component=aircraft.comps['Wing'].comps['Right Flap'],
                                    hl_engine_count=12,cm_engine_count=2, high_lift_blower_component=hlb)
    x57_controls.update_high_lift_control(flap_flag=False, blower_flag=False)

    cruise = aircraft_conditions.CruiseCondition(
        fd_axis=axis_dict['fd_axis'],
        controls=x57_controls,
        altitude=Q_(8000, 'ft'),
        range=Q_(160, 'km'),
        speed=Q_(76.8909, 'm/s'),
        pitch_angle=Q_(0, 'deg'))

    wing_num_chordwise_vlm = 121
    wing_num_spanwise_vlm = 6

    wing_mesh = get_airfoil_mesh(
        geometry=geo_dict['geometry'],
        wing=geo_dict['wing'],
        wing_num_spanwise_vlm=wing_num_spanwise_vlm,
        wing_num_chordwise_vlm=wing_num_chordwise_vlm  
    )

    # Export airfoil coordinates for each spanwise index
    for spanwise_index in range(wing_num_spanwise_vlm):
        filename = f"airfoil_coords_{spanwise_index}.dat"
        export_airfoil_coords(
            wing_mesh=wing_mesh,
            wing_num_chordwise_vlm=wing_num_chordwise_vlm,
            wing_num_spanwise_vlm=wing_num_spanwise_vlm,
            spanwise_index=spanwise_index,
            filename=filename
        )

    # Use all spanwise airfoil files
    airfoil_files = [os.path.join('airfoils', f"airfoil_coords_{i}.dat") for i in range(wing_num_spanwise_vlm)]


    coord = airfoil_files[0]  # Use the first airfoil file for XFoil    
    alphas = 2  # Define a range of angles of attack to test
    Re = 1e6
    Mach = 0.2
    extra_cmds = [
    'test_x57_wing_airfoil',
    'ppar\nn\n200',  
    '\n',  # Repanel based on curvature
    '\n',
    'gdes\ntgap\n0.0\nexec\ngset', # to ensure no gap between top and bottom surfaces of airfoil',
    '\n',

]
    pol, foil = aero.xfoil(coord, alphas, Re, Mach, *extra_cmds)





    recorder.stop()