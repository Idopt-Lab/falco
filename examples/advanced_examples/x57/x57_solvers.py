import csdl_alpha as csdl
import numpy as np
from typing import Union
from scipy.interpolate import Akima1DInterpolator
from flight_simulator.core.loads.loads import Loads
import os
import scipy.io as sio
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator import REPO_ROOT_FOLDER, Q_, ureg


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
            u = controls.u
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
            daileron = controls.roll_control['Left Aileron'].deflection            
            dtrim = controls.pitch_control['Trim Tab'].deflection
            drudder = controls.yaw_control['Rudder'].deflection

            # blowing affect if HL engines are active
            blow_num = 0
            for engine in controls.hl_engines:
                if engine.throttle.value != 0:
                    blow_num += 1


            CL = self.AC_CL(alpha=AOA, i_w=i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            CD = self.AC_CD(alpha=AOA, i_w=i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            CM = self.AC_CM(alpha=AOA, i_w=i_wing, i_stab=dstab, AR=self.AR_wing, flap=dflap, blow_num=blow_num, trimtab=dtrim)
            L = 0.5 * density * velocity**2 * self.Sref_wing * CL
            D = 0.5 * density * velocity**2 * self.Sref_wing * CD
            M = 0.5 * density * velocity**2 * self.Sref_wing * CM * self.cref_wing
            # print(f"CL: {CL.value}, CD: {CD.value}, CM: {CM.value}, L: {L.value}, D: {D.value}, M: {M.value}")


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
            return loads_wind_axis
    

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

        # Obtained Mod-IV Propeller Data from CFD database
        J_data = np.array(
            [0.5490,0.5966,0.6860,0.8250,1.0521,1.4595,1.6098])
        Ct_data = np.array(
            [0.3125,0.3058,0.2848,0.2473,0.1788,0.0366,-0.0198])
        self.ct = Akima1DInterpolator(J_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)

        Cp_data = np.array([0.3134,0.3152,0.3075,0.2874,0.2367,0.0809,0.0018])
        self.cp = Akima1DInterpolator(J_data, Cp_data, method="akima")
        self.cp_derivative = Akima1DInterpolator.derivative(self.cp)

        Cq_data = np.array([0.0499,0.0502,0.0489,0.0457,0.0377,0.0129,0.0003])
        self.cq = Akima1DInterpolator(J_data, Cq_data, method="akima")
        self.cq_derivative = Akima1DInterpolator.derivative(self.cq)

        # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, J_data: csdl.Variable=None):
        outputs = csdl.VariableGroup()

        if J_data is not None:
            self.declare_input('J_data', J_data)
            cp = self.create_output('cp', J_data.shape)
            cq = self.create_output('cq', J_data.shape)
            ct = self.create_output('ct', J_data.shape)

            outputs.ct = ct
            outputs.cp = cp
            outputs.cq = cq

        return outputs
            
    def compute(self, input_vals, output_vals):

        if 'J_data' in input_vals:
            J_data = input_vals['J_data']
            output_vals['ct'] = self.ct(J_data)
            output_vals['cp'] = self.cp(J_data)
            output_vals['cq'] = self.cq(J_data)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        J_data = input_vals['J_data']

        derivatives['ct', 'J_data'] = np.diag(self.ct_derivative(J_data))
        derivatives['cp', 'J_data'] = np.diag(self.cp_derivative(J_data))
        derivatives['cq', 'J_data'] = np.diag(self.cq_derivative(J_data))


class CruisePropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        # Obtained Mod-III Propeller Data from CFD database

        # Pull the data for blade pitch is 26 and alpha is 2, beta is 0, aileron is 0, and flap is 0.
        # From 1.6 onwards, it is at a 30 degree blade pitch, with everything else the same as the condition above.

        J_data = np.array(
            [0.1,0.4,0.6,0.8, 1.0, 1.2, 1.3, 1.4, 1.6, 1.8])
        Ct_data = np.array(
            [0.1897,0.1857,0.1744,0.1466, 0.1021, 0.0498, 0.0223, 0.0567, 0.0030, 0.1234])
        self.ct = Akima1DInterpolator(J_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)
        Cp_data = np.array([0.1360,0.1487,0.1579,0.1514, 0.1201, 0.0686, 0.0364, 0.0003, 0.0134, 0.2537])
        self.cp = Akima1DInterpolator(J_data, Cp_data, method="akima")
        self.cp_derivative = Akima1DInterpolator.derivative(self.cp)
        Cq_data = np.array([0.0216,0.0237,0.0251,0.0241, 0.0191, 0.0109, 0.0058, 0.0001, 0.0021, 0.0404])
        self.cq = Akima1DInterpolator(J_data, Cq_data, method="akima")
        self.cq_derivative = Akima1DInterpolator.derivative(self.cq)

    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, J_data: csdl.Variable=None):
        outputs = csdl.VariableGroup()

        if J_data is not None:
            self.declare_input('J_data', J_data)
            cp = self.create_output('cp', J_data.shape)
            cq = self.create_output('cq', J_data.shape)
            ct = self.create_output('ct', J_data.shape)

            outputs.ct = ct
            outputs.cp = cp
            outputs.cq = cq

        return outputs
            
    def compute(self, input_vals, output_vals):

        if 'J_data' in input_vals:
            J_data = input_vals['J_data']
            output_vals['ct'] = self.ct(J_data)
            output_vals['cp'] = self.cp(J_data)
            output_vals['cq'] = self.cq(J_data)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        J_data = input_vals['J_data']

        derivatives['ct', 'J_data'] = np.diag(self.ct_derivative(J_data))
        derivatives['cp', 'J_data'] = np.diag(self.cp_derivative(J_data))
        derivatives['cq', 'J_data'] = np.diag(self.cq_derivative(J_data))


class X57Propulsion(Loads):

    def __init__(self, RPMmin, RPMmax, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:Union[HLPropCurve, CruisePropCurve], engine_index: int = 0, **kwargs):
        self.prop_curve = prop_curve
        self.engine_index = engine_index  # Save the index for later lookup
        self.RPMmin = RPMmin
        self.RPMmax = RPMmax

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=(1.89/2)*0.308) 
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units().magnitude)
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
            - rpm

        Returns
        -------
        loads : ForcesMoments
            Computed forces and moments about the reference point.
        """
        u = controls.u
        throttle = u[5+self.engine_index]  # Get the throttle for the specific engine
        density = states.atmospheric_states.density  # kg/m^3
        velocity = states.VTAS  # m/s 
        rpm = self.RPMmin + (self.RPMmax - self.RPMmin) * throttle  # RPM based on throttle input
        # print(f"RPM: {rpm.value}, Density: {density.value}, Velocity: {velocity.value}")

        omega = (rpm / 60)  # Convert RPM to rev/s

        # Compute advance ratio
        J = (velocity) / ((omega) * (self.radius * 2))   # non-dimensional
        # print(f"Advance Ratio J: {J.value}")

        # Compute Ct
        prop_curve_obj = type(self.prop_curve)()
        prop_out = prop_curve_obj.evaluate(J_data=J)
        ct = prop_out.ct
        cp = prop_out.cp
        cq = prop_out.cq
        # print(f"Computed Ct: {ct.value}")

        # Compute Thrust
        T = (ct * density * (omega)**2 * ((self.radius*2)**4)) # N
    
        # print(f"Computed Thrust T: {T.value} Newtons")
        if np.isnan(T.value):
            T = csdl.Variable(shape=(1,), value=0.)
        
        # print(f"Computed Thrust T: {T.value} Newtons")
    
        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads

   

    def get_torque_power(self, states, controls):
        """
        Compute power available for the propulsion system.

        Parameters
        ----------
        shaft_power : 
            Shaft power for the propulsion system (P) which should include:
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
            Computed torque for the propulsion system.
        power_avail : csdl.VariableGroup
            Computed power available for the propulsion system.
        power_shaft : csdl.VariableGroup
            Computed shaft power for the propulsion system.
        """
        u = controls.u
        throttle = u[5+self.engine_index]  # Get the rpm for the specific engine
        density = states.atmospheric_states.density
        velocity = states.VTAS # m/s to ft/s
        rpm = self.RPMmin + (self.RPMmax - self.RPMmin) * throttle  # RPM based on throttle input
        omega = (rpm / 60)  # Convert RPM to rev/s


        J = (velocity) / ((omega) * (self.radius * 2))   # non-dimensional

        prop_curve_obj = type(self.prop_curve)()
        prop_out = prop_curve_obj.evaluate(J_data=J)
        ct = prop_out.ct
        cp = prop_out.cp
        cq = prop_out.cq

        torque = cq*(density * (omega)**2 * ((self.radius*2)**5)) #  Nm

        if np.isnan(torque.value):
            torque = csdl.Variable(shape=(1,), value=0.)

        shaft_power = (torque * (omega) * 2 * np.pi) # Watts 

        if np.isnan(shaft_power.value):
            shaft_power = csdl.Variable(shape=(1,), value=0.)

        if np.isnan(cp.value) and np.isnan(ct.value):
            eta = csdl.Variable(shape=(1,), value=0.)
        else:
            eta = J * (ct / cp)  # Efficiency of the propeller

        power_avail = shaft_power * eta  # Power available for the propulsion system

        return {
            'torque': torque,
            'cq': cq,
            'shaft_power': shaft_power,
            'ct': ct,
            'cp': cp,
            'eta': eta,
            'J': J,
            'power_avail': power_avail
        }
    

