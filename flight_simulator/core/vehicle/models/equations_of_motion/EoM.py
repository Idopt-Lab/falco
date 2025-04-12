from abc import abstractmethod
import numpy as np

import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.loads.mass_properties import MassProperties
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from typing import Union
from dataclasses import dataclass



class StateVectorDot(csdl.VariableGroup):
    du_dt: csdl.Variable
    dv_dt: csdl.Variable
    dw_dt: csdl.Variable
    dp_dt: csdl.Variable
    dq_dt: csdl.Variable
    dr_dt: csdl.Variable
    dphi_dt: csdl.Variable
    dtheta_dt: csdl.Variable
    dpsi_dt: csdl.Variable
    dx_dt: csdl.Variable
    dy_dt: csdl.Variable
    dz_dt: csdl.Variable



    def __repr__(self):
        return (f"du_dt={self.du_dt}\n"
                f"dv_dt={self.dv_dt}\n"
                f"dw_dt={self.dw_dt}\n"
                f"dp_dt={self.dp_dt}\n"
                f"dq_dt={self.dq_dt}\n"
                f"dr_dt={self.dr_dt}\n"
                f"dphi_dt={self.dphi_dt}\n"
                f"dtheta_dt={self.dtheta_dt}\n"
                f"dpsi_dt={self.dpsi_dt}\n"
                f"dx_dt={self.dx_dt}\n"
                f"dy_dt={self.dy_dt}\n"
                f"dz_dt={self.dz_dt}")

class DynamicSystem:
    def __init__(self, x, t0=0, origin='ref'):
        self.state_vector = x
        self.state_vector_dot = csdl.Variable(shape=x.shape, value=0)
        self.time = t0
        self.origin = origin


    
class Aircraft6DOF(DynamicSystem):
    """
    Classical Aircraft Implementation:
    6DOF equations of motion for aircraft dynamics.
    Uses Euler Flat Earth model for aircraft dynamics.
    Vehicle position expressed in inertial axis

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight", Courier Corporation,
        p. 149 (5.8 The Flat-Earth Approximation), 2012.

    """
    #TODO: CONFIRM WHETHER BODY FIXED AXIS IS AT CG OR ORIGIN
    def __init__(self, x, t0=0, origin='fd-axis'):
        super().__init__(x, t0, origin=origin)

    def _EoM(aircraft_states, mass_properties, total_forces,total_moments):
        """

        """

        Fx=total_forces[0]
        Fy=total_forces[1]
        Fz=total_forces[2]
        Mx=total_moments[0]
        My=total_moments[1]
        Mz=total_moments[2]

        u = aircraft_states.states.u
        v = aircraft_states.states.v
        w = aircraft_states.states.w
        p = aircraft_states.states.p
        q = aircraft_states.states.q
        r = aircraft_states.states.r
        phi = aircraft_states.states.phi
        theta = aircraft_states.states.theta
        psi = aircraft_states.states.psi
        x = aircraft_states.states.x
        y = aircraft_states.states.y
        z = aircraft_states.states.z

        # Extract mass properties
        m = mass_properties.mass
        cg_vector = mass_properties.cg_vector
        inertia_tensor = mass_properties.inertia_tensor.inertia_tensor

        xcg = cg_vector[0]
        ycg = cg_vector[1]
        zcg = cg_vector[2]

        Ixx = inertia_tensor[0, 0]
        Iyy = inertia_tensor[1, 1]
        Izz = inertia_tensor[2, 2]
        Ixy = inertia_tensor[0, 1]
        Ixz = inertia_tensor[0, 2]
        Iyz = inertia_tensor[1, 2]
        Idot = csdl.Variable(shape=(3, 3),value=0)

        # CG Offset from ref point (inertial axis)
        Rbc = csdl.Variable(shape=(3,), value=0)
        Rbc = Rbc.set(csdl.slice[0], xcg)
        Rbc = Rbc.set(csdl.slice[1], ycg)
        Rbc = Rbc.set(csdl.slice[2], zcg)

        xcgdot = csdl.Variable(shape=(1, ), value=0.)
        ycgdot = csdl.Variable(shape=(1, ), value=0.)
        zcgdot = csdl.Variable(shape=(1, ), value=0.)
        xcgddot = csdl.Variable(shape=(1, ), value=0.)
        ycgddot = csdl.Variable(shape=(1, ), value=0.)
        zcgddot = csdl.Variable(shape=(1, ), value=0.)

        # Generate the 6x6 mp matrix
        mp_matrix = csdl.Variable(shape=(3, 3), value=0)
        mp_matrix = mp_matrix.set(csdl.slice[0, 0], m)
        mp_matrix = mp_matrix.set(csdl.slice[1, 1], m)
        mp_matrix = mp_matrix.set(csdl.slice[2, 2], m)
        mp_matrix = mp_matrix.set(csdl.slice[0, 4], m * zcg)
        mp_matrix = mp_matrix.set(csdl.slice[0, 5], -m * ycg)
        mp_matrix = mp_matrix.set(csdl.slice[1, 3], -m * zcg)
        mp_matrix = mp_matrix.set(csdl.slice[1, 5], m * xcg)
        mp_matrix = mp_matrix.set(csdl.slice[2, 3], m * ycg)
        mp_matrix = mp_matrix.set(csdl.slice[2, 4], -m * xcg)
        mp_matrix = mp_matrix.set(csdl.slice[3, 1], -m * zcg)
        mp_matrix = mp_matrix.set(csdl.slice[3, 2], m * ycg)
        mp_matrix = mp_matrix.set(csdl.slice[4, 0], m * zcg)
        mp_matrix = mp_matrix.set(csdl.slice[4, 2], -m * xcg)
        mp_matrix = mp_matrix.set(csdl.slice[5, 0], -m * ycg)
        mp_matrix = mp_matrix.set(csdl.slice[5, 1], m * xcg)
        mp_matrix = mp_matrix.set(csdl.slice[3, 3], Ixx)
        mp_matrix = mp_matrix.set(csdl.slice[3, 4], Ixy)
        mp_matrix = mp_matrix.set(csdl.slice[3, 5], Ixz)
        mp_matrix = mp_matrix.set(csdl.slice[4, 3], Ixy)
        mp_matrix = mp_matrix.set(csdl.slice[4, 4], Iyy)
        mp_matrix = mp_matrix.set(csdl.slice[4, 5], Iyz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 3], Ixz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 4], Iyz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 5], Izz)

        lambda_x = Fx + m * (r * v - q * w - xcgdot - 2 * q * zcgdot 
                             + 2 * r * ycgdot + xcg * (q ** 2 + r ** 2)
                             - ycg * p * q - zcg * p * r)
        lambda_y = Fy + m * (p * w - r * u - ycgdot - 2 * r * xcgdot
                             + 2 * p * zcgdot - xcg * p * q 
                             + ycg * (p ** 2+ r ** 2) - zcg * q * r)
        lambda_z = Fz + m * (q * u - p * v - zcgdot - 2 * p * ycgdot
                             + 2 * q * xcgdot - xcg * p * r
                             - ycg * q * r + zcg * (p ** 2 + q ** 2))
        
        angvel_vector = csdl.Variable(shape=(3,), value=0)
        angvel_vector = angvel_vector.set(csdl.slice[0], p)
        angvel_vector = angvel_vector.set(csdl.slice[1], q)
        angvel_vector = angvel_vector.set(csdl.slice[2], r)

        angvel_ssym = csdl.Variable(shape=(3, 3), value=0)
        angvel_ssym = angvel_ssym.set(csdl.slice[0, 1], -r)
        angvel_ssym = angvel_ssym.set(csdl.slice[0, 2], q)
        angvel_ssym = angvel_ssym.set(csdl.slice[1, 0], r)
        angvel_ssym = angvel_ssym.set(csdl.slice[1, 2], -p)
        angvel_ssym = angvel_ssym.set(csdl.slice[2, 0], -q)
        angvel_ssym = angvel_ssym.set(csdl.slice[2, 1], p)

        Rbc_ssym = csdl.Variable(shape=(3, 3), value=0)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[0, 1], -zcg)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[0, 2], ycg)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[1, 0], zcg)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[1, 2], -xcg)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[2, 0], -ycg)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[2, 1], xcg)

        mu_vec = total_moments - csdl.matvec(Idot, angvel_vector) - csdl.matvec(csdl.matmat(angvel_ssym, inertia_tensor), angvel_vector) - m * csdl.matvec(csdl.matmat(Rbc_ssym, angvel_ssym), angvel_vector)


        # RHS Vector setup
        rhs = csdl.Variable(shape=(6,), value=0)
        rhs = rhs.set(csdl.slice[0], lambda_x)
        rhs = rhs.set(csdl.slice[1], lambda_y)
        rhs = rhs.set(csdl.slice[2], lambda_z)
        rhs = rhs.set(csdl.slice[3], mu_vec[0])
        rhs = rhs.set(csdl.slice[4], mu_vec[1])
        rhs = rhs.set(csdl.slice[5], mu_vec[2])

        accelerations = csdl.VariableGroup(shape=(6,), value=0)
        accelerations = accelerations.set(csdl.solve_linear(mp_matrix, rhs))

        du_dt = accelerations[0]
        dv_dt = accelerations[1]
        dw_dt = accelerations[2]
        dp_dt = accelerations[3]
        dq_dt = accelerations[4]
        dr_dt = accelerations[5]

        dphi_dt = p + q * csdl.sin(phi) * csdl.tan(theta) + r * csdl.tan(theta) * csdl.cos(phi)
        dtheta_dt = q * csdl.cos(phi) - r * csdl.sin(phi)
        dpsi_dt = q * csdl.sin(phi) / csdl.cos(theta) + r * csdl.cos(phi) / csdl.cos(theta)
        dx_dt = u * csdl.cos(theta) * csdl.cos(psi) \
                + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.cos(psi) - csdl.cos(phi) * csdl.sin(psi)) \
                + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.cos(psi) + csdl.sin(phi) * csdl.sin(psi))
        dy_dt = u * csdl.cos(theta) * csdl.sin(psi) \
                + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.sin(psi) + csdl.cos(phi) * csdl.cos(psi)) \
                + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.sin(psi) - csdl.sin(phi) * csdl.cos(psi))
        dz_dt = -u * csdl.sin(theta) + v * csdl.sin(phi) * csdl.cos(theta) + w * csdl.cos(phi) * csdl.cos(theta)

        dstate_output = StateVectorDot(du_dt=du_dt, dv_dt=dv_dt, dw_dt=dw_dt,
                                                    dp_dt=dp_dt, dq_dt=dq_dt, dr_dt=dr_dt,
                                                    dphi_dt=dphi_dt, dtheta_dt=dtheta_dt, dpsi_dt=dpsi_dt,
                                                    dx_dt=dx_dt, dy_dt=dy_dt, dz_dt=dz_dt)
        return dstate_output
    
    def EoM_steady_state_trim(self, x, component):
        total_forces = component.quantities.total_forces
        total_moments = component.quantities.total_moments
        mass_properties = component.quantities.mass_properties
        aircraft_states = component.quantities.ac_states

        res = self._EoM(aircraft_states, mass_properties, total_forces, total_moments)

        du_dt = res.du_dt
        dv_dt = res.dv_dt
        dw_dt = res.dw_dt
        dp_dt = res.dp_dt
        dq_dt = res.dq_dt
        dr_dt = res.dr_dt

        xddot = csdl.Variable(shape=(6,), value=0)
        xddot = xddot.set(csdl.slice[0], du_dt)
        xddot = xddot.set(csdl.slice[1], dv_dt)
        xddot = xddot.set(csdl.slice[2], dw_dt)
        xddot = xddot.set(csdl.slice[3], dp_dt)
        xddot = xddot.set(csdl.slice[4], dq_dt)
        xddot = xddot.set(csdl.slice[5], dr_dt)
        xddotT = xddot.T()

        W= csdl.Variable(shape=(6, 6), value=0)
        W = W.set(csdl.slice[0, 0], 1)
        W = W.set(csdl.slice[1, 1], 1)
        W = W.set(csdl.slice[2, 2], 1)
        W = W.set(csdl.slice[3, 3], 1)
        W = W.set(csdl.slice[4, 4], 1)
        W = W.set(csdl.slice[5, 5], 1)

        J = csdl.Variable(shape=(1,), value=0)
        #todo: verify that xddot and xddotT shouldnt be flipped
        J = J.set(csdl.slice[0], csdl.matmat(xddot, csdl.matvec(W, xddotT)))
        return J
