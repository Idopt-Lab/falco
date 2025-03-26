import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.loads.mass_properties import MassProperties
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from typing import Union
from dataclasses import dataclass


@dataclass
class LinAngAccel(csdl.VariableGroup):
    du_dt: csdl.Variable
    dv_dt: csdl.Variable
    dw_dt: csdl.Variable
    dp_dt: csdl.Variable
    dq_dt: csdl.Variable
    dr_dt: csdl.Variable
    accel_norm: csdl.Variable


class SixDoFModel:

    def __init__(self, num_nodes: int = 1, stability_flag: bool = False):
        self.num_nodes = num_nodes
        self.stability_flag = stability_flag


    def evaluate(self, 
        total_forces: csdl.Variable, 
        total_moments: csdl.Variable, 
        ac_states: AircraftStates,
        ac_mass_properties: MassProperties,
        ref_pt: Union[csdl.Variable, np.ndarray] = np.array([0., 0., 0.])
    ):
        
        if self.num_nodes == 1:
            total_forces = csdl.reshape(total_forces, shape=(1, 3))
            total_moments = csdl.reshape(total_moments, shape=(1, 3))

        Fx = total_forces[:, 0]
        Fy = total_forces[:, 1]
        Fz = total_forces[:, 2]
        Mx = total_moments[:, 0] 
        My = total_moments[:, 1] 
        Mz = total_moments[:, 2] 

        # Get mass, cg, I and decompose into components
        m = ac_mass_properties.mass.value
        cg_vector =  ac_mass_properties.cg_vector
        inertia_tensor = ac_mass_properties.inertia_tensor.inertia_tensor

        cgx = cg_vector.vector[0].value
        cgy = cg_vector.vector[1].value
        cgz = cg_vector.vector[2].value


        Ixx = inertia_tensor[0, 0].value
        Iyy = inertia_tensor[1, 1].value
        Izz = inertia_tensor[2, 2].value
        Ixy = inertia_tensor[0, 1].value
        Ixz = inertia_tensor[0, 2].value
        Iyz = inertia_tensor[1, 2].value

        # Get aircraft states
        u = ac_states.states.u
        v = ac_states.states.v
        w = ac_states.states.w
        p = ac_states.states.p
        q = ac_states.states.q
        r = ac_states.states.r
        phi = ac_states.states.phi
        theta = ac_states.states.theta
        psi = ac_states.states.psi
        x = ac_states.states.x
        y = ac_states.states.y
        z = ac_states.states.z


        Idot = csdl.Variable(shape=(3, 3), value=0.)

        # cg offset from reference point
        Rbcx = cgx - ref_pt[0]
        Rbcy = cgy - ref_pt[1]
        Rbcz = cgz - ref_pt[2]

        xcgdot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        ycgdot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        zcgdot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        xcgddot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        ycgddot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        zcgddot = csdl.Variable(shape=(self.num_nodes, ), value=0.)

        # fill in (6 x 6) mp matrix
        mp_matrix = csdl.Variable(shape=(6, 6), value=0)
        
        mp_matrix = mp_matrix.set(csdl.slice[0, 0], m)
        mp_matrix = mp_matrix.set(csdl.slice[0, 4], m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[0, 5], -m * Rbcy)

        mp_matrix = mp_matrix.set(csdl.slice[1, 1], m)
        mp_matrix = mp_matrix.set(csdl.slice[1, 3], -m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[1, 5], m * Rbcx)

        mp_matrix = mp_matrix.set(csdl.slice[2, 2], m)
        mp_matrix = mp_matrix.set(csdl.slice[2, 3], m * Rbcy)
        mp_matrix = mp_matrix.set(csdl.slice[2, 4], -m * Rbcx)

        mp_matrix = mp_matrix.set(csdl.slice[3, 1], -m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[3, 2], m * Rbcy)
        mp_matrix = mp_matrix.set(csdl.slice[3, 3], Ixx)
        mp_matrix = mp_matrix.set(csdl.slice[3, 4], Ixy)
        mp_matrix = mp_matrix.set(csdl.slice[3, 5], Ixz)

        mp_matrix = mp_matrix.set(csdl.slice[4, 0], m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[4, 2], -m * Rbcx)
        mp_matrix = mp_matrix.set(csdl.slice[4, 3], Ixy)
        mp_matrix = mp_matrix.set(csdl.slice[4, 4], Iyy)
        mp_matrix = mp_matrix.set(csdl.slice[4, 5], Iyz)

        mp_matrix = mp_matrix.set(csdl.slice[5, 0], -m * Rbcy)
        mp_matrix = mp_matrix.set(csdl.slice[5, 1], m * Rbcx)
        mp_matrix = mp_matrix.set(csdl.slice[5, 3], Ixz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 4], Iyz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 5], Izz)

        lambda_x = Fx + m * (r * v - q * w - xcgdot - 2 * q * zcgdot
                            + 2 * r * ycgdot + Rbcx * (q ** 2 + r ** 2)
                            - Rbcy * p * q - Rbcz * p * r)

        lambda_y = Fy + m * (p * w - r * u - ycgddot - 2 * r * xcgdot 
                            + 2 * p * zcgdot - Rbcx * p * q
                            + Rbcy * (p ** 2 + r ** 2) - Rbcz * q * r)

        lambda_z = Fz + m * (q * u - p * v - zcgddot - 2 * p * ycgdot 
                            + 2 * q * xcgdot - Rbcx * p * r 
                            - Rbcy * q * r + Rbcz * (p ** 2 + q ** 2))
        

        ang_vel_vec = csdl.Variable(shape=(self.num_nodes, 3), value=0.)
        ang_vel_vec = ang_vel_vec.set(csdl.slice[:, 0], p)
        ang_vel_vec = ang_vel_vec.set(csdl.slice[:, 1], q)
        ang_vel_vec = ang_vel_vec.set(csdl.slice[:, 2], r)


        angvel_ssym = csdl.Variable(shape=(self.num_nodes, 3, 3), value=0.)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 0, 1], -r)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 0, 2], q)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 1, 0], r)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 1, 2], -p)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 2, 0], -q)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 2, 1], p)


        Rbc_ssym = csdl.Variable(shape=(self.num_nodes, 3, 3), value=0.)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 0, 1], -Rbcz)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 0, 2], Rbcy)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 1, 0], Rbcz)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 1, 2], -Rbcx)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 2, 0], -Rbcy)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 2, 1], Rbcx)


        mu_vec = csdl.Variable(shape=(self.num_nodes, 3), value=0.)
        for i in csdl.frange(self.num_nodes):
            t1 = csdl.matvec(Idot, ang_vel_vec[i, :])
            
            var_1 = csdl.matmat(angvel_ssym[i, :, :], inertia_tensor)

            var_2 = csdl.matvec(var_1, ang_vel_vec[i, :])

            var_3 = csdl.matmat(angvel_ssym[i, :, :], Rbc_ssym[i, :, :])

            var_4 = csdl.matvec(var_3, ang_vel_vec[i, :])

            var_5 = m * var_4

            mu_vec = mu_vec.set(
                slices=csdl.slice[i, :],
                value=total_moments[i, :] - t1 - var_2 - var_5,
            )

        
        # Assemble the right hand side vector
        rhs = csdl.Variable(shape=(self.num_nodes, 6), value=0.)
        rhs = rhs.set(csdl.slice[:, 0], lambda_x)
        rhs = rhs.set(csdl.slice[:, 1], lambda_y)
        rhs = rhs.set(csdl.slice[:, 2], lambda_z)
        rhs = rhs.set(csdl.slice[:, 3], mu_vec[:, 0])
        rhs = rhs.set(csdl.slice[:, 4], mu_vec[:, 1])
        rhs = rhs.set(csdl.slice[:, 5], mu_vec[:, 2])

        # Initialize the state vector (acceleration) and the residual
        state = csdl.ImplicitVariable(shape=(6, self.num_nodes), value=0.)
        residual = mp_matrix @ state - rhs.T()

        accel = csdl.solve_linear(mp_matrix, rhs.T())

        lin_and_ang_accel = accel.T()

        lin_and_ang_accel_output = LinAngAccel(
            du_dt=lin_and_ang_accel[:, 0],
            dv_dt=lin_and_ang_accel[:, 1],
            dw_dt=lin_and_ang_accel[:, 2],
            dp_dt=lin_and_ang_accel[:, 3],
            dq_dt=lin_and_ang_accel[:, 4],
            dr_dt=lin_and_ang_accel[:, 5],
            accel_norm=csdl.norm(
                lin_and_ang_accel,
                axes=(1, ),
            )
        )

        return lin_and_ang_accel_output
    






        
    
