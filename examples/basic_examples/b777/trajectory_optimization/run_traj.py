import ozone
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem, IPOPT, SLSQP, SNOPT
import matplotlib.pyplot as plt

def get_trajectory_recorder(approach: ozone.approaches._Approach,
                            method: str,
                            num: int = 40) -> csdl.Recorder:
    """
    Build and return a CSDL recorder for trajectory optimization.
    :param approach: Ozone approach to use for trajectory optimization.
    :param method: Ozone method to use for trajectory optimization.
    :param num: Number of discretization points.
    :return: CSDL recorder containing the trajectory optimization model.
    """

    recorder, ode_problem, num = build_recorder(num, approach, method)

    return recorder

if __name__ == '__main__':

    approach = ozone.approaches.Collocation()
    method = ozone.methods.ImplicitMidpoint()
    num = 40
    max_iter = 200

    rec = get_trajectory_recorder(
        approach=approach,
        method=method,
        num=num,
    )

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        gpu=False,
        derivatives_kwargs={'loop': False},
        additional_outputs=[]
        )

    sim.run()

    prob = CSDLAlphaProblem(
        problem_name='trajectory_opt',
        simulator=sim
    )

    optimizer = IPOPT(prob, turn_off_outputs=False, solver_options={'maxiter': max_iter})
    # optimizer = SLSQP(prob, turn_off_outputs=False, solver_options={'maxiter': max_iter})
    # optimizer = SNOPT(prob, turn_off_outputs=False, solver_options={'Major optimality': 2e-4, 'Major feasibility': 1e-4, 'Major iterations': 600, 'Iteration limit': 100000, 'Verbose': False})

    optimizer.solve()
    optimizer.print_results()

    # Plot results
    rec.execute()
