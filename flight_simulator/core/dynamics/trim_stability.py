import time
import csdl_alpha as csdl
from dataclasses import dataclass


@dataclass
class LinearStabilityMetrics:
    A_mat_longitudinal : csdl.Variable
    real_eig_short_period : csdl.Variable
    imag_eig_short_period : csdl.Variable
    nat_freq_short_period : csdl.Variable
    damping_ratio_short_period : csdl.Variable
    time_2_double_short_period : csdl.Variable

    real_eig_phugoid : csdl.Variable
    imag_eig_phugoid : csdl.Variable
    nat_freq_phugoid : csdl.Variable
    damping_ratio_phugoid : csdl.Variable
    time_2_double_phugoid : csdl.Variable

    A_mat_lateral_directional : csdl.Variable
    real_eig_spiral : csdl.Variable
    imag_eig_spiral : csdl.Variable
    nat_freq_spiral : csdl.Variable
    damping_ratio_spiral : csdl.Variable
    time_2_double_spiral : csdl.Variable

    real_eig_dutch_roll : csdl.Variable
    imag_eig_dutch_roll : csdl.Variable
    nat_freq_dutch_roll : csdl.Variable
    damping_ratio_dutch_roll : csdl.Variable
    time_2_double_dutch_roll : csdl.Variable

    
