import time
import csdl_alpha as csdl
from dataclasses import dataclass


@dataclass
class TrimStabilityMetrics:
    A_mat_long : csdl.Variable
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