
import csdl_alpha as csdl
import numpy as np
from typing import Union
from scipy.interpolate import Akima1DInterpolator
from falco.core.loads.loads import Loads
import os
import scipy.io as sio
from falco.core.loads.forces_moments import Vector, ForcesMoments
from falco import Q_, ureg

from pint import UnitRegistry
from pathlib import Path

from julia import Julia, Main
jl = Julia(compiled_modules=False)



# Get the path to the GitHub folder (five levels up from this file)
ROOT_FOLDER = Path(__file__).resolve().parents[4]
# CCBlade.jl root as a sibling to Sarojini_Research
ccblade_root = ROOT_FOLDER / "CCBlade.jl"
# Path to the airfoil data file
ccblade_data_path = ccblade_root / "data" / "naca4412.dat"


# --- Julia/CCBlade.jl integration ---

Main.eval('using Pkg')
Main.eval('Pkg.activate("{ccblade_root}")')
Main.eval('using CCBlade')


# Example function to call CCBlade.jl from Python
def run_ccblade_example():
# Replace with actual CCBlade.jl function call and arguments
    result = Main.eval('CCBlade.some_function()')
    return result


# --- CCBlade.jl full example ---
def run_ccblade_full_example():
    airfoil_path = str(ccblade_data_path).replace("\\", "/")
    julia_code = f'''
using CCBlade
using PyPlot
Rtip = 10/2.0 * 0.0254  # inches to meters
Rhub = 0.10*Rtip
B = 2  # number of blades

rotor = Rotor(Rhub, Rtip, B)
propgeom = [
0.15   0.130   32.76
0.20   0.149   37.19
0.25   0.173   33.54
0.30   0.189   29.25
0.35   0.197   25.64
0.40   0.201   22.54
0.45   0.200   20.27
0.50   0.194   18.46
0.55   0.186   17.05
0.60   0.174   15.97
0.65   0.160   14.87
0.70   0.145   14.09
0.75   0.128   13.39
0.80   0.112   12.84
0.85   0.096   12.25
0.90   0.081   11.37
0.95   0.061   10.19
1.00   0.041   8.99
]

r = propgeom[:, 1] * Rtip
chord = propgeom[:, 2] * Rtip
theta = propgeom[:, 3] * pi/180
af = AlphaAF("{airfoil_path}")
sections = Section.(r, chord, theta, Ref(af))
Vinf = 5.0
Omega = 5400*pi/30  # convert to rad/s
rho = 1.225

op = simple_op.(Vinf, Omega, r, rho)
out = solve.(Ref(rotor), sections, op)
alpha_deg = out.alpha*180/pi  # angle of attack in degrees

figure()
plot(r/Rtip, out.Np)
plot(r/Rtip, out.Tp)
xlabel("r/Rtip")
ylabel("distributed loads (N/m)")
legend(["flapwise", "lead-lag"])
show()

figure()
plot(r/Rtip, out.u/Vinf)
plot(r/Rtip, out.v/Vinf)
xlabel("r/Rtip")
ylabel("(normalized) induced velocity at rotor disk")
legend(["axial velocity", "swirl velocity"])
show()

nJ = 20  # number of advance ratios

J = range(0.1, 0.6, length=nJ)  # advance ratio

Omega = 5400.0*pi/30
n = Omega/(2*pi)
D = 2*Rtip

eff = zeros(nJ)
CT = zeros(nJ)
CQ = zeros(nJ)

for i = 1:nJ
local Vinf = J[i] * D * n

local op = simple_op.(Vinf, Omega, r, rho)
outputs = solve.(Ref(rotor), sections, op)
T, Q = thrusttorque(rotor, sections, outputs)
eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, rotor, "propeller")

end

exp = [
0.113   0.0912   0.0381   0.271
0.145   0.0890   0.0386   0.335
0.174   0.0864   0.0389   0.387
0.200   0.0834   0.0389   0.429
0.233   0.0786   0.0387   0.474
0.260   0.0734   0.0378   0.505
0.291   0.0662   0.0360   0.536
0.316   0.0612   0.0347   0.557
0.346   0.0543   0.0323   0.580
0.375   0.0489   0.0305   0.603
0.401   0.0451   0.0291   0.620
0.432   0.0401   0.0272   0.635
0.466   0.0345   0.0250   0.644
0.493   0.0297   0.0229   0.640
0.519   0.0254   0.0210   0.630
0.548   0.0204   0.0188   0.595
0.581   0.0145   0.0162   0.520
]
Jexp = exp[:, 1]
CTexp = exp[:, 2]
CPexp = exp[:, 3]
etaexp = exp[:, 4]

figure()
plot(J, CT)
plot(J, CQ*2*pi)
plot(Jexp, CTexp, "ko")
plot(Jexp, CPexp, "ko")
xlabel(L"J")
legend([L"C_T", L"C_P", "experimental"])
show()

figure()
plot(J, eff)
plot(Jexp, etaexp, "ko")
xlabel(L"J")
ylabel(L"\eta")
legend(["CCBlade", "experimental"])
show()
'''  # End Julia code
# Run the Julia code
    Main.eval(julia_code)
# Retrieve alpha_deg for inspection in Python
    alpha_deg = Main.eval('out.alpha*180/pi')
    return alpha_deg




if __name__ == "__main__":
    alpha_deg = run_ccblade_full_example()
    print("Angle of attack (deg):", alpha_deg)