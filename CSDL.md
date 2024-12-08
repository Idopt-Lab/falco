# CSDL Packages

## Discilpinary Packages

### Beam model
Simple Euler-Bernoulli beam model for structural analysis.
It is restricted to Tube cross-sections as of 8 Decemeber 2024.

[Beam](https://github.com/nichco/pyframe/tree/main/pyframe)

### VLM
Vortex lattice method for aerodynamic analysis.

[VLM](https://github.com/lscotzni/VortexAD_temp)

### BEM
Blade Element Momentum theory for rotor analysis.

[VLM](https://github.com/lscotzni/VortexAD_temp)

### Stability
Linear stability analysis for a 6 DoF model. 
This is written in an old version of CSDL.

[Stability](https://github.com/nichco/tc1-stability/tree/main/tc1_stability)

### Motor Model
ECM model for a motor. 
It is written in old CSDL. 
I recall this code being a mess.

[ECM motor](https://github.com/lscotzni/TC1_motor_model_solver/tree/master)

### Enforcing spatial constraints
Ryan's MS thesis code.
Also in old CSDL.

[spatial-constraints](https://github.com/RyanDunn729/spatial-constraints/tree/main)


### Weight Regression
[eVToL weight regressions](https://github.com/MariusLRuh/lift_plus_cruise_weights/blob/main/lift_plus_cruise_weights/core/components/gross_weight_regression_model.py)

## Aircraft Design Studies

### Drone
DARPA Drone design study.
This appears to be in a old version of CSDL but might still be useful for data.

[DARPA Drone](https://github.com/nichco/darpa/tree/main)

### Trajectory optimization
This is quite a detailed example. 
It uses surrogates for aero and propulsion. Has models for rotor noise. Uses a 3 DoD EoM.
Shows how to use OZONE for time-marching.
Written in new CSDL.

[L+C trajectory opt](https://github.com/LSDOlab/ozone_alpha/tree/main/ozone_alpha/paper_examples/trajectory_optimization)

