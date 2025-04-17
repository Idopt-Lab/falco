import numpy as np
from scipy.interpolate import Akima1DInterpolator

from flight_simulator import Q_, ureg
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.dynamics.vector import Vector
from flight_simulator.core.loads.forces_moments import ForcesMoments
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.vehicle.controls.vehicle_control_system import (
    VehicleControlSystem, ControlSurface, PropulsiveControl)
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.components.aircraft import Aircraft
from typing import Union
from typing import List
import csdl_alpha as csdl


recorder = csdl.Recorder(inline=True)
recorder.start()

ft2m = 0.3048

# region Axis

# region Inertial Axis
# I am picking the inertial axis location as the OpenVSP (0,0,0)
inertial_axis = Axis(
    name='Inertial Axis',
    origin=ValidOrigins.Inertial.value
)
# endregion

# region Aircraft FD Axis

fd_axis = Axis(
    name='Flight Dynamics Body Fixed Axis',
    x=Q_(0, 'ft'),
    y=Q_(0, 'ft'),
    z=Q_(-12000, 'ft'),  # z is positive down in FD axis
    phi=Q_(0, 'deg'),
    theta=Q_(4, 'deg'),
    psi=Q_(0, 'deg'),
    sequence=np.array([3, 2, 1]),
    reference=inertial_axis,
    origin=ValidOrigins.Inertial.value
)
# endregion

# region Aircraft Wind Axis

wind_axis = Axis(
        name='Wind Axis',
        x=Q_(0, 'ft'),
        y=Q_(0, 'ft'),
        z=Q_(0, 'ft'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=fd_axis,
        origin=ValidOrigins.Inertial.value
    )
# endregion

# region Aircraft Component

# Create a Mass Properties object with given values
x57_mi = MassMI(axis=fd_axis,
                 Ixx=Q_(4314, 'kg*(m*m)'),
                 Iyy=Q_(18657, 'kg*(m*m)'),
                 Izz=Q_(22340, 'kg*(m*m)'),
                 Ixy=Q_(-233, 'kg*(m*m)'),
                 Ixz=Q_(-2563, 'kg*(m*m)'),
                 Iyz=Q_(-62, 'kg*(m*m)'))
x57_mass_properties = MassProperties(mass=Q_(1360.77, 'kg'),
                                      inertia=x57_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))

aircraft_component = Aircraft()
aircraft_component.quantities.mass_properties = x57_mass_properties
# endregion

# region Fuselage Component
fuselage_component = Component(name='Fuselage')

# Horizontal Tail
ht_component = Component(name='Horizontal Tail')
fuselage_component.add_subcomponent(ht_component)
# Elevator
elevator_component = Component(name='Elevator')
elevator_component.parameters.actuate_angle = csdl.Variable(name="Elevator Actuate Angle", shape=(1,), value=np.deg2rad(0))
ht_component.add_subcomponent(elevator_component)

aircraft_component.add_subcomponent(fuselage_component)

# endregion

# region Wing Component
wing_component = Component(name='Wing')
aircraft_component.add_subcomponent(wing_component)


cruise_motor_left_component = Component(name='Cruise Motor Left')
radius_x57_cml = csdl.Variable(name='cmr_radius', shape=(1,), value=2.5*ft2m)
cruise_motor_left_component.parameters.radius = radius_x57_cml
wing_component.add_subcomponent(cruise_motor_left_component)

cruise_motor_right_component = Component(name='Cruise Motor Right')
radius_x57_cmr = csdl.Variable(name='cmr_radius', shape=(1,), value=2.5*ft2m)
cruise_motor_left_component.parameters.radius = radius_x57_cmr
wing_component.add_subcomponent(cruise_motor_right_component)
# endregion

# region Aircraft Controls
class X57Control(VehicleControlSystem):

    def __init__(self, hlp_count) -> None:
        self.elevator = ControlSurface(name='Elevator', lb=-26, ub=28, component=elevator_component)

        self.cml = PropulsiveControl(name='CML', throttle=1)
        self.cmr = PropulsiveControl(name='CMR', throttle=1)

        self.u = csdl.concatenate((self.elevator.deflection,
                                   self.cml.throttle, self.cmr.throttle), axis=0)

    @property
    def lower_bounds(self):
        raise NotImplementedError

    @property
    def upper_bounds(self):
        raise NotImplementedError