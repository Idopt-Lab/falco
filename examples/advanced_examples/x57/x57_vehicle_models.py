import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
from flight_simulator import REPO_ROOT_FOLDER, Q_, ureg
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from flight_simulator.core.loads.mass_properties import MassProperties, MassMI
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.vehicle.components.wing import Wing as WingComp
from flight_simulator.core.vehicle.components.fuselage import Fuselage as FuseComp
from flight_simulator.core.vehicle.components.aircraft import Aircraft as AircraftComp
from flight_simulator.core.vehicle.components.rotor import Rotor as RotorComp
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
from typing import List
from flight_simulator.core.vehicle.controls.vehicle_control_system import VehicleControlSystem, ControlSurface, PropulsiveControl
from typing import Union
from scipy.interpolate import Akima1DInterpolator
from flight_simulator.core.loads.loads import Loads



import sys
from flight_simulator import REPO_ROOT_FOLDER
x57_folder_path = REPO_ROOT_FOLDER / 'examples' / 'advanced_examples' / 'x57'
sys.path.append(str(x57_folder_path))


from x57_geometry import get_geometry

debug = False


do_geo_param = False


## AXIS/AXISLSDOGEO CREATION



def axes_create():
    geo = get_geometry()

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    # OpenVSP Model Axis
    openvsp_axis = Axis(
        name='OpenVSP Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        origin=ValidOrigins.OpenVSP.value

    )
    
    wing_axis = AxisLsdoGeo(
        name='Wing Axis',
        geometry=geo['wing'],
        parametric_coords=geo['wing_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    left_flap_axis = AxisLsdoGeo(
        name='Left Flap Axis',
        geometry=geo['flapL'],
        parametric_coords=geo['left_flap_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_flap_axis = AxisLsdoGeo(
        name='Right Flap Axis',
        geometry=geo['flapR'],
        parametric_coords=geo['right_flap_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    left_aileron_axis = AxisLsdoGeo(
        name='Left Aileron Axis',
        geometry=geo['aileronL'],
        parametric_coords=geo['left_aileron_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    right_aileron_axis = AxisLsdoGeo(
        name='Right Aileron Axis',
        geometry=geo['aileronR'],
        parametric_coords=geo['right_aileron_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Tail Region Axis

    ht_tail_axis = AxisLsdoGeo(
        name='Horizontal Tail Axis',
        geometry=geo['h_tail'],
        parametric_coords=geo['ht_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    trimTab_axis = AxisLsdoGeo(
        name='Trim Tab Axis',
        geometry=geo['trimTab'],
        parametric_coords=geo['trimTab_le_center_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=ht_tail_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    vt_tail_axis = AxisLsdoGeo(
        name='Vertical Tail Axis',
        geometry=geo['vertTail'],
        parametric_coords=geo['vt_le_mid_parametric'],
        sequence=np.array([3, 2, 1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    rudder_axis = AxisLsdoGeo(
        name= 'Rudder Axis',
        geometry=geo['rudder'],
        parametric_coords=geo['rudder_le_mid_parametric'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    ## Distributed Propulsion Motors Axes

    M1_axis = AxisLsdoGeo(
        name= 'Motor 1 Axis',
        geometry=geo['spinner1'],
        parametric_coords=geo['M1_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    M2_axis = AxisLsdoGeo(
        name= 'Motor 2 Axis',
        geometry=geo['spinner2'],
        parametric_coords=geo['M2_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    M3_axis = AxisLsdoGeo(
        name= 'Motor 3 Axis',
        geometry=geo['spinner3'],
        parametric_coords=geo['M3_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M4_axis = AxisLsdoGeo(
        name= 'Motor 4 Axis',
        geometry=geo['spinner4'],
        parametric_coords=geo['M4_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M5_axis = AxisLsdoGeo(
        name= 'Motor 5 Axis',
        geometry=geo['spinner5'],
        parametric_coords=geo['M5_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M6_axis = AxisLsdoGeo(
        name= 'Motor 6 Axis',
        geometry=geo['spinner6'],
        parametric_coords=geo['M6_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M7_axis = AxisLsdoGeo(
        name= 'Motor 7 Axis',
        geometry=geo['spinner7'],
        parametric_coords=geo['M7_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M8_axis = AxisLsdoGeo(
        name= 'Motor 8 Axis',
        geometry=geo['spinner8'],
        parametric_coords=geo['M8_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M9_axis = AxisLsdoGeo(
        name= 'Motor 9 Axis',
        geometry=geo['spinner9'],
        parametric_coords=geo['M9_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M10_axis = AxisLsdoGeo(
        name= 'Motor 10 Axis',
        geometry=geo['spinner10'],
        parametric_coords=geo['M10_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M11_axis = AxisLsdoGeo(
        name= 'Motor 11 Axis',
        geometry=geo['spinner11'],
        parametric_coords=geo['M11_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    M12_axis = AxisLsdoGeo(
        name= 'Motor 12 Axis',
        geometry=geo['spinner12'],
        parametric_coords=geo['M12_disk_on_wing'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )


    HL_motor_axes = [M1_axis, M2_axis, M3_axis, M4_axis, M5_axis, M6_axis, M7_axis, M8_axis, M9_axis, M10_axis, M11_axis, M12_axis]
    # Cruise Motor Region


    cruise_motor1_axis = AxisLsdoGeo(
        name= 'Cruise Motor 1 Axis',
        geometry=geo['cruise_spinner1'],
        parametric_coords=geo['cruise_motor1_tip_parametric'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    cruise_motor2_axis = AxisLsdoGeo(
        name= 'Cruise Motor 2 Axis',
        geometry=geo['cruise_spinner2'],
        parametric_coords=geo['cruise_motor2_tip_parametric'],
        sequence=np.array([3,2,1]),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        reference=openvsp_axis,
        origin=ValidOrigins.OpenVSP.value
    )

    cruise_motor_axes = [cruise_motor1_axis, cruise_motor2_axis]


    fd_axis = Axis(
        name='Flight Dynamics Body Fixed Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    wind_axis = Axis(
        name='Wind Axis',
        x=Q_(0, 'm'),
        y=Q_(0, 'm'),
        z=Q_(0, 'm'),
        phi=Q_(0, 'deg'),
        theta=Q_(0, 'deg'),
        psi=Q_(0, 'deg'),
        sequence=np.array([3, 2, 1]),
        reference=inertial_axis,
        origin=ValidOrigins.Inertial.value
    )

    return openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis, geo, left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis, rudder_axis

openvsp_axis, wing_axis, ht_tail_axis, trimTab_axis, vt_tail_axis, HL_motor_axes, cruise_motor_axes, inertial_axis, fd_axis, wind_axis,geo,left_flap_axis, right_flap_axis, left_aileron_axis, right_aileron_axis, rudder_axis = axes_create()





## Aircraft Component Creation

parameterization_solver = ParameterizationSolver()
ffd_geometric_variables = GeometricVariables()


geometry = geo['geometry']

Aircraft = AircraftComp(geometry=geometry, compute_surface_area_flag=False, 
                        parameterization_solver=parameterization_solver,
                        ffd_geometric_variables=ffd_geometric_variables)



Fuselage = FuseComp(
    length=csdl.Variable(name="fuselage_length", shape=(1, ), value=8.2242552),
    max_height=csdl.Variable(name="fuselage_max_height", shape=(1, ), value=1.09),
    max_width=csdl.Variable(name="fuselage_max_width", shape=(1, ), value=1.24070602),
    geometry=geo['fuselage'], skip_ffd=False, 
    parameterization_solver=parameterization_solver,
    ffd_geometric_variables=ffd_geometric_variables)


Aircraft.add_subcomponent(Fuselage)






wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=15)
wing_span = csdl.Variable(name="wing_span", shape=(1, ), value=9.6)
wing_sweep = csdl.Variable(name="wing_sweep", shape=(1, ), value=0)
wing_dihedral = csdl.Variable(name="wing_dihedral", shape=(1, ), value=0)

Wing = WingComp(AR=wing_AR,
                span=wing_span,
                sweep=wing_sweep,
                dihedral=wing_dihedral,
                geometry=geo['wingALL'],
                parametric_geometry=geo['wing_parametric_geometry'],
                tight_fit_ffd=False, 
                orientation='horizontal', 
                name='Wing', parameterization_solver=parameterization_solver, 
                ffd_geometric_variables=ffd_geometric_variables
                )


Aircraft.add_subcomponent(Wing)
wing_qc_fuse_connection = geometry.evaluate(geo['wing_qc_center_parametric']) - geometry.evaluate(geo['fuselage_wing_qc_center_parametric'])
parameterization_solver.add_variable(computed_value=wing_qc_fuse_connection, desired_value=wing_qc_fuse_connection.value)

LeftAileron = Component(name='Left Aileron')
LeftAileron.parameters.actuate_angle = csdl.Variable(name="left_aileron_actuate_angle", shape=(1,), value=np.deg2rad(15))
Aircraft.add_subcomponent(LeftAileron)


RightAileron = Component(name='Right Aileron')
RightAileron.parameters.actuate_angle = csdl.Variable(name="right_aileron_actuate_angle", shape=(1,), value=np.deg2rad(15))
Aircraft.add_subcomponent(RightAileron)


LeftFlap = Component(name='Left Flap')
LeftFlap.parameters.actuate_angle = csdl.Variable(name="left_flap_actuate_angle", shape=(1,), value=np.deg2rad(10))
Aircraft.add_subcomponent(LeftFlap)

RightFlap = Component(name='Right Flap')
RightFlap.parameters.actuate_angle = csdl.Variable(name="right_flap_actuate_angle", shape=(1,), value=np.deg2rad(10))
Aircraft.add_subcomponent(RightFlap)


HorTailArea = geo['ht_span']*geo['ht_chord']
htAR = geo['ht_span']**2/HorTailArea
HorTail_AR = csdl.Variable(name="HT_AR", shape=(1, ), value=4)
HT_span = csdl.Variable(name="HT_span", shape=(1, ), value=3.14986972)
HT_sweep = csdl.Variable(name="HT_sweep", shape=(1, ), value=0)

HorTail = WingComp(AR=HorTail_AR, span=HT_span, sweep=HT_sweep,
                   geometry=geo['htALL'], parametric_geometry=geo['ht_parametric_geometry'],
                   tight_fit_ffd=False, skip_ffd=False,
                   name='Elevator', orientation='horizontal', 
                   parameterization_solver=parameterization_solver,
                   ffd_geometric_variables=ffd_geometric_variables)
Aircraft.add_subcomponent(HorTail)

HorTail.parameters.actuate_angle = csdl.Variable(name="elevator_actuate_angle", shape=(1,), value=np.deg2rad(0))


tail_moment_arm_computed = csdl.norm(geometry.evaluate(geo['ht_qc_center_parametric']) - geometry.evaluate(geo['wing_qc_center_parametric']))
h_tail_fuselage_connection = geometry.evaluate(geo['ht_te_center_parametric']) - geometry.evaluate(geo['fuselage_tail_te_center_parametric'])
# parameterization_solver.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm_computed.value)
parameterization_solver.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)



vt_AR= csdl.Variable(name="VT_AR", shape=(1, ), value=1.998)
VT_span = csdl.Variable(name="VT_span", shape=(1, ), value=1.6191965383361169)
VT_sweep = csdl.Variable(name="VT_sweep", shape=(1, ), value=-30)


VertTail = WingComp(AR=vt_AR, span=VT_span, sweep=VT_sweep,
                    geometry=geo['vtALL'], parametric_geometry=geo['vt_parametric_geometry'],
                    tight_fit_ffd=False, 
                    name='Vertical Tail', orientation='vertical',
                    parameterization_solver=parameterization_solver,
                    ffd_geometric_variables=ffd_geometric_variables)
Aircraft.add_subcomponent(VertTail)

Rudder = Component(name='Rudder')
Rudder.parameters.actuate_angle = csdl.Variable(name="Rudder Actuate Angle", shape=(1,), value=np.deg2rad(0))
Aircraft.add_subcomponent(Rudder)

vtail_fuselage_connection = geometry.evaluate(geo['fuselage_rear_pts_parametric']) - geometry.evaluate(geo['vt_qc_base_parametric'])
parameterization_solver.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

lift_rotors = []
for i in range(1, 13):
    HL_motor = RotorComp(name=f'HL Motor {i}',radius=geo['MotorDisks'][i-1])
    lift_rotors.append(HL_motor)
    Aircraft.add_subcomponent(HL_motor)

cruise_motors = []
for i in range(1, 3):
    cruise_motor = RotorComp(name=f'Cruise Motor {i}',radius=geo['cruise_motors_base'][i-1])
    cruise_motors.append(cruise_motor)
    Aircraft.add_subcomponent(cruise_motor)

Battery = Component(name='Battery')
Aircraft.add_subcomponent(Battery)

LandingGear = Component(name='Landing Gear')
Aircraft.add_subcomponent(LandingGear)

if do_geo_param is True:
    parameterization_solver.evaluate(ffd_geometric_variables)
    geometry.plot(camera=dict(pos=(12, 15, -12),  # Camera position 
                            focal_point=(-Fuselage.parameters.length.value/2, 0, 0),  # Point camera looks at
                            viewup=(0, 0, -1)),    # Camera up direction
                            title= f'X-57 Maxwell Aircraft Geometry\nWing Span: {Wing.parameters.span.value[0]:.2f} m\nWing AR: {Wing.parameters.AR.value[0]:.2f}\nWing Area S: {Wing.parameters.S_ref.value[0]:.2f} m^2\nWing Sweep: {Wing.parameters.sweep.value[0]:.2f} deg',
                            #  title=f'X-57 Maxwell Aircraft Geometry\nFuselage Length: {Fuselage.parameters.length.value[0]:.2f} m\nFuselage Height: {Fuselage.parameters.max_height.value[0]:.2f} m\nFuselage Width: {Fuselage.parameters.max_width.value[0]:.2f} m',
                            screenshot= REPO_ROOT_FOLDER / 'examples'/ 'advanced_examples' / 'Joeys_X57'/ 'images' / f'x_57_{Wing.parameters.span.value[0]}_AR_{Wing.parameters.AR.value[0]}_S_ref_{Wing.parameters.S_ref.value[0]}_sweep_{Wing.parameters.sweep.value[0]}.png')






### FORCES AND MOMENTS MODELLING


x_57_states = AircraftStates(axis=fd_axis,u=Q_(120, 'mph')) # stall speed
x_57_mi = MassMI(axis=fd_axis,
                 Ixx=Q_(4314.08, 'kg*(m*m)'),
                 Ixy=Q_(-232.85, 'kg*(m*m)'),
                 Ixz=Q_(-2563.29, 'kg*(m*m)'),
                 Iyy=Q_(18656.93, 'kg*(m*m)'),
                 Iyz=Q_(-62.42, 'kg*(m*m)'),
                 Izz=Q_(22340.21, 'kg*(m*m)'),
                 )



x57_mass_properties = MassProperties(mass=Q_(1360.77, 'kg'),
                                      inertia=x_57_mi,
                                      cg=Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis))

atmospheric_states = x_57_states.atmospheric_states






class X57ControlSystem(VehicleControlSystem):

    def __init__(self, engine_count: int, symmetrical: bool = True)-> None:
        self.symmetrical = symmetrical
        self.elevator = ControlSurface(name='Elevator',lb=-26, ub=28)
        
        if symmetrical:
            self._init_symmetrical_controls()
        else:
            self._init_asymmetrical_controls()

        self.rudder = ControlSurface(name='Rudder',lb=-15, ub=15)
        self.engines = self._init_engines(engine_count)
        self.u = self._assemble_control_vector()


        if symmetrical:
            super().__init__(
                pitch_control=[self.elevator],
                roll_control=[self.aileron],
                yaw_control=[self.rudder],
                throttle_control=[self.engines[0]]
            )
        else:
            super().__init__(
                pitch_control=[self.elevator],
                roll_control=[self.left_aileron, self.right_aileron],
                yaw_control=[self.rudder],
                throttle_control=self.engines
            )


    def _init_symmetrical_controls(self) -> None:
        """Initialize controls for a symmetrical configuration."""
        self.aileron = ControlSurface(name='Aileron', lb=-15, ub=20)
        self.flap = ControlSurface(name='Flap', lb=-15, ub=20)
        self.left_aileron = self.aileron
        self.right_aileron = self.aileron
        self.left_flap = self.flap
        self.right_flap = self.flap
    
    def _init_asymmetrical_controls(self) -> None:
        """Initialize controls for an asymmetrical configuration."""
        self.left_aileron = ControlSurface(name='Left Aileron', lb=-15, ub=20)
        self.right_aileron = ControlSurface(name='Right Aileron', lb=-15, ub=20)
        self.left_flap = ControlSurface(name='Left Flap', lb=-15, ub=20)
        self.right_flap = ControlSurface(name='Right Flap', lb=-15, ub=20)
    

    def _init_engines(self, engine_count: int) -> List[PropulsiveControl]:
        """
        Initialize the propulsion controls.

        Returns:
            List[PropulsiveControl]: List of propulsive control instances.
        """
        return [
            PropulsiveControl(name=f'Motor{i+1}', throttle=1.0)
            for i in range(engine_count)
        ]
        


    def _assemble_control_vector(self):
        """
        Assemble the control vector using the deflections of the surfaces.
        
        Returns:
            The concatenated control vector.
        """
        controls = (
            self.left_aileron.deflection,
            self.right_aileron.deflection,
            self.left_flap.deflection,
            self.right_flap.deflection,
            self.elevator.deflection,
            self.rudder.deflection
        )
        # Use all engine throttle values for control vector
        engine_controls = tuple(engine.throttle for engine in self.engines)
        return csdl.concatenate(controls + engine_controls, axis=0)
        
    
    @property
    def control_order(self)-> List[str]:
        return ['roll', 'pitch', 'yaw', 'throttle']


    @property
    def min_values(self):
        return np.array([
            self.left_aileron.min_value,
            self.right_aileron.min_value,
            self.left_flap.min_value,
            self.right_flap.min_value,
            self.elevator.min_value,
            self.rudder.min_value
        ] + [engine.min_value for engine in self.engines])
    

    @property
    def max_values(self):
        return np.array([
            self.left_aileron.max_value,
            self.right_aileron.max_value,
            self.left_flap.max_value,
            self.right_flap.max_value,
            self.elevator.max_value,
            self.rudder.max_value
        ] + [engine.max_value for engine in self.engines])

x57_controls = X57ControlSystem(engine_count=14,symmetrical=True)
x57_controls.elevator.deflection = HorTail.parameters.actuate_angle
x57_controls.rudder.deflection = Rudder.parameters.actuate_angle
x57_controls.aileron.deflection = LeftAileron.parameters.actuate_angle
x57_controls.flap.deflection = LeftFlap.parameters.actuate_angle

x57_aircraft = Component(name='X-57')
x57_aircraft.mass_properties = x57_mass_properties
HL_radius_x57 = csdl.Variable(name="high_lift_motor_radius",shape=(1,), value=1.89/2) # HL propeller radius in ft
cruise_radius_x57 = csdl.Variable(name="cruise_lift_motor_radius",shape=(1,), value=5/2) # cruise propeller radius in ft

e_x57 = csdl.Variable(name="wing_e",shape=(1,), value=0.87) # Oswald efficiency factor
CD0_x57 = csdl.Variable(name="wing_CD0",shape=(1,), value=0.001) # Zero-lift drag coefficient
Wing.parameters.actuate_angle = csdl.Variable(name="wing_incidence",shape=(1,), value=np.deg2rad(2)) # Wing incidence angle in radians


## Aerodynamic Forces - from Modification IV



# TODO: IMPROVE AERODYNAMIC MODEL TO INCLUDE MORE COMPLEX AERODYNAMIC EFFECTS

class LiftModel:
    def __init__(self, AR:Union[ureg.Quantity, csdl.Variable], e:Union[ureg.Quantity, csdl.Variable], CD0:Union[ureg.Quantity, csdl.Variable], 
                 S:Union[ureg.Quantity, csdl.Variable], incidence:Union[ureg.Quantity, csdl.Variable]):
        super().__init__()


        if AR is None:
            self.AR = csdl.Variable(name='AR', shape=(1,), value=15)
        else:
            self.AR = AR
        

        if e is None:
            self.e = csdl.Variable(name='e', shape=(1,), value=0.87)
        else:
            self.e = e

        if CD0 is None:
            self.CD0 = csdl.Variable(name='CD0', shape=(1,), value=0.001)
        else:
            self.CD0 = CD0

        if S is None:
            self.S = csdl.Variable(name='S', shape=(1,), value=6.22)
        elif isinstance(S, ureg.Quantity):
            self.S = csdl.Variable(name='S', shape=(1,), value=S.to_base_units())
        else:
            self.S = S

        if incidence is None:
            self.incidence = csdl.Variable(name='incidence', shape=(1,), value=2*np.pi/180)
        elif isinstance(incidence, ureg.Quantity):
            self.incidence = csdl.Variable(name='incidence', shape=(1,), value=incidence.to_base_units())
        else:
            self.incidence = incidence
        
    


class AircraftAerodynamics(Loads):

    # TODO: Improve aerodynamic model to include more complex aerodynamic effects

    def __init__(self, component, lift_model:LiftModel):
        self.lift_model = lift_model
        self.wing_axis = component.mass_properties.cg_vector.axis
        

    def get_FM_localAxis(self, states, controls):
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
            density = states.atmospheric_states.density
            velocity = states.VTAS
            theta = states.state_vector.theta
            alpha = theta + self.lift_model.incidence

            CL = 2*np.pi*alpha + 0.3
            CD = self.lift_model.CD0 + (1 / (self.lift_model.e * self.lift_model.AR * np.pi)) * CL**2  
            L = 0.5 * density * velocity**2 * self.lift_model.S * CL
            D = 0.5 * density * velocity**2 * self.lift_model.S * CD

            aero_force = csdl.Variable(shape=(3,), value=0.)
            aero_force = aero_force.set(csdl.slice[0], -D)
            aero_force = aero_force.set(csdl.slice[2], -L)
            force_vector = Vector(vector=aero_force, axis=self.wing_axis)

            moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=self.wing_axis)
            loads = ForcesMoments(force=force_vector, moment=moment_vector)
            return loads
    



lift_models = []
for comp in Aircraft.comps.values():
    if isinstance(comp, WingComp):
        # Build the lift model with the wing's parameters
        lift_model = LiftModel(
            AR=comp.parameters.AR,
            e=e_x57,
            CD0=CD0_x57,
            S=comp.parameters.S_ref,
            incidence=comp.parameters.actuate_angle,
        )
        lift_models.append(lift_model)







class HLPropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        V_inf_data = np.array(
            [0,61.3,87.6,101.6,113.8,131.3,157.6,175])
        RPM_data = np.array(
        [0,3545,4661,4702,4379,3962,3428,3451])
        self.rpm = Akima1DInterpolator(V_inf_data, RPM_data, method="akima")
        self.rpm_derivative = Akima1DInterpolator.derivative(self.rpm)
        self.min_RPM = min(RPM_data)
        self.max_RPM = max(RPM_data)
        # Obtained Mod-IV Propeller Data from CFD database
        J_data = np.array(
            [0,0.5490,0.5966,0.6860,0.8250,1.0521,1.4595,1.6098])
        Ct_data = np.array(
            [0,0.3125,0.3058,0.2848,0.2473,0.1788,0.0366,-0.0198])
        self.ct = Akima1DInterpolator(J_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)


    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, advance_ratio: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('advance_ratio', advance_ratio)

        # declare output variables
        ct = self.create_output('ct', advance_ratio.shape)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.ct = ct

        return outputs
    

    def evaluate_rpm(self, velocity: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('velocity', velocity)

        # declare output variables
        rpm = self.create_output('rpm', velocity.shape)

        if hasattr(velocity, 'value') and (velocity.value is not None):
            rpm.set_value(self.rpm(velocity.value))

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.rpm = rpm

        return outputs
    
    def compute(self, input_vals, output_vals):
        advance_ratio = input_vals['advance_ratio']
        output_vals['ct'] = self.ct(advance_ratio)

    def compute_rpm(self, input_vals, output_vals):
        velocity = input_vals['velocity']
        output_vals['rpm'] = self.rpm(velocity)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        advance_ratio = input_vals['advance_ratio']
        derivatives['ct', 'advance_ratio'] = np.diag(self.ct_derivative(advance_ratio))

    def compute_rpm_derivatives(self, input_vals, outputs_vals, derivatives):
        velocity = input_vals['velocity']
        derivatives['rpm', 'velocity'] = np.diag(self.rpm_derivative(velocity))



class CruisePropCurve(csdl.CustomExplicitOperation):

    def __init__(self):
        super().__init__()

        # Obtained Mod-III Propeller Data from CFD database
        V_inf_data = np.array(
            [0,18.75,75,112.5,150,187.5,225,243.75,262.5,266.67,300])
        RPM_data = np.array(
        [0,2250,2250,2250,2250,2250,2250,2250,2250,2000,2000])
        self.rpm = Akima1DInterpolator(V_inf_data, RPM_data, method="akima")
        self.rpm_derivative = Akima1DInterpolator.derivative(self.rpm)
        self.min_RPM = min(RPM_data)
        self.max_RPM = max(RPM_data)
        J_data = np.array(
            [0,0.1,0.4,0.6,0.8,1.0,1.2,1.3,1.4,1.6,1.8])
        Ct_data = np.array(
            [0,0.1831,0.1673,0.1422,0.1003,0.0479,-0.0085,-0.0366,-0.0057,0.0030,-0.0504])
        self.ct = Akima1DInterpolator(J_data, Ct_data, method="akima")
        self.ct_derivative = Akima1DInterpolator.derivative(self.ct)


    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, advance_ratio: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('advance_ratio', advance_ratio)

        # declare output variables
        ct = self.create_output('ct', advance_ratio.shape)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.ct = ct

        return outputs
    

    def evaluate_rpm(self, velocity: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('velocity', velocity)

        # declare output variables
        rpm = self.create_output('rpm', velocity.shape)
    
        if hasattr(velocity, 'value') and (velocity.value is not None):
            rpm.set_value(self.rpm(velocity.value))

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.rpm = rpm

        return outputs
    
    def compute(self, input_vals, output_vals):
        advance_ratio = input_vals['advance_ratio']
        output_vals['ct'] = self.ct(advance_ratio)

    def compute_rpm(self, input_vals, output_vals):
        velocity = input_vals['velocity']
        output_vals['rpm'] = self.rpm(velocity)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        advance_ratio = input_vals['advance_ratio']
        derivatives['ct', 'advance_ratio'] = np.diag(self.ct_derivative(advance_ratio))

    def compute_rpm_derivatives(self, input_vals, outputs_vals, derivatives):
        velocity = input_vals['velocity']
        derivatives['rpm', 'velocity'] = np.diag(self.rpm_derivative(velocity))



class AircraftPropulsion(Loads):

    def __init__(self, component, radius:Union[ureg.Quantity, csdl.Variable], prop_curve:Union[HLPropCurve, CruisePropCurve], **kwargs):
        self.prop_curve = prop_curve
        self.prop_axis = component.mass_properties.cg_vector.axis

        if radius is None:
            self.radius = csdl.Variable(name='radius', shape=(1,), value=1.89/2) 
        elif isinstance(radius, ureg.Quantity):
            self.radius = csdl.Variable(name='radius', shape=(1,), value=radius.to_base_units())
        else:
            self.radius = radius



    def get_FM_localAxis(self, states, controls):
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
            - throttle

        Returns
        -------
        loads : ForcesMoments
            Computed forces and moments about the reference point.
        """
        throttle = controls.u[6]
        density = states.atmospheric_states.density * 0.00194032  # kg/m^3 to slugs/ft^3
        velocity = states.VTAS * 3.281  # m/s to ft/s

        # Compute RPM
        rpm_curve = type(self.prop_curve)() 
        rpm = rpm_curve.evaluate_rpm(velocity=velocity).rpm * throttle
        omega_RAD = (rpm * 2 * np.pi) / 60.0  # rad/s


        # Compute advance ratio
        J = (np.pi * velocity) / (omega_RAD * self.radius)  # non-dimensional

        # Compute Ct
        ct_curve = type(self.prop_curve)()
        ct = ct_curve.evaluate(advance_ratio=J).ct

        # Compute Thrust
        T = ct * density * (rpm/60)**2 * ((self.radius*2)**4) * 4.44822 # lbf to N
    
        

        force_vector = Vector(vector=csdl.concatenate((T,
                                                       csdl.Variable(shape=(1,), value=0.),
                                                       csdl.Variable(shape=(1,), value=0.)),
                                                      axis=0), axis=self.prop_axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=self.prop_axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)
        return loads

   
    def get_power(self, rpm, cp, states):
        """
        Compute power required for the propulsion system.

        Parameters
        ----------
        cp : 
            Power coefficient (C_p) which should include:
            - cp
        
        rpm :
            Rotational speed of the propeller (RPM) which should include:
            - rpm

        x_bar : csdl.VariableGroup
            Flight-dynamic state (x̄) which should include:
            - density
            - VTAS
            - states.theta

        Returns
        -------
        power : csdl.VariableGroup
            Computed power required for the propulsion system.
        """
        density = states.atmospheric_states.density * 0.00194032

        power = (cp * density * (rpm/60)**3 * (self.radius*2)**5)/737.56  # should be kW

        return power

    def get_torque(self, rpm, power, states):
        """
        Compute torque required for the propulsion system.

        Parameters
        ----------
        power : 
            Power required for the propulsion system (P) which should include:
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
            Computed torque required for the propulsion system.
        """
        torque = (power * 737.56) / ((rpm/60) * 2 * np.pi) # should be lbf-ft

        density = states.atmospheric_states.density * 0.00194032


        torque_coeff = torque/(density*(rpm/60)**2*(self.radius*2)**5)

        return torque
        

 
Wing.mass_properties.mass = Q_(152.88, 'kg')
Wing.mass_properties.cg_vector = Vector(vector=Q_(geo['wing_le_center'].value, 'm'), axis=wing_axis)
Wing_aerodynamics = AircraftAerodynamics(lift_model=lift_models[0], component=Wing)
Wing.load_solvers.append(Wing_aerodynamics)


LeftAileron.mass_properties.mass = Q_(1, 'kg')
LeftAileron.mass_properties.cg_vector = Vector(vector=Q_(geo['left_aileron_le_center'].value, 'm'), axis=left_aileron_axis)

RightAileron.mass_properties.mass = Q_(1, 'kg')
RightAileron.mass_properties.cg_vector = Vector(vector=Q_(geo['right_aileron_le_center'].value, 'm'), axis=right_aileron_axis)

LeftFlap.parameters.actuate_angle = csdl.Variable(name="left_flap_actuate_angle", shape=(1, ), value=0)  
LeftFlap.mass_properties.mass = Q_(1, 'kg')
LeftFlap.mass_properties.cg_vector = Vector(vector=Q_(geo['left_flap_le_center'].value, 'm'), axis=left_flap_axis)

RightFlap.parameters.actuate_angle = csdl.Variable(name="right_flap_actuate_angle", shape=(1, ), value=0)
RightFlap.mass_properties.mass = Q_(1, 'kg')
RightFlap.mass_properties.cg_vector = Vector(vector=Q_(geo['right_flap_le_center'].value, 'm'), axis=right_flap_axis)


Fuselage.mass_properties.mass = Q_(235.87, 'kg')
Fuselage.mass_properties.cg_vector =  Vector(vector=Q_(geo['wing_le_center'].value + np.array([0,0,1]), 'm'), axis=wing_axis) # cg is at the wing cg but shifted down 


HorTail.mass_properties.mass = Q_(27.3/2, 'kg')
HorTail.mass_properties.cg_vector = Vector(vector=Q_(geo['ht_le_center'].value, 'm'), axis=ht_tail_axis)
HorTail_aerodynamics = AircraftAerodynamics(lift_model=lift_models[1], component=HorTail)
HorTail.load_solvers.append(HorTail_aerodynamics)


VertTail.mass_properties.mass = Q_(27.3/2, 'kg')
VertTail.mass_properties.cg_vector = Vector(vector=Q_(geo['vt_le_mid'].value, 'm'), axis=vt_tail_axis)
VertTail_aerodynamics = AircraftAerodynamics(lift_model=lift_models[2], component=VertTail)
VertTail.load_solvers.append(VertTail_aerodynamics)


Rudder.mass_properties.mass = Q_(1, 'kg')
Rudder.mass_properties.cg_vector = Vector(vector=Q_(geo['rudder_le_mid'].value, 'm'), axis=rudder_axis)

Battery.mass_properties.mass = Q_(390.08, 'kg')
Battery.mass_properties.cg_vector = Vector(vector=Q_(geo['wing_le_center'].value + np.array([0,0,2]), 'm'), axis=wing_axis)


LandingGear.mass_properties.mass = Q_(61.15, 'kg')
LandingGear.mass_properties.cg_vector = Vector(vector=Q_(geo['wing_le_center'].value + np.array([0,0,2]), 'm'), axis=wing_axis)


for i, HL_motor in enumerate(lift_rotors):
    HL_motor.mass_properties.mass = Q_(81.65/12, 'kg')
    HL_motor.mass_properties.cg_vector = Vector(vector=Q_(geo['MotorDisks'][i].value, 'm'), axis=HL_motor_axes[i])
    HL_motor_propulsion = AircraftPropulsion(radius=HL_radius_x57, prop_curve=HLPropCurve(), component=HL_motor)
    HL_motor.load_solvers.append(HL_motor_propulsion)


for i, cruise_motor in enumerate(cruise_motors):
    cruise_motor.mass_properties.mass = Q_(106.14/2, 'kg')
    cruise_motor.mass_properties.cg_vector = Vector(vector=Q_(geo['cruise_motors_base'][i].value, 'm'), axis=cruise_motor_axes[i])
    cruise_motor_propulsion = AircraftPropulsion(radius=cruise_radius_x57, prop_curve=CruisePropCurve(), component=cruise_motor)
    cruise_motor.load_solvers.append(cruise_motor_propulsion)


Aircraft.mass_properties.mass = Q_(0, 'kg')
Aircraft.mass_properties.cg_vector = Vector(vector=Q_(np.array([0, 0, 0]), 'm'), axis=fd_axis)
Aircraft.mass_properties = Aircraft.compute_mass_properties()
print(repr(Aircraft))


