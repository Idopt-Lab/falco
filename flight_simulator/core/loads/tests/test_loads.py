from unittest import TestCase

from sympy.physics.paulialgebra import delta

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.loads.loads import CsdlLoads, NonCsdlLoads
from flight_simulator.core.vehicle.components.component import Component
from flight_simulator.core.vehicle.models.aerodynamics.aerodynamic_model import LiftModel
from flight_simulator.core.vehicle.controls.aircraft_control_system import AircraftControlSystem
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
import NRLMSIS2


class TestCsdlLoads(TestCase):
    class DummyStates:
        def __init__(self):
            self.state_vector = csdl.Variable(
                name='aircraft_state',
                shape=(12,),
                value=np.zeros(12)
            )
            self.state_vector = self.state_vector.set(slices=csdl.slice[7, ], value=1.0)

    class DummyControls:
        def __init__(self):
            self.control_vector = csdl.Variable(
                name='aircraft_control',
                shape=(2,),
                value=np.array([0.85, -2.0])
            )

    class DummyLoads(CsdlLoads):
        def __init__(self, states, controls):
            super().__init__(states=states, controls=controls)

        def get_FM_refPoint(self):
            state_vector: csdl.Variable = self.states.state_vector
            control_vector: csdl.Variable = self.controls.control_vector

            alpha = state_vector[7]
            deltae = control_vector[1]

            CLalpha = 10
            CLdeltae = 1

            CL = CLalpha * alpha + CLdeltae * deltae
            L = 0.5*CL
            return L

    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.states = self.DummyStates()
        self.controls = self.DummyControls()
        self.loads = self.DummyLoads(states=self.states, controls=self.controls)

    def test_get_FM_refPoint(self):
        L = self.loads.get_FM_refPoint()
        self.assertIsInstance(L, csdl.Variable)
        self.assertEqual(L.value, 4)

    def test_derivative_of_load(self):
        L = self.loads.get_FM_refPoint()
        state_vector: csdl.Variable = self.states.state_vector
        dydx = csdl.derivative(L, state_vector)
        self.assertEqual(dydx.value.max(), 5)
        


class TestNonCsdlLoads(TestCase):
    class DummyStates:
        def __init__(self):
            self.state_vector = csdl.Variable(
                name='aircraft_state',
                shape=(12,),
                value=np.zeros(12)
            )
            self.state_vector = self.state_vector.set(slices=csdl.slice[7, ], value=1.0)

    class DummyControls:
        def __init__(self):
            self.control_vector = csdl.Variable(
                name='aircraft_control',
                shape=(2,),
                value=np.array([0.85, -2.0])
            )

    # I want a dummy class that produces a mesh of a rectangular wing
    class DummyMesh:
        def __init__(self):
            self.mesh_nodes = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]])

        def compute_area(self):
            # Assuming the mesh is a rectangle and the nodes are ordered
            # Calculate the vectors for two sides of the rectangle
            vec1 = self.mesh_nodes[1] - self.mesh_nodes[0]
            vec2 = self.mesh_nodes[3] - self.mesh_nodes[0]

            # The area of the rectangle is the magnitude of the cross product of vec1 and vec2
            area = np.linalg.norm(np.cross(vec1, vec2))
            return area

    class DummyLoads(NonCsdlLoads):
        def __init__(self, states, controls):
            super().__init__(states=states, controls=controls)

        def compute_loads_as_pint_quantities(self,
                                             state_vector:np.array,
                                             control_vector:np.array,
                                             mesh):

            area = Q_(mesh.compute_area(), 'meter*meter')
            rho = Q_(1, 'kg/meter**3')
            V = Q_(1, 'm/s')


            x = csdl.VariableGroup()
            x.a = 1
            x.b = csdl.Variable(shape=(3,), value=np.array([1, 2, 3]))

            alpha = Q_(state_vector[7], 'rad')
            deltae = Q_(control_vector[1], 'rad')

            CLalpha = Q_(10, '1/rad')
            CLdeltae = Q_(1, '1/rad')

            CL = CLalpha * alpha + CLdeltae * deltae
            L = 0.5*rho*V**2*area*CL
            return L


        def get_FM_refPoint(self, mesh):
            state_vector: csdl.Variable = self.states.state_vector
            control_vector: csdl.Variable = self.controls.control_vector

            L = self.compute_loads_as_pint_quantities(
                state_vector, control_vector,
                mesh=mesh
            )
            return L

    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.states = self.DummyStates()
        self.controls = self.DummyControls()
        self.mesh = self.DummyMesh()
        self.loads = self.DummyLoads(states=self.states, controls=self.controls)

    def test_get_FM_refPoint(self):
        L = self.loads.get_FM_refPoint(mesh=self.mesh)
        self.assertIsInstance(L, ureg.Quantity)
        self.assertIsInstance(L.magnitude, csdl.Variable)
        self.assertEqual(L.magnitude.value, 8)

    def test_get_FM_refPoint_derivative(self):
        L = self.loads.get_FM_refPoint(mesh=self.mesh)
        state_vector: csdl.Variable = self.states.state_vector
        dydx = csdl.derivative(L.magnitude, state_vector)
        self.assertEqual(dydx.value.max(), 10)


class TestInertialLoads(TestCase):
    
    def test_inertial_loads_buildup(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        inertial_axis = Axis(
            name='Inertial Axis',
            origin=ValidOrigins.Inertial.value)

        axis = Axis(
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

        parent_component = Component(name="Aircraft")
        sub_component1 = Component(name="Wing")
        sub_component2 = Component(name="Motor")
        sub_component3 = Component(name="Fuselage")


        atmos_model = NRLMSIS2.Atmosphere()

        # Add subcomponents to the parent component
        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)
        parent_component.add_subcomponent(sub_component3)

        sub_component1.quantities.mass_properties.cg_vector.axis = axis
        sub_component2.quantities.mass_properties.cg_vector.axis = axis
        sub_component3.quantities.mass_properties.cg_vector.axis = axis
        parent_component.quantities.mass_properties.cg_vector.axis = axis
        lift_model = LiftModel(
            AR=csdl.Variable(name='AR', shape=(1,), value=np.array([10,])),
            e=csdl.Variable(name='e', shape=(1,), value=np.array([0.87,])),
            CD0=csdl.Variable(name='CD0', shape=(1,), value=np.array([0.001,])),
            S=csdl.Variable(name='S', shape=(1,), value=np.array([100,])),
            incidence=csdl.Variable(name='incidence', shape=(1,), value=np.deg2rad(-2))
        )

        atmos_states = atmos_model.evaluate(axis.translation_from_origin.z)
        sub_component1.quantities.lift_model = lift_model
        v = w = p = q = r = csdl.Variable(value=0.)

        ac_states = AircraftStates(
            u=csdl.Variable(value=50), v=v, w=w, p=p, q=q, r=r, axis=axis)
        
        controls = AircraftControlSystem(engine_count=1,symmetrical=True)

        force, moment = parent_component.compute_total_loads(fd_state=ac_states, controls=controls)

        alpha = ac_states.states.theta + lift_model.incidence + controls.elevator.deflection

        cl = 2 * np.pi * alpha
        cd = lift_model.CD0 + (1 / (lift_model.e * lift_model.AR * np.pi)) * cl**2  
        l = 0.5 * ac_states.atmospheric_states.density * ac_states.VTAS**2 * lift_model.S * cl
        d = 0.5 * ac_states.atmospheric_states.density * ac_states.VTAS**2 * lift_model.S * cd


        aero_force = csdl.Variable(shape=(3,), value=0.)
        aero_force = aero_force.set(csdl.slice[0], -d)
        aero_force = aero_force.set(csdl.slice[2], -l)
        force_vector = Vector(vector=aero_force, axis=axis)

        moment_vector = Vector(vector=csdl.Variable(shape=(3,), value=0.), axis=axis)
        loads = ForcesMoments(force=force_vector, moment=moment_vector)

        loads_new = loads.rotate_to_axis(axis)

        self.assertEqual(force[0].value, loads_new.F.vector[0].value)
