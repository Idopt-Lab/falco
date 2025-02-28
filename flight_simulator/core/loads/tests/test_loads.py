from unittest import TestCase

from sympy.physics.paulialgebra import delta

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.loads.loads import CsdlLoads, NonCsdlLoads


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


