from unittest import TestCase

from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments


class TestVector(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        # Setup a mock axis for testing
        self.axis = Axis(
            name='Inertial Axis',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value,
        )

    def test_vector_initialization_with_quantity(self):
        # Create a Quantity vector with units
        vector_quantity = Q_([10, 20, 30], 'newton')
        vector = Vector(vector_quantity, self.axis)

        # Check if vector was initialized with correct values and units
        self.assertEqual(vector.vector.shape, (3,))
        np.testing.assert_array_equal(vector.vector.value, [10, 20, 30])
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')
        self.assertEqual(vector.axis.name, "Inertial Axis")

    def test_vector_initialization_with_nonSI_quantity(self):
        # Create a Quantity vector with units
        vector_quantity = Q_([0, 100, 0], 'lbf')
        vector = Vector(vector_quantity, self.axis)

        # Check if vector was initialized with correct values and units
        self.assertEqual(vector.vector.shape, (3,))
        np.testing.assert_almost_equal(vector.vector.value, [0, 100*4.44822162, 0], decimal=5)
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')
        self.assertEqual(vector.axis.name, "Inertial Axis")

    def test_vector_initialization_with_variable(self):
        # Create a csdl.Variable with a 'newton' tag
        vector_variable = csdl.Variable(shape=(3,), value=np.array([10, 20, 30]))
        vector_variable.add_tag('kilogram * meter / second ** 2')
        vector = Vector(vector_variable, self.axis)

        # Check if vector was initialized with correct values and tags
        np.testing.assert_array_equal(vector.vector.value, [10, 20, 30])
        self.assertEqual(vector.vector.tags[0], 'kilogram * meter / second ** 2')

    def test_vector_magnitude(self):
        vector_quantity = Q_([3, 4, 0], 'newton')
        vector = Vector(vector_quantity, self.axis)
        np.testing.assert_array_equal(vector.magnitude.value, np.array([5, ]))

    def test_vector_str(self):
        vector_quantity = Q_([10, 20, 30], 'newton')
        vector = Vector(vector_quantity, self.axis)
        expected_output = "Vector: [10. 20. 30.] \nUnit: kilogram * meter / second ** 2 \nAxis: Inertial Axis"
        self.assertEqual(str(vector), expected_output)


class ForcesMomentsWithoutGeometryTests(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        self.inertial_axis = Axis(
            name='Inertial Axis',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,            
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        self.fd_axis = Axis(
            name='Flight Dynamics Body Fixed Axis',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([5000, ]) * ureg.meter,            
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            sequence=np.array([3, 2, 1]),
            reference=self.inertial_axis,
            origin=ValidOrigins.Inertial.value,
        )
        
        self.axis1 = Axis(
            name='Axis 1',
            x=np.array([5, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            sequence=np.array([3, 2, 1]),
            reference=self.fd_axis,
            origin=ValidOrigins.Inertial.value,
        )
        
        self.axis2 = Axis(
            name='Axis 2',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([5, ]) * ureg.meter,
            z=np.array([-2, ]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            sequence=np.array([3, 2, 1]),
            reference=self.fd_axis,
            origin=ValidOrigins.Inertial.value,
        )

    def test_forces_moments_initialization(self):
        # Initialize Force and Moment Vectors
        force = Vector(Q_([5, 10, 15], 'newton'), self.inertial_axis)
        moment = Vector(Q_([1, 2, 3], 'newton * meter'), self.inertial_axis)
        # Create ForcesMoments object
        forces_moments = ForcesMoments(force=force, moment=moment)

        # Check if forces and moments have the correct axis and vector values
        self.assertEqual(forces_moments.F.vector.value.tolist(), [5, 10, 15])
        self.assertEqual(forces_moments.M.vector.value.tolist(), [1, 2, 3])
        self.assertEqual(forces_moments.axis.name, "Inertial Axis")

    def test_rotating_a_vector_1(self):
         # Setup another axis
        new_axis = Axis(
            name='New Axis 1',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([90, ]) * ureg.degree,
            reference=self.inertial_axis,
            sequence=np.array([3, 2, 1]),
            origin=ValidOrigins.Inertial.value,
        )
    
        # Initialize Force and Moment Vectors
        force = Vector(Q_([5, 10, 15], 'newton'), new_axis)
        moment = Vector(Q_([1, 2, 3], 'newton * meter'), new_axis)
        # Create ForcesMoments object
        forces_moments = ForcesMoments(force=force, moment=moment)

    
        # Perform rotation
        new_load = forces_moments.rotate_to_axis(self.inertial_axis)
    
        # Expected values after 90-degree rotation about Z-axis
        expected_force = np.array([-10, 5, 15]) * ureg.newton
        expected_moment = np.array([-2, 1, 3]) * ureg.newton * ureg.meter
    
        np.testing.assert_array_almost_equal(new_load.F.vector.value, expected_force, decimal=5)
        np.testing.assert_array_almost_equal(new_load.M.vector.value, expected_moment, decimal=5)


    def test_rotating_a_vector_2(self):
        # Setup another axis
        new_axis = Axis(
            name='New Axis 2',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,            
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([90, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            reference=self.inertial_axis,
            sequence=np.array([3, 2, 1]),
            origin=ValidOrigins.Inertial.value,
        )

        # Initialize Force and Moment Vectors
        force = Vector(Q_([10, 0, 0], 'newton'), new_axis)
        moment = Vector(Q_([0, 5, 0], 'newton * meter'), new_axis)
        # Create ForcesMoments object
        forces_moments = ForcesMoments(force=force, moment=moment)

        # Perform rotation
        new_load = forces_moments.rotate_to_axis(self.inertial_axis)

        # Expected values after 90-degree rotation about the Y-axis
        # The initial force vector [10, 0, 0] should become [0, 0, -10]
        # The initial moment vector [0, 5, 0] should remain [0, 5, 0]
        expected_force = np.array([0, 0, -10])
        expected_moment = np.array([0, 5, 0])

        np.testing.assert_array_almost_equal(new_load.F.vector.value, expected_force, decimal=5)
        np.testing.assert_array_almost_equal(new_load.M.vector.value, expected_moment, decimal=5)

    def test_loads_transfer_1(self):
        force_vector_1 = Vector(vector=np.array([0, 0, 10]) * ureg.newton, axis=self.axis1)
        moment_vector_1 = Vector(vector=np.array([0, 0, 0]) * ureg.newton*ureg.meter, axis=self.axis1)
        load1 = ForcesMoments(force=force_vector_1, moment=moment_vector_1)
    
        new_load = load1.rotate_to_axis(parent_or_child_axis=self.fd_axis)
        print(new_load.F.vector.value)
        print(new_load.M.vector.value)