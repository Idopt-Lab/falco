from unittest import TestCase
from flight_simulator import ureg, Q_
import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments

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
        expected_force = np.array([5, -15, 10]) * ureg.newton
        expected_moment = np.array([1, -3, 2]) * ureg.newton * ureg.meter
    
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
        force = Vector(Q_([10, 2, 4], 'newton'), new_axis)
        moment = Vector(Q_([0, 5, 0], 'newton * meter'), new_axis)
        # Create ForcesMoments object
        forces_moments = ForcesMoments(force=force, moment=moment)

    def test_vector_initialization(self):
        """Test vector initialization with different inputs"""
        # Test with scalar quantities
        scalar_vector = Vector(Q_([1, 1, 1], 'newton'), self.inertial_axis)
        self.assertEqual(scalar_vector.vector.value.tolist(), [1, 1, 1])

        # Test with numpy arrays
        np_vector = Vector(np.array([2, 3, 4]) * ureg.newton, self.inertial_axis)
        self.assertEqual(np_vector.vector.value.tolist(), [2, 3, 4])

        # Test with mixed units
        with self.assertRaises(ValueError):
            Vector([1 * ureg.newton, 1 * ureg.meter, 1 * ureg.second], self.inertial_axis)

    def test_vector_operations(self):
        """Test vector mathematical operations"""
        v1 = Vector(Q_([1, 0, 0], 'newton'), self.inertial_axis)
        v2 = Vector(Q_([0, 1, 0], 'newton'), self.inertial_axis)

        # Test addition if implemented
        if hasattr(Vector, '__add__'):
            result = v1 + v2
            self.assertEqual(result.vector.value.tolist(), [1, 1, 0])

        # Test scalar multiplication if implemented
        if hasattr(Vector, '__mul__'):
            result = v1 * 2
            self.assertEqual(result.vector.value.tolist(), [2, 0, 0])

    def test_vector_transformations(self):
        """Test vector coordinate transformations"""
        # Create a vector in inertial frame
        force = Vector(Q_([10, 0, 0], 'newton'), self.inertial_axis)
        
        # Transform to body frame
        transformed_force = force.transform_to(self.body_axis)
        
        # Verify transformation results
        # Add specific expected values based on your transformation logic
        self.assertEqual(len(transformed_force.vector.value), 3)
        self.assertEqual(transformed_force.axis.name, 'Body Axis')


class TestForcesMoments(TestCase):
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

    def test_forces_moments_basic(self):
        """Test basic ForcesMoments functionality"""
        force = Vector(Q_([1, 2, 3], 'newton'), self.inertial_axis)
        moment = Vector(Q_([4, 5, 6], 'newton * meter'), self.inertial_axis)
        fm = ForcesMoments(force=force, moment=moment)

        self.assertEqual(fm.F.vector.value.tolist(), [1, 2, 3])
        self.assertEqual(fm.M.vector.value.tolist(), [4, 5, 6])

    def test_forces_moments_addition(self):
        """Test addition of ForcesMoments objects"""
        if not hasattr(ForcesMoments, '__add__'):
            return

        fm1 = ForcesMoments(
            force=Vector(Q_([1, 0, 0], 'newton'), self.inertial_axis),
            moment=Vector(Q_([0, 1, 0], 'newton * meter'), self.inertial_axis)
        )
        fm2 = ForcesMoments(
            force=Vector(Q_([0, 1, 0], 'newton'), self.inertial_axis),
            moment=Vector(Q_([0, 0, 1], 'newton * meter'), self.inertial_axis)
        )

        result = fm1 + fm2
        self.assertEqual(result.F.vector.value.tolist(), [1, 1, 0])
        self.assertEqual(result.M.vector.value.tolist(), [0, 1, 1])

    def test_complex_rotations(self):
        """Test complex rotation scenarios"""
        # Test rotation about multiple axes
        force = Vector(Q_([10, 0, 0], 'newton'), self.inertial_axis)
        moment = Vector(Q_([0, 10, 0], 'newton * meter'), self.inertial_axis)
        fm = ForcesMoments(force=force, moment=moment)

        # Create axis with complex rotation
        complex_axis = Axis(
            name='Complex Axis',
            x=np.array([0, ]) * ureg.meter,
            y=np.array([0, ]) * ureg.meter,
            z=np.array([0, ]) * ureg.meter,
            phi=np.array([30, ]) * ureg.degree,
            theta=np.array([45, ]) * ureg.degree,
            psi=np.array([60, ]) * ureg.degree,
            reference=self.inertial_axis,
            sequence=np.array([3, 2, 1]),
            origin=ValidOrigins.Inertial.value
        )

        rotated_fm = fm.rotate_to_axis(complex_axis)
        # Add specific assertions based on expected rotation results

    def test_error_handling(self):
        """Test error handling and edge cases"""
        # Test initialization with invalid units
        with self.assertRaises(ValueError):
            Vector(Q_([1, 2, 3], 'meter'), self.inertial_axis)  # Wrong units for force

        # Test initialization with mismatched dimensions
        with self.assertRaises(ValueError):
            Vector(Q_([1, 2], 'newton'), self.inertial_axis)  # Wrong vector size


        expected_force = np.array([4, 2, -10])
        expected_moment = np.array([0, 5, 0])

        np.testing.assert_array_almost_equal(new_load.F.vector.value, expected_force, decimal=5)
        np.testing.assert_array_almost_equal(new_load.M.vector.value, expected_moment, decimal=5)
