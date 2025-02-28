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

        # Perform rotation
        new_load = forces_moments.rotate_to_axis(self.inertial_axis)

        # Expected values after 90-degree rotation about the Y-axis
        # The initial force vector [10, 0, 0] should become [0, 0, -10]
        # The initial moment vector [0, 5, 0] should remain [0, 5, 0]
        expected_force = np.array([4, 2, -10])
        expected_moment = np.array([0, 5, 0])

        np.testing.assert_array_almost_equal(new_load.F.vector.value, expected_force, decimal=5)
        np.testing.assert_array_almost_equal(new_load.M.vector.value, expected_moment, decimal=5)
