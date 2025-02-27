from unittest import TestCase
import numpy as np
import csdl_alpha as csdl
from flight_simulator.utils.euler_rotations import build_rotation_matrix


class TestEulerRotations(TestCase):
    def test_build_rotation_matrix_identity(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()        
        angles = csdl.Variable(shape=(3,), value=np.zeros(3))
        seq = np.array([1, 2, 3])
        R = build_rotation_matrix(angles, seq)
        print(R.value)
        expected = np.identity(3)
        np.testing.assert_array_almost_equal(R.value, expected)
        recorder.stop()

    def test_build_rotation_matrix_90_deg_x(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start() 
        angles = csdl.Variable(shape=(3,), value=np.array([np.pi/2, 0, 0]))
        seq = np.array([1, 2, 3])
        R = build_rotation_matrix(angles, seq)
        print("R.value:", R.value)
        expected = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])
        np.testing.assert_array_almost_equal(R.value, expected)
        recorder.stop()

    def test_build_rotation_matrix_90_deg_y(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start() 
        angles = csdl.Variable(shape=(3,), value=np.array([0, np.pi/2, 0]))
        seq = np.array([1, 2, 3])
        R = build_rotation_matrix(angles, seq)
        expected = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [-1, 0, 0]])
        np.testing.assert_array_almost_equal(R.value, expected)
        recorder.stop()

    def test_build_rotation_matrix_90_deg_z(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        angles = csdl.Variable(shape=(3,), value=np.array([0, 0, np.pi/2]))
        seq = np.array([1, 2, 3])
        R = build_rotation_matrix(angles, seq)
        expected = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])
        np.testing.assert_array_almost_equal(R.value, expected)
        recorder.stop()

if __name__ == '__main__':
    import unittest
    unittest.main()
