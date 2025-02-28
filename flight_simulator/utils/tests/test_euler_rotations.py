import unittest
import numpy as np
import csdl_alpha as csdl
from flight_simulator.utils.euler_rotations import build_rotation_matrix

class TestEulerRotations(unittest.TestCase):
    def test_build_rotation_matrix(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        angles = csdl.Variable(shape=(3,), value=np.array([0., np.pi/2, 0.]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)

        expected_R = np.array([[0., 0., 1.],
                               [0., 1., 0.],
                               [-1., 0., 0.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)
    
    def test_build_rotation_matrix_2(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        angles = csdl.Variable(shape=(3,), value=np.array([0., 0., np.pi/2]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)
        expected_R = np.array([[1., 0., 0.],
                               [0., 0., -1.],
                               [0., 1., 0.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)

    def test_build_rotation_matrix_3(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        angles = csdl.Variable(shape=(3,), value=np.array([np.pi/2, 0., 0.]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)
        expected_R = np.array([[0., -1., 0.],
                               [1., 0., 0.],
                               [0., 0, 1.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)

    def test_build_rotation_matrix_4(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        angles = csdl.Variable(shape=(3,), value=np.array([np.pi/4, np.pi/2, np.pi/6]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)
        expected_R = np.array([[0., -0.258819045102521, 0.965925826289068],
                               [0., 0.965925826289068, 0.258819045102521],
                               [-1., 0., 0.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)





