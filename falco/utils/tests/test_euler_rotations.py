import unittest
import numpy as np
import csdl_alpha as csdl
from flight_simulator.utils.euler_rotations import build_rotation_matrix

class TestEulerRotations(unittest.TestCase):

    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_build_rotation_matrix(self):

        angles = csdl.Variable(shape=(3,), value=np.array([0., np.pi/2, 0.]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)

        expected_R = np.array([[0., 0., 1.],
                               [0., 1., 0.],
                               [-1., 0., 0.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)
    
    def test_build_rotation_matrix_2(self):

        angles = csdl.Variable(shape=(3,), value=np.array([0., 0., np.pi/2]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)
        expected_R = np.array([[1., 0., 0.],
                               [0., 0., -1.],
                               [0., 1., 0.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)

    def test_build_rotation_matrix_3(self):

        angles = csdl.Variable(shape=(3,), value=np.array([np.pi/2, 0., 0.]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)
        expected_R = np.array([[0., -1., 0.],
                               [1., 0., 0.],
                               [0., 0, 1.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)

    def test_build_rotation_matrix_4(self):

        angles = csdl.Variable(shape=(3,), value=np.array([np.pi/4, np.pi/2, np.pi/6]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)
        expected_R = np.array([[0., -0.258819045102521, 0.965925826289068],
                               [0., 0.965925826289068, 0.258819045102521],
                               [-1., 0., 0.]])
        np.testing.assert_array_almost_equal(R.value, expected_R)

    def test_derivative_of_rotation_matrix(self):
        angles = csdl.Variable(shape=(3,), value=np.array([np.pi / 4, np.pi / 3, np.pi / 6]))
        seq = np.array([3, 2, 1])
        R = build_rotation_matrix(angles, seq)

        dRda = csdl.derivative(R, angles)

        np.testing.assert_array_almost_equal(dRda[csdl.slice[6, 1]].value, -np.cos(np.pi / 3))
        np.testing.assert_array_almost_equal(dRda[csdl.slice[0, 0]].value, -np.sin(np.pi/4)*np.cos(np.pi/3))
