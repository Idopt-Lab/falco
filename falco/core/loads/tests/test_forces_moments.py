import unittest
import numpy as np
import csdl_alpha as csdl
from falco import ureg, Q_
from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.loads.forces_moments import ForcesMoments
from falco.core.dynamics.vector import Vector


class TestForcesMoments(unittest.TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        # simple inertial axis
        self.inertial = Axis(
            name='Inertial',
            origin=ValidOrigins.Inertial.value
        )

    def test_require_same_axis(self):
        a1 = Axis(name='A1', origin=ValidOrigins.Inertial.value)
        a2 = Axis(name='A2', origin=ValidOrigins.Inertial.value)

        F = Vector(vector=np.array([1., 0., 0.]) * ureg.newton, axis=a1)
        M = Vector(vector=np.array([0., 0., 0.]) * ureg.newton, axis=a2)

        with self.assertRaises(AssertionError):
            ForcesMoments(force=F, moment=M)

    def test_rotate_to_axis_identity_preserves_values_and_tags(self):
        # zero Euler angles -> identity rotation
        euler = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))
        seq = np.array([3, 2, 1])

        F = csdl.Variable(shape=(3,), value=np.array([1., 2., 3.]), tags=['N'])
        M = csdl.Variable(shape=(3,), value=np.array([4., 5., 6.]), tags=['N*m'])

        F_rot, M_rot = ForcesMoments.rotate_to_axis(F, M, euler, seq, reverse=False)

        np.testing.assert_array_almost_equal(F_rot.value, F.value)
        np.testing.assert_array_almost_equal(M_rot.value, M.value)
        # tags preserved
        self.assertIn('N', F_rot.tags)
        self.assertIn('N*m', M_rot.tags)

    def test_translate_to_axis_translates_moment_correctly(self):
        r = csdl.Variable(shape=(3,), value=np.array([4., -2., 2.])) # translation vector
        F = csdl.Variable(shape=(3,), value=np.array([4., 8, 4.]))
        M_local = csdl.Variable(shape=(3,), value=np.array([0, 0, 0]), tags=['N*m'])  # local moment when located at an axis is 0

        F_out, M_out = ForcesMoments.translate_to_axis(F, M_local, r, reverse=False)

        np.testing.assert_array_almost_equal(F_out.value, F.value)  # force unchanged
        np.testing.assert_array_almost_equal(M_out.value, np.array([-24, -8, 40]))  # expected cross result
        self.assertIn('N*m', M_out.tags)

    def test_translate_to_axis_reverse_negates_r(self):
        F = csdl.Variable(shape=(3,), value=np.array([0., 1., 0.]), tags=['N'])
        M = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]), tags=['N*m'])
        r = csdl.Variable(shape=(3,), value=np.array([1., 0., 0.]), tags=['m'])

        # reverse=True should use -r -> cross(-r, F) = -cross(r, F) = [0,0,-1]
        F_out, M_out = ForcesMoments.translate_to_axis(F, M, r, reverse=True)

        np.testing.assert_array_almost_equal(M_out.value, np.array([0., 0., -1.]))

    def test_transform_to_axis_parent_to_child(self):
        parent = Axis(
            name='Parent Axis Test',
            origin=ValidOrigins.Inertial.value
        )
        child = Axis(
            name='Child Axis Test',
            x=Q_(-1, 'm'),
            y=Q_(0, 'm'),
            z=Q_(0, 'm'),  
            phi=Q_(0, 'deg'),
            theta=Q_(0, 'deg'),
            psi=Q_(0, 'deg'),
            sequence=np.array([3, 2, 1]),
            reference=parent,
            origin=ValidOrigins.Inertial.value
        )

        F_vec = Vector(vector=np.array([0., 1., 0.]) * ureg.newton, axis=parent)
        M_vec = Vector(vector=np.array([0., 0., 0.]) * ureg.newton, axis=parent)

        loads = ForcesMoments(force=F_vec, moment=M_vec)

        # translate_flag=True, rotate_flag=False so only translation is applied
        new_loads = loads.transform_to_axis(child, translate_flag=True, rotate_flag=False, reverse_flag=False)

        # expected moment = original + cross(r, F) where r = child.translation ([-1,0,0])
        # cross([-1,0,0], [0,1,0]) = [0,0,-1]
        expected_M = np.array([0., 0., -1.])
        np.testing.assert_array_almost_equal(new_loads.M.vector.value, expected_M)
        # force unchanged
        np.testing.assert_array_almost_equal(new_loads.F.vector.value, F_vec.vector.value)
        self.assertEqual(new_loads.axis.name, child.name)

    def test_transform_to_axis_child_to_parent(self):
        parent = Axis(
            name='Parent Axis Test',
            origin=ValidOrigins.Inertial.value
        )
        child = Axis(
            name='Child Axis Test',
            x=Q_(1, 'm'),
            y=Q_(0, 'm'),
            z=Q_(0, 'm'),
            phi=Q_(0, 'deg'),
            theta=Q_(0, 'deg'),
            psi=Q_(0, 'deg'),
            sequence=np.array([3, 2, 1]),
            reference=parent,
            origin=ValidOrigins.Inertial.value
        )

        F_vec = Vector(vector=np.array([0., 1., 0.]) * ureg.newton, axis=child)
        M_vec = Vector(vector=np.array([0., 0., 0.]) * ureg.newton, axis=child)

        loads = ForcesMoments(force=F_vec, moment=M_vec)

        # translate_flag=True, rotate_flag=False so only translation is applied
        new_loads = loads.transform_to_axis(parent, translate_flag=True, rotate_flag=False, reverse_flag=False)

        # expected moment = original + cross(r, F) where r = -child.translation = [-1,0,0] 
        # cross([-1,0,0], [0,1,0]) = [0,0,-1]
        expected_M = np.array([0., 0., -1.])
        np.testing.assert_array_almost_equal(new_loads.M.vector.value, expected_M)
        # force unchanged
        np.testing.assert_array_almost_equal(new_loads.F.vector.value, F_vec.vector.value)
        self.assertEqual(new_loads.axis.name, parent.name)

    def test_transform_to_axis_parent_with_rotation(self):
        parent = Axis(
            name='Parent Axis Test',
            origin=ValidOrigins.Inertial.value
        )
        child = Axis(
            name='Child Axis Test',
            x=Q_(1, 'm'),
            y=Q_(0, 'm'),
            z=Q_(0, 'm'),
            phi=Q_(0, 'deg'),
            theta=Q_(90, 'deg'),
            psi=Q_(0, 'deg'),
            sequence=np.array([3, 2, 1]),
            reference=parent,
            origin=ValidOrigins.Inertial.value
        )

        F_vec = Vector(vector=np.array([1., 0., 0.]) * ureg.newton, axis=parent)
        M_vec = Vector(vector=np.array([0., 0., 0.]) * ureg.newton, axis=parent)

        loads = ForcesMoments(force=F_vec, moment=M_vec)

        new_loads = loads.transform_to_axis(child)

        # compute rotation matrix from child's Euler angles (sequence Z, Y, X -> 3,2,1)
        phi = child.euler_angles.phi.value 
        theta = child.euler_angles.theta.value 
        psi = child.euler_angles.psi.value 

        # extract scalars (phi/theta/psi are 1-element arrays)
        phi = float(np.asarray(phi).item())
        theta = float(np.asarray(theta).item())
        psi = float(np.asarray(psi).item())

        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(phi), -np.sin(phi)],
                       [0.0, np.sin(phi),  np.cos(phi)]])
        Ry = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                       [0.0,            1.0, 0.0],
                       [-np.sin(theta), 0.0, np.cos(theta)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                       [np.sin(psi),  np.cos(psi), 0.0],
                       [0.0,          0.0,         1.0]])
        R = Rz @ Ry @ Rx

        F_parent = F_vec.vector.value
        M_parent = M_vec.vector.value

        expected_F = np.dot(R.T, F_parent)
        x_trans = float(np.asarray(child.translation_from_origin.x.value).item())
        y_trans = float(np.asarray(child.translation_from_origin.y.value).item())
        z_trans = float(np.asarray(child.translation_from_origin.z.value).item())
        r = np.array([x_trans, y_trans, z_trans])
        expected_M = np.dot(R.T, M_parent) + np.cross(r, expected_F)
        np.testing.assert_array_almost_equal(new_loads.M.vector.value, expected_M)
        self.assertEqual(new_loads.axis.name, child.name)

if __name__ == "__main__":
    unittest.main()