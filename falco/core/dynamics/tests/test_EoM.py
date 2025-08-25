from unittest import TestCase
import numpy as np
import csdl_alpha as csdl
from falco import ureg, Q_

from falco.core.dynamics.EoM import EquationsOfMotion, DynamicSystem, StateVectorDot
from falco.core.dynamics.aircraft_states import AircraftStates
from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.dynamics.vector import Vector
from falco.core.loads.mass_properties import MassProperties, MassMI


class TestStateVectorDot(TestCase):
    """Test cases for the StateVectorDot dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_state_vector_dot_initialization(self):
        """Test initialization of StateVectorDot with all required variables."""
        # Create test variables for each component
        du_dt = csdl.Variable(shape=(1,), value=1.0, name='du_dt')
        dv_dt = csdl.Variable(shape=(1,), value=2.0, name='dv_dt')
        dw_dt = csdl.Variable(shape=(1,), value=3.0, name='dw_dt')
        dp_dt = csdl.Variable(shape=(1,), value=0.1, name='dp_dt')
        dq_dt = csdl.Variable(shape=(1,), value=0.2, name='dq_dt')
        dr_dt = csdl.Variable(shape=(1,), value=0.3, name='dr_dt')
        dphi_dt = csdl.Variable(shape=(1,), value=0.01, name='dphi_dt')
        dtheta_dt = csdl.Variable(shape=(1,), value=0.02, name='dtheta_dt')
        dpsi_dt = csdl.Variable(shape=(1,), value=0.03, name='dpsi_dt')
        dx_dt = csdl.Variable(shape=(1,), value=10.0, name='dx_dt')
        dy_dt = csdl.Variable(shape=(1,), value=20.0, name='dy_dt')
        dz_dt = csdl.Variable(shape=(1,), value=30.0, name='dz_dt')
        
        # Create StateVectorDot instance
        state_dot = StateVectorDot(
            du_dt=du_dt, dv_dt=dv_dt, dw_dt=dw_dt,
            dp_dt=dp_dt, dq_dt=dq_dt, dr_dt=dr_dt,
            dphi_dt=dphi_dt, dtheta_dt=dtheta_dt, dpsi_dt=dpsi_dt,
            dx_dt=dx_dt, dy_dt=dy_dt, dz_dt=dz_dt
        )
        
        # Verify all attributes are set correctly
        self.assertEqual(state_dot.du_dt.value, 1.0)
        self.assertEqual(state_dot.dv_dt.value, 2.0)
        self.assertEqual(state_dot.dw_dt.value, 3.0)
        self.assertEqual(state_dot.dp_dt.value, 0.1)
        self.assertEqual(state_dot.dq_dt.value, 0.2)
        self.assertEqual(state_dot.dr_dt.value, 0.3)
        self.assertEqual(state_dot.dphi_dt.value, 0.01)
        self.assertEqual(state_dot.dtheta_dt.value, 0.02)
        self.assertEqual(state_dot.dpsi_dt.value, 0.03)
        self.assertEqual(state_dot.dx_dt.value, 10.0)
        self.assertEqual(state_dot.dy_dt.value, 20.0)
        self.assertEqual(state_dot.dz_dt.value, 30.0)


class TestDynamicSystem(TestCase):
    """Test cases for the DynamicSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
    
    def test_dynamic_system_initialization_default(self):
        """Test DynamicSystem initialization with default parameters."""
        # Create initial state vector
        x = csdl.Variable(shape=(12,), value=np.zeros(12), name='state')
        
        # Initialize dynamic system
        system = DynamicSystem(x)
        
        # Check default values
        self.assertEqual(system.time, 0)
        self.assertEqual(system.origin, 'ref')
        np.testing.assert_array_equal(system.state_vector.value, np.zeros(12))
        np.testing.assert_array_equal(system.state_vector_dot.value, np.zeros(12))
    
    def test_dynamic_system_initialization_custom(self):
        """Test DynamicSystem initialization with custom parameters."""
        # Create initial state vector
        initial_state = np.array([1, 2, 3, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 10, 20, 30])
        x = csdl.Variable(shape=(12,), value=initial_state, name='state')
        
        # Initialize dynamic system with custom parameters
        system = DynamicSystem(x, t0=5.0, origin='custom')
        
        # Check custom values
        self.assertEqual(system.time, 5.0)
        self.assertEqual(system.origin, 'custom')
        np.testing.assert_array_equal(system.state_vector.value, initial_state)
        np.testing.assert_array_equal(system.state_vector_dot.value, np.zeros(12))
    
    def test_dynamic_system_state_vector_shapes(self):
        """Test that state vector and its derivative have matching shapes."""
        # Test with different state vector sizes
        for size in [6, 12, 18]:
            x = csdl.Variable(shape=(size,), value=np.zeros(size), name=f'state_{size}')
            system = DynamicSystem(x)
            
            self.assertEqual(system.state_vector.shape, (size,))
            self.assertEqual(system.state_vector_dot.shape, (size,))


class TestEquationsOfMotion(TestCase):
    """Test cases for the EquationsOfMotion class."""
    
    def setUp(self):
        """Set up test fixtures for EoM tests."""
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        
        # Create inertial axis
        self.inertial_axis = Axis(
            name='Inertial Axis',
            origin=ValidOrigins.Inertial.value
        )
        
        # Create body-fixed axis with small Euler angles
        phi = csdl.Variable(shape=(1,), value=np.array([0.0]), name='phi')
        theta = csdl.Variable(shape=(1,), value=np.array([0.0]), name='theta')
        psi = csdl.Variable(shape=(1,), value=np.array([0.0]), name='psi')
        
        self.body_axis = Axis(
            name='Body Axis',
            x=Q_(0, 'm'),
            y=Q_(0, 'm'),
            z=Q_(0, 'm'),
            phi=phi,
            theta=theta,
            psi=psi,
            sequence=np.array([3, 2, 1]),
            reference=self.inertial_axis,
            origin=ValidOrigins.Inertial.value,
        )
        
        # Create aircraft states
        self.aircraft_states = AircraftStates(axis=self.body_axis)
        
        # Create mass properties
        cg_vector = Vector(Q_(np.array([0.0, 0.0, 0.0]), 'm'), self.body_axis)
        inertia = MassMI(
            axis=self.body_axis,
            Ixx=Q_(1000, 'kg*m^2'),
            Iyy=Q_(2000, 'kg*m^2'),
            Izz=Q_(3000, 'kg*m^2'),
            Ixy=Q_(0, 'kg*m^2'),
            Ixz=Q_(0, 'kg*m^2'),
            Iyz=Q_(0, 'kg*m^2')
        )
        self.mass_properties = MassProperties(
            cg=cg_vector,
            inertia=inertia,
            mass=Q_(1000, 'kg')
        )
        
        # Create EoM instance
        self.eom = EquationsOfMotion()
    
    def test_eom_initialization(self):
        """Test EquationsOfMotion class can be instantiated."""
        eom = EquationsOfMotion()
        self.assertIsInstance(eom, EquationsOfMotion)
    
    def test_eom_res_zero_forces_moments(self):
        """Test EoM with zero external forces and moments."""
        # Zero forces and moments
        total_forces = csdl.Variable(shape=(3,), value=np.zeros(3), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
        
        # Call EoM residual function
        dstate_vector, state_vector = self.eom._EoM_res(
            self.aircraft_states, 
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Check that we get valid outputs
        self.assertEqual(dstate_vector.shape, (12,))
        self.assertEqual(state_vector.shape, (12,))
        
        # For zero initial velocities and zero forces, accelerations should be zero
        # (except for gravitational effects which are not included in this test)
        np.testing.assert_array_almost_equal(dstate_vector.value[:6], np.zeros(6), decimal=10)
    
    def test_eom_res_constant_forces(self):
        """Test EoM with constant external forces."""
        # Constant forces
        total_forces = csdl.Variable(shape=(3,), value=np.array([100.0, 0.0, 0.0]), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Check accelerations
        du_dt = dstate_vector[0]
        dv_dt = dstate_vector[1]
        dw_dt = dstate_vector[2]
        
        # For 100N force in x-direction and 1000kg mass, expect 0.1 m/s^2 acceleration
        self.assertAlmostEqual(du_dt.value.item(), 0.1, places=6)
        self.assertAlmostEqual(dv_dt.value.item(), 0.0, places=6)
        self.assertAlmostEqual(dw_dt.value.item(), 0.0, places=6)
    
    def test_eom_res_constant_moments(self):
        """Test EoM with constant external moments."""
        # Constant moments
        total_forces = csdl.Variable(shape=(3,), value=np.zeros(3), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.array([1000.0, 0.0, 0.0]), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Check angular accelerations
        dp_dt = dstate_vector[3]
        dq_dt = dstate_vector[4]
        dr_dt = dstate_vector[5]
        
        # For 1000 N⋅m moment about x-axis and Ixx=1000 kg⋅m², expect 1.0 rad/s² acceleration
        self.assertAlmostEqual(dp_dt.value.item(), 1.0, places=6)
        self.assertAlmostEqual(dq_dt.value.item(), 0.0, places=6)
        self.assertAlmostEqual(dr_dt.value.item(), 0.0, places=6)
    
    def test_eom_res_with_initial_velocities(self):
        """Test EoM with non-zero initial velocities."""
        # Set initial velocities
        self.aircraft_states.states.u.set_value(10.0)  # 10 m/s forward velocity
        self.aircraft_states.states.v.set_value(0.0)
        self.aircraft_states.states.w.set_value(8.0) # 8 m/s upward velocity

        # Zero external forces and moments
        total_forces = csdl.Variable(shape=(3,), value=np.zeros(3), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Check position derivatives (should reflect velocity)
        dx_dt = dstate_vector[9]
        dy_dt = dstate_vector[10]
        dz_dt = dstate_vector[11]
        
        # For zero Euler angles, body x-velocity should translate directly to inertial x-velocity
        self.assertAlmostEqual(dx_dt.value.item(), 10.0, places=6)
        self.assertAlmostEqual(dy_dt.value.item(), 0.0, places=6)
        self.assertAlmostEqual(dz_dt.value.item(), 8.0, places=6)
    
    def test_eom_res_with_angular_rates(self):
        """Test EoM with non-zero angular rates."""
        # Set initial angular rates
        self.aircraft_states.states.p.set_value(0.1)  # 0.1 rad/s roll rate
        self.aircraft_states.states.q.set_value(0.0)
        self.aircraft_states.states.r.set_value(0.0)
        
        # Zero external forces and moments
        total_forces = csdl.Variable(shape=(3,), value=np.zeros(3), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Check Euler angle derivatives
        dphi_dt = dstate_vector[6]
        dtheta_dt = dstate_vector[7]
        dpsi_dt = dstate_vector[8]
        
        # For zero theta and phi, dphi_dt should equal p
        self.assertAlmostEqual(dphi_dt.value.item(), 0.1, places=6)
        self.assertAlmostEqual(dtheta_dt.value.item(), 0.0, places=6)
        self.assertAlmostEqual(dpsi_dt.value.item(), 0.0, places=6)
    
    def test_eom_res_euler_angle_kinematics(self):
        """Test Euler angle kinematic equations."""
        # Set non-zero angular rates and Euler angles
        self.aircraft_states.states.p.set_value(0.1)
        self.aircraft_states.states.q.set_value(0.2)
        self.aircraft_states.states.r.set_value(0.3)
        self.aircraft_states.states.phi.set_value(np.deg2rad(10))
        self.aircraft_states.states.theta.set_value(np.deg2rad(5))
        self.aircraft_states.states.psi.set_value(np.deg2rad(15))
        
        # Zero external forces and moments
        total_forces = csdl.Variable(shape=(3,), value=np.zeros(3), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Extract values for manual calculation
        p, q, r = 0.1, 0.2, 0.3
        phi, theta = np.deg2rad(10), np.deg2rad(5)
        
        # Expected Euler angle derivatives from kinematic equations
        expected_dphi_dt = p + q * np.sin(phi) * np.tan(theta) + r * np.tan(theta) * np.cos(phi)
        expected_dtheta_dt = q * np.cos(phi) - r * np.sin(phi)
        expected_dpsi_dt = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        
        # Check Euler angle derivatives
        dphi_dt = dstate_vector[6]
        dtheta_dt = dstate_vector[7]
        dpsi_dt = dstate_vector[8]
        
        self.assertAlmostEqual(dphi_dt.value.item(), expected_dphi_dt, places=6)
        self.assertAlmostEqual(dtheta_dt.value.item(), expected_dtheta_dt, places=6)
        self.assertAlmostEqual(dpsi_dt.value.item(), expected_dpsi_dt, places=6)
    
    def test_eom_res_position_kinematics(self):
        """Test position kinematic equations with Euler angles."""
        # Set initial velocities and Euler angles
        self.aircraft_states.states.u.set_value(10.0)
        self.aircraft_states.states.v.set_value(5.0)
        self.aircraft_states.states.w.set_value(2.0)
        self.aircraft_states.states.phi.set_value(np.deg2rad(0))
        self.aircraft_states.states.theta.set_value(np.deg2rad(10))
        self.aircraft_states.states.psi.set_value(np.deg2rad(20))
        
        # Zero external forces and moments
        total_forces = csdl.Variable(shape=(3,), value=np.zeros(3), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Extract values for manual calculation
        u, v, w = 10.0, 5.0, 2.0
        phi, theta, psi = 0.0, np.deg2rad(10), np.deg2rad(20)
        
        # Expected position derivatives from kinematic equations
        expected_dx_dt = (u * np.cos(theta) * np.cos(psi) +
                         v * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) +
                         w * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)))
        
        expected_dy_dt = (u * np.cos(theta) * np.sin(psi) +
                         v * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) +
                         w * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)))
        
        expected_dz_dt = (-u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(phi) * np.cos(theta))
        
        # Check position derivatives
        dx_dt = dstate_vector[9]
        dy_dt = dstate_vector[10]
        dz_dt = dstate_vector[11]
        
        self.assertAlmostEqual(dx_dt.value.item(), expected_dx_dt, places=6)
        self.assertAlmostEqual(dy_dt.value.item(), expected_dy_dt, places=6)
        self.assertAlmostEqual(dz_dt.value.item(), expected_dz_dt, places=6)
    
    def test_eom_res_with_cg_offset(self):
        """Test EoM with center of gravity offset from reference point."""
        # Create mass properties with CG offset
        cg_vector = Vector(Q_(np.array([1.0, 0.5, -0.2]), 'm'), self.body_axis)
        inertia = MassMI(
            axis=self.body_axis,
            Ixx=Q_(1000, 'kg*m^2'),
            Iyy=Q_(2000, 'kg*m^2'),
            Izz=Q_(3000, 'kg*m^2'),
            Ixy=Q_(100, 'kg*m^2'),
            Ixz=Q_(50, 'kg*m^2'),
            Iyz=Q_(25, 'kg*m^2')
        )
        mass_properties = MassProperties(
            cg=cg_vector,
            inertia=inertia,
            mass=Q_(1000, 'kg')
        )
        
        # Set some initial angular rates to test coupling effects
        self.aircraft_states.states.p.set_value(0.1)
        self.aircraft_states.states.q.set_value(0.2)
        self.aircraft_states.states.r.set_value(0.1)
        
        # Apply external forces
        total_forces = csdl.Variable(shape=(3,), value=np.array([100.0, 50.0, -25.0]), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.array([10.0, 20.0, 15.0]), name='moments')
        
        # Call EoM residual function
        dstate_vector, state_vector = self.eom._EoM_res(
            self.aircraft_states,
            mass_properties,
            total_forces,
            total_moments
        )
        
        # Check that we get valid outputs (more complex verification would require detailed calculations)
        self.assertEqual(dstate_vector.shape, (12,))
        self.assertEqual(state_vector.shape, (12,))
        
        # Check that accelerations are non-zero due to forces
        du_dt = dstate_vector[0]
        dv_dt = dstate_vector[1]
        dw_dt = dstate_vector[2]
        
        self.assertNotAlmostEqual(du_dt.value.item(), 0.0, places=6)
        self.assertNotAlmostEqual(dv_dt.value.item(), 0.0, places=6)
        self.assertNotAlmostEqual(dw_dt.value.item(), 0.0, places=6)
    
    def test_eom_res_state_vector_consistency(self):
        """Test that the returned state vector matches the input aircraft states."""
        # Set various state values
        self.aircraft_states.states.u.set_value(15.0)
        self.aircraft_states.states.v.set_value(-2.0)
        self.aircraft_states.states.w.set_value(3.0)
        self.aircraft_states.states.p.set_value(0.05)
        self.aircraft_states.states.q.set_value(-0.1)
        self.aircraft_states.states.r.set_value(0.15)
        self.aircraft_states.states.phi.set_value(np.deg2rad(5))
        self.aircraft_states.states.theta.set_value(np.deg2rad(-3))
        self.aircraft_states.states.psi.set_value(np.deg2rad(45))
        self.aircraft_states.states.x.set_value(1000.0)
        self.aircraft_states.states.y.set_value(-500.0)
        self.aircraft_states.states.z.set_value(2000.0)
        
        # External forces and moments
        total_forces = csdl.Variable(shape=(3,), value=np.array([50.0, -25.0, 100.0]), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.array([5.0, -10.0, 8.0]), name='moments')
        
        # Call EoM residual function
        _, state_vector = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Expected state vector values
        expected_state = np.array([
            15.0, -2.0, 3.0,  # u, v, w
            0.05, -0.1, 0.15,  # p, q, r
            np.deg2rad(5), np.deg2rad(-3), np.deg2rad(45),  # phi, theta, psi
            1000.0, -500.0, 2000.0  # x, y, z
        ])
        
        # Check state vector consistency
        np.testing.assert_array_almost_equal(state_vector.value, expected_state, decimal=6)
    
    def test_eom_res_forces_moments_array_indexing(self):
        """Test that forces and moments are correctly indexed in the EoM."""
        # Create forces and moments with distinct values for each component
        total_forces = csdl.Variable(shape=(3,), value=np.array([100.0, 200.0, 300.0]), name='forces')
        total_moments = csdl.Variable(shape=(3,), value=np.array([10.0, 20.0, 30.0]), name='moments')
        
        # Call EoM residual function
        dstate_vector, _ = self.eom._EoM_res(
            self.aircraft_states,
            self.mass_properties,
            total_forces,
            total_moments
        )
        
        # Check that forces are correctly applied (simple check for non-zero accelerations)
        du_dt = dstate_vector[0]
        dv_dt = dstate_vector[1]
        dw_dt = dstate_vector[2]
        dp_dt = dstate_vector[3]
        dq_dt = dstate_vector[4]
        dr_dt = dstate_vector[5]
        
        # All accelerations should be non-zero
        self.assertNotAlmostEqual(du_dt.value.item(), 0.0, places=10)
        self.assertNotAlmostEqual(dv_dt.value.item(), 0.0, places=10)
        self.assertNotAlmostEqual(dw_dt.value.item(), 0.0, places=10)
        self.assertNotAlmostEqual(dp_dt.value.item(), 0.0, places=10)
        self.assertNotAlmostEqual(dq_dt.value.item(), 0.0, places=10)
        self.assertNotAlmostEqual(dr_dt.value.item(), 0.0, places=10)
    
    def test_eom_res_mass_matrix_properties(self):
        """Test various mass and inertia configurations."""
        # Test with different mass values
        for mass_val in [500.0, 1000.0, 2000.0]:
            cg_vector = Vector(Q_(np.array([0.0, 0.0, 0.0]), 'm'), self.body_axis)
            inertia = MassMI(
                axis=self.body_axis,
                Ixx=Q_(1000, 'kg*m^2'),
                Iyy=Q_(2000, 'kg*m^2'),
                Izz=Q_(3000, 'kg*m^2')
            )
            mass_properties = MassProperties(
                cg=cg_vector,
                inertia=inertia,
                mass=Q_(mass_val, 'kg')
            )
            
            # Apply unit force
            total_forces = csdl.Variable(shape=(3,), value=np.array([1.0, 0.0, 0.0]), name='forces')
            total_moments = csdl.Variable(shape=(3,), value=np.zeros(3), name='moments')
            
            # Call EoM residual function
            dstate_vector, _ = self.eom._EoM_res(
                self.aircraft_states,
                mass_properties,
                total_forces,
                total_moments
            )
            
            # Check that acceleration is inversely proportional to mass
            du_dt = dstate_vector[0]
            expected_acceleration = 1.0 / mass_val
            self.assertAlmostEqual(du_dt.value.item(), expected_acceleration, places=6)


if __name__ == '__main__':
    import unittest
    unittest.main()
