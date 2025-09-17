from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

from falco import ureg, Q_
import csdl_alpha as csdl
import numpy as np

from falco.core.dynamics.axis import Axis, ValidOrigins
from falco.core.loads.forces_moments import Vector
from falco.core.loads.mass_properties import MassProperties, MassMI
from falco.core.vehicle.components.component import Component, ComponentParameters
from falco.utils.import_geometry import import_geometry


class TestComponentInitialization(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_default_initialization(self):
        """Test default initialization of a Component."""
        component = Component(name="custom_component")
        self.assertEqual(component._name, "custom_component")
        self.assertIsNone(component.geometry)
        self.assertFalse(component.compute_surface_area_flag)
        self.assertIsNone(component._parameterization_solver)
        self.assertIsNone(component.mass_properties)
        self.assertIsNone(component._ffd_geometric_variables)
        self.assertIsInstance(component.parameters, ComponentParameters)
        self.assertEqual(component.comps, {})
        self.assertEqual(component.surface_mesh, [])
        self.assertEqual(component.load_solvers, [])
        self.assertIsNone(component.surface_area)


    def test_kwargs_parameters(self):
        """Test initialization with user-defined parameters using kwargs."""
        component = Component(name="custom_component",
                              param1="value1", param2=123)
        self.assertEqual(component.parameters.param1, "value1")
        self.assertEqual(component.parameters.param2, 123)

    def test_parent_attribute(self):
        """Test the parent attribute initialization."""
        component = Component(name="custom_component")
        self.assertIsNone(component.parent)

    def test_empty_subcomponents(self):
        """Test that the subcomponents dictionary is empty upon initialization."""
        component = Component(name="custom_component")
        self.assertEqual(component.comps, {})

    # def test_providing_mp(self):
    #     component = Component(name="custom_component")

        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create a Quantity vector with units
        cg = Q_([0, 0, 0], 'newton')
        cg = Vector(cg, axis)
        # Create a mass moment of inertia object
        mi = MassMI(axis=axis)
        # Create a mass properties object
        mp = MassProperties(cg=cg, inertia=mi, mass=Q_(10, 'lb'))


class TestComponentGeometry(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        from falco import GEOMETRIES_ROOT_FOLDER
        self.wing_geometry = import_geometry("simple_wing.stp", file_path=GEOMETRIES_ROOT_FOLDER)

    def test_geometry_surface_area(self):
        wing_component = Component(name="Wing",
                                   geometry=self.wing_geometry,
                                   compute_surface_area_flag=True)
        np.testing.assert_almost_equal(wing_component.surface_area.value, 94.82, decimal=3)


class TestComponentHierarchy(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_adding_subcomponent(self):
        """Test adding a subcomponent to a Component."""
        parent_component = Component(name="parent_component")
        sub_component = Component(name="sub_component")

        parent_component.add_subcomponent(sub_component)

        self.assertIn("sub_component", parent_component.comps)
        self.assertEqual(parent_component.comps["sub_component"], sub_component)
        self.assertEqual(sub_component.parent, parent_component)

    def test_adding_subcomponent_that_already_exists(self):
        """Test adding a subcomponent that already exists in the parent Component."""
        parent_component = Component(name="parent_component")
        sub_component = Component(name="sub_component")

        parent_component.add_subcomponent(sub_component)

        with self.assertRaises(KeyError):
            parent_component.add_subcomponent(sub_component)

    def test_removing_subcomponent(self):
        """Test removing a subcomponent from a Component."""
        parent_component = Component(name="parent_component")
        sub_component1 = Component(name="sub_component1")
        sub_component2 = Component(name="sub_component2")

        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)

        parent_component.remove_subcomponent(sub_component1)

        self.assertNotIn("sub_component1", parent_component.comps)
        self.assertIsNone(sub_component1.parent)
        self.assertIn("sub_component2", parent_component.comps)

    def test_viz_component_hierarchy(self):
        parent_component = Component(name="parent_component")
        sub_component1 = Component(name="sub_component1")
        sub_component2 = Component(name="sub_component2")

        parent_component.add_subcomponent(sub_component1)
        parent_component.add_subcomponent(sub_component2)

        parent_component.visualize_component_hierarchy(filepath=Path.cwd()/"python_test_outputs")
        self.assertTrue((Path.cwd()/"python_test_outputs/component_hierarchy.png").exists())
        # Delete the folder and all its files after the test
        (Path.cwd()/"python_test_outputs/component_hierarchy.png").unlink()
        (Path.cwd()/"python_test_outputs/component_hierarchy").unlink()
        (Path.cwd() / "python_test_outputs/").rmdir()


class TestComponentRepr(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_repr_with_mass_properties(self):
        """Test __repr__ method with mass properties."""
        # Create axis
        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create mass properties
        cg = Q_([1, 2, 3], 'meter')
        cg_vector = Vector(cg, axis)
        mi = MassMI(axis=axis)
        mp = MassProperties(cg=cg_vector, inertia=mi, mass=Q_(100, 'kg'))
        
        component = Component(name="test_component", mass_properties=mp)
        
        repr_str = repr(component)
        self.assertIn("Component Mass: [100.] kg", repr_str)
        self.assertIn("Component CG:", repr_str)
        self.assertIn("Component Inertia:", repr_str)



class TestComponentSetupGeometry(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_setup_geometry_without_geometry(self):
        """Test _setup_geometry when component has no geometry."""
        component = Component(name="test_component")
        mock_solver = Mock()
        mock_variables = Mock()
        
        # Should return early without error
        component._setup_geometry(mock_solver, mock_variables)
        
        # Verify no operations were performed
        mock_solver.add_parameter.assert_not_called()

    @patch('falco.core.vehicle.components.component.csdl')
    def test_setup_geometry_with_geometry(self, mock_csdl):
        """Test _setup_geometry with geometry."""
        # Mock geometry with functions
        mock_function = Mock()
        mock_function.name = "rigid_body_translation"
        mock_coefficients = Mock()
        mock_coefficients.shape = (3, 4, 5)
        mock_function.coefficients = mock_coefficients
        
        mock_geometry = Mock()
        mock_geometry.functions = {"func1": mock_function}
        
        component = Component(name="test_component", geometry=mock_geometry)
        
        mock_solver = Mock()
        mock_variables = Mock()
        
        # Mock csdl operations
        mock_implicit_var = Mock()
        mock_expand_result = Mock()
        mock_csdl.ImplicitVariable.return_value = mock_implicit_var
        mock_csdl.expand.return_value = mock_expand_result
        
        # Mock the addition operation
        mock_coefficients.__add__ = Mock(return_value=Mock())
        
        component._setup_geometry(mock_solver, mock_variables, plot=True)
        
        # Verify operations
        mock_csdl.ImplicitVariable.assert_called_once()
        mock_solver.add_parameter.assert_called_once()


class TestComponentComputeTotalLoads(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_compute_total_loads_no_load_solvers_no_subcomponents(self):
        """Test compute_total_loads with no load solvers and no subcomponents."""
        component = Component(name="test_component")
        
        mock_fd_state = Mock()
        mock_controls = Mock()
        
        total_forces, total_moments = component.compute_total_loads(mock_fd_state, mock_controls)
        
        # Should return zero vectors
        self.assertEqual(total_forces.shape, (3,))
        self.assertEqual(total_moments.shape, (3,))

    def test_compute_total_loads_with_subcomponents(self):
        """Test compute_total_loads with subcomponents."""
        parent = Component(name="parent")
        child = Component(name="child")
        parent.add_subcomponent(child)
        
        mock_fd_state = Mock()
        mock_controls = Mock()
        
        # Mock child's compute_total_loads
        with patch.object(child, 'compute_total_loads', return_value=(csdl.Variable(shape=(3,), value=1.), csdl.Variable(shape=(3,), value=2.))):
            total_forces, total_moments = parent.compute_total_loads(mock_fd_state, mock_controls)
            
            self.assertEqual(total_forces.shape, (3,))
            self.assertEqual(total_moments.shape, (3,))

    def test_compute_total_loads_with_load_solvers(self):
        """Test compute_total_loads with load solvers."""
        # Create axis
        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create mass properties
        cg = Q_([0, 0, 0], 'meter')
        cg_vector = Vector(cg, axis)
        mi = MassMI(axis=axis)
        mp = MassProperties(cg=cg_vector, inertia=mi, mass=Q_(100, 'kg'))
        
        component = Component(name="test_component", mass_properties=mp)
        
        # Mock load solver
        from falco.core.loads.loads import Loads
        mock_load_solver = Mock(spec=Loads)
        
        # Create proper CSDL Variables for the mock
        force_vector = csdl.Variable(shape=(3,), value=np.array([1, 2, 3]))
        moment_vector = csdl.Variable(shape=(3,), value=np.array([4, 5, 6]))
        
        # Create mock forces/moments object
        mock_fm = Mock()
        mock_fm.F.axis.reference.name = 'Inertial Axis'
        mock_fm.F.vector = force_vector
        mock_fm.M.vector = moment_vector
        
        # Mock the transform_to_axis method to return the same vectors
        mock_fm.transform_to_axis.return_value = mock_fm
        
        mock_load_solver.get_FM_localAxis.return_value = mock_fm
        component.load_solvers = [mock_load_solver]
        
        mock_fd_state = Mock()
        mock_fd_state.theta = csdl.Variable(shape=(1,), value=0.0)
        mock_fd_state.phi = csdl.Variable(shape=(1,), value=0.0)
        mock_fd_state.psi = csdl.Variable(shape=(1,), value=0.0)
        mock_controls = Mock()
        
        total_forces, total_moments = component.compute_total_loads(mock_fd_state, mock_controls)
        
        self.assertEqual(total_forces.shape, (3,))
        self.assertEqual(total_moments.shape, (3,))

    def test_compute_total_loads_with_gravity_loads(self):
        """Test compute_total_loads with gravity loads for root component."""
        # Create axis
        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create mass properties
        cg = Q_([0, 0, 0], 'meter')
        cg_vector = Vector(cg, axis)
        mi = MassMI(axis=axis)
        mp = MassProperties(cg=cg_vector, inertia=mi, mass=Q_(100, 'kg'))
        
        component = Component(name="test_component", mass_properties=mp)
        
        mock_fd_state = Mock()
        mock_controls = Mock()
        
        # Mock GravityLoads
        with patch('falco.core.vehicle.components.component.GravityLoads') as mock_gravity_class:
            mock_gravity_instance = Mock()
            mock_gfm = Mock()
            mock_gfm.F.vector = csdl.Variable(shape=(3,), value=np.array([0, 0, -981]))
            mock_gfm.M.vector = csdl.Variable(shape=(3,), value=np.array([0, 0, 0]))
            mock_gravity_instance.get_FM_localAxis.return_value = mock_gfm
            mock_gravity_class.return_value = mock_gravity_instance
            
            # Mock the fd_state to have proper attributes for gravity loads
            mock_fd_state.theta = csdl.Variable(shape=(1,), value=0.0)
            mock_fd_state.phi = csdl.Variable(shape=(1,), value=0.0)
            mock_fd_state.psi = csdl.Variable(shape=(1,), value=0.0)
            
            total_forces, total_moments = component.compute_total_loads(mock_fd_state, mock_controls)
            
            self.assertEqual(total_forces.shape, (3,))
            self.assertEqual(total_moments.shape, (3,))


class TestComponentComputeTotalTorquePower(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_compute_total_torque_power_no_load_solvers(self):
        """Test compute_total_torque_total_power with no load solvers."""
        component = Component(name="test_component")
        
        mock_fd_state = Mock()
        mock_controls = Mock()
        
        total_torque, total_power = component.compute_total_torque_total_power(mock_fd_state, mock_controls)
        
        self.assertEqual(total_torque.shape, (1,))
        self.assertEqual(total_power.shape, (1,))

    def test_compute_total_torque_power_with_load_solver(self):
        """Test compute_total_torque_total_power with load solver that has torque/power."""
        component = Component(name="test_component")
        
        # Mock load solver with torque/power capability
        from falco.core.loads.loads import Loads
        mock_load_solver = Mock(spec=Loads)
        # Add the get_torque_power method to the mock
        mock_load_solver.get_torque_power = Mock(return_value={
            'torque': csdl.Variable(shape=(1,), value=10.),
            'power_avail': csdl.Variable(shape=(1,), value=100.)
        })
        component.load_solvers = [mock_load_solver]
        
        mock_fd_state = Mock()
        mock_controls = Mock()
        
        total_torque, total_power = component.compute_total_torque_total_power(mock_fd_state, mock_controls)
        
        mock_load_solver.get_torque_power.assert_called_once_with(mock_fd_state, mock_controls)
        self.assertEqual(total_torque.shape, (1,))
        self.assertEqual(total_power.shape, (1,))

    def test_compute_total_torque_power_with_subcomponents(self):
        """Test compute_total_torque_total_power with subcomponents."""
        parent = Component(name="parent")
        child = Component(name="child")
        parent.add_subcomponent(child)
        
        mock_fd_state = Mock()
        mock_controls = Mock()
        
        # Mock child's compute_total_torque_total_power
        with patch.object(child, 'compute_total_torque_total_power', return_value=(csdl.Variable(shape=(1,), value=5.), csdl.Variable(shape=(1,), value=50.))):
            total_torque, total_power = parent.compute_total_torque_total_power(mock_fd_state, mock_controls)
            
            self.assertEqual(total_torque.shape, (1,))
            self.assertEqual(total_power.shape, (1,))


class TestComponentComputeTotalMassProperties(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_compute_total_mass_properties_with_subcomponents(self):
        """Test compute_total_mass_properties with subcomponents."""
        # Create axis
        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create parent mass properties
        cg_parent = Q_([0, 0, 0], 'meter')
        cg_vector_parent = Vector(cg_parent, axis)
        mi_parent = MassMI(axis=axis)
        mp_parent = MassProperties(cg=cg_vector_parent, inertia=mi_parent, mass=Q_(100, 'kg'))
        
        parent = Component(name="parent", mass_properties=mp_parent)
        
        # Create child mass properties
        cg_child = Q_([1, 1, 1], 'meter')
        cg_vector_child = Vector(cg_child, axis)
        mi_child = MassMI(axis=axis)
        mp_child = MassProperties(cg=cg_vector_child, inertia=mi_child, mass=Q_(50, 'kg'))
        
        child = Component(name="child", mass_properties=mp_child)
        parent.add_subcomponent(child)
        
        total_props = parent.compute_total_mass_properties()
        
        self.assertIsInstance(total_props, MassProperties)
        self.assertEqual(total_props.mass.value, 150.0)  # 100 + 50

    def test_compute_total_mass_properties_single_component(self):
        """Test compute_total_mass_properties for single component."""
        # Create axis
        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create mass properties
        cg = Q_([1, 2, 3], 'meter')
        cg_vector = Vector(cg, axis)
        mi = MassMI(axis=axis)
        mp = MassProperties(cg=cg_vector, inertia=mi, mass=Q_(100, 'kg'))
        
        component = Component(name="test_component", mass_properties=mp)
        
        total_props = component.compute_total_mass_properties()
        
        self.assertIsInstance(total_props, MassProperties)
        self.assertEqual(total_props.mass.value, 100.0)



class TestComponentFFDBlockCreation(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    @patch('falco.core.vehicle.components.component.construct_ffd_block_around_entities')
    def test_ffd_block_creation_with_function_set(self, mock_construct_ffd):
        """Test FFD block creation when geometry is FunctionSet."""
        from lsdo_function_spaces import FunctionSet
        
        # Create a mock that properly simulates FunctionSet
        mock_geometry = Mock(spec=FunctionSet)
        mock_geometry.__class__ = FunctionSet
        
        component = Component(name="test_component", geometry=mock_geometry)
        
        mock_construct_ffd.assert_called_once()
        self.assertTrue(hasattr(component, '_ffd_block'))

    @patch('falco.core.vehicle.components.component.construct_ffd_block_around_entities')
    def test_ffd_block_creation_skipped_with_flag(self, mock_construct_ffd):
        """Test FFD block creation is skipped with do_not_remake_ffd_block flag."""
        mock_geometry = Mock()
        mock_geometry.__class__.__name__ = 'FunctionSet'
        
        component = Component(name="test_component", geometry=mock_geometry, do_not_remake_ffd_block=True)
        
        mock_construct_ffd.assert_not_called()


class TestComponentErrorConditions(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_add_subcomponent_wrong_type(self):
        """Test adding subcomponent with wrong type."""
        parent = Component(name="parent")
        
        with self.assertRaises(TypeError) as context:
            parent.add_subcomponent("not_a_component")
        
        self.assertIn("Subcomponent must be of type 'Component'", str(context.exception))

    def test_add_subcomponent_already_has_parent(self):
        """Test adding subcomponent that already has a parent."""
        parent1 = Component(name="parent1")
        parent2 = Component(name="parent2")
        child = Component(name="child")
        
        parent1.add_subcomponent(child)
        
        with self.assertRaises(ValueError) as context:
            parent2.add_subcomponent(child)
        
        self.assertIn("already a subcomponent of another component", str(context.exception))

    def test_remove_subcomponent_not_found(self):
        """Test removing subcomponent that doesn't exist."""
        parent = Component(name="parent")
        child = Component(name="child")
        
        with self.assertRaises(KeyError) as context:
            parent.remove_subcomponent(child)
        
        self.assertIn("Subcomponent 'child' not found", str(context.exception))


class TestComponentVisualizeHierarchy(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_visualize_hierarchy_import_error(self):
        """Test visualize_component_hierarchy when graphviz is not available."""
        component = Component(name="test_component")
        
        # Mock the import to raise ImportError when trying to import graphviz
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'graphviz':
                raise ImportError("graphviz not found")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with self.assertRaises(ImportError) as context:
                component.visualize_component_hierarchy()
            
            # The actual error message from the component is "Must install graphviz via application and do 'pip install graphviz'"
            self.assertIn("Must install graphviz", str(context.exception))

    @patch('graphviz.Graph')
    def test_visualize_hierarchy_success(self, mock_graph_class):
        """Test successful visualization of component hierarchy."""
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        
        component = Component(name="test_component")
        
        component.visualize_component_hierarchy(file_name="test", file_format="pdf", show=True)
        
        mock_graph_class.assert_called_once()
        mock_graph.node.assert_called()
        mock_graph.render.assert_called_once()


class TestComponentParameters(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_component_parameters_default(self):
        """Test ComponentParameters default initialization."""
        params = ComponentParameters()
        self.assertIsNone(params.actuate_angle)

    def test_component_parameters_with_values(self):
        """Test ComponentParameters with values."""
        mock_variable = Mock()
        params = ComponentParameters(actuate_angle=mock_variable)
        self.assertEqual(params.actuate_angle, mock_variable)


class TestComponentInitializationEdgeCases(TestCase):
    def setUp(self):
        recorder = csdl.Recorder(inline=True)
        recorder.start()

    def test_initialization_with_all_parameters(self):
        """Test initialization with all optional parameters."""
        # Create axis
        axis = Axis(
            name='Inertial Axis',
            x=np.array([0]) * ureg.meter,
            y=np.array([0]) * ureg.meter,
            z=np.array([0]) * ureg.meter,
            phi=np.array([0, ]) * ureg.degree,
            theta=np.array([0, ]) * ureg.degree,
            psi=np.array([0, ]) * ureg.degree,
            origin=ValidOrigins.Inertial.value
        )

        # Create mass properties
        cg = Q_([0, 0, 0], 'meter')
        cg_vector = Vector(cg, axis)
        mi = MassMI(axis=axis)
        mp = MassProperties(cg=cg_vector, inertia=mi, mass=Q_(100, 'kg'))
        
        mock_geometry = Mock()
        mock_geometry.functions = {"surface1": Mock()}  # Add functions attribute
        mock_solver = Mock()
        mock_variables = Mock()
        
        # Mock the _compute_surface_area method to avoid geometry processing
        with patch.object(Component, '_compute_surface_area', return_value=csdl.Variable(shape=(1,), value=100.0)):
            component = Component(
                name="test_component",
                geometry=mock_geometry,
                compute_surface_area_flag=True,
                parameterization_solver=mock_solver,
                mass_properties=mp,
                ffd_geometric_variables=mock_variables,
                custom_param="test_value"
            )
        
        self.assertEqual(component._name, "test_component")
        self.assertEqual(component.geometry, mock_geometry)
        self.assertTrue(component.compute_surface_area_flag)
        self.assertEqual(component._parameterization_solver, mock_solver)
        self.assertEqual(component.mass_properties, mp)
        self.assertEqual(component._ffd_geometric_variables, mock_variables)
        self.assertEqual(component.parameters.custom_param, "test_value")

    def test_initialization_with_geometry_and_surface_area_flag(self):
        """Test initialization with geometry and surface area computation."""
        mock_geometry = Mock()
        
        with patch.object(Component, '_compute_surface_area', return_value=csdl.Variable(shape=(1,), value=100.0)) as mock_compute:
            component = Component(
                name="test_component",
                geometry=mock_geometry,
                compute_surface_area_flag=True
            )
            
            mock_compute.assert_called_once_with(mock_geometry)
            self.assertEqual(component.surface_area.value, 100.0)

    def test_initialization_without_geometry_and_surface_area_flag(self):
        """Test initialization without geometry but with surface area flag."""
        component = Component(
            name="test_component",
            geometry=None,
            compute_surface_area_flag=True
        )
        
        self.assertIsNone(component.surface_area)


