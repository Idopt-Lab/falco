import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector, ForcesMoments
from flight_simulator.core.dynamics.axis import Axis, ValidOrigins


class MassMI:

    @dataclass
    class MomentOfInertiaComponents(csdl.VariableGroup):
        Ixx: csdl.Variable
        Iyy: csdl.Variable
        Izz: csdl.Variable
        Ixy: csdl.Variable
        Ixz: csdl.Variable
        Iyz: csdl.Variable

        def define_checks(self):
            self.add_check('Ixx', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Iyy', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Izz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Ixy', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Ixz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('Iyz', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

        def _check_parameters(self, name, value):
            if self._metadata[name]['type'] is not None:
                if type(value) not in self._metadata[name]['type']:
                    raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")

            if self._metadata[name]['variablize']:
                if isinstance(value, ureg.Quantity):
                    value_si = value.to_base_units()
                    value = csdl.Variable(value=value_si.magnitude, shape=(1,), name=name)
                    value.add_tag(tag=str(value_si.units))

            if self._metadata[name]['shape'] is not None:
                if value.shape != self._metadata[name]['shape']:
                    raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
            return value

    def __init__(
            self,
            axis: Union[Axis, AxisLsdoGeo],
            Ixx: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Iyy: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Izz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Ixy: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Ixz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
            Iyz: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg*(m*m)'),
    ):

        self.axis = axis

        self.mass_mi_components = self.MomentOfInertiaComponents(
            Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=Ixy, Ixz=Ixz, Iyz=Iyz
        )

        self.inertia_tensor = csdl.Variable(shape=(3, 3), value=0.)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[0, 0], self.mass_mi_components.Ixx)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[1, 1], self.mass_mi_components.Iyy)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[2, 2], self.mass_mi_components.Izz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[0, 1], -self.mass_mi_components.Ixy)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[1, 0], -self.mass_mi_components.Ixy)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[0, 2], -self.mass_mi_components.Ixz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[2, 0], -self.mass_mi_components.Ixz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[1, 2], -self.mass_mi_components.Iyz)
        self.inertia_tensor = self.inertia_tensor.set(csdl.slice[2, 1], -self.mass_mi_components.Iyz)
        return


class MassProperties:
    def __init__(self,
                 cg: Vector, 
                 inertia: MassMI,
                 mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg')):

        assert cg.axis.name == inertia.axis.name

        if isinstance(mass, ureg.Quantity):
            value_si = mass.to_base_units()
            self.mass = csdl.Variable(value=value_si.magnitude, shape=(1,), name='mass')
            self.mass.add_tag(tag=str(value_si.units))
        elif isinstance(mass, csdl.Variable):
            self.mass = mass
        else:
            raise IOError

        self.cg_vector = cg
        self.inertia_tensor = inertia


class GravityLoads(Loads):

    def __init__(self, fd_state, controls, component):
        super().__init__(states=fd_state, controls=controls)

        # Store the states and mass properties
        self.states = fd_state
        self.load_axis = fd_state.axis
        self.cg = component.quantities.mass_properties.cg_vector
        self.mass = component.quantities.mass_properties.mass

        if self.mass is None:
            self.mass = csdl.Variable(name='mass', shape=(1,), value=1630) 
        elif isinstance(self.mass, ureg.Quantity):
            self.mass = csdl.Variable(name='mass', shape=(1,), value=self.mass.to_base_units().magnitude)
        else:
            self.mass = self.mass




    def get_FM_refPoint(self, x_bar, u_bar):
        """Use vehicle state and control objects to generate an estimate
        of gravity forces and moments about a reference point."""
        # Gravity FM
        g=9.81
        cg_vals = self.cg.vector.value 
        x = cg_vals[0]
        y = cg_vals[1]
        z = cg_vals[2]
        
        Rbc = np.array([x,y,z])


        m = self.mass
        th = self.states.states.theta
        ph = self.states.states.phi

        Fxg = -m * g * csdl.sin(th)
        Fyg = m * g * csdl.cos(th) * csdl.sin(ph)
        Fzg = m * g * csdl.cos(th) * csdl.cos(ph)
        forceVec = csdl.concatenate([Fxg, Fyg, Fzg])

        Rbsksym = np.array([[0, -Rbc[2], Rbc[1]],
                            [Rbc[2], 0, -Rbc[0]],
                            [-Rbc[1], Rbc[0], 0]])
        
        Mgrav = np.dot(Rbsksym, np.array([Fxg, Fyg, Fzg]))



        F_FD_BodyFixed = Vector(forceVec,axis=self.load_axis)
        M_FD_BodyFixed = Vector(csdl.concatenate([Mgrav[0],Mgrav[1],Mgrav[2]]),axis=self.load_axis)

        loads = ForcesMoments(force=F_FD_BodyFixed, moment=M_FD_BodyFixed)
        return loads
    
    

if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    mi = MassMI(axis=inertial_axis)
    cg = Vector(vector=np.array([0, 0, 0])*ureg.meter, axis=inertial_axis)
    mass_properties = MassProperties(cg=cg, inertia=mi)
