import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector
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

        def _check_pamaeters(self, name, value):
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
                 cg_vector: Vector, inertia_tensor: MassMI,
                 mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg')):

        assert cg_vector.axis.name == inertia_tensor.axis.name

        if isinstance(mass, ureg.Quantity):
            value_si = mass.to_base_units()
            self.mass = csdl.Variable(value=value_si.magnitude, shape=(1,), name='mass')
            self.mass.add_tag(tag=str(value_si.units))
        elif isinstance(mass, csdl.Variable):
            self.mass = mass
        else:
            raise IOError

        self.cg_vector = cg_vector
        self.inertia_tensor = inertia_tensor


class GravityLoads(Loads):

    # TODO: Implement this class

    def get_FM_refPoint(self):
        """Use vehicle state and control objects to generate an estimate
        of gravity forces and moments about a reference point."""
        # Gravity FM
        g = states_obj.atmosphere_properties['g']
        FD_body_fixed_axis = states_obj.return_fd_bodyfixed_axis()

        Rbc = np.array([self.cg.x.magnitude,
                        self.cg.y.magnitude,
                        self.cg.z.magnitude])

        m = self.mass
        th = states_obj.Theta
        ph = states_obj.Phi

        Fxg = -m * g * np.sin(th)
        Fyg = m * g * np.cos(th) * np.sin(ph)
        Fzg = m * g * np.cos(th) * np.cos(ph)

        Rbsksym = np.array([[0, -Rbc[2], Rbc[1]],
                            [Rbc[2], 0, -Rbc[0]],
                            [-Rbc[1], Rbc[0], 0]])
        Mgrav = np.dot(Rbsksym, np.array([Fxg.magnitude,
                                          Fyg.magnitude,
                                          Fzg.magnitude]))
        Mgrav = Mgrav * ureg.newton * ureg.meter

        F_FD_BodyFixed = Vector(x=Fxg,
                                y=Fyg,
                                z=Fzg, axis=FD_body_fixed_axis)
        M_FD_BodyFixed = Vector(x=Mgrav[0],
                                y=Mgrav[1],
                                z=Mgrav[2], axis=FD_body_fixed_axis)
        FM_grav_FDbodyfixed = ForcesMoments(F=F_FD_BodyFixed, M=M_FD_BodyFixed)
        return FM_grav_FDbodyfixed



if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    mi = MassMI(axis=inertial_axis)
    cg = Vector(vector=np.array([0, 0, 0])*ureg.meter, axis=inertial_axis)
    mass_properties = MassProperties(cg_vector=cg, inertia_tensor=mi)
