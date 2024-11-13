import csdl_alpha as csdl
from typing import Union
import numpy as np
from flight_simulator import ureg, Q_
from flight_simulator.core.loads.loads import Loads
from flight_simulator.core.dynamics.axis import Axis
from flight_simulator.core.dynamics.axis_lsdogeo import AxisLsdoGeo
from flight_simulator.core.loads.forces_moments import Vector


class MassMI:
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
        self.Ixx = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.Iyy = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.Izz = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.Ixy = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.Ixz = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.Iyz = csdl.Variable(shape=(1,), value=np.array([0, ]))

        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixy = Ixy
        self.Ixz = Ixz
        self.Iyz = Iyz

    @property
    def Ixx(self):
        return self._Ixx

    @Ixx.setter
    def Ixx(self, Ixx):
        if isinstance(Ixx, ureg.Quantity):
            Ixx_si = Ixx.to_base_units()
            self._Ixx.set_value(Ixx_si.magnitude)
            self._Ixx.add_tag(str(Ixx_si.units))
        elif isinstance(Ixx, csdl.Variable):
            self._Ixx = Ixx
        else:
            raise IOError

    @property
    def Iyy(self):
        return self._Ixx

    @Iyy.setter
    def Iyy(self, Iyy):
        if isinstance(Iyy, ureg.Quantity):
            Iyy_si = Iyy.to_base_units()
            self._Iyy.set_value(Iyy_si.magnitude)
            self._Iyy.add_tag(str(Iyy_si.units))
        elif isinstance(Iyy, csdl.Variable):
            self._Iyy = Iyy
        else:
            raise IOError

    @property
    def Izz(self):
        return self._Izz

    @Izz.setter
    def Izz(self, Izz):
        if isinstance(Izz, ureg.Quantity):
            Izz_si = Izz.to_base_units()
            self._Izz.set_value(Izz_si.magnitude)
            self._Izz.add_tag(str(Izz_si.units))
        elif isinstance(Izz, csdl.Variable):
            self._Izz = Izz
        else:
            raise IOError

    @property
    def Ixy(self):
        return self._Ixy

    @Ixy.setter
    def Ixy(self, Ixy):
        if isinstance(Ixy, ureg.Quantity):
            Ixy_si = Ixy.to_base_units()
            self._Ixy.set_value(Ixy_si.magnitude)
            self._Ixy.add_tag(str(Ixy_si.units))
        elif isinstance(Ixy, csdl.Variable):
            self._Ixy = Ixy
        else:
            raise IOError

    @property
    def Ixz(self):
        return self._Ixz

    @Ixz.setter
    def Ixz(self, Ixz):
        if isinstance(Ixz, ureg.Quantity):
            Ixz_si = Ixz.to_base_units()
            self._Ixz.set_value(Ixz_si.magnitude)
            self._Ixz.add_tag(str(Ixz_si.units))
        elif isinstance(Ixz, csdl.Variable):
            self._Ixz = Ixz
        else:
            raise IOError

    @property
    def Iyz(self):
        return self._Iyz

    @Iyz.setter
    def Iyz(self, Iyz):
        if isinstance(Iyz, ureg.Quantity):
            Iyz_si = Iyz.to_base_units()
            self._Iyz.set_value(Iyz_si.magnitude)
            self._Iyz.add_tag(str(Iyz_si.units))
        elif isinstance(Iyz, csdl.Variable):
            self._Iyz = Iyz
        else:
            raise IOError

    @property
    def inertia_tensor(self):
        inertia_tensor = csdl.Variable(shape=(3, 3), value=0.)
        inertia_tensor = inertia_tensor.set(csdl.slice[0, 0], self.Ixx)
        inertia_tensor = inertia_tensor.set(csdl.slice[1, 1], self.Iyy)
        inertia_tensor = inertia_tensor.set(csdl.slice[2, 2], self.Izz)
        inertia_tensor = inertia_tensor.set(csdl.slice[0, 1], -self.Ixy)
        inertia_tensor = inertia_tensor.set(csdl.slice[1, 0], -self.Ixy)
        inertia_tensor = inertia_tensor.set(csdl.slice[0, 2], -self.Ixz)
        inertia_tensor = inertia_tensor.set(csdl.slice[2, 0], -self.Ixz)
        inertia_tensor = inertia_tensor.set(csdl.slice[1, 2], -self.Iyz)
        inertia_tensor = inertia_tensor.set(csdl.slice[2, 1], -self.Iyz)
        return inertia_tensor


class MassProperties(Loads):
    def __init__(self, cg_vector: Vector, inertia_tensor: MassMI,
                 mass: Union[ureg.Quantity, csdl.Variable] = Q_(0, 'kg')):
        assert cg_vector.axis.name == inertia_tensor.axis.name
        super().__init__()
        self.mass = mass
        self.cg_vector = cg_vector
        self.inertia_tensor = inertia_tensor

    def get_FM_refPoint(self, states_obj, controls_obj):
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