import numpy as np
from flight_simulator import ureg, Q_
import csdl_alpha as csdl
from typing import Union, Literal
from enum import Enum
from dataclasses import dataclass


class ValidOrigins(Enum):
    Inertial = "inertial"
    OpenVSP = "openvsp_rotated_to_fd"


def axis_checkers(func):
    def test_origin_value(*args, **kwargs):
        # Check kwargs for origin
        origin_in = kwargs.get('origin')
        if origin_in not in ValidOrigins._value2member_map_:
            print('Axis origin "%s" not permitted' % origin_in)
            raise IOError
            kwargs['origin'] = ValidOrigins.Inertial.value
        func(*args, **kwargs)

    # def test_reference(*args, **kwargs):
    #     # Check kwargs for name
    #     name = kwargs.get('name')
    #     if name in ValidOrigins._value2member_map_:
    #         # todo: check that reference is None
    #         pass

    return test_origin_value


class Axis:

    @dataclass
    class euler_angles(csdl.VariableGroup):
        phi : csdl.Variable
        theta : csdl.Variable
        psi: csdl.Variable

        def define_checks(self):
            self.add_check('phi', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('theta', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('psi', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

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

    @dataclass
    class translation_from_origin(csdl.VariableGroup):
        x: csdl.Variable
        y: csdl.Variable
        z: csdl.Variable

        def define_checks(self):
            self.add_check('x', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('y', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('z', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

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

    @axis_checkers
    def __init__(self, name: str,
                 origin: str,
                 x: Union[ureg.Quantity, csdl.Variable] = None,
                 y: Union[ureg.Quantity, csdl.Variable] = None,
                 z: Union[ureg.Quantity, csdl.Variable] = None,
                 phi: Union[ureg.Quantity, csdl.Variable] = None,
                 theta: Union[ureg.Quantity, csdl.Variable] = None,
                 psi: Union[ureg.Quantity, csdl.Variable] = None,
                 sequence = None,
                 reference=None):

        self.name = name

        if x is not None:
            self.translation_from_origin = self.translation_from_origin(
                x=x, y=y, z=z
            )
            self.translation_from_origin_vector = csdl.concatenate(
                (self.translation_from_origin.x, self.translation_from_origin.y, self.translation_from_origin.z),
                axis=0
            )
            self.translation = self.translation_from_origin_vector
        else:
            self.translation_from_origin = None
            self.translation_from_origin_vector = None

        if phi is not None:
            self.euler_angles = self.euler_angles(phi=phi, theta=theta, psi=psi)
            self.euler_angles_vector = csdl.concatenate(
                (self.euler_angles.phi, self.euler_angles.theta, self.euler_angles.psi), axis=0)
        else:
            self.euler_angles = None
            self.euler_angles_vector = None

        self.sequence = sequence
        self.reference = reference
        self.origin = origin


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        origin=ValidOrigins.Inertial.value
    )

    axis = Axis(name='Reference Axis',
                x=np.array([10, ]) * ureg.meter,
                y=np.array([0, ]) * ureg.meter,
                z=np.array([0, ]) * ureg.meter,
                phi=np.array([0, ]) * ureg.degree,
                theta=np.array([5, ]) * ureg.degree,
                psi=np.array([0, ]) * ureg.degree,
                reference=inertial_axis,
                origin=ValidOrigins.Inertial.value)

    print('Axis translation: ', axis.translation_from_origin_vector)
    print('Axis translation value: ', axis.translation_from_origin_vector.value)
    print('Axis angles: ', axis.euler_angles_vector)
    print('Axis angles value: ', axis.euler_angles_vector.value)
    pass