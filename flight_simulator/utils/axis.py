import numpy as np
from flight_simulator import ureg, Q_
import csdl_alpha as csdl
from typing import Union, Literal
from enum import Enum


class ValidOrigins(Enum):
    Inertial = "inertial"
    Reference = "ref"
    CenterOfGravity = "cg"


def validate_origin(func):
    def test_origin_value(*args, **kwargs):
        origin_valid = {'inertial', 'ref', 'cg', 'openvsp'}
        # Check kwargs for origin
        origin_in = kwargs.get('origin')
        if origin_in not in ValidOrigins._value2member_map_ and origin_in is not None:
            print('Axis origin "%s" not permitted, defaulting to reference origin' % origin_in)
            kwargs['origin'] = 'ref'
        func(*args, **kwargs)

    return test_origin_value


class Axis:
    @validate_origin
    def __init__(self, name: str,
                 translation: Union[ureg.Quantity, csdl.Variable],
                 phi: Union[ureg.Quantity, csdl.Variable] = np.array([0., ])*ureg.radian,
                 theta: Union[ureg.Quantity, csdl.Variable] = np.array([0., ])*ureg.radian,
                 psi: Union[ureg.Quantity, csdl.Variable] = np.array([0., ])*ureg.radian,
                 origin: Union[str, None] = None,
                 sequence=np.array([3, 2, 1]),
                 reference = None):

        self.translation = csdl.Variable(
            shape=(3,),
            value=np.array([0, 0, 0]),
        )
        self.phi = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.theta = csdl.Variable(shape=(1,), value=np.array([0, ]))
        self.psi = csdl.Variable(shape=(1,), value=np.array([0, ]))

        self.name = name
        self.translation = translation
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.sequence = sequence
        self.reference = reference
        self.origin = origin

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name_string):
        self._name = name_string

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation_vector):
        if translation_vector is None:
            self._translation = None
        elif isinstance(translation_vector, ureg.Quantity):
            vector_si = translation_vector.to_base_units()
            self._translation.set_value(vector_si.magnitude)
            self._translation.add_tag(str(vector_si.units))
        elif isinstance(translation_vector, csdl.Variable):
            self._translation = translation_vector
        else:
            raise IOError

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi_value):
        if phi_value is None:
            self._phi = None
        elif isinstance(phi_value, ureg.Quantity):
            scalar_si = phi_value.to_base_units()
            assert scalar_si.shape[0] == 1
            self._phi.set_value(scalar_si.magnitude)
            self._phi.add_tag(str(scalar_si.units))
        elif isinstance(phi_value, csdl.Variable):
            assert phi_value.shape[0] == 1
            self._phi = phi_value
        else:
            raise IOError

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta_value):
        if theta_value is None:
            self._theta = None
        elif isinstance(theta_value, ureg.Quantity):
            scalar_si = theta_value.to_base_units()
            assert scalar_si.shape[0] == 1
            self._theta.set_value(scalar_si.magnitude)
            self._theta.add_tag(str(scalar_si.units))
        elif isinstance(theta_value, csdl.Variable):
            assert theta_value.shape[0] == 1
            self._theta = theta_value
        else:
            raise IOError

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi_value):
        if psi_value is None:
            self._psi = None
        elif isinstance(psi_value, ureg.Quantity):
            scalar_si = psi_value.to_base_units()
            assert scalar_si.shape[0] == 1
            self._psi.set_value(scalar_si.magnitude)
            self._psi.add_tag(str(scalar_si.units))
        elif isinstance(psi_value, csdl.Variable):
            assert psi_value.shape[0] == 1
            self._psi = psi_value
        else:
            raise IOError

    @property
    def euler_angles(self):
        euler_angles = csdl.Variable(shape=(3,), value=0.)
        euler_angles = euler_angles.set(csdl.slice[0], self.phi)
        euler_angles = euler_angles.set(csdl.slice[1], self.theta)
        euler_angles = euler_angles.set(csdl.slice[2], self.psi)
        return euler_angles

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, sequence_vector):
        if sequence_vector is None:
            self._sequence = None
        else:
            # todo: check if the sequence vector is one of the two possibilities
            self._sequence = sequence_vector

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, axis_obj):
        if axis_obj is None:
            self._reference = None
        else:
            isinstance(axis_obj, Axis)
            self._reference = axis_obj


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inertial_axis = Axis(
        name='Inertial Axis',
        translation=np.array([0, 0, 0]) * ureg.meter,
        origin=ValidOrigins.Inertial.value
    )

    axis = Axis(name='Reference Axis',
                translation=np.array([0, 0, 0])*ureg.meter,
                phi=np.array([0, ])*ureg.degree,
                theta=np.array([5, ]) * ureg.degree,
                psi=np.array([0, ]) * ureg.degree,
                reference=inertial_axis,
                origin=ValidOrigins.Reference.value)

    # Translation automatically creates a CSDL variable
    print('Axis translation: ', axis.translation)
    print('Axis translation value: ', axis.translation.value)
    print('Axis angles value: ', axis.euler_angles.value)

    # Update using the class' setter.
    # Here we need it to be a Pint Quantity with units
    # It will automatically be converted to base SI units

    print('\n Using class setter to update')

    axis.translation = np.array([3, 0, 0])*ureg.meter
    print('Axis translation: ', axis.translation)
    print('Axis translation value: ', axis.translation.value)

    axis.phi = np.array([4, ]) * ureg.degree
    print('Changed phi value: ', axis.phi.value)
    print('Changed Euler angles value: ', axis.euler_angles.value)

    # Update by directly accessing the CSDL Variable and using the set_value() method
    print('\n Using set_value() to update')

    axis.translation.set_value(np.array([0, 5, 0]))
    print('Axis translation value: ', axis.translation.value)

    # Since the shape is given as (3, ), giving a scalar will broadcast
    axis.translation.set_value(5)
    print('Axis translation value: ', axis.translation.value)

    axis.psi.set_value(np.array([np.deg2rad(-3.)]))
    print('Changed psi value: ', axis.psi.value)
    print('Changed Euler angles value: ', axis.euler_angles.value)
    pass