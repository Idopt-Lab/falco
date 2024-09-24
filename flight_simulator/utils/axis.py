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
        origin_valid = {'inertial', 'ref', 'cg'}
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
                 angles: Union[ureg.Quantity, csdl.Variable],
                 origin: str = 'ref',
                 sequence=np.array([3, 2, 1]),
                 reference = None):

        self.translation = csdl.Variable(
            shape=(3,),
            value=np.array([0, 0, 0]),
        )
        self.angles = csdl.Variable(
            shape=(3,),
            value=np.array([0, 0, 0]),
        )

        self.name = name
        self.translation = translation
        self.angles = angles
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
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, angles_vector):
        if angles_vector is None:
            self._angles = None
        elif isinstance(angles_vector, ureg.Quantity):
            vector_si = angles_vector.to_base_units()
            self._angles.set_value(vector_si.magnitude)
            self._angles.add_tag(str(vector_si.units))
        elif isinstance(angles_vector, csdl.Variable):
            self._angles = angles_vector
        else:
            raise IOError

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
        angles=np.array([0, 0, 0]) * ureg.degree,
        origin=ValidOrigins.Inertial.value
    )

    axis = Axis(name='Reference Axis',
                translation=np.array([0, 0, 0])*ureg.meter,
                angles=np.array([0, 0, 0])*ureg.degree,
                reference=inertial_axis,
                origin=ValidOrigins.Reference.value)

    # Translation automatically creates a CSDL variable
    print(axis.translation)
    print(axis.translation.value)

    # Update using the class' setter.
    # Here we need it to be a Pint Quantity with units
    # It will automatically be converted to base SI units
    axis.translation = np.array([3, 0, 0])*ureg.meter
    print(axis.translation)
    print(axis.translation.value)

    # Update by directly accessing the CSDL Variable and using the set_value() method
    axis.translation.set_value(np.array([0, 5, 0]))
    print(axis.translation)
    print(axis.translation.value)

    # Since the shape is given as (3, ), giving a scalar will broadcast
    axis.translation.set_value(5)
    print(axis.translation)
    print(axis.translation.value)
    pass