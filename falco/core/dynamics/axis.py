import numpy as np
from flight_simulator import ureg
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
    """Represents a coordinate axis with translation and orientation.

    Supports translation from an origin and orientation via Euler angles.
    Used as a reference for expressing positions, velocities, and rotations.

    Attributes
    ----------
    name : str
        Name of the axis.
    origin : str
        Origin identifier (must be a ValidOrigins value).
    translation_from_origin : Axis.translation_from_origin or None
        Translation from the origin.
    translation_from_origin_vector : csdl.Variable or None
        Translation vector [x, y, z] from the origin.
    translation : csdl.Variable or None
        Alias for translation_from_origin_vector.
    euler_angles : Axis.euler_angles or None
        Euler angles (phi, theta, psi) for orientation.
    euler_angles_vector : csdl.Variable or None
        Euler angles as a vector.
    sequence : any
        Euler rotation sequence.
    reference : object or None
        Reference axis or frame.
    """
    @dataclass
    class euler_angles(csdl.VariableGroup):
        """Euler angles for axis orientation.

        Attributes
        ----------
        phi : csdl.Variable
            Roll angle.
        theta : csdl.Variable
            Pitch angle.
        psi : csdl.Variable
            Yaw angle.
        """
        phi: csdl.Variable
        theta: csdl.Variable
        psi: csdl.Variable

        def define_checks(self):
            self.add_check('phi', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('theta', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('psi', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

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

    @dataclass
    class translation_from_origin(csdl.VariableGroup):
        """Translation from the origin for the axis.

        Attributes
        ----------
        x : csdl.Variable
            X-coordinate of translation.
        y : csdl.Variable
            Y-coordinate of translation.
        z : csdl.Variable
            Z-coordinate of translation.
        """
        x: csdl.Variable
        y: csdl.Variable
        z: csdl.Variable

        def define_checks(self):
            self.add_check('x', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('y', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)
            self.add_check('z', type=[csdl.Variable, ureg.Quantity], shape=(1,), variablize=True)

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

    @axis_checkers
    def __init__(self, name: str,
                 origin: str,
                 x: Union[ureg.Quantity, csdl.Variable] = None,
                 y: Union[ureg.Quantity, csdl.Variable] = None,
                 z: Union[ureg.Quantity, csdl.Variable] = None,
                 phi: Union[ureg.Quantity, csdl.Variable] = None,
                 theta: Union[ureg.Quantity, csdl.Variable] = None,
                 psi: Union[ureg.Quantity, csdl.Variable] = None,
                 sequence=None,
                 reference=None):
        """Initialize an Axis object.

        Parameters
        ----------
        name : str
            Name of the axis.
        origin : str
            Origin identifier (must be a ValidOrigins value).
        x, y, z : ureg.Quantity or csdl.Variable, optional
            Translation from the origin.
        phi, theta, psi : ureg.Quantity or csdl.Variable, optional
            Euler angles for orientation.
        sequence : any, optional
            Euler rotation sequence.
        reference : object, optional
            Reference axis or frame.
        """

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

    def copy(self, new_name: str = None):
        """Create a copy of the Axis object.

        Parameters
        ----------
        new_name : str, optional
            Name for the new Axis object.

        Returns
        -------
        Axis
            A new Axis object with the same properties as the original.
        """
        if new_name is None:
            self.name = self.name + "_copy"
        else:
            self.name = new_name

        # Copy translation variables if set
        if self.translation_from_origin is not None:
            new_x = csdl.Variable(
                value=self.translation_from_origin.x.value,
                shape=self.translation_from_origin.x.shape,
                name=self.translation_from_origin.x.name + "_copy"
            )
            new_y = csdl.Variable(
                value=self.translation_from_origin.y.value,
                shape=self.translation_from_origin.y.shape,
                name=self.translation_from_origin.y.name + "_copy"
            )
            new_z = csdl.Variable(
                value=self.translation_from_origin.z.value,
                shape=self.translation_from_origin.z.shape,
                name=self.translation_from_origin.z.name + "_copy"
            )
        else:
            new_x = new_y = new_z = None

        # Copy Euler angle variables if set
        if hasattr(self, 'euler_angles') and self.euler_angles is not None:
            new_phi = csdl.Variable(
                value=self.euler_angles.phi.value,
                shape=self.euler_angles.phi.shape,
                name=self.euler_angles.phi.name + "_copy"
            )
            new_theta = csdl.Variable(
                value=self.euler_angles.theta.value,
                shape=self.euler_angles.theta.shape,
                name=self.euler_angles.theta.name + "_copy"
            )
            new_psi = csdl.Variable(
                value=self.euler_angles.psi.value,
                shape=self.euler_angles.psi.shape,
                name=self.euler_angles.psi.name + "_copy"
            )
        else:
            new_phi = new_theta = new_psi = None

        return Axis(
            name=self.name,
            origin=self.origin,
            x=new_x,
            y=new_y,
            z=new_z,
            phi=new_phi,
            theta=new_theta,
            psi=new_psi,
            sequence=self.sequence,
            reference=self.reference
        )

    def csdl_copy(self, new_name: str = None):
        """
        Create a deep copy of the current Axis object.

        This method replicates all the Axis properties, including translation,
        Euler angles, sequence, reference, and origin, producing a new instance
        with the same configuration using the csdl copyvar functionality.

        Returns
        -------
        Axis
            A new Axis instance identical to the original.
        """
        if new_name is None:
            self.name = self.name + "_copy"
        else:
            self.name = new_name

        # Copy translation variables if set
        if self.translation_from_origin is not None:

            new_x = csdl.copyvar(self.translation_from_origin.x)
            new_y = csdl.copyvar(self.translation_from_origin.y)
            new_z = csdl.copyvar(self.translation_from_origin.z)
        else:
            new_x = new_y = new_z = None

        # Copy Euler angle variables if set
        if hasattr(self, 'euler_angles') and self.euler_angles is not None:
            new_phi = csdl.copyvar(self.euler_angles.phi)
            new_theta = csdl.copyvar(self.euler_angles.theta)
            new_psi = csdl.copyvar(self.euler_angles.psi)
        else:
            new_phi = new_theta = new_psi = None

        return Axis(
            name=self.name,
            origin=self.origin,
            x=new_x,
            y=new_y,
            z=new_z,
            phi=new_phi,
            theta=new_theta,
            psi=new_psi,
            sequence=self.sequence,
            reference=self.reference)


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