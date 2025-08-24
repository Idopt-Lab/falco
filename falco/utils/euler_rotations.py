import numpy as np
import csdl_alpha as csdl


def build_rotation_matrix(angles: csdl.Variable, seq:np.ndarray):

    assert angles.shape == (3,)
    if np.all(seq == np.array([3, 2, 1])):
        """
        The rotation matrix R can be constructed as follows from
        input eul = [tz ty tx] and
        ct = cos(eul) = [cz cy cx] and
        st = sin(eul) = [sz sy sx]
        R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
               cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
                 -sy            cy*sx             cy*cx]
          = Rz(tz) * Ry(ty) * Rx(tx)
          
        The seq array [3, 2, 1] indicates:
        - seq[0] = 3 -> first rotation is about Z-axis (psi)
        - seq[1] = 2 -> second rotation is about Y-axis (theta)  
        - seq[2] = 1 -> third rotation is about X-axis (phi)
        
        If angles are provided as [phi, theta, psi], we need to reorder to [psi, theta, phi]
        """
        psi = angles[2]
        theta = angles[1]
        phi = angles[0]
        # Compute trigonometric functions
        cz = csdl.cos(phi)
        cy = csdl.cos(theta)
        cx = csdl.cos(psi)
        sz = csdl.sin(phi)
        sy = csdl.sin(theta)
        sx = csdl.sin(psi)

        R = csdl.Variable(shape=(3, 3), value=np.identity(3))

        R = R.set(csdl.slice[0, 0], cy * cz)
        R = R.set(csdl.slice[0, 1], sy * sx * cz - sz * cx)
        R = R.set(csdl.slice[0, 2],sy * cx * cz + sz * sx)
        R = R.set(csdl.slice[1, 0], cy * sz)
        R = R.set(csdl.slice[1, 1], sy * sx * sz + cz * cx)
        R = R.set(csdl.slice[1, 2], sy * cx * sz - cz * sx)
        R = R.set(csdl.slice[2, 0], -sy)
        R = R.set(csdl.slice[2, 1], cy * sx)
        R = R.set(csdl.slice[2, 2], cy * cx)

    else:
        raise NotImplementedError
    
    return R
