import numpy as np
import jax
import jax.numpy as jnp


def build_rotation_matrix(angles, seq:np.ndarray):
    """
    Build a rotation matrix from Euler angles.
    
    Parameters:
    -----------
    angles : jnp.ndarray
        Array of 3 Euler angles in radians
    seq : np.ndarray
        Sequence of rotation axes
        
    Returns:
    --------
    R : jnp.ndarray
        3x3 rotation matrix
    """
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
        """

        ct = jnp.cos(angles)
        st = jnp.sin(angles)

        cz = ct[0]
        cx = ct[2]
        cy = ct[1]
        sx = st[2]
        sy = st[1]
        sz = st[0]

        # Create rotation matrix using JAX
        R = jnp.array([
            [cy * cz, sy * sx * cz - sz * cx, sy * cx * cz + sz * sx],
            [cy * sz, sy * sx * sz + cz * cx, sy * cx * sz - cz * sx],
            [-sy, cy * sx, cy * cx]
        ])
    else:
        raise NotImplementedError
    
    return R


if __name__ == "__main__":
    # Example usage of the build_rotation_matrix function
    import jax
    
    # Enable 64-bit precision for more accurate results
    jax.config.update("jax_enable_x64", True)
    
    # Define Euler angles (in radians) for a sample rotation [tz, ty, tx]
    # This represents a 30° rotation about z-axis, 45° about y-axis, and 60° about x-axis
    angles = jnp.array([np.pi/6, np.pi/4, np.pi/3])  # [30°, 45°, 60°]
    
    # Define rotation sequence (3,2,1 corresponds to z-y-x sequence)
    seq = np.array([3, 2, 1])
    
    # Compute the rotation matrix
    R = build_rotation_matrix(angles, seq)
    
    # Print the rotation matrix
    print("Euler angles (radians):", angles)
    print("Euler angles (degrees):", jnp.degrees(angles))
    print("\nRotation matrix:")
    print(R)
    
    # Test the rotation on a sample vector
    test_vector = jnp.array([1.0, 0.0, 0.0])
    rotated_vector = jnp.matmul(R, test_vector)
    
    print("\nOriginal vector:", test_vector)
    print("Rotated vector:", rotated_vector)
    
    # Verify orthogonality of the rotation matrix (R^T * R should be identity)
    identity_check = jnp.matmul(R.T, R)
    print("\nOrthogonality check (should be identity):")
    print(identity_check)
    print("Determinant (should be 1.0):", jnp.linalg.det(R))
