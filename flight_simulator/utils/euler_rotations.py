import numpy as np
import csdl_alpha as csdl


def build_rotation_matrix(angles: csdl.Variable, seq:np.ndarray):
    # todo: check this rotation again
    #  https://www.mathworks.com/help/robotics/ref/eul2rotm.html#d126e44534
    R = csdl.Variable(shape=(3, 3), value=np.identity(3))
    for dimen in seq:
        ang = angles[dimen - 1]
        # print(ang)
        if dimen == 1:
            D = csdl.Variable(shape=(3, 3), value=0.)
            D = D.set(csdl.slice[0, 0], 1.)
            D = D.set(csdl.slice[1, 1], csdl.cos(ang))
            D = D.set(csdl.slice[1, 2], csdl.sin(ang))
            D = D.set(csdl.slice[2, 1], -csdl.sin(ang))
            D = D.set(csdl.slice[2, 2], csdl.cos(ang))
        elif dimen == 2:
            D = csdl.Variable(shape=(3, 3), value=0.)
            D = D.set(csdl.slice[0, 0], csdl.cos(ang))
            D = D.set(csdl.slice[0, 2], csdl.sin(ang))
            D = D.set(csdl.slice[1, 1], 1.)
            D = D.set(csdl.slice[2, 0], -csdl.sin(ang))
            D = D.set(csdl.slice[2, 2], csdl.cos(ang))
        elif dimen == 3:
            D = csdl.Variable(shape=(3, 3), value=0.)
            D = D.set(csdl.slice[0, 0], csdl.cos(ang))
            D = D.set(csdl.slice[0, 1], -csdl.sin(ang))
            D = D.set(csdl.slice[1, 0], csdl.sin(ang))
            D = D.set(csdl.slice[1, 1], csdl.cos(ang))
            D = D.set(csdl.slice[2, 2], 1.)
        else:
            D = csdl.Variable(shape=(3, 3), value=np.identity(3))
        # print(D)
        R = csdl.matmat(D, R)
        # print(R)
    # R = csdl.transpose(R)
    return R