import numpy as np

from utils import *


def inverseHomogeneousMatrix(T):
    """! Compute the inverse of an homogeneous matrix `T`.
    @param[in] T  4-by-4 numpy array corresponding to an homogeneous matrix.
    @exception When `T` is not a 4-by-4 array raise an exception.
    @return  The inverse of the homogenous matrix `T`.
    """

    if T.shape != (4, 4):
        raise ValueError("Matrix T must be a 4x4 array.")

    # Inverse for a homogeneous transformation matrix
    R = T[:3, :3]
    t = T[:3, 3]

    # Compute the inverse
    inv_R = R.T
    inv_t = -inv_R @ t
    inv_T = np.eye(4)
    inv_T[:3, :3] = inv_R
    inv_T[:3, 3] = inv_t

    return inv_T


def multiplyHomogeneousMatrix(T1, T2):
    """! Return the product of two homogenous matrices.
    @param[in] T1  First homogeneous matrix to consider.
    @param[in] T2  Second homogeneous matrix to consider.
    @exception When  `T1` or `T2` are not a 4-by-4 array raise an exception.
    @return The homogeneous matrix corresponding to the product `T1 * T2`.
    """

    if T1.shape != (4, 4) or T2.shape != (4, 4):
        raise ValueError("Both T1 and T2 must be 4x4 matrices.")

    T12 = np.dot(T1, T2)

    return T12


def skew(t):
    """! Compute the skew symmetric matrix of vector `t`.
    @param[in] t  3-dim vector corresponding to a translation vector `(t_x, t_y, t_z)`.
    @exception When  `t` is not a 3-dim vector raise an exception.
    @return The 3-by-3 skew symmetric matrix.
    """

    if len(t) != 3:
        raise ValueError("Vector t must be 3-dimensional.")

    # Create skew-symmetric matrix
    sk = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    return sk


def DLT(u1, v1, u2, v2):
    """! From couples of matched points `p1=(u1,v1,1)` in image 1 and p2=(u2,v2,1)` in image 2
    with homogeneous coordinates in pixels, computes the homography matrix by resolving `p2 = H21 * p1`
    using the DLT (Direct Linear Transform) algorithm.

    At least 4 couples of points are needed.
    @param[in] u1  n-dim vector corresponding to pixel coordinate `u` of the n points `p1`.
    @param[in] v1  n-dim vector corresponding to pixel coordinate `v` of the n points `p1`.
    @param[in] u2  n-dim vector corresponding to pixel coordinate `u` of the n points `p2`.
    @param[in] v2  n-dim vector corresponding to pixel coordinate `v` of the n points `p2`.
    @exception When less than 4 points are given or when the size of the input vectors are not the same.
    @return The 3-by-3 homogenous matrix matrix H21.
    """

    if len(u1) < 4 or len(u1) != len(v1) or len(u2) != len(v2) or len(u1) != len(u2):
        raise ValueError("There must be at least 4 points and all input vectors must have the same length.")

    n = len(u1)
    A = []

    # Create matrix A for DLT
    for i in range(n):
        x1, y1 = u1[i], v1[i]
        x2, y2 = u2[i], v2[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

    A = np.array(A)

    # Perform SVD on A
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)  # Solution is the last row of V in SVD

    return H / H[2, 2]  # Normalize to make H[2,2] = 1