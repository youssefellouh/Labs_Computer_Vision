import numpy as np

from utils import *

def inverseHomogeneousMatrix(T):
    """! Compute the inverse of an homogeneous matrix `T`.
    @param[in] T  4-by-4 numpy array corresponding to an homogeneous matrix.
    @exception When `T` is not a 4-by-4 array raise an exception.
    @return  The inverse of the homogenous matrix `T`.
    """

    inv_T = np.eye(4)
    # BEGIN TO COMPLETE

    # END TO COMPLETE

    return inv_T

def multiplyHomogeneousMatrix(T1, T2):
    """! Return the product of two homogenous matrices.
    @param[in] T1  First homogeneous matrix to consider.
    @param[in] T2  Second homogeneous matrix to consider.
    @exception When  `T1` or `T2` are not a 4-by-4 array raise an exception.
    @return The homogeneous matrix corresponding to the product `T1 * T2`.
    """

    T12 = np.eye(4)
    # BEGIN TO COMPLETE

    # END TO COMPLETE

    return T12

def skew(t):
    """! Compute the skew symmetric matrix of vector `t`.
    @param[in] t  3-dim vector corresponding to a translation vector `(t_x, t_y, t_z)`.
    @exception When  `t` is not a 3-dim vector raise an exception.
    @return The 3-by-3 skew symmetric matrix.
    """

    sk = np.zeros((3,3))
    # BEGIN TO COMPLETE

    # END TO COMPLETE

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

    c2Hc1 = np.eye(3)
    # BEGIN TO COMPLETE

    # END TO COMPLETE

    return c2Hc1
