import numpy as np
import math
import sys

# -----------  Basic Math Stuff  ---------------
def sign(x):
    """! Return the sign of `x`.
    @param[in] x  Value to test.
    @return  When value is negative return -1. When value is positive return 1. When value is equal to zero return 0.
    """
    if abs(x) < sys.float_info.epsilon:
        return 0
    else:
        return -1 if x < 0 else 1

def sinc(x):
    """! Compute the sinus cardinal of `x`.
    @param[in] x  Value to consider.
    @return  Sinus cardinal of `x` as `sin(x)/x`.
    """
    if abs(x) < 1e-8:
        return 1.0
    else:
        return math.sin(x) / x

def mcosc(cosx, x):
    """! Compute `(1-cos(x))/x^2`.
    @param[in] cosx  Value of `cos(x)`.
    @param[in] x  Value of `x`.
    @return  Value of `(1-cos(x))/x^2`.
    """
    if abs(x) < 2.5e-4:
        return 0.5
    else:
        return (1 - cosx) / x / x

def msinc(sinx, x):
    """! Compute `(1-sinc(x))/x^2` where `sinc(x) = sin(x)/x`.
    @param[in] sinx  Value of `sin(x)`.
    @param[in] x  Value of `x`.
    @return  Value of `(1-sin(x))/x^2`.
    """
    if abs(x) < 2.5e-4:
        return 1 / 6.0
    else:
        return (1 - sinx / x) / x /x

def deg2rad(angle_deg):
    """! Convert an angle from degrees to radians.
    @param[in] angle_deg  Angle in degrees.
    @return  Corresponding value in radians.
    """
    return angle_deg * math.pi / 180.0

def min(a,b):
    """! Find the minimum between two values `a` and `b`.
    @param[in] a  First value to consider.
    @param[in] b  Second value to consider.
    @return  The minimum of the two numbers.
    """
    if a <= b:
        return a
    else:
        return b

# -----------  Rotation Matrix  ----------------

def buildRotationFromThetaUVector(thetau):
    """! Build a rotation matrix from an angle-axis minimal representation.
    @param[in] thetau  3-dim numpy vector that contains `(thetau_x, thetau_y, thetau_z)`.
    @return  The corresponding 3x3 rotation matrix.
    """
    v =  np.array([0.0, 0.0, 0.0])

    v = thetau
    theta = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    si = math.sin(theta)
    co = math.cos(theta)
    sc = sinc(theta)
    mcc = mcosc(co,theta)

    R = np.eye(3)

    R[0][0] = co + mcc * v[0]**2
    R[0][1] = -sc * v[2] + mcc * v[0] * v[1]
    R[0][2] = sc * v[1] + mcc * v[0] * v[2]
    R[1][0] = sc * v[2] + mcc * v[1] * v[0]
    R[1][1] = co + mcc * v[1]**2
    R[1][2] = -sc * v[0] + mcc * v[1] * v[2]
    R[2][0] = -sc * v[1] + mcc * v[2] * v[0]
    R[2][1] = sc * v[0] + mcc * v[2] * v[1]
    R[2][2] = co + mcc * v[2]**2

    return R

def buildThetaUVectorFromRotation(R):
    """! Build an angle-axis minimal representation from a rotation matrix.
    @param[in] R  3x3 rotation matrix as a numpy array. .
    @return  3-dim numpy vector that corresponds to the angle-axis minimal representation which contains `(thetau_x, thetau_y, thetau_z)`.
    """

    s, c, theta = 0.0, 0.0, 0.0

    s = ((R[1][0] - R[0][1]) ** 2 + (R[2][0] - R[0][2]) ** 2 + (R[2][1] - R[1][2]) ** 2) ** 0.5 / 2.0
    c = (np.trace(R) - 1.0) / 2.0
    theta = math.atan2(s, c) # theta in [0, PI] since s > 0

    thetau = np.array([0,0,0]) # thetau is the rotation vector

    minimum = 0.0001
    # General case when theta != pi. If theta=pi, c=-1
    if (1 + c) > minimum: # Since -1 <= c <= 1, no fabs(1+c) is required
        sc = sinc(theta)
        #print("sinc = ", sc)
        #print(R)
        #print(R[2][1])

        thetau = np.array([(R[2][1] - R[1][2]) / (2 * sc),
                            (R[0][2] - R[2][0]) / (2 * sc),
                            (R[1][0] - R[0][1]) / (2 * sc)])

    else: # theta near PI
        x = 0
        if (R[0][0] - c) > sys.float_info.epsilon:
            x = math.sqrt((R[0][0] - c) / (1 - c))

        y = 0
        if (R[1][1] - c) > sys.float_info.epsilon:
            y = math.sqrt((R[1][1] - c) / (1 - c))

        z = 0
        if (R[2][2] - c) > sys.float_info.epsilon:
            z = math.sqrt((R[2][2] - c) / (1 - c))

        if x > y and x > z:
            if (R[2][1] - R[1][2]) < 0:
                x = -x
            if sign(x) * sign(y) != sign(R[0][1] + R[1][0]):
                y = -y
            if sign(x) * sign(z) != sign(R[0][2] + R[2][0]):
                z = -z
        elif y > z:
            if (R[0][2] - R[2][0]) < 0:
                y = -y
            if sign(y) * sign(x) != sign(R[1][0] + R[0][1]):
                x = -x
            if sign(y) * sign(z) != sign(R[1][2] + R[2][1]):
                z = -z
        else:
            if (R[1][0] - R[0][1]) < 0:
                z = -z
            if sign(z) * sign(x) != sign(R[2][0] + R[0][2]):
                x = -x
            if sign(z) * sign(y) != sign(R[2][1]+R[1][2]):
                y = -y
        thetau[0] = theta*x
        thetau[1] = theta*y
        thetau[2] = theta*z

    return thetau

#-----------  Exponential Map  ----------------

def exp(v, delta_t):
    """! Compute the exponential map of a vector.
    @param[in] v  Instantaneous velocity skew represented by a 6 dimension vector \f$ {\bf v} = [v, \omega] \f$
                  where \f$ v \f$ is a translation velocity vector and \f$ \omega \f$ is a rotation velocity vector.
    @param[in] delta_t  Sampling time in seconds corresponding to the time during which the velocity $ \bf v $ is applied.
    @return  The exponential map of vector `v` corresponding to an homogeneous matrix represented by a 4-by-4 numpy array.
    """

    v_dt = v * delta_t

    u = v_dt[3:]

    rd = buildRotationFromThetaUVector(u)

    theta = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    si = math.sin(theta)
    co = math.cos(theta)
    sinc = sinc(theta)
    mcosc = mcosc(co, theta)
    msinc = msinc(si, theta)

    dt = np.array([0.0, 0.0, 0.0])

    dt[0] = v_dt[0] * (sinc + u[0] * u[0] * msinc) + v_dt[1] * (u[0] * u[1] * msinc - u[2] * mcosc) + v_dt[2] * (u[0] * u[2] * msinc + u[1] * mcosc);

    dt[1] = v_dt[0] * (u[0] * u[1] * msinc + u[2] * mcosc) + v_dt[1] * (sinc + u[1] * u[1] * msinc) + v_dt[2] * (u[1] * u[2] * msinc - u[0] * mcosc);

    dt[2] = v_dt[0] * (u[0] * u[2] * msinc - u[1] * mcosc) + v_dt[1] * (u[1] * u[2] * msinc + u[0] * mcosc) + v_dt[2] * (sinc + u[2] * u[2] * msinc);

    Delta = np.eye(4)

    Delta[0:3,0:3] = rd

    Delta[0][3] = dt[0]
    Delta[1][3] = dt[1]
    Delta[2][3] = dt[2]

    return Delta

#------------ Homogeneous Matrix ----------------

def initHomogeneousMatrix(t, thetau):
    """! Initialize an homogeneous matrix from a translation vector and an angle-axis vector.
    @param[in] t  3-dim vector corresponding to the translation (t_x, t_y, t_z) in meter.
    @param[in] thetau  3-dim vector corresponding to the angle-axis minimal representation (thetau_x, thetau_y, thetau_z) in radians.
    @return  The corresponding 4-by-4 homogenous matrix as a numpy array.
    """
    T = np.eye(4)
    R = buildRotationFromThetaUVector(thetau)
    T[0:3,0:3] = R
    T[0][3] = t[0]
    T[1][3] = t[1]
    T[2][3] = t[2]

    return T

def printHomogeneousMatrix(T, header=''):
    """! Print the content of an homogeneous matrix `T` preceded by the `header` string content.
    @param[in] T  Homogeneous matrix to print.
    @param[in] header  Header printed before the homogenous matrix content.
    @exception When  `T` is not a 4-by-4 array raise an exception.
    """
    CRED = '\033[91m'
    CEND = '\033[0m'
    print(header)

    if T.shape != (4,4):
        raise ValueError(f"Matrix has shape {T.shape}, but should be (4,4)")

    for i in range(3):
        print(CRED, '\t', T[i][0],'\t', T[i][1],'\t', T[i][2], '\t',  CEND, T[i][3])
    print ('\t0\t 0\t 0\t 1')

#------------ Image Processing ----------------

def bilinearInterpolation(img, u, v):
    """! Compute the grey level value of pixel `(u,v)` in image `img` using bilinear interpolation.
    @param[in] img  Grey level image to consider.
    @param[in] u,v  Coordinates of the pixel `(u,v)` where `u` is the coordinate along the columns,
                    and `v` along the lines.
    @return The bilinear interpolated grey level value of pixel `(u,v)` in image `img` or O when the pixel `(u,v)` is outside the image.
    """
    rows, cols = img.shape

    if v < 0 or v > rows - 1 or u < 0 or u > cols - 1:
        return 0
    else:
        vround = math.floor(v)
        uround = math.floor(u)

        rratio = v - vround
        cratio = u - uround

        rfrac = 1.0 - rratio
        cfrac = 1.0 - cratio

        vround_1 = min(rows - 1, vround + 1)
        uround_1 = min(cols - 1, uround + 1)

        value = (img[vround][uround] * rfrac + img[vround_1][uround] * rratio) * cfrac + (img[vround][uround_1] * rfrac + (img[vround_1][uround_1]) * rratio) * cratio

        return round(value)
