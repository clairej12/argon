import argon.numpy as npx
import argon.transforms as agt
import argon.typing as atp
import shapely.geometry as sg

def _polygon_overlap_cb(pointsA, pointsB):
    polyA = sg.Polygon(pointsA)
    polyB = sg.Polygon(pointsB)
    return npx.array(polyA.intersection(polyB).area / polyA.area, dtype=npx.float32)

def polygon_overlap(pointsA, pointsB):
    return agt.pure_callback(
        _polygon_overlap_cb,
        atp.ShapeDtypeStruct((), npx.float32),
        pointsA, pointsB
    )

def quat_to_angle(quat):
    w0 = quat[0] # cos(theta/2)
    w3 = quat[3] # sin(theta/2)
    angle = 2*npx.atan2(w3, w0)
    return angle

def angle_to_rot2d(angle):
    return npx.array([
        [npx.cos(angle), -npx.sin(angle)],
        [npx.sin(angle), npx.cos(angle)]
    ])

def quat_to_mat(quat):
    """
    Adapted from diffusion_policy quatmath.py.
    Converts given quaternion to matrix.

    Args:
        quat (npx.array): (x,y,z,w) vec4 float angles

    Returns:
        npx.array: 3x3 rotation matrix
    """
    quat = npx.asarray(quat, dtype=npx.float32)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = npx.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = npx.empty(quat.shape[:-1] + (3, 3), dtype=npx.float32)
    mat = mat.at[..., 0, 0].set(1.0 - (yY + zZ))
    mat = mat.at[..., 0, 1].set(xY - wZ)
    mat = mat.at[..., 0, 2].set(xZ + wY)
    mat = mat.at[..., 1, 0].set(xY + wZ)
    mat = mat.at[..., 1, 1].set(1.0 - (xX + zZ))
    mat = mat.at[..., 1, 2].set(yZ - wX)
    mat = mat.at[..., 2, 0].set(xZ - wY)
    mat = mat.at[..., 2, 1].set(yZ + wX)
    mat = mat.at[..., 2, 2].set(1.0 - (xX + yY))
    return npx.where((Nq > 1e-6)[..., npx.newaxis, npx.newaxis], mat, npx.eye(3))

def mat_to_quat(rmat):
    """
    Adapted from robosuite transform_utils.py
    Converts given rotation matrix to quaternion.

    Args:
        rmat (npx.array): 3x3 rotation matrix

    Returns:
        npx.array: (x,y,z,w) float quaternion angles
    """
    M = npx.asarray(rmat).astype(npx.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = npx.array(
        [
            [m00 - m11 - m22, npx.float32(0.0), npx.float32(0.0), npx.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, npx.float32(0.0), npx.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, npx.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = npx.linalg.eigh(K)
    inds = npx.array([3, 0, 1, 2])
    q1 = V[inds, npx.argmax(w)]
    q1 *= npx.sign(q1[0])
    inds = npx.array([1, 2, 3, 0])
    return q1[inds]

def mat_to_euler(mat):
    """ 
    Adapted from diffusion_policy quatmath.py. 
    Convert Rotation Matrix to Euler Angles. 
    """
    mat = npx.asarray(mat, dtype=npx.float32)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = npx.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > 1e-6
    euler = npx.empty(mat.shape[:-1], dtype=npx.float32)
    euler = euler.at[..., 2].set(npx.where(condition,
                             -npx.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -npx.arctan2(-mat[..., 1, 0], mat[..., 1, 1])))
    euler = euler.at[..., 1].set(npx.where(condition,
                             -npx.arctan2(-mat[..., 0, 2], cy),
                             -npx.arctan2(-mat[..., 0, 2], cy)))
    euler = euler.at[..., 0].set(npx.where(condition,
                             -npx.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0))
    return euler

def orientation_error(desired, current):
    """
    Adapted from robosuite control_utils.py
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (npx.cross(rc1, rd1) + npx.cross(rc2, rd2) + npx.cross(rc3, rd3))

    return error