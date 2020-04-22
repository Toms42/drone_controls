import numpy as np


def normalize(pos_list):
    """
    Normalizes a set of points with respect to their centroid.
    Args:
        pos_list: 2D list of 3D coordinates: [[x, y, z]...]

    Returns:

    """
    N = len(pos_list)
    norm_pos_list = []
    assert(N > 0)
    cx, cy, cz = 0, 0, 0
    for [x, y, z] in pos_list:
        cx += float(x) / N
        cy += float(y) / N
        cz += float(z) / N
    for i in range(N):
        [x, y, z] = pos_list[i]
        norm_pos_list.append([x - cx, y - cy, z - cz])
    return norm_pos_list, np.array([cx, cy, cz])


def best_fit_plane(pos_list):
    """
    Find the best fit plane given a list points.
    Args:
        pos_list: 2D list of 3D coordinates.
        visualize(default): set True to visualize the generated plane

    Return:
        normal vector of plane as a (3 x 1) numpy array of [x, y, z]
        particular solution, d to plane EQN: ax + by + cz = d
        centroid: single point guaranteed to be on the plane
    """
    # normalize each marker's position wrt to collective centroid
    norm_pos_list, centroid = normalize(pos_list)

    # store in 3 x N matrix with each col representing a marker
    point_matrix = np.array([pos for pos in norm_pos_list])

    # Calculate reduced SVD
    try:
        U, D, V = np.linalg.svd(point_matrix, full_matrices=False)
        normal = V[2].reshape(3,)
    except np.linalg.LinAlgError:
        print("SVD Failed, input matrix: ", point_matrix)
        return

    # Given any point on plane and normal vec, find standard EQN
    # ax + by + cz = d, or [x, y, z] * [a, b, c]^T
    d = np.array(centroid).dot(normal)
    return normal, d, centroid


def quat_from_vecs(v1, v2):
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    v1 = v1 / v1_mag
    v2 = v2 / v2_mag

    cos_theta = v1.T.dot(v2)
    if np.isclose(abs(cos_theta), 1.0, rtol=1e-5):
        if cos_theta > 0:  # if near 1, return identity
            return [0, 0, 0, 1]
        else:  # 180 deg rotation about any axis
            return [1, 0, 0, 0]

    quat = np.cross(v1, v2)  # xyz
    w = (v1_mag * v2_mag) + cos_theta
    return np.append(quat, [w])
