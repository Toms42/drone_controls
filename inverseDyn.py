import math
import numpy as np
from scipy.spatial.transform import Rotation

def inverse_dyn(x, u, m):
    up = u[:3]  # linear accelerations
    thrust = m * (up.T @ up)
    psid = x[6]
    rotz = Rotation.from_euler(
        seq="ZYX", angles=[-psid, 0, 0]).as_matrix()
    z = rotz @ up / np.linalg.norm(up)
    phid = -math.atan2(z[1], z[2])
    thetad = math.atan2(z[0], z[2])
    return [thrust, phid, thetad, psid]