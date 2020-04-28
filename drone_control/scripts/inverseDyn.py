import math
import numpy as np
from scipy.spatial.transform import Rotation

def inverse_dyn(q, x_ref, u, m):
    up = u[:3] # linear accelerations
    # up = np.array(
    #     np.ndarray.flatten(up).tolist()[0])

    # thrust = float(m * np.linalg.norm(up))
    normal_measured = Rotation.from_quat(q).apply([0, 0, 1])
    thrust = max(0, np.dot(up, normal_measured))
    psid = x_ref[6]
    rotz = Rotation.from_euler(
        seq="ZYX", angles=[-psid, 0, 0]).as_dcm()
    z = rotz.dot(up) / np.linalg.norm(up)
    phid = -math.atan2(z[1], z[2])
    thetad = math.atan2(z[0], z[2])
    return [thrust, phid, thetad, psid]
