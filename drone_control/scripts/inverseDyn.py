import math
import numpy as np
from scipy.spatial.transform import Rotation

def inverse_dyn(xref, u, m, current_ori):
    up = u[:3] # linear accelerations
    
    psid = xref[6]
    rotz = Rotation.from_euler(
        seq="ZYX", angles=[-psid, 0, 0]).as_dcm()
    z = rotz.dot(up) / np.linalg.norm(up)
    phid = -math.atan2(z[1], z[2])
    thetad = math.atan2(z[0], z[2])
    rotx = Rotation.from_euler(
        seq="ZYX", angles=[0, 0, phid]).as_dcm()
    roty = Rotation.from_euler(
        seq="ZYX", angles=[0, thetad, 0]).as_dcm()

    new_rot = Rotation.from_quat(current_ori).as_dcm()
    up = new_rot.T.dot(rotx.dot(roty.dot(rotz.dot(up))))
    thrust = float(m * np.linalg.norm(up))
    return [thrust, phid, thetad, psid]
