import numpy as np 
import control
from scipy.spatial.transform import Rotation
from simple_pid import PID
from collections import namedtuple

import matplotlib.pyplot as plt

import python_optimal_splines.OptimalSplineGen as OptimalSplineGen
from inverseDyn import inverse_dyn

# Global Constants
m = 5  # kg
g = 9.81
SPLINE_ORDER = 5  # to minimize snap(5th order)

# Flat Dynamics
FLAT_STATES = 7
FLAT_CTRLS = 4
A = np.zeros((FLAT_STATES, FLAT_STATES))
A[0:3,3:6] = np.eye(3)
B = np.zeros((FLAT_STATES, FLAT_CTRLS))
B[3:,:] = np.eye(4)
G = np.array([[0, 0, 0, 0, 0, -1, 0]]).T * g
Gff = np.array([[0, 0, g, 0]]).T  # gravity compensation
Q = np.diag([10, 10, 10, 0.1, 0.1, 0.1, 1])
R = np.eye(FLAT_CTRLS) * 0.1

# Trajectory generation
wpts = []
pts = []  # FILL IN
v0 = np.zeros((3,1))
a0 = np.zeros((3,1))
vg = np.zeros((3,1))
ag = np.zeros((3,1))
for pt in pts:
    (x, y, z, t) = pt
    wp = OptimalSplineGen.Waypoint(t)
    wp.add_hard_constraint(0, x, y, z)
    wpts.append(wp)

# traj = OptimalSplineGen.compute_min_derivative_spline(5, 3, 2, wpts)

# Controls Calculation
dt = 0.01
# N = len(pts) // dt
T = 10  # seconds
N = int((T+dt) // dt)  # num samples

# PID Controller for setting torques
pidX = PID(Kp=1, Ki=0.1, Kd=0.05)
pidY = PID(Kp=1, Ki=0.1, Kd=0.05)
pidZ = PID(Kp=1, Ki=0.1, Kd=0.05)

# Optimal control law for flat system
K, S, E = control.lqr(A, B, Q, R)

x0 = np.zeros((FLAT_STATES, ))
x_traj = np.zeros((FLAT_STATES, N))
# get flightgoggles' odometry to get new state
# velocity given in body frame, need to change to world frame
x_traj[:, 0] = x0
xref = np.array([[3, 5, 8, 0, 0, 0, 0]]).T
for i in range(1,N):
    t = i*dt
    # TODO: order = 0 for x, y, z...
    # xref = traj.val(order=0, t=t)
    # ff = traj.val(order=2, t=t)  # feedfwd accel
    ff = 0
    x = np.reshape(x_traj[:,i-1], newshape=(FLAT_STATES, 1))
    u = -K*(x-xref) + ff + Gff
    x = x + dt*(A.dot(x) + B.dot(u) + G)

    [Td, phid, thetad, psid] = inverse_dyn(x, u, m)
    x_traj[:,i] = np.reshape(x, newshape=(FLAT_STATES,))

    target_rot = Rotation.from_euler(seq="ZYX", angles=[psid, thetad, phid]).as_dcm()

    # Pid to estimate torque moments

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
time_axis = np.arange(0, T, dt)
ax1.plot(time_axis, x_traj[0,:])
ax1.set_title('X v.s time')
ax2.plot(time_axis, x_traj[1,:])
ax2.set_title('Y v.s time')
ax3.plot(time_axis, x_traj[2,:])
ax3.set_title('Z v.s time')
plt.show()
