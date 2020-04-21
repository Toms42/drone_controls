import numpy as np 
import control
from scipy.spatial.transform import Rotation
from simple_pid import PID
from collections import namedtuple
import matplotlib.pyplot as plt
import time

import roslib
import rospy
import tf
from mav_msgs.msg import RateThrust
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu

import python_optimal_splines.OptimalSplineGen as OptimalSplineGen
from inverseDyn import inverse_dyn

imu_data = None
def imu_cb(data):
    imu_data = data

# init rosnodes
rospy.init_node('lqr_controller')
tf_listener = tf.TransformListener()
imu_sub = rospy.Subscriber('/uav/sensors/imu', Imu, callback=imu_data, queue_size=1)
start_sim_pub = rospy.Publisher('/uav/input/arm', Empty, queue_size=1)
ctrl_pub = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=10)

# Global Constants
Ixx = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_inertia_xx")
Iyy = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_inertia_yy")
Izz = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_inertia_zz")
m = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_mass")
g = 9.81
SPLINE_ORDER = 5  # to minimize snap(5th order)

# Flat Dynamics
FLAT_STATES = 7
FLAT_CTRLS = 4
A = np.zeros((FLAT_STATES, FLAT_STATES))
A[0:3,3:6] = np.eye(3)
B = np.zeros((FLAT_STATES, FLAT_CTRLS))
B[3:,:] = np.eye(4)
Gff = np.array([[0, 0, g, 0]]).T  # gravity compensation
Q = np.diag([10, 10, 10, 0.1, 0.1, 0.1, 1])
R = np.eye(FLAT_CTRLS)

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
rate = rospy.Rate(int(1./dt))
# N = len(pts) // dt
T = 10  # seconds
N = int((T+dt) // dt)  # num samples

# PID Controller for setting angular rates
pid_phi = PID(Kp=2, Ki=0, Kd=2, setpoint=0)
pid_theta = PID(Kp=2, Ki=0, Kd=2, setpoint=0)

# Optimal control law for flat system
K, S, E = control.lqr(A, B, Q, R)

# plotting
time_axis = []
start_time = time.time()

# start simulation
trans, rot = [0, 0, 0], [0, 0, 0, 1]
x = np.zeros((FLAT_STATES, 1))
xref = np.array([[1, 0, 1, 0, 0, 0, 0]]).T
while not rospy.is_shutdown():
    start_sim_pub.publish(Empty())
    try:
        (trans, rot) = tf_listener.lookupTransform('world', 'uav/imu', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
    cur_time = time.time()
    # print(trans)

    # calculate linear velocities, define new state
    lin_vel = (np.array(trans) - x[:3,0]) / dt
    x[:3,0] = trans  # new position
    x[3:6,0] = lin_vel  # new linear velocity
    # print(lin_vel)
    [psi, theta, phi] = Rotation.from_quat(rot).as_euler("ZYX")
    x[6] = psi

    u = -K*(x-xref) + Gff
    # ideal_x = x
    [thrustd, phid, thetad, psid] = inverse_dyn(x, u, m)
    # print(u[2], thrustd)
    # print(xref[2], x[2], u[2])

    # generate desired roll, pitch rates, minimize error btwn desired and current
    dphi = pid_phi(phi - phid)
    dtheta = pid_theta(theta - thetad)
    dpsi = u[3]
    # print("ang rates:")

    # convert rotation quaternion to euler angles and rotation matrix
    
    # rpy rates around around body axis
    new_ctrl = RateThrust()
    new_ctrl.angular_rates.x = dphi
    new_ctrl.angular_rates.y = dtheta
    new_ctrl.angular_rates.z = dpsi
    new_ctrl.thrust.z = thrustd
    ctrl_pub.publish(new_ctrl)

    # Plot results
    # time_axis.append(cur_time - start_time)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # time_axis = np.arange(0, T, dt)
    # ax1.plot(time_axis, x_traj[0,:])
    # ax1.set_title('X v.s time')
    # ax2.plot(time_axis, x_traj[1,:])
    # ax2.set_title('Y v.s time')
    # ax3.plot(time_axis, x_traj[2,:])
    # ax3.set_title('Z v.s time')
    # plt.show()

    rate.sleep()
