import numpy as np 
import math
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
end_sim_pub = rospy.Publisher('/uav/input/reset', Empty, queue_size=1)
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
R = np.eye(FLAT_CTRLS) * 10

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

# gate position and quaternion
# traj = OptimalSplineGen.compute_min_derivative_spline(5, 3, 2, wpts)

# Controls Calculation
dt = 0.01
rate = rospy.Rate(int(1./dt))
# N = len(pts) // dt
T = 10  # seconds
N = int((T+dt) // dt)  # num samples

# PID Controller for setting angular rates
pid_phi = PID(Kp=7, Ki=0, Kd=1, setpoint=0)
pid_theta = PID(Kp=7, Ki=0, Kd=1, setpoint=0)

# Optimal control law for flat system
K, S, E = control.lqr(A, B, Q, R)

# plotting
time_axis = []
start_time = time.time()
fig = plt.figure()
roll_data, pitch_data = [], []
rolld_data, pitchd_data = [], []
roll_plot = fig.add_subplot(1, 2, 1)
roll_plot.set_ylabel('Roll')
pitch_plot = fig.add_subplot(1, 2, 2)
pitch_plot.set_xlabel('Time (s)')
pitch_plot.set_ylabel('Pitch')
fig.suptitle('Target(red) v.s actual(green) roll and pitch')

# start simulation
trans, rot = [0, 0, 0], [0, 0, 0, 1]
x = np.array([[0., 0., 1., 0., 0., 0., 0.]]).T

# oscillating btwn two rolls
target_i = 0
theta_targets = [math.pi/4, -math.pi/4]
thetad = theta_targets[target_i]
phid = 0

# oscillating between two positions
targets = [
    np.array([[2, 0, 1, 0, 0, 0, 0]]).T,
    np.array([[-2, 0, 1, 0, 0, 0, 0]]).T
]
xref_i = 0
xref = np.array([[-1.16694409e-03, -4.42462022e+00, 1.11156116e+00, 0, 0, 0, 0]]).T

iter = 0
while iter < 14/dt and not rospy.is_shutdown():
    start_sim_pub.publish(Empty())
    try:
        (trans, rot) = tf_listener.lookupTransform('world', 'uav/imu', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
    cur_time = time.time()

    # calculate linear velocities, define new state
    lin_vel = (np.array(trans) - x[:3,0]) / dt
    x[:3,0] = trans  # new position
    x[3:6,0] = lin_vel  # new linear velocity
    [psi, theta, phi] = Rotation.from_quat(rot).as_euler("ZYX")
    x[6] = psi

    u = -K*(x-xref) + Gff
    # [thrustd, phid, thetad, psid] = inverse_dyn(x, u, m)
    [thrustd, phid, thetad, psid] = inverse_dyn(x, u, m)

    # if iter % int(2.0/dt) == 0:
    #     target_i = 1 - target_i
    #     thetad = theta_targets[target_i]
        # xref_i = 1 - xref_i
        # xref = targets[xref_i]
        # print("New target: (%d,%d,%d)" % (xref[0], xref[1], xref[2]))

    # generate desired roll, pitch rates, minimize error btwn desired and current
    dphi = pid_phi(phi - phid)
    dtheta = pid_theta(theta - thetad)
    # print(dtheta)
    dpsi = u[3]

    # convert rotation quaternion to euler angles and rotation matrix
    
    # rpy rates around around body axis
    new_ctrl = RateThrust()
    new_ctrl.angular_rates.x = dphi
    new_ctrl.angular_rates.y = dtheta
    new_ctrl.angular_rates.z = dpsi
    new_ctrl.thrust.z = thrustd
    ctrl_pub.publish(new_ctrl)

    # oscillate btwn two points to get step responses of angles
    error = np.linalg.norm((x-xref)[:3])

    # Plot results
    time_axis.append(iter)
    roll_data.append(phi)
    rolld_data.append(phid)
    pitch_data.append(theta)
    pitchd_data.append(thetad)

    iter += 1
    rate.sleep()

end_sim_pub.publish(Empty())
roll_plot.scatter(time_axis, rolld_data, c = 'r')
roll_plot.scatter(time_axis, roll_data, c = 'g')

pitch_plot.scatter(time_axis, pitchd_data, c = 'r')
pitch_plot.scatter(time_axis, pitch_data, c = 'g')
plt.show()