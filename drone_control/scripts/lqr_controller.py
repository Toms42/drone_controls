#!/usr/bin/env python
import numpy as np
import math
import control
from scipy.spatial.transform import Rotation
from simple_pid import PID
from collections import namedtuple
import matplotlib.pyplot as plt
import time
import copy

import roslib
import rospy
import tf
from mav_msgs.msg import RateThrust
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path

from python_optimal_splines.DroneTrajectory import DroneTrajectory
from inverseDyn import inverse_dyn

imu_data = None
def imu_cb(data):
    imu_data = data

def get_gate_positions(tf_listener, gate_ids, ref_frame='world', max_attempts=10):
    delay = 0.1
    gate_transforms = dict()
    attempts = max_attempts
    while (attempts > 0) and (len(gate_ids)) > 0:
        temp_gates = []
        for gate_id in gate_ids:
            try:
                (trans, rot) = tf_listener.lookupTransform(ref_frame, 'gate%d' % gate_id, rospy.Time(0))
                gate_transforms[gate_id] = (trans, rot)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                temp_gates.append(gate_id)
                continue
        
        print("Missing the following gates: ", temp_gates)
        gate_ids = temp_gates
        attempts -= 1
        time.sleep(delay)
    
    return gate_transforms

# rosparams
aggr = 0.1

# init rosnodes
rospy.init_node('lqr_controller')
tf_listener = tf.TransformListener()
imu_sub = rospy.Subscriber('/uav/sensors/imu', Imu, callback=imu_data, queue_size=1)
start_sim_pub = rospy.Publisher('/uav/input/arm', Empty, queue_size=1)
end_sim_pub = rospy.Publisher('/uav/input/reset', Empty, queue_size=1)
ctrl_pub = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=10)
path_pub = rospy.Publisher('ref_traj', Path, queue_size=1)

# Global Constants
Ixx = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_inertia_xx")
Iyy = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_inertia_yy")
Izz = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_inertia_zz")
m = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_mass")
g = 9.81

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
# get gate poses
num_gates = 3
gate_ids = list(range(1, num_gates+1))
gate_transforms = get_gate_positions(tf_listener, gate_ids)


# inital drone pose and generated spline of waypoints
print("Solving for optimal trajectory...")
dt = 0.01
rate = rospy.Rate(int(1./dt))
x0 = np.array([[0, 0, 1, 0, 0, 0, 0]]).T
drone_traj = DroneTrajectory()
drone_traj.set_start(position=list(x0[:3]), velocity=list(x0[3:6]))
for (trans, rot) in gate_transforms.values():
    drone_traj.add_gate(trans, rot)
drone_traj.solve(aggr)
print(drone_traj.trajectory.splines[0].ts)


# PID Controller for setting angular rates
pid_phi = PID(Kp=12, Ki=0, Kd=4, setpoint=0)
pid_theta = PID(Kp=12, Ki=0, Kd=4, setpoint=0)

# Optimal control law for flat system
print("Solving linear feedback system...")
K, S, E = control.lqr(A, B, Q, R)

# Generate trajectory (starttime-sensitive)
print("Generating optimal trajectory...")
start_time = rospy.get_time()
xref_traj = drone_traj.as_path(dt=dt, frame='world', start_time=rospy.Time.now())
path_pub.publish(xref_traj)

# plotting
N = len(xref_traj.poses)
time_axis = []
xref_traj_series = np.zeros((FLAT_STATES, N))
x_traj_series = np.zeros((FLAT_STATES, N))
x = x0

# run simulation
print("Running simulation and executing controls...")
iter = 0

# for pose in xref_traj.poses:
#     # publish arm command and ref traj
#     start_sim_pub.publish(Empty())
#     path_pub.publish(xref_traj)

start_sim_pub.publish(Empty())

#     # get next target waypoint
pose = xref_traj.poses[0]
t = rospy.get_time() - start_time
while not rospy.is_shutdown():
    start_sim_pub.publish(Empty())
    path_pub.publish(xref_traj)

    vx = drone_traj.val(t=t, order=1, dim=0)
    vy = drone_traj.val(t=t, order=1, dim=1)
    vz = drone_traj.val(t=t, order=1, dim=2)
    target_ori = [
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w]

    # TODO: Use these desired roll/pitch or the ones generated from fdbk law?
    [psid, thetad, phid] = Rotation.from_quat(target_ori).as_euler('ZYX')
    
    xref = np.array([[
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z,
        vx,
        vy,
        vz,
        psid]]).T
    xref_traj_series[:, iter] = np.ndarray.flatten(xref)

    # feedforward acceleration
    ff = np.array([[
        drone_traj.val(t=t, order=2, dim=0),
        drone_traj.val(t=t, order=2, dim=1),
        drone_traj.val(t=t, order=2, dim=2),
        0]]).T
    
    try:
        (trans, rot) = tf_listener.lookupTransform('world', 'uav/imu', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue

    # calculate linear velocities, define new state
    lin_vel = (np.array(trans) - x[:3,0]) / dt
    x[:3,0] = trans  # new position
    x[3:6,0] = lin_vel  # new linear velocity
    [psi, theta, phi] = Rotation.from_quat(rot).as_euler("ZYX")
    x[6] = psi
    x_traj_series[:, iter] = np.ndarray.flatten(x)

    u = -K*(x-xref) + ff + Gff
    [thrustd, phid, thetad, psid] = inverse_dyn(x, u, m)

    # generate desired roll, pitch rates, minimize error btwn desired and current
    dphi = pid_phi(phi - phid)
    dtheta = pid_theta(theta - thetad)
    dpsi = u[3]

    # convert rotation quaternion to euler angles and rotation matrix
    
    # rpy rates around around body axis
    new_ctrl = RateThrust()
    new_ctrl.angular_rates.x = dphi
    new_ctrl.angular_rates.y = dtheta
    new_ctrl.angular_rates.z = dpsi
    new_ctrl.thrust.z = thrustd
    ctrl_pub.publish(new_ctrl)

    # Plot results
    time_axis.append(t)
    iter += 1
    rate.sleep()

end_sim_pub.publish(Empty())


# plot x, y, z, vx, vy, vz
fig, axs = plt.subplots(3, 3)
axs[0, 0].set_title('X')
axs[0, 1].set_title('Y')
axs[0, 2].set_title('Z')
axs[1, 0].set_title('VX')
axs[1, 1].set_title('VY')
axs[1, 2].set_title('VZ')
for ax in axs.flat:
    ax.set(xlabel='Time(s)')
fig.suptitle('Target(red) v.s actual(green) roll and pitch')
for i in range(3):
    # position
    axs[0, i].scatter(time_axis, xref_traj_series[i,:], c = 'r')
    axs[0, i].scatter(time_axis, x_traj_series[i,:], c = 'g')

    # velocity
    axs[1, i].scatter(time_axis, xref_traj_series[i+3,:], c = 'r')
    axs[1, i].scatter(time_axis, x_traj_series[i+3,:], c = 'g')

plt.show()