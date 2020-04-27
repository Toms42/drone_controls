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


def get_gate_positions(gate_ids, ref_frame='world', max_attempts=10):
    tf_listener = tf.TransformListener()
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


def main():
    # rosparams
    aggr = .1

    # init rosnodes
    rospy.init_node('lqr_controller')
    tf_listener = tf.TransformListener()
    imu_sub = rospy.Subscriber('/uav/sensors/imu', Imu, callback=imu_data, queue_size=1)
    start_sim_pub = rospy.Publisher('/uav/input/arm', Empty, queue_size=1)
    end_sim_pub = rospy.Publisher('/uav/input/reset', Empty, queue_size=1)
    ctrl_pub = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=10)
    path_pub = rospy.Publisher('ref_traj', Path, queue_size=1)
    tf_br = tf.TransformBroadcaster()

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
    A[0:3, 3:6] = np.eye(3)
    B = np.zeros((FLAT_STATES, FLAT_CTRLS))
    B[3:, :] = np.eye(4)
    Gff = np.array([[0, 0, g, 0]]).T  # gravity compensation
    Q = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 10])
    R = np.eye(FLAT_CTRLS) * 5

    # Trajectory generation
    # get gate poses
    num_gates = 4
    gate_ids = list(range(0, num_gates))
    gate_transforms = get_gate_positions(gate_ids)

    # inital drone pose and generated spline of waypoints
    print("Solving for optimal trajectory...")
    dt = 0.05
    rate = rospy.Rate(int(1./dt))
    x0 = np.array([[0., 0., 1., 0., 0., 0., 0.]]).T
    drone_traj = DroneTrajectory()
    drone_traj.set_start(position=x0[:3], velocity=x0[3:6])
    drone_traj.set_end(position=x0[:3], velocity=x0[3:6])
    for (trans, rot) in gate_transforms.values():
        drone_traj.add_gate(trans, rot)
    drone_traj.solve(aggr)

    # PID Controller for setting angular rates
    pid_phi = PID(Kp=7, Ki=0, Kd=1, setpoint=0)
    pid_theta = PID(Kp=7, Ki=0, Kd=1, setpoint=0)

    # Optimal control law for flat system
    print("Solving linear feedback system...")
    K, S, E = control.lqr(A, B, Q, R)

    # Generate trajectory (starttime-sensitive)
    print("Generating optimal trajectory...")
    start_time = rospy.get_time()
    xref_traj = drone_traj.as_path(dt=dt, frame='world', start_time=rospy.Time.now())
    max_time = xref_traj.poses[-1].header.stamp.to_sec() - start_time

    # plotting
    N = 500
    time_axis = []
    xref_traj_series = np.zeros((FLAT_STATES, N))
    x_traj_series = np.zeros((FLAT_STATES, N))

    # run simulation
    print("Running simulation and executing controls...")
    iter = 0
    x = x0
    prev_pose = None

    phid_traj = []
    thetad_traj = []
    psid_traj = []

    phi_traj = []
    theta_traj = []
    psi_traj = []
    # for pose in xref_traj.poses:
    while not rospy.is_shutdown():
        # publish arm command and ref traj
        start_sim_pub.publish(Empty())
        path_pub.publish(xref_traj)

        # get next target waypoint
        t = (rospy.get_time() - start_time) % max_time
        pos_g, vel_g, ori_g = drone_traj.full_pose(t)
        # vx = drone_traj.val(t=t, order=1, dim=0)
        # vy = drone_traj.val(t=t, order=1, dim=1)
        # vz = drone_traj.val(t=t, order=1, dim=2)

        # print(pose.pose.position.x, drone_traj.val(t=t, order=0, dim=0))
        # print(pose.pose.position.y, drone_traj.val(t=t, order=0, dim=1))
        # print(pose.pose.position.z, drone_traj.val(t=t, order=0, dim=2))
        # if prev_pose is not None:
        #     dx = pose.pose.position.

        # TODO: Use these desired roll/pitch or the ones generated from fdbk law?
        [psid, _, _] = Rotation.from_quat(ori_g).as_euler('ZYX')
        psid = 0
        
        xref = np.array([[
            pos_g[0], pos_g[1], pos_g[2],
            vel_g[0], vel_g[1], vel_g[2],
            psid]]).T
        # xref_traj_series[:, iter] = np.ndarray.flatten(xref)
        tf_br.sendTransform((xref[0][0], xref[1][0], xref[2][0]),
            ori_g,
            rospy.Time.now(),
            "xref_pose",
            "world")

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
        phi_traj.append(phi)
        theta_traj.append(theta)
        psi_traj.append(psi)
        # x_traj_series[:, iter] = np.ndarray.flatten(x)

        u = -K*(x-xref) + Gff + ff
        # print("%.3f, %.3f, %.3f" % (ff[0][0], ff[1][0], ff[2][0]))
        [thrustd, phid, thetad, psid] = inverse_dyn(x, u, m, rot)
        # [psid, thetad, phid] = Rotation.from_quat(ori_g).as_euler('ZYX')
        phid_traj.append(phid)
        thetad_traj.append(thetad)
        psid_traj.append(psid)

        # generate desired roll, pitch rates, minimize error btwn desired and current
        dphi = pid_phi(phi - phid)
        dtheta = pid_theta(theta - thetad)
        dpsi = u[3]

        # convert rotation quaternion to euler angles and rotation matrix
        
        # rpy rates around around body axis
        # print(thrustd)
        new_ctrl = RateThrust()
        new_ctrl.angular_rates.x = dphi
        new_ctrl.angular_rates.y = dtheta
        new_ctrl.angular_rates.z = dpsi
        new_ctrl.thrust.z = thrustd
        ctrl_pub.publish(new_ctrl)

        # Plot results
        # time_axis.append(t)
        iter += 1
        rate.sleep()

    end_sim_pub.publish(Empty())


    # fig, axs = plt.subplots(1, 3)
    # fig.suptitle('Target(red) v.s actual(green) roll and pitch')
    # axs[0].set_title('phi')
    # axs[1].set_title('theta')
    # axs[2].set_title('psi')
    # axs[0].scatter(time_axis, phid_traj, c = 'r')
    # axs[0].scatter(time_axis, phi_traj, c = 'g')
    # axs[1].scatter(time_axis, thetad_traj, c = 'r')
    # axs[1].scatter(time_axis, theta_traj, c = 'g')
    # axs[2].scatter(time_axis, psid_traj, c = 'r')
    # axs[2].scatter(time_axis, psi_traj, c = 'g')
    # plt.show()

    # plot x, y, z, vx, vy, vz
    # fig, axs = plt.subplots(3, 3)
    # axs[0, 0].set_title('X')
    # axs[0, 1].set_title('Y')
    # axs[0, 2].set_title('Z')
    # axs[1, 0].set_title('VX')
    # axs[1, 1].set_title('VY')
    # axs[1, 2].set_title('VZ')
    # for ax in axs.flat:
    #     ax.set(xlabel='Time(s)')
    # fig.suptitle('Target(red) v.s actual(green) roll and pitch')
    # for i in range(3):
    #     # position
    #     axs[0, i].scatter(time_axis, xref_traj_series[i,:iter], c = 'r')
    #     axs[0, i].scatter(time_axis, x_traj_series[i,:iter], c = 'g')

    #     # velocity
    #     axs[1, i].scatter(time_axis, xref_traj_series[i+3,:iter], c = 'r')
    #     axs[1, i].scatter(time_axis, x_traj_series[i+3,:iter], c = 'g')

    # plt.show()


if __name__ == '__main__':
    main()