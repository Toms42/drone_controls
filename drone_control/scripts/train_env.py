#!/usr/bin/env python

# things needed:
# Class Environment
# reset()
# start()
# step([thrust, dphi, dtheta, dpsi])
# get_obs() --> [new state, reference state] ... + possibly other things with stability of angular rates

# initially test with the exact same track and starting position, just to see if agent can learn to imitate


import numpy as np 
import math
import control
from scipy.spatial.transform import Rotation
from simple_pid import PID
from collections import namedtuple
import matplotlib.pyplot as plt
import time
import copy
import threading

import roslib
import rospy
import tf
from mav_msgs.msg import RateThrust
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path

from python_optimal_splines.DroneTrajectory import DroneTrajectory
from inverseDyn import inverse_dyn


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class Environment(object):

    def __init__(self, aggr=None, env_name="DroneEnv", gate_ids=None):
        self.imu_data = None
        self.is_viz_ref_traj = True
        self.is_viz_ref_pose = True

        # init rosnodes
        rospy.init_node(env_name)
        self.imu_sub = rospy.Subscriber('/uav/sensors/imu', Imu, callback=self.imu_cb, queue_size=1)
        self.tf_listener = tf.TransformListener()
        self.start_sim_pub = rospy.Publisher('/uav/input/arm', Empty, queue_size=1)
        self.end_sim_pub = rospy.Publisher('/uav/input/reset', Empty, queue_size=1)
        self.ctrl_pub = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=10)
        self.path_pub = rospy.Publisher('ref_traj', Path, queue_size=1)
        self.tf_br = tf.TransformBroadcaster()

        if aggr is None: self.aggr = rospy.get_param("/splinegen/aggr")
        else: self.aggr = aggr

        # Drone Constants
        self.m = rospy.get_param("/uav/flightgoggles_uav_dynamics/vehicle_mass")
        self.g = 9.81
        self.x = self.xref = self.x0 = np.array([[0., 0., 1., 0., 0., 0., 0.]]).T
        self.xrot = [0, 0, 0, 1]  # zero rotation quat
        
        # Gate Positions
        self.available_gates = list(range(23))  # TODO: probably not the best method, but temporary
        self.gate_ids = list(range(0, 4))
        self.gate_transforms = self.get_gate_positions()

        # Simulation params
        self.dt = 0.05
        self.rate = rospy.Rate(int(1./self.dt))

        # Generate Trajectory
        self.generate_trajectory()

        # Safety -- ensure only access sim data when sim is running
        self.sim_running = False
        self.handle = None

    
    def start(self, Nsecs=5):
        assert(not self.sim_running)
        if self.handle is not None: 
            self.handle.join()
        # Generate trajectory (starttime-sensitive)
        print("Generating optimal trajectory...")
        self.start_time = rospy.get_time()
        self.track_time = 0
        self.sim_running = True
        
        # start sim in another thread
        self.handle = self.run_sim(Nsecs)


    @threaded
    def run_sim(self, Nsecs):
        run_forever = (Nsecs == -1)
        elapsed_time = rospy.get_time() - self.start_time
        while not rospy.is_shutdown() and (elapsed_time < Nsecs or run_forever):
            # publish arm command and ref traj
            self.start_sim_pub.publish(Empty())
            if self.is_viz_ref_traj: self.path_pub.publish(self.xref_traj)

            # update time
            elapsed_time = rospy.get_time() - self.start_time
            self.track_time = elapsed_time % self.max_time

            # get next target waypoint
            pos_g, vel_g, ori_g = self.drone_traj.full_pose(self.track_time)
            psid = 0
            
            self.xref = np.array([[
                pos_g[0], pos_g[1], pos_g[2],
                vel_g[0], vel_g[1], vel_g[2],
            psid]]).T

            if self.is_viz_ref_pose:
                self.tf_br.sendTransform((self.xref[0][0], self.xref[1][0], self.xref[2][0]),
                    ori_g,
                    rospy.Time.now(),
                    "xref_pose",
                    "world")

            self._get_agent_pose()

            self.rate.sleep()
        
        self.end()

    def end(self):
        self.sim_running = False
        self.end_sim_pub.publish(Empty())


    def step(self, action):
        [thrustd, dphi, dtheta, dpsi] = action
        new_ctrl = RateThrust()
        new_ctrl.angular_rates.x = dphi
        new_ctrl.angular_rates.y = dtheta
        new_ctrl.angular_rates.z = dpsi
        new_ctrl.thrust.z = thrustd
        self.ctrl_pub.publish(new_ctrl)

    def get_xref(self):
        assert(self.sim_running)
        return self.xref

    def get_feedfwd(self):
        # Used only by expert
        assert(self.sim_running)
        ff = np.array([[
            self.drone_traj.val(t=self.track_time, order=2, dim=0),
            self.drone_traj.val(t=self.track_time, order=2, dim=1),
            self.drone_traj.val(t=self.track_time, order=2, dim=2),
            0]]).T
        return ff

    def get_agent_pose(self):
        # external calls to this
        assert(self.sim_running)
        return self.x, self.xrot

    def _get_agent_pose(self):
        # only called within simulation
        assert(self.sim_running)
        try:
            (trans, self.xrot) = self.tf_listener.lookupTransform('world', 'uav/imu', rospy.Time(0))
            lin_vel = (np.array(trans) - self.x[:3,0]) / self.dt
            self.x[:3,0] = trans  # new position
            self.x[3:6,0] = lin_vel  # new linear velocity
            [psi, theta, phi] = Rotation.from_quat(self.xrot).as_euler("ZYX")
            self.x[6] = psi
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Couldn't listen to current state...")
            return

    def generate_trajectory(self):
        print("Solving for optimal trajectory...")
        self.drone_traj = DroneTrajectory()
        self.drone_traj.set_start(position=self.x0[:3], velocity=self.x0[3:6])
        self.drone_traj.set_end(position=self.x0[:3], velocity=self.x0[3:6])
        for gate_id in self.gate_ids:
            try:
                (trans, rot) = self.gate_transforms[gate_id]
                self.drone_traj.add_gate(trans, rot)
            except KeyError:
                print("Gate %d doesn't exist, skipping..." % gate_id)
                continue
                
        self.drone_traj.solve(self.aggr)
        self.xref_traj = self.drone_traj.as_path(dt=self.dt, frame='world', start_time=rospy.Time.now())
        self.max_time = self.xref_traj.poses[-1].header.stamp.to_sec() - rospy.get_time()
        

    def change_gate_ids(self, targets):
        print("Setting new gate targets, solving new trajectory...")
        self.gate_ids = targets
        self.gate_transforms = self.get_gate_positions()
        self.generate_trajectory()

    def imu_cb(self, data):
        self.imu_data = data

    def get_gate_positions(self, ref_frame='world', max_attempts=10):
        delay = 0.1
        gate_transforms = dict()
        attempts = max_attempts
        gate_ids = copy.copy(self.gate_ids)
        while (attempts > 0) and (len(gate_ids)) > 0:
            temp_gates = []
            for gate_id in gate_ids:
                try:
                    (trans, rot) = self.tf_listener.lookupTransform(ref_frame, 'gate%d' % gate_id, rospy.Time(0))
                    gate_transforms[gate_id] = (trans, rot)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    temp_gates.append(gate_id)
                    continue
            
            print("Missing the following gates: ", temp_gates)
            gate_ids = temp_gates
            attempts -= 1
            time.sleep(delay)
        
        return gate_transforms

class Expert(object):
    def __init__(self, dt, m):
        self.dt = dt
        self.m = m

        # Flat Dynamics
        g = 9.81
        self.FLAT_STATES = 7
        self.FLAT_CTRLS = 4
        self.A = np.zeros((self.FLAT_STATES, self.FLAT_STATES))
        self.A[0:3, 3:6] = np.eye(3)
        self.B = np.zeros((self.FLAT_STATES, self.FLAT_CTRLS))
        self.B[3:, :] = np.eye(4)
        self.Gff = np.array([[0, 0, g, 0]]).T  # gravity compensation
        self.Q = np.diag([10, 10, 10, 0.01, 0.01, 0.01, 10])
        self.R = np.eye(self.FLAT_CTRLS) * 5
        
        # Optimal control law for flat system
        print("Solving linear feedback system...")
        self.K, S, E = control.lqr(self.A, self.B, self.Q, self.R)

        # PID Controller for setting angular rates
        self.pid_phi = PID(Kp=7, Ki=0, Kd=1, setpoint=0)
        self.pid_theta = PID(Kp=7, Ki=0, Kd=1, setpoint=0)

    def change_pids(self, phi_params, theta_params):
        self.pid_phi = PID(
            Kp=phi_params[0], 
            Ki=phi_params[1], 
            Kd=phi_params[2], 
            setpoint=0)
        self.pid_theta = PID(
            Kp=theta_params[0], 
            Ki=theta_params[1], 
            Kd=theta_params[2], 
            setpoint=0)

    def change_controller_weights(self, Q=None, R=None):
        if Q is not None: self.Q = Q
        if R is not None: self.R = R
        print("Updating linear feedback system...")
        self.K, S, E = control.lqr(self.A, self.B, self.Q, self.R)

    def gen_action(self, x, rot, xref, ff):
        u = -self.K*(x-xref) + self.Gff + ff
        [thrustd, phid, thetad, psid] = inverse_dyn(x, u, self.m, rot)

        # generate desired roll, pitch rates, minimize error btwn desired and current
        [psi, theta, phi] = Rotation.from_quat(rot).as_euler('ZYX')
        dphi = self.pid_phi(phi - phid)
        dtheta = self.pid_theta(theta - thetad)
        dpsi = u[3]

        # take action
        action = [thrustd, dphi, dtheta, dpsi]
        return action

def test():
    env = Environment(aggr=1)
    expert = Expert(env.dt, env.m)
    Kp_vals = np.arange(start=5, stop=10, step=1.25)
    Kd_vals = np.arange(start=0, stop=3, step=1.5)
    for kp in Kp_vals:
        for kd in Kd_vals:
            expert.change_pids(
                phi_params=[kp, 0, kd],
                theta_params=[kp, 0, kd]
            )
            print("Kp: %.3f, Kd: %.3f" % (kp, kd))
            env.start(Nsecs=10)
            while env.sim_running:
                try:
                    # feedforward acceleration
                    ff = env.get_feedfwd()

                    # get target pose
                    xref = env.get_xref()
                    
                    # get current state
                    x, rot = env.get_agent_pose()
                except Exception as e:
                    print(e)
                    continue

                action = expert.gen_action(x, rot, xref, ff)
                env.step(action)
                env.rate.sleep()

if __name__=='__main__':
    test()