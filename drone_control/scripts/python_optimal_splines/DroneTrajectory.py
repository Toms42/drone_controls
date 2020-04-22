from OptimalTrajectory import OptimalTrajectory, TrajectoryWaypoint
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from std_msgs.msg import Header
import rospy
from math import atan2, asin


class DroneGate:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class DroneTrajectory:
    def __init__(self):
        self.gates = []
        self.waypoints = []
        self.splines = []
        self.start_pos = None
        self.start_velocity = None
        self.end_pos = None
        self.end_velocity = None
        self.trajectory = None

        self.spacing = 0.5
        self.entry_radius = 0.25  # ensure that this is less than spacing/sqrt(3) to properly handle diagonal gates.

    def set_start(self, position, velocity):
        self.start_pos = position
        self.start_velocity = velocity

    def set_end(self, position, velocity):
        self.end_pos = position
        self.end_velocity = velocity

    def clear_gates(self):
        self.gates = []

    def add_gate(self, position, orientation):
        self.gates.append(DroneGate(position, orientation))

    def solve(self, aggressiveness):
        if self.start_pos is None:
            return None
        if len(self.gates) == 0 and self.end_pos is None:
            return None

        self.waypoints = []

        gate_waypoints = []
        for gate in self.gates:
            rotm = Rotation.from_quat(gate.orientation).as_dcm()
            entry_gate_pos = rotm.dot(np.array([-self.spacing, 0, 0]).transpose() + np.array(gate.position).transpose())
            exit_gate_pos = rotm.dot(np.array([self.spacing, 0, 0]).transpose() + np.array(gate.position).transpose())
            middle_gate_pos = np.array(gate.position).transpose()

            middle_gate_wp = TrajectoryWaypoint(tuple(middle_gate_pos.ravel()))
            entry_gate_wp = TrajectoryWaypoint(3)
            entry_gate_wp.add_soft_constraints(0, tuple(entry_gate_pos.ravel()), (self.entry_radius, self.entry_radius, self.entry_radius))
            exit_gate_wp = TrajectoryWaypoint(3)
            exit_gate_wp.add_soft_constraints(0, tuple(exit_gate_pos.ravel()), (self.entry_radius, self.entry_radius, self.entry_radius))

            gate_waypoints.extend([entry_gate_wp, middle_gate_wp, exit_gate_wp])

        start_waypoint = TrajectoryWaypoint(tuple(self.start_pos))
        start_waypoint.add_hard_constraints(1, tuple(self.start_velocity))

        self.waypoints.append(start_waypoint)
        self.waypoints.extend(gate_waypoints)

        if self.end_pos is not None:
            end_waypoint = TrajectoryWaypoint(tuple(self.end_pos))
            if self.end_velocity is not None:
                end_waypoint.add_hard_constraints(1, tuple(self.end_velocity))
            self.waypoints.append(end_waypoint)

        self.trajectory = OptimalTrajectory(5, 3, self.waypoints)
        self.trajectory.solve(aggressiveness)

    def val(self, t, order=0, dim=None):
        if self.trajectory is None:
            return None
        return self.trajectory.val(t, dim, order)

    def as_path(self, dt, start_time, frame='odom'):
        if self.trajectory is None:
            return None
        ts = np.arange(0, self.trajectory.end_time(), dt)

        poses = []
        for t in ts:
            pos = self.val(t)
            vel = self.val(t, order=1)

            pose = PoseStamped()
            pose.header = Header()
            pose.header.frame_id = frame
            pose.header.stamp = start_time + rospy.Duration(t)
            pose.pose = Pose()
            pose.pose.position = Point(pos[0], pos[1], pos[2])

            vel = np.array(vel) / np.linalg.norm(np.array(vel))

            psi = atan2(vel[1], vel[0])
            theta = asin(-vel[2])
            q = Rotation.from_euler('ZYX', [psi, theta, 0]).as_quat()
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            poses.append(pose)

        path = Path()
        path.header = Header
        path.header.frame_id = frame
        path.header.stamp = start_time
        path.poses = poses

        return path
