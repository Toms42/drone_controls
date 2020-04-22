#!/usr/bin/env python
import numpy as np

import rospy
import tf2_ros
import geometry_msgs.msg

import geometry_helpers as geom


if __name__ == '__main__':
    rospy.init_node('gate_publisher')
    rate = rospy.Rate(10)
    br = tf2_ros.TransformBroadcaster()
    drone_init_pos = np.zeros((3,))
    world_vec = np.array([1, 0, 0])
    num_gates = 23
    gate_ids = list(range(1, num_gates+1))
    gate_transforms = []
    gate_normals = []

    # get all gate static transforms
    for gate_id in gate_ids:
        # TODO: Check if true gate locations(with perturbation included) are published
        try:
            corners = rospy.get_param("/uav/Gate%d/nominal_location" % gate_id)
        except:
            rospy.logwarn("Couldn't obtain get gate %d position" % gate_id)

        # corners of gates marked from left to right, top to bottom
        # assumes we move through gates in order of their id
        normal, d, centroid = geom.best_fit_plane(corners)

        # make sure all gates facing same direction
        if len(gate_normals) != 0:
            prev_normal = gate_normals[-1]
            if normal.dot(prev_normal) < 0:
                normal *= -1  # flip so same direction as previous gate
        # make sure first gate points away from drone initial position
        else:
            v_drone_to_gate = centroid - drone_init_pos
            if (normal.dot(v_drone_to_gate)) < 0:
                normal *= -1

        # get not-unique quaternion from world frame to this normal
        quat = geom.quat_from_vecs(normal, world_vec)
        gate_normals.append(normal)

        # represent gate as centroid
        t = geometry_msgs.msg.TransformStamped()
        t.transform.translation.x = centroid[0]
        t.transform.translation.y = centroid[1]
        t.transform.translation.z = centroid[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        t.header.frame_id = "world"
        t.child_frame_id = "gate%d" % gate_id

        # don't assign timestamp or id, reuse
        gate_transforms.append(t)

    while not rospy.is_shutdown():
        for t in gate_transforms:
            t.header.stamp = rospy.Time.now()
            br.sendTransform(t)

        rate.sleep()