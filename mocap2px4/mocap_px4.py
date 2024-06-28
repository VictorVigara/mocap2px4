import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, TransformStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
from mocap4r2_msgs.msg import RigidBodies
from tf2_ros import TransformBroadcaster

class MocapConversionNode(Node):
    def __init__(self):
        super().__init__('mocap_conversion_node')

        self.R_wFLU_wFRD = R.from_euler('x', np.pi).as_matrix()
        self.R_bFRD_bFLU = R.from_euler('x', np.pi).as_matrix()

        #Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.mocap_subscriber = self.create_subscription(
            RigidBodies,
            '/rigid_bodies',
            self.rigid_bodies_callback,
            10)
        
        self.local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_position_callback,
            qos_profile)

        self.pose_publisher = self.create_publisher(PoseStamped, '/px4/pose', 10)
        self.odom_px4_publisher = self.create_publisher(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', qos_profile)
        self.ekf_pose_publisher = self.create_publisher(PoseStamped, '/px4_ekf_pose', 10)
        self.mocap_initial_publisher = self.create_publisher(PoseStamped, '/mocap_initial', 10)
        self.mocap_publisher = self.create_publisher(PoseStamped, '/mocap_original', 10)
        self.pipe_publisher = self.create_publisher(PoseStamped, '/pipe_initial', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.initial_position = None
        self.initial_orientation_inv = None

    def rigid_bodies_callback(self, rb_msg: RigidBodies) -> None: 
        rbs = rb_msg.rigidbodies
        rb_pose = None
        rb_p = None
        rb_q = None
        for rb in rbs: 
            if rb.rigid_body_name == '47': 
                rb_pose = rb.pose
                rb_q = [rb_pose.orientation.x, rb_pose.orientation.y, rb_pose.orientation.z, rb_pose.orientation.w]
                rb_p = [rb_pose.position.x, rb_pose.position.y, rb_pose.position.z]
            if rb.rigid_body_name == '50': 
                
                pipe_pose = rb.pose
                pipe_q = [pipe_pose.orientation.x, pipe_pose.orientation.y, pipe_pose.orientation.z, pipe_pose.orientation.w]
                pipe_p = [pipe_pose.position.x, pipe_pose.position.y, pipe_pose.position.z]
                #print(f"Receiving pipe optitrack: {pipe_p}")

                if self.initial_position is not None: 
                    #print(f"Adjusting pipe optitrack")
                    pipe_q_wxyz = [pipe_q[-1], pipe_q[0], pipe_q[1], pipe_q[2]]
                    pos_adjusted, quat_adjusted = self.adjust_initial_position_and_orientation(pipe_p, pipe_q_wxyz)
                    opt_pipe_pose = PoseStamped()
                    opt_pipe_pose.header.stamp = self.get_clock().now().to_msg()
                    opt_pipe_pose.header.frame_id = "world"

                    opt_pipe_pose.pose.position.x = pos_adjusted[0]
                    opt_pipe_pose.pose.position.y = pos_adjusted[1]
                    opt_pipe_pose.pose.position.z = pos_adjusted[2]

                    opt_pipe_pose.pose.orientation.w = quat_adjusted[0]
                    opt_pipe_pose.pose.orientation.x = quat_adjusted[1]
                    opt_pipe_pose.pose.orientation.y = quat_adjusted[2]
                    opt_pipe_pose.pose.orientation.z = quat_adjusted[3]

                    self.pipe_publisher.publish(opt_pipe_pose)



        if self.initial_position is None and self.initial_orientation_inv is None and rb_pose != None:

            self.initial_position = np.array([rb_p[0], rb_p[1], 0.0])
            self.initial_orientation = R.from_quat(rb_q)    #xyzw

            self.initial_orientation_inv = self.initial_orientation.inv()
        
        if rb_pose != None:
            opt_pose = PoseStamped()
            opt_pose.header.stamp = self.get_clock().now().to_msg()
            opt_pose.header.frame_id = "world"

            opt_pose.pose.position.x = rb_p[0]
            opt_pose.pose.position.y = rb_p[1]
            opt_pose.pose.position.z = rb_p[2]

            opt_pose.pose.orientation.w = rb_q[3]
            opt_pose.pose.orientation.x = rb_q[0]
            opt_pose.pose.orientation.y = rb_q[1]
            opt_pose.pose.orientation.z = rb_q[2]

            self.mocap_publisher.publish(opt_pose)

            px4_odom_msg = self.convert_mocap_2_px4(rb_p, rb_q)
            self.odom_px4_publisher.publish(px4_odom_msg)
        

    def local_position_callback(self, msg):
        # Convert NED to ENU
        pos_ned = np.array([msg.x, msg.y, msg.z])
        
        # Rotation matrix for NED to ENU
        rotation_matrix = R.from_euler('x', np.pi).as_matrix()
        pos_enu = rotation_matrix.dot(pos_ned)
        
        # Convert heading to quaternion for yaw
        heading_enu = -msg.heading  # Convert NED heading to ENU heading
        quaternion = R.from_euler('z', heading_enu).as_quat()

        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = pos_enu[0]
        pose_msg.pose.position.y = pos_enu[1]
        pose_msg.pose.position.z = pos_enu[2]
        pose_msg.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
        
        self.ekf_pose_publisher.publish(pose_msg)

        self.publish_transform(pose_msg)

    def convert_mocap_2_px4(self, pose_received, q_received):
        """ Convert FLU to FRD by rotating around X-axis by π radians """
        quat_received = [q_received[-1], q_received[0], q_received[1], q_received[2]]   # w, x, y, z
        
        # Adjust current position with initial position
        pos_adjusted, quat_adjusted = self.adjust_initial_position_and_orientation(
            pose_received, 
            quat_received,
        )

        # Fill odom msg with initial pose

        opt_pose = PoseStamped()
        opt_pose.header.stamp = self.get_clock().now().to_msg()
        opt_pose.header.frame_id = "world"

        opt_pose.pose.position.x = pos_adjusted[0]
        opt_pose.pose.position.y = pos_adjusted[1]
        opt_pose.pose.position.z = pos_adjusted[2]

        opt_pose.pose.orientation.w = quat_adjusted[0]
        opt_pose.pose.orientation.x = quat_adjusted[1]
        opt_pose.pose.orientation.y = quat_adjusted[2]
        opt_pose.pose.orientation.z = quat_adjusted[3]

        self.mocap_initial_publisher.publish(opt_pose)

        ### TRANSFORM FROM ROS (W:FLU)
        q_adj_xyzw = [quat_adjusted[1], quat_adjusted[2], quat_adjusted[3], quat_adjusted[0]]

        # Calculate ROS FLUFLU to px4 FRDFRD
        p_b_wFRD = np.float32([pos_adjusted[0], -pos_adjusted[1], -pos_adjusted[2]])
        R_bFLU_wFLU = R.from_quat(q_adj_xyzw).as_matrix()
        q_bFRD_wFRD = R.from_matrix(self.R_wFLU_wFRD @ R_bFLU_wFLU @ self.R_bFRD_bFLU).as_quat()    # xyzw
        q_px4_wxyz = [q_bFRD_wFRD[3], q_bFRD_wFRD[0], q_bFRD_wFRD[1], q_bFRD_wFRD[2]]

        # Fill px4 odometry msg    
        odometry_px4_msg = VehicleOdometry()
        odometry_px4_msg.timestamp = int(self.get_clock().now().nanoseconds*0.001)   # microseconds
        
        odometry_px4_msg.pose_frame = 1  # NED world-fixed-frame
        odometry_px4_msg.position = p_b_wFRD
        odometry_px4_msg.position_variance = np.float32([0.001, 0.001, 0.001])

        odometry_px4_msg.q = np.float32(q_px4_wxyz)
        odometry_px4_msg.orientation_variance = np.float32([0.001, 0.001, 0.001])

        odometry_px4_msg.velocity_frame = 0  # Unknown
        odometry_px4_msg.velocity = np.float32([math.nan, math.nan, math.nan])
        odometry_px4_msg.velocity_variance = np.float32([math.nan, math.nan, math.nan])
        
        odometry_px4_msg.angular_velocity = np.float32([math.nan, math.nan, math.nan])  # body fixed frame
    
        return odometry_px4_msg

    def adjust_initial_position_and_orientation(self, pos, q_curr_wxyz):
        """ Adjust position and orientation based on initial values """
        pos = np.array(pos)
        initial_pos = np.array(self.initial_position)

        # Adjust position by subtracting the initial position
        pos_adjusted = pos - initial_pos

        # Rotate the adjusted position by the inverse of the initial orientation
        pos_adjusted = self.initial_orientation_inv.apply(pos_adjusted)

        q_curr_xyzw = [q_curr_wxyz[1], q_curr_wxyz[2], q_curr_wxyz[3], q_curr_wxyz[0]]
        current_orientation = R.from_quat(q_curr_xyzw)

        adjusted_orientation = self.initial_orientation_inv * current_orientation
        q_adjusted_xyzw = adjusted_orientation.as_quat()
        q_adj_wxyz = [q_adjusted_xyzw[3], q_adjusted_xyzw[0], q_adjusted_xyzw[1], q_adjusted_xyzw[2]]

        return pos_adjusted, q_adj_wxyz
    
    def publish_transform(self, pose):
        t = TransformStamped()
        
        t.header.stamp = pose.header.stamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = pose.pose.position.x
        t.transform.translation.y = pose.pose.position.y
        t.transform.translation.z = pose.pose.position.z
        
        t.transform.rotation.x = pose.pose.orientation.x
        t.transform.rotation.y = pose.pose.orientation.y
        t.transform.rotation.z = pose.pose.orientation.z
        t.transform.rotation.w = pose.pose.orientation.w
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MocapConversionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()