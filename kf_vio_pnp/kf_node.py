import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry  # Using the PX4 message based on your definition

from kf_vio_pnp.kf_vio_pnp import VioAugmentedKalmanFilter, KFConfig


class KFNode(Node):
    def __init__(self):
        super().__init__('kf_vio_pnp_node')

        # Declare parameters
        self.declare_parameter('accel_noise_std', 1.0)
        self.declare_parameter('bias_rw_std', 0.02)
        self.declare_parameter('vio_pos_std', 0.20)
        self.declare_parameter('vio_vel_std', 0.30)
        self.declare_parameter('pnp_pos_std', 0.03)

        # Get parameters
        cfg = KFConfig(
            accel_noise_std=self.get_parameter('accel_noise_std').value,
            bias_rw_std=self.get_parameter('bias_rw_std').value,
            vio_pos_std=self.get_parameter('vio_pos_std').value,
            vio_vel_std=self.get_parameter('vio_vel_std').value,
            pnp_pos_std=self.get_parameter('pnp_pos_std').value
        )
        self.kf = VioAugmentedKalmanFilter(cfg)
        self.initialized = False

        # Subscribers
        self.sub_vio = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_visual_odometry',
            self.vio_callback,
            10
        )
        self.sub_pnp = self.create_subscription(
            PoseStamped,
            '/pnp_pose',
            self.pnp_callback,
            10
        )

        # Publisher for the fused state
        self.pub_fused = self.create_publisher(Odometry, '/fused_odom', 10)

    def init_filter(self, t, pos, vel):
        self.kf.init_state(pos=pos, vel=vel, t0=t)
        self.initialized = True
        self.get_logger().info(f"Kalman filter initialized at t={t}")

    def vio_callback(self, msg: VehicleOdometry):
        # Convert timestamp [us] to float seconds
        t = msg.timestamp / 1e6
        pos = [msg.position[0], msg.position[1], msg.position[2]]
        vel = [msg.velocity[0], msg.velocity[1], msg.velocity[2]]

        if not self.initialized:
            self.init_filter(t, pos, vel)
            return

        event = {"t": t, "type": "vio", "pos": pos, "vel": vel}
        self.kf.process_event(event)
        self.publish_fused(t)

    def pnp_callback(self, msg: PoseStamped):
        # Convert std_msgs/Header timestamp to float seconds
        t = (msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec) / 1e9
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        if not self.initialized:
            self.init_filter(t, pos, [0.0, 0.0, 0.0])
            return

        event = {"t": t, "type": "pnp", "pos": pos}
        self.kf.process_event(event)
        self.publish_fused(t)

    def publish_fused(self, t):
        p, v, b, P = self.kf.get_state()
        msg = Odometry()
        msg.header.stamp = Time(seconds=t).to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        
        msg.pose.pose.position.x = p[0]
        msg.pose.pose.position.y = p[1]
        msg.pose.pose.position.z = p[2]
        
        msg.twist.twist.linear.x = v[0]
        msg.twist.twist.linear.y = v[1]
        msg.twist.twist.linear.z = v[2]
        
        self.pub_fused.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = KFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()