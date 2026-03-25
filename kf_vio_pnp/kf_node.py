import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry

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
        self.declare_parameter('initial_pos_x', 0.0)
        self.declare_parameter('initial_pos_y', 0.0)
        self.declare_parameter('initial_pos_z', 0.0)

        # Get parameters
        cfg = KFConfig(
            accel_noise_std=self.get_parameter('accel_noise_std').value,
            bias_rw_std=self.get_parameter('bias_rw_std').value,
            vio_pos_std=self.get_parameter('vio_pos_std').value,
            vio_vel_std=self.get_parameter('vio_vel_std').value,
            pnp_pos_std=self.get_parameter('pnp_pos_std').value,
            initial_pos=[
                self.get_parameter('initial_pos_x').value,
                self.get_parameter('initial_pos_y').value,
                self.get_parameter('initial_pos_z').value
            ]
        )
        self.kf = VioAugmentedKalmanFilter(cfg)
        self.initialized = False

        # Subscribers
        self.sub_vio = self.create_subscription(
            Odometry,
            '/d2vins/odometry',
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
        self.pub_fused = self.create_publisher(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', 10)

    def init_filter(self, t, pos, vel):
        self.kf.init_state(pos=pos, vel=vel, t0=t)
        self.initialized = True
        self.get_logger().info(f"Kalman filter initialized at t={t}")

    def vio_callback(self, msg: Odometry):
        # Convert timestamp to float seconds
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]

        if not self.initialized:
            self.init_filter(t, pos, vel)
            return

        event = {"t": t, "type": "vio", "pos": pos, "vel": vel}
        self.kf.process_event(event)
        self.publish_fused(t)

    def pnp_callback(self, msg: PoseStamped):
        # Convert std_msgs/Header timestamp to float seconds
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        if not self.initialized:
            self.init_filter(t, pos, [0.0, 0.0, 0.0])
            return

        event = {"t": t, "type": "pnp", "pos": pos}
        self.kf.process_event(event)
        self.publish_fused(t)

    def publish_fused(self, t):
        p, v, b, P = self.kf.get_state()
        msg = VehicleOdometry()
        msg.timestamp = int(t * 1e6)
        msg.position = p
        msg.velocity = v
        
        self.pub_fused.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = KFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()