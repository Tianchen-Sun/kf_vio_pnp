import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
import csv
import os
from datetime import datetime
import numpy as np

from kf_vio_pnp.kf_vio_pnp import VioAugmentedKalmanFilter, KFConfig
from kf_vio_pnp.transform import Transform

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
        self.declare_parameter('mocap_as_pnp', False)

        # Get parameters
        cfg = KFConfig(
            accel_noise_std=self.get_parameter('accel_noise_std').value,
            bias_rw_std=self.get_parameter('bias_rw_std').value,
            vio_pos_std=self.get_parameter('vio_pos_std').value,
            vio_vel_std=self.get_parameter('vio_vel_std').value,
            pnp_pos_std=self.get_parameter('pnp_pos_std').value,
        )
        self.mocap_as_pnp=self.get_parameter('mocap_as_pnp').value

        self.transform = Transform(
            vio_yaw_rel_pnp=-1.57, # rad
            vio_translation_rel_pnp=[
                self.get_parameter('initial_pos_x').value,
                self.get_parameter('initial_pos_y').value,
                self.get_parameter('initial_pos_z').value
            ]
        )

        
        self.init_vel = [0.0, 0.0, 0.0]

        self.kf = VioAugmentedKalmanFilter(cfg)
        self.initialized = False

        # CSV logging setup
        self.csv_file = None
        self.csv_writer = None
        self.bias_log_path = None

        # Subscribers
        self.sub_vio = self.create_subscription(
            Odometry,
            '/d2vins/odometry',
            self.vio_callback,
            10
        )

        if self.mocap_as_pnp:
            self.sub_pnp = self.create_subscription(
                PoseStamped,
                '/mavros/vision_pose/pose',
                self.pnp_callback,
                10
            )

            # skip 100 count mocap data to simulate low-frequency PnP
            self.mocap_count = 0

        else:
            self.sub_pnp = self.create_subscription(
                PoseStamped,
                '/pnp_pose',
                self.pnp_callback,
                10
            )
   
        # Publisher for the fused state
        self.pub_fused = self.create_publisher(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', 10)

    def start_bias_logging(self, t):
        """Initialize CSV file for bias data logging"""
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.expanduser('./logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.bias_log_path = os.path.join(log_dir, f'bias_data_{timestamp}.csv')
            
            # Open CSV file and write header
            self.csv_file = open(self.bias_log_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'bias_x', 'bias_y', 'bias_z'])
            
            self.get_logger().info(f"Bias logging started: {self.bias_log_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to start bias logging: {e}")

    def log_bias(self, t):
        """Log current bias state to CSV file"""
        if self.csv_writer is None:
            return
        
        try:
            p, v, b, P = self.kf.get_state()
            self.csv_writer.writerow([t, b[0], b[1], b[2]])
            self.csv_file.flush()
        except Exception as e:
            self.get_logger().error(f"Failed to log bias: {e}")

    def stop_bias_logging(self):
        """Close CSV file"""
        if self.csv_file is not None:
            try:
                self.csv_file.close()
                self.get_logger().info(f"Bias logging stopped. Data saved to: {self.bias_log_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to close bias log: {e}")

    def init_filter(self, t, pos, vel):
        self.kf.init_state(pos=pos, vel=vel, t0=t)
        self.initialized = True
        self.start_bias_logging(t)
        self.get_logger().info(f"Kalman filter initialized at t={t}")
        # ros output rotation matrix
        self.get_logger().info(f"Rotation matrix from VIO to PnP:\n{self.transform.R_vio_to_pnp}")

    def vio_callback(self, msg: Odometry):
        # Convert timestamp to float seconds
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]

        # transform vio to pnp frame
        pos_transformed = self.transform.vio_to_pnp(pos)
        vel_transformed = self.transform.vio_to_pnp(vel)

        if not self.initialized:
            self.init_filter(t, pos_transformed, vel_transformed)
            return

        event = {"t": t, "type": "vio", "pos": pos_transformed, "vel": vel_transformed}
        self.kf.process_event(event)
        self.publish_fused(t)
        self.log_bias(t)

    def pnp_callback(self, msg: PoseStamped):
        # Convert std_msgs/Header timestamp to float seconds
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        if not self.initialized:
            self.init_filter(
                t, pos, self.init_vel
            )
            
            return
        
        if self.mocap_as_pnp:
            # Simulate low-frequency PnP by skipping some mocap messages
            self.mocap_count += 1
            
            # avoid mocap_count too large                
            if self.mocap_count > 10000:
                self.mocap_count = 1

            if self.mocap_count % 100 != 0:   
                return
            
        event = {"t": t, "type": "pnp", "pos": pos}
        self.kf.process_event(event)
        self.publish_fused(t)
        self.log_bias(t)

    def publish_fused(self, t):
        p, v, b, P = self.kf.get_state()
        msg = VehicleOdometry()
        msg.timestamp = int(t * 1e6)
        
        # convert p, v to float32 for VehicleOdometry message
        msg.position = np.array(p, dtype=np.float32).tolist()
        msg.velocity = np.array(v, dtype=np.float32).tolist()
        
        self.pub_fused.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = KFNode()
    try:
        rclpy.spin(node)
    finally:
        node.stop_bias_logging()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()