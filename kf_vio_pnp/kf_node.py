import csv
import os
from datetime import datetime
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from message_filters import ApproximateTimeSynchronizer, Subscriber as MFSubscriber
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition

from kf_vio_pnp.kf_vio_pnp import VioAugmentedKalmanFilter, KFConfig
from kf_vio_pnp.transform import Transform, ENUtoNEDTransform
from kf_vio_pnp.pnp import GateMap, PnPPoseCompose

class KFNode(Node):
    def __init__(self):
        """
        sub:
            - /d2vins/odometry (Odometry): 
                VIO pose and velocity
            - /mavros/vision_pose/pose (PoseStamped) (optional): 
                MoCap quadrotor pose to the world for debug
            
            - /gate_pose/pose (PoseStamped) (optional): 
                PnP gate pose to quadrotor
            - /gate_pose/id (Int32) (bind with gate_pose/pose): 
                gate id for PnP pose estimation
        """
        super().__init__('kf_vio_pnp_node')

        # Declare parameters
        self.declare_parameter('vio_delay_compensation', True)
        self.declare_parameter('vio_delay_sec', 0.05)
        self.declare_parameter('vio_bias_rw_std', 0.01)
        self.declare_parameter('vio_pos_std', 0.20)
        self.declare_parameter('vio_vel_std', 0.30)
        self.declare_parameter('imu_accel_noise_std', 0.2)
        
        self.declare_parameter('pnp_pos_std', 0.03)
        
        self.declare_parameter('init_pos_x', 0.0)
        self.declare_parameter('init_pos_y', 0.0)
        self.declare_parameter('init_pos_z', 0.0)
        self.declare_parameter('init_yaw_rad', 0.0)
        
        self.declare_parameter('gate_max_distance', 3.0) # m
        self.declare_parameter('gate_min_distance', 0.5) # m
        
        self.declare_parameter('mocap_as_pnp', False)
        self.declare_parameter('mocap_mode', 'continuous')
        self.declare_parameter('mocap_freq', 100.0)
        self.declare_parameter('mocap_available_duration', 2.0)
        self.declare_parameter('mocap_unavailable_duration', 5.0)

        # Get parameters
        cfg = KFConfig(
            imu_accel_noise_std=self.get_parameter('imu_accel_noise_std').value,
            vio_bias_rw_std=self.get_parameter('vio_bias_rw_std').value,
            vio_pos_std=self.get_parameter('vio_pos_std').value,
            vio_vel_std=self.get_parameter('vio_vel_std').value,
            pnp_pos_std=self.get_parameter('pnp_pos_std').value,
            vio_delay_compensation=self.get_parameter('vio_delay_compensation').value,
            vio_delay_sec=self.get_parameter('vio_delay_sec').value
        )
        
        self.mocap_mode = self.get_parameter('mocap_mode').value
        self.mocap_freq = self.get_parameter('mocap_freq').value
        self.mocap_available_duration = self.get_parameter('mocap_available_duration').value
        self.mocap_unavailable_duration = self.get_parameter('mocap_unavailable_duration').value
        
        self.gate_min_distance = self.get_parameter('gate_min_distance').value
        self.gate_max_distance = self.get_parameter('gate_max_distance').value
        self.gate_max_speed = self.get_parameter('gate_max_speed').value
        self.last_gate_pos_quad = {}  # Stores last valid (time, position) per gate ID

        # init gate pose map and pose composer
        self.gate_map = GateMap()
        self.pnp_pose_composer = PnPPoseCompose(self.gate_map)
        
        self.init_transform = Transform(
            vio_yaw_rel_pnp=self.get_parameter('init_yaw_rad').value,  # rad
            vio_translation_rel_pnp=[
                self.get_parameter('init_pos_x').value,
                self.get_parameter('init_pos_y').value,
                self.get_parameter('init_pos_z').value
            ]
        )
        self.enu_to_ned = ENUtoNEDTransform()
        
        # Init param and variable
        self.init_vel = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.last_accel_meas = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_imu_time = None

        self.kf = VioAugmentedKalmanFilter(cfg)
        self.initialized = False

        # CSV logging setup
        self.csv_file = None
        self.csv_writer = None
        self.bias_log_path = None

        # Subscribers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # IMU subscription - get linear acceleration
        # self.sub_imu = self.create_subscription(
        #     VehicleLocalPosition,
        #     '/fmu/out/vehicle_local_position',
        #     self.imu_callback,
        #     px4_qos
        # )
        
        mf_sub_pose = MFSubscriber(
            self,
            PoseStamped,
            '/gate_pose/pose',
            qos_profile=qos
        )

        mf_sub_id = MFSubscriber(
            self,
            Int32,
            '/gate_pose/id',
            qos_profile=qos
        )
        
        self.sync = ApproximateTimeSynchronizer(
            [mf_sub_pose, mf_sub_id],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        
        self.sync.registerCallback(self.gate_pose_callback)

        self.sub_vio = self.create_subscription(
            Odometry,
            '/d2vins/odometry',
            self.vio_callback,
            qos
        )
        
        self.sub_mocap = self.create_subscription(
            PoseStamped,
            '/mavros/vision_pose/pose',
            self.mocap_callback,
            qos
        )

        # Downsampling parameters for different modes
        self.mocap_count = 0
        self.mocap_init_time = None

        self.get_logger().info(f"MoCap mode: {self.mocap_mode}")
        if self.mocap_mode == 'downsampled':
            self.get_logger().info(f"Downsampling to {self.mocap_freq}Hz")
        elif self.mocap_mode == 'periodic':
            self.get_logger().info(f"Periodic mode: {self.mocap_available_duration}s available, "
                                    f"{self.mocap_unavailable_duration}s unavailable")
        
        # Publisher for the fused state
        self.pub_fused = self.create_publisher(
            VehicleOdometry, 
            '/fmu/in/vehicle_visual_odometry', 
            10
        )

        self.get_logger().info("KF Node initialized")


    def should_skip_mocap(self, t):
        """
        Determine if mocap measurement should be skipped based on mode
        """
        if self.mocap_mode == 'continuous':
            return False
        
        elif self.mocap_mode == 'downsampled':
            self.mocap_count += 1
            if self.mocap_count > 10000:
                self.mocap_count = 1
            
            skip_interval = int(100 * (1 / self.mocap_freq))
            return self.mocap_count % skip_interval != 0
        
        elif self.mocap_mode == 'periodic':
            if self.mocap_init_time is None:
                self.mocap_init_time = t
            
            elapsed_time = t - self.mocap_init_time
            cycle_duration = self.mocap_available_duration + self.mocap_unavailable_duration
            position_in_cycle = elapsed_time % cycle_duration
            
            should_skip = position_in_cycle >= self.mocap_available_duration
            
            if not should_skip:
                self.mocap_count += 1
                if self.mocap_count > 10000:
                    self.mocap_count = 1
                
                skip_interval = int(100 * (1 / self.mocap_freq))
                should_skip = self.mocap_count % skip_interval != 0
            
            return should_skip
        
        else:
            self.get_logger().warn(f"Unknown mocap_mode: {self.mocap_mode}")
            return False


    def init_log_bias(self, t):
        """Initialize CSV file for bias data logging"""
        try:
            # Get source code path directly
            # Get workspace root: install/share/kf_vio_pnp -> src/kf_vio_pnp/data/logs
            package_dir = get_package_share_directory('kf_vio_pnp')
            workspace_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            package_dir
                        )
                    )
                )
            )
        
            self.get_logger().info(f"Workspace root: {workspace_root}")
            log_dir = os.path.join(workspace_root, 'src', 'kf_vio_pnp', 'data', 'logs')
            
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
        """Log current VIO bias state to CSV file"""
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


    def is_valid_gate_pose(self, gate_id, gate_pos_quad, t):
        """
        Filter out gate poses that are too far, too close, or change too fast (m/s) between consecutive messages.
        Arg:
            - gate_id: ID of the current single detected gate, single int 
            - gate_pos_quad: position of the gate in quadrotor frame (3,)
            - t: timestamp of the measurement (s)
        """
        # Check if the overall distance to the gate is too small
        dist = np.linalg.norm(gate_pos_quad)
        if dist < self.gate_min_distance:
            self.get_logger().warn(f"Filtered gate {gate_id}: distance too small {dist:.2f}m < {self.gate_min_distance}m")
            return False
            
        # Check if the absolute distance in any axis is larger than the threshold
        if any(abs(axis) > self.gate_max_distance for axis in gate_pos_quad):
            self.get_logger().warn(f"Filtered gate {gate_id}: distance too large {gate_pos_quad}")
            return False
            
        # Update the last valid position and timestamp for this gate
        self.last_gate_pos_quad[gate_id] = (t, gate_pos_quad)
        return True


    def init_filter(self, t, pos, vel):
        self.kf.init_state(pos=pos, vel=vel, t0=t)
        self.initialized = True
        self.init_log_bias(t)
        self.get_logger().info(f"Kalman filter initialized at t={t}")
        self.get_logger().info(f"Rotation matrix from VIO to PnP:\n{self.init_transform.R_vio_to_pnp}")


    # def imu_callback(self, msg: VehicleLocalPosition):
    #     """
    #     IMU callback to get linear acceleration.
        
    #     Args:
    #         msg: VehicleLocalPosition message containing ax, ay, az
    #     """
    #     # Get timestamp
    #     t = msg.timestamp / 1e6  # Convert from microseconds to seconds
        
    #     # Get linear acceleration from IMU [m/s^2]
    #     accel_meas = np.array([msg.ax, msg.ay, msg.az], dtype=float)
        
    #     # Store for use in process_event
    #     self.last_accel_meas = accel_meas
    #     self.last_imu_time = t

    #     # update kalman filter with IMU measurement
    #     if self.initialized:
    #         self.kf.predict_with_imu(t, accel_meas)

    
    
    def gate_pose_callback(self, pose_msg: PoseStamped, id_msg: Int32):
        """
        Synchronized callback for gate pose and gate id. Use the gate pose as PnP measurement to update the filter. 
        """
        t = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec / 1e9
        gate_id = id_msg.data
        gate_pos_quad = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        
        if not self.is_valid_gate_pose(gate_id, gate_pos_quad, t):
            return
            
        T_g_to_q = self.pnp_pose_composer.get_T_g_to_q(np.array(gate_pos_quad, dtype=float))
        
        self.get_logger().info(f"Gate ID: {gate_id}, Position: {gate_pos_quad}")
        self.get_logger().info(f"T_g_to_q:\n{T_g_to_q}")
        
        # compute quadrotor pose in world frame using PnP pose composer
        result = self.pnp_pose_composer.comp_quadrotor_pose(gate_id, T_g_to_q)
        quad_pos_world = result.quadrotor_pose_world[:3].tolist()
    
        if not self.initialized:
            self.init_filter(t, quad_pos_world, self.init_vel)
            return
        
        event = {"t": t, "type": "pnp", "pos": quad_pos_world}
        self.kf.process_event(event, accel_meas=self.last_accel_meas)
        self.publish_fused(t)
        self.log_bias(t)
        

    def vio_callback(self, msg: Odometry):
        """
        VIO callback with IMU acceleration input.
        """
        # Convert timestamp to float seconds
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        ori = msg.pose.pose.orientation

        # Get yaw from VIO orientation
        self.yaw = self.init_transform.quaternion_to_yaw([ori.x, ori.y, ori.z, ori.w])  
        
        # Transform VIO to PnP frame
        pos_transformed = self.init_transform.vio_to_pnp(pos)
        vel_transformed = self.init_transform.vio_to_pnp(vel)
        self.yaw = self.init_transform.yaw_vio_to_pnp(self.yaw)

        if not self.initialized:
            self.init_filter(t, pos_transformed, vel_transformed)
            return

        event = {"t": t, "type": "vio", "pos": pos_transformed, "vel": vel_transformed}
        # Use IMU acceleration measurement from callback
        self.kf.process_event(event, accel_meas=self.last_accel_meas)
        self.publish_fused(t)
        self.log_bias(t)


    def mocap_callback(self, msg: PoseStamped):
        """
        PnP callback with IMU acceleration input.
        """
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        if not self.initialized:
            self.init_filter(t, pos, self.init_vel)
            return
        
        if self.mocap_as_pnp:
            if self.should_skip_mocap(t):
                return
            
        event = {"t": t, "type": "pnp", "pos": pos}
        
        # Use IMU acceleration measurement
        self.kf.process_event(event, accel_meas=self.last_accel_meas)
        self.publish_fused(t)
        self.log_bias(t)


    def publish_fused(self, t):
        """Publish fused odometry estimate"""
        p, v, b, P = self.kf.get_state()
        msg = VehicleOdometry()
        msg.timestamp = int(t * 1e6)
        
        # ENU to NED transformation
        p_ned = self.enu_to_ned.enu_to_ned_position(p)
        v_ned = self.enu_to_ned.enu_to_ned_position(v)
        self.yaw = self.enu_to_ned.enu_to_ned_yaw(self.yaw)
        
        # Convert p, v to float32
        msg.position = np.array(p_ned, dtype=np.float32)
        msg.velocity = np.array(v_ned, dtype=np.float32)
        
        # Convert yaw to quaternion
        quat = self.init_transform.yaw_to_quaternion(self.yaw)
        msg.q[0] = quat[0]
        msg.q[1] = quat[1]
        msg.q[2] = quat[2]
        msg.q[3] = quat[3]
        
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