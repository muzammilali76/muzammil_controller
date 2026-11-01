"""
ROS 2 autonomous racing controller node for the CIE-KFUPM F1TENTH competition.

Subscribes to:
  - /scan (sensor_msgs/msg/LaserScan) with sensor_data QoS
  - /ego_racecar/odom (nav_msgs/msg/Odometry)

Publishes to:
  - /drive (ackermann_msgs/msg/AckermannDriveStamped)
"""

import math
import numpy as np

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .horizon_guidance import GuidanceConfig, HorizonGapGuidance
from .perception_processor import LidarPerceptionProcessor, ScanConfig


class AutonomousRaceController(Node):
    """
    Horizon-weighted disparity gap controller with lateral corridor centering,
    odometry yaw-rate damping, and Time-to-Collision (TTC) speed regulation.
    """

    def __init__(self) -> None:
        super().__init__("muzammil_f1tenth_controller")

        # Declare configurable ROS 2 parameters
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("base_frame", "ego_racecar/base_link")

        # Perception parameters
        self.declare_parameter("fov_half_angle_deg", 100.0)
        self.declare_parameter("min_range_m", 0.08)
        self.declare_parameter("max_horizon_m", 11.5)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("disparity_threshold_m", 0.35)
        self.declare_parameter("car_half_width_m", 0.165)
        self.declare_parameter("lateral_safety_margin_m", 0.115)
        self.declare_parameter("bubble_radius_m", 0.25)

        # Guidance & speed regulation parameters
        self.declare_parameter("max_steer_rad", 0.4189)
        self.declare_parameter("max_steer_rate_rad_s", 3.5)
        self.declare_parameter("kp_steer", 0.78)
        self.declare_parameter("kd_steer", 0.08)
        self.declare_parameter("kyaw_damping", 0.04)
        self.declare_parameter("corridor_centering_gain", 0.24)
        self.declare_parameter("min_gap_width_beams", 12)
        self.declare_parameter("gap_depth_floor_m", 1.10)
        self.declare_parameter("v_max_mps", 4.35)
        self.declare_parameter("v_min_mps", 1.50)
        self.declare_parameter("v_corner_mps", 1.85)
        self.declare_parameter("brake_decel_distance_m", 4.6)
        self.declare_parameter("ttc_critical_s", 0.18)
        self.declare_parameter("wall_repulsion_dist_m", 0.38)
        self.declare_parameter("wall_repulsion_gain", 0.48)
        self.declare_parameter("scan_timeout_s", 0.25)

        scan_cfg = ScanConfig(
            fov_half_angle_rad=math.radians(
                float(self.get_parameter("fov_half_angle_deg").value)
            ),
            min_range_m=float(self.get_parameter("min_range_m").value),
            max_horizon_m=float(self.get_parameter("max_horizon_m").value),
            smoothing_window=int(self.get_parameter("smoothing_window").value),
            disparity_threshold_m=float(
                self.get_parameter("disparity_threshold_m").value
            ),
            car_half_width_m=float(self.get_parameter("car_half_width_m").value),
            lateral_safety_margin_m=float(
                self.get_parameter("lateral_safety_margin_m").value
            ),
            bubble_radius_m=float(self.get_parameter("bubble_radius_m").value),
        )

        guidance_cfg = GuidanceConfig(
            max_steer_rad=float(self.get_parameter("max_steer_rad").value),
            max_steer_rate_rad_s=float(
                self.get_parameter("max_steer_rate_rad_s").value
            ),
            kp_steer=float(self.get_parameter("kp_steer").value),
            kd_steer=float(self.get_parameter("kd_steer").value),
            kyaw_damping=float(self.get_parameter("kyaw_damping").value),
            corridor_centering_gain=float(
                self.get_parameter("corridor_centering_gain").value
            ),
            min_gap_width_beams=int(
                self.get_parameter("min_gap_width_beams").value
            ),
            gap_depth_floor_m=float(
                self.get_parameter("gap_depth_floor_m").value
            ),
            v_max_mps=float(self.get_parameter("v_max_mps").value),
            v_min_mps=float(self.get_parameter("v_min_mps").value),
            v_corner_mps=float(self.get_parameter("v_corner_mps").value),
            brake_decel_distance_m=float(
                self.get_parameter("brake_decel_distance_m").value
            ),
            ttc_critical_s=float(self.get_parameter("ttc_critical_s").value),
            wall_repulsion_dist_m=float(
                self.get_parameter("wall_repulsion_dist_m").value
            ),
            wall_repulsion_gain=float(
                self.get_parameter("wall_repulsion_gain").value
            ),
        )

        self.perception = LidarPerceptionProcessor(scan_cfg)
        self.guidance = HorizonGapGuidance(guidance_cfg)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.scan_timeout_s = float(self.get_parameter("scan_timeout_s").value)

        self.linear_speed_mps: float = 0.0
        self.yaw_rate_rad_s: float = 0.0
        self.last_scan_time_ns: int | None = None

        scan_topic = str(self.get_parameter("scan_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        drive_topic = str(self.get_parameter("drive_topic").value)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            20,
        )

        self.watchdog_timer = self.create_timer(
            0.05, self.watchdog_callback
        )

        self.get_logger().info(
            f"Controller online | scan: {scan_topic}, odom: {odom_topic}, "
            f"drive: {drive_topic} | v_max={guidance_cfg.v_max_mps:.2f} m/s"
        )

    def odom_callback(self, msg: Odometry) -> None:
        """Update longitudinal speed and yaw rate from simulator odometry."""
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.linear_speed_mps = math.hypot(vx, vy)
        self.yaw_rate_rad_s = float(msg.twist.twist.angular.z)

    def scan_callback(self, msg: LaserScan) -> None:
        """Process incoming LaserScan and publish AckermannDriveStamped command."""
        now_ns = self.get_clock().now().nanoseconds
        if self.last_scan_time_ns is None:
            dt = 0.025
        else:
            dt = max((now_ns - self.last_scan_time_ns) * 1e-9, 1e-3)
        self.last_scan_time_ns = now_ns

        if len(msg.ranges) == 0 or msg.angle_increment <= 0.0:
            self.publish_drive_command(0.0, 0.0)
            return

        try:
            ranges_np = np.asarray(msg.ranges, dtype=np.float64)
            processed = self.perception.process(
                ranges_np,
                float(msg.angle_min),
                float(msg.angle_increment),
            )
            steer_rad, speed_mps = self.guidance.compute_command(
                processed,
                self.linear_speed_mps,
                self.yaw_rate_rad_s,
                dt,
            )
            self.publish_drive_command(steer_rad, speed_mps)
        except Exception as exc:
            self.get_logger().error(
                f"Scan processing fault; issuing safe stop: {exc}"
            )
            self.publish_drive_command(0.0, 0.0)

    def watchdog_callback(self) -> None:
        """Command zero velocity if sensor measurements time out."""
        if self.last_scan_time_ns is None:
            return
        elapsed_s = (
            self.get_clock().now().nanoseconds - self.last_scan_time_ns
        ) * 1e-9
        if elapsed_s > self.scan_timeout_s:
            self.publish_drive_command(0.0, 0.0)

    def publish_drive_command(
        self, steering_angle_rad: float, speed_mps: float
    ) -> None:
        """Publish stamped Ackermann command in radians and metres per second."""
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.base_frame
        cmd.drive.steering_angle = float(steering_angle_rad)
        cmd.drive.speed = float(speed_mps)
        self.drive_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutonomousRaceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_drive_command(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
