# `muzammil_controller` — CIE-KFUPM F1TENTH Autonomous Racing Controller

Autonomous racing controller package developed for the **CIE-KFUPM F1TENTH Autonomous Racing Competition (V1.1)** in ROS 2 Jazzy.

- **Author:** Muzammil Ali (MS in Systems & Control Engineering, Control & Instrumentation Engineering Dept., KFUPM)
- **Package Name:** `muzammil_controller`
- **Build Type:** `ament_python` (ROS 2 Jazzy)
- **Executable Command:** `ros2 run muzammil_controller controller`
- **Maximum Requested Speed (`v_max_mps`):** `4.35 m/s` (configurable via ROS 2 parameters; minimum cornering speed `1.50 m/s`, with `0.0 m/s` emergency Time-to-Collision stop)

## 1. Control Architecture Overview

The controller implements a map-free, reactive-plus-odometry control pipeline so that the exact same package operates across **Stage 1 (Levine clean-lap qualification)**, **Stage 2 (`levine_blocked` obstacle course)**, and **Stage 3 (hidden-track adaptability)** without relying on precomputed waypoints or external map coordinates:

1. **LiDAR Horizon Conditioning & Radial Disparity Inflation (`perception_processor.py`)**
   - Crops incoming 2D `LaserScan` returns to the forward hemisphere (`±100°`), sanitizes `NaN`/`Inf` returns, and applies a 5-point moving-average spatial filter.
   - Identifies sharp radial range discontinuities (`|r[i+1] - r[i]| > 0.35 m`) from hallway corners or track obstacles and inflates the closer boundary laterally by the vehicle half-width plus safety margin (`0.165 m + 0.115 m = 0.28 m`).
   - Masks a physical safety bubble (`0.25 m` radius) around the global nearest obstacle in the forward field of view.
2. **Horizon-Weighted Gap Selection & Corridor Centering (`horizon_guidance.py`)**
   - Extracts contiguous navigable free-space sectors above an adaptive depth threshold and computes a squared-depth, angular-continuity-weighted heading target.
   - Blends a symmetric lateral/diagonal wall-centering term along straight corridors (`forward_clearance > 3.8 m`) to eliminate wall-hugging before 90° turns, paired with a repulsive barrier term if any obstacle approaches within `0.38 m`.
3. **Odometry Yaw-Rate Damped PD Steering & Slew-Rate Limiter (`horizon_guidance.py`)**
   - Computes desired steering angle using a proportional-derivative heading law augmented with angular velocity damping (`ω_z` from `/ego_racecar/odom`) and bounded steering slew rate (`3.5 rad/s`) to suppress high-speed oscillation on Levine straightaways.
4. **Directional Clearance & Time-to-Collision (TTC) Speed Profile (`horizon_guidance.py` & `controller_node.py`)**
   - Modulates longitudinal speed smoothly between `v_min_mps = 1.50 m/s` and `v_max_mps = 4.35 m/s` based on steering angle magnitude and directional range clearance along the steered heading cone.
   - Triggers an immediate `0.0 m/s` command if directional Time-to-Collision drops below `0.18 s`, if any obstacle breaches `0.14 m`, if `/scan` messages time out (`> 0.25 s` watchdog), or upon node shutdown (`Ctrl+C`).

## 2. ROS 2 Interface & Dependencies

### Topic Interface

| Topic | Message Type | Direction | QoS / Notes |
|---|---|---|---|
| `/scan` | `sensor_msgs/msg/LaserScan` | Simulator → Controller | `qos_profile_sensor_data` |
| `/ego_racecar/odom` | `nav_msgs/msg/Odometry` | Simulator → Controller | Depth 20 |
| `/drive` | `ackermann_msgs/msg/AckermannDriveStamped` | Controller → Simulator | Radians (`steering_angle`) and m/s (`speed`) |

### Declared Dependencies (`package.xml`)

All dependencies are standard ROS 2 Jazzy / competition environment packages:
- `rclpy`
- `sensor_msgs`
- `nav_msgs`
- `ackermann_msgs`
- `std_msgs`
- `launch`
- `launch_ros`
- `ament_index_python`
- `python3-numpy`

## 3. Build and Run Instructions

Assume the competition starter repository is located at `<competition-repo>` (e.g., `~/CIE-KFUPM-F1tenth-Autonomous-Racing-Competition`).

### Step 1: Clone into your controller workspace and build

```bash
source /opt/ros/jazzy/setup.bash
source <competition-repo>/starter_ws/.venv/bin/activate
source <competition-repo>/starter_ws/install/setup.bash

mkdir -p ~/controller_ws/src
cd ~/controller_ws/src
git clone <repository-url> muzammil_controller

cd ~/controller_ws
python /usr/bin/colcon build --symlink-install
source install/setup.bash
```

Verify that ROS 2 registers the `controller` executable:

```bash
ros2 pkg executables muzammil_controller
# Expected output:
# muzammil_controller controller
```

### Step 2: Launch Simulator & Run Controller

**Terminal 1 — Start the competition simulator:**

```bash
cd <competition-repo>
./scripts/launch_sim.sh
```

**Terminal 2 — Source overlays and start the controller:**

```bash
source /opt/ros/jazzy/setup.bash
source <competition-repo>/starter_ws/.venv/bin/activate
source <competition-repo>/starter_ws/install/setup.bash
source ~/controller_ws/install/setup.bash

ros2 run muzammil_controller controller
```

*(Optional)* To run with a lower speed cap or custom parameter override from the command line:

```bash
ros2 run muzammil_controller controller --ros-args -p v_max_mps:=3.50
```

Or launch using the included launch file and YAML config:

```bash
ros2 launch muzammil_controller controller_launch.py
```

## 4. Configurable Parameters (`config/controller_params.yaml`)

| Parameter | Default | Units | Description |
|---|---|---|---|
| `scan_topic` | `"/scan"` | string | Subscribed 2D LiDAR scan topic |
| `odom_topic` | `"/ego_racecar/odom"` | string | Subscribed vehicle odometry topic |
| `drive_topic` | `"/drive"` | string | Published Ackermann drive topic |
| `fov_half_angle_deg` | `100.0` | deg | Forward LiDAR field-of-view half-angle |
| `smoothing_window` | `5` | beams | Spatial moving-average filter width |
| `disparity_threshold_m` | `0.35` | m | Radial jump threshold for disparity extension |
| `car_half_width_m` | `0.165` | m | Physical half-width of the F1TENTH chassis |
| `lateral_safety_margin_m` | `0.115` | m | Additional lateral clearance buffer around corners |
| `bubble_radius_m` | `0.25` | m | Closest-point obstacle zeroing radius |
| `max_steer_rad` | `0.4189` | rad | Physical steering angle saturation limit |
| `max_steer_rate_rad_s` | `3.5` | rad/s | Maximum steering slew rate |
| `kp_steer` | `0.78` | — | Proportional heading gain |
| `kd_steer` | `0.08` | s | Derivative heading damping gain |
| `kyaw_damping` | `0.04` | s | Odometry yaw-rate feedback damping gain |
| `corridor_centering_gain` | `0.24` | — | Lateral symmetry centering weight on straights |
| `v_max_mps` | `4.35` | m/s | **Maximum speed requested on clear straights** |
| `v_min_mps` | `1.50` | m/s | Minimum cornering speed at full steering lock |
| `v_corner_mps` | `1.85` | m/s | Target entry speed when approaching 90° turns |
| `brake_decel_distance_m` | `4.6` | m | Forward clearance distance to begin corner deceleration |
| `ttc_critical_s` | `0.18` | s | Directional Time-to-Collision threshold for zero-speed brake |
| `scan_timeout_s` | `0.25` | s | Sensor watchdog timeout before automatic stop |
