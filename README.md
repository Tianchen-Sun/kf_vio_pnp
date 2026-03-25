# Kalman Filter for VIO correction in ROS2

## TODO

- [ ] verify the msg timestamp
- [ ] add data visualization

## Dependency

- python==3.13
- px4_msgs

## run the node

```
ros2 launch kf_vio_pnp kf_vio_pnp.python.launch.py
```

## IO

- input
  - the VIO odometry in ros2 msg: `nav_msgs/msg/odometry`
  - the pnp position in ros2 msg: `geometry_msgs/msg/pose_stamped`
- output
  - `px4_msgs/msg/vehicle_odometry`
