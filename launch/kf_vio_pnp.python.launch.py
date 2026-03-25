import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('kf_vio_pnp'),
        'config',
        'kf_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='kf_vio_pnp',
            executable='kf_node',
            name='kf_vio_pnp_node',
            output='screen',
            parameters=[config]
        )
    ])