import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

node_params = os.path.join(
    get_package_share_directory('armor_detector_nn'), 'config', 'node_params.yaml')
def generate_launch_description():


    armor_detector_nn_node = Node(
        package='armor_detector_nn',
        executable='armor_detector_nn_node',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[node_params]
    )

    return LaunchDescription([armor_detector_nn_node])
