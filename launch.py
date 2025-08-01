from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments untuk port custom
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/ttyUSB0',
        description='Serial port untuk RPLiDAR A2M12 (contoh: /dev/ttyUSB0, /dev/ttyUSB1)'
    )
    
    lidar_baudrate_arg = DeclareLaunchArgument(
        'lidar_baudrate',
        default_value='115200',
        description='Baudrate untuk RPLiDAR A2M12'
    )
    
    lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='laser',
        description='Frame ID untuk LiDAR data'
    )

    return LaunchDescription([
        # Declare arguments
        lidar_port_arg,
        lidar_baudrate_arg,
        lidar_frame_arg,
        
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            respawn=True,
            respawn_delay=2.0
        ),
        
        Node(
            package='gamepad_robot_controller',
            executable='gamepad_controller_node2',
            name='manual_controller',
            output='screen',
            respawn=True,
            respawn_delay=2.0
        ),
        
        Node(
            package='uwbSENSOR',
            executable='lidar_uwb_node2',
            name='auto_controller',
            output='screen',
            respawn=True,
            respawn_delay=3.0
        ),
        
        # RPLiDAR dengan konfigurasi port custom
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('lidar_port'),
                'serial_baudrate': LaunchConfiguration('lidar_baudrate'),
                'frame_id': LaunchConfiguration('lidar_frame'),
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Sensitivity',  # Mode untuk A2M12
            }],
            respawn=True,
            respawn_delay=5.0,
            # Remapping topik jika diperlukan
            remappings=[
                ('scan', '/scan'),
            ]
        ),
        
        Node(
            package='FollowME',
            executable='mode_switcher',
            name='mode_switcher',
            output='screen',
            respawn=True,
            respawn_delay=1.0
        ),
    ])
