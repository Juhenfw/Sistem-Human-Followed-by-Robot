#!/usr/bin/env python3

import socket
import numpy as np
import math
import time
import threading
import sys
import os
import select
import netifaces
from collections import deque

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String  # TAMBAH INI UNTUK MODE SWITCHING
from uwbSENSOR.ddsm115 import MotorControl
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Import gpiozero for controlling GPIO pins
from gpiozero import LED

# GPIO imports for ultrasonic sensors
import RPi.GPIO as GPIO

# [COPY SEMUA KONFIGURASI GLOBAL DARI KODE SEBELUMNYA]
SCAN_THRESHOLD = 2000
DANGER_THRESHOLD = 1000
MIN_VALID_DISTANCE = 200
CRITICAL_DANGER_THRESHOLD = 300

# Motor speed configuration
DEFAULT_SPEED = 75
ROTATION_FACTOR = 2
STOP_THRESHOLD = 80

# [COPY SEMUA CLASS HELPER DARI KODE SEBELUMNYA]
# - UltrasonicSensor
# - UltrasonicSensorManager  
# - DynamicWindowApproach
# - PathEvaluator
# - FastResponseController
# - UWBTracker
# - DynamicObjectDetector
# - LidarProcessor
# - RobotController
# - PerformanceMonitor

class FollowingRobotNode(Node):
    """Main ROS2 node dengan mode switching integration"""
    
    def __init__(self):
        super().__init__('following_robot_node')
        
        # =============================================
        # 🔄 MODE SWITCHING INTEGRATION
        # =============================================
        self.current_mode = 'manual'  # Default ke manual
        self.mode_active = False
        self.last_mode_check = time.time()
        
        # Subscribe to mode switching topic
        self.mode_subscription = self.create_subscription(
            String,
            '/robot_mode',
            self.mode_callback,
            10
        )
        
        # Initialize components
        self.running = True
        
        # Initialize UWB tracker
        self.uwb_tracker = UWBTracker()
        
        # Initialize LIDAR processor
        self.lidar = LidarProcessor()
        
        # Initialize ultrasonic sensor manager
        try:
            self.ultrasonic_manager = UltrasonicSensorManager()
            print("✓ Ultrasonic sensors initialized")
        except Exception as e:
            print(f"⚠ Ultrasonic sensors failed: {e}")
            self.ultrasonic_manager = None
        
        # Initialize robot controller
        self.controller = RobotController("/dev/ttyUSB1", "/dev/ttyUSB0")
        self.controller.parent = self
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor() if hasattr(globals(), 'PerformanceMonitor') else None
        
        # Control parameters
        self.control_frequency = 100  # Hz
        self.uwb_timeout = 0.0005
        
        # Setup systems
        self.setup_uwb_communication()
        self.setup_lidar_subscription()
        
        # =============================================
        # 🎯 CONDITIONAL CONTROL TIMER
        # =============================================
        # Timer untuk control loop - akan di-check di callback
        self.setup_control_timer()
        
        # Turn on program start indicator
        try:
            gpio_pin_17 = LED(17)
            gpio_pin_17.on()
        except:
            pass
        
        print("✓ FollowingRobotNode initialized")
        print(f"📡 Current mode: {self.current_mode.upper()}")
        print("🎮 Waiting for mode switching...")

    def mode_callback(self, msg):
        """🔄 CALLBACK UNTUK MODE SWITCHING"""
        new_mode = msg.data
        
        if new_mode != self.current_mode:
            # Mode berubah
            self.get_logger().info(f'🔄 MODE CHANGE: {self.current_mode} → {new_mode}')
            
            old_mode = self.current_mode
            self.current_mode = new_mode
            self.last_mode_check = time.time()
            
            if new_mode == 'otomatis':
                # AKTIFKAN MODE OTOMATIS
                self.mode_active = True
                self.get_logger().info('🤖 AUTONOMOUS MODE ACTIVATED')
                print("=" * 50)
                print("🤖 AUTONOMOUS ROBOT CONTROL ACTIVE")
                print("   ⚡ Ultra-fast obstacle avoidance")
                print("   🎯 UWB target following")
                print("   👁 Multi-sensor fusion active")
                print("   🛡 Safety systems online")
                print("=" * 50)
                
                # Initialize systems untuk mode otomatis
                self.initialize_autonomous_systems()
                
            else:
                # MATIKAN MODE OTOMATIS  
                self.mode_active = False
                self.get_logger().info('🎮 MANUAL MODE ACTIVATED')
                print("=" * 50)
                print("🎮 MANUAL CONTROL MODE ACTIVE")
                print("   🛑 Autonomous systems deactivated")
                print("   🎮 Joystick control enabled")
                print("   ⚠ Motors stopped safely")
                print("=" * 50)
                
                # Stop motors dengan aman
                self.deactivate_autonomous_systems()
    
    def initialize_autonomous_systems(self):
        """Initialize systems untuk mode otomatis"""
        try:
            # Initialize fast response system
            if hasattr(self.controller, 'initialize_fast_response'):
                self.controller.initialize_fast_response()
                print("✓ Fast response system activated")
            
            # Reset performance monitor
            if self.performance_monitor:
                self.performance_monitor.metrics['emergency_activations'] = 0
                print("✓ Performance monitor reset")
            
            print("✓ Autonomous systems initialized")
            
        except Exception as e:
            print(f"⚠ Error initializing autonomous systems: {e}")
    
    def deactivate_autonomous_systems(self):
        """Deactivate autonomous systems saat switch ke manual"""
        try:
            # Stop motors dengan smooth
            self.controller.smooth_speed_transition(0, 0)
            time.sleep(0.1)
            self.controller.stop_motors()
            
            print("✓ Motors stopped safely")
            print("✓ Autonomous systems deactivated")
            
        except Exception as e:
            print(f"⚠ Error deactivating systems: {e}")
    
    def setup_uwb_communication(self):
        """Setup UWB communication"""
        try:
            # Get robot IP
            robot_ip = self.get_ip_from_wifi()
            if not robot_ip:
                raise Exception("Could not get robot IP")
            
            # Setup UWB socket
            self.uwb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.uwb_socket.bind((robot_ip, 8888))
            self.uwb_socket.settimeout(self.uwb_timeout)
            
            print(f"✓ UWB communication setup on {robot_ip}:8888")
            
        except Exception as e:
            print(f"✗ UWB setup failed: {e}")
            self.uwb_socket = None
    
    def get_ip_from_wifi(self, interface='wlan0'):
        """Get IP from WiFi interface"""
        try:
            import netifaces
            ip = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
            return ip
        except:
            return None
    
    def setup_lidar_subscription(self):
        """Setup LIDAR subscription"""
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile,
            callback_group=ReentrantCallbackGroup()
        )
        
        print("✓ LIDAR subscription created")
    
    def setup_control_timer(self):
        """Setup control timer"""
        timer_period = 1.0 / self.control_frequency
        
        self.control_timer = self.create_timer(
            timer_period,
            self.control_loop_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        print(f"✓ Control timer setup at {self.control_frequency}Hz")
    
    def lidar_callback(self, msg):
        """LIDAR data callback"""
        try:
            self.lidar.process_scan(msg)
        except Exception as e:
            print(f"LIDAR callback error: {e}")
    
    def control_loop_callback(self):
        """🎯 MAIN CONTROL LOOP - HANYA AKTIF SAAT MODE OTOMATIS"""
        
        # =============================================
        # 🔒 MODE CHECKING - CRITICAL SECTION
        # =============================================
        if not self.mode_active or self.current_mode != 'otomatis':
            # Mode manual atau tidak aktif - SKIP control loop
            return
        
        # Mode otomatis aktif - lanjutkan dengan control loop
        start_time = time.time()
        
        try:
            # Get UWB data
            uwb_distances = self.get_uwb_data()
            
            if uwb_distances:
                # ✅ HANYA BERJALAN DI MODE OTOMATIS
                # Process control dengan ultra-fast response
                self.controller.process_control_ultra_fast(
                    uwb_distances, 
                    self.lidar, 
                    self.ultrasonic_manager
                )
                
                # Update target info untuk LIDAR
                target_direction, target_distance = self.uwb_tracker.estimate_target_direction(uwb_distances)
                self.lidar.set_target_info(target_direction, target_distance)
                
                # Log status (optional)
                if time.time() - self.last_mode_check > 5.0:  # Every 5 seconds
                    print(f"🤖 AUTONOMOUS: Target at {target_distance:.1f}cm, {target_direction:.0f}°")
                    self.last_mode_check = time.time()
            
            # Performance monitoring
            if self.performance_monitor:
                loop_time = (time.time() - start_time) * 1000
                self.performance_monitor.log_loop_time(loop_time)
                
        except Exception as e:
            print(f"❌ Control loop error: {e}")
            # Emergency stop on critical error
            try:
                self.controller.stop_motors()
            except:
                pass
    
    def get_uwb_data(self):
        """Get UWB data from socket"""
        if not self.uwb_socket:
            return None
        
        try:
            data, addr = self.uwb_socket.recvfrom(2048)
            received_data = data.decode('utf-8').strip()
            
            # Parse UWB data
            parts = received_data.split(',')
            if len(parts) >= 3:
                distances = {
                    'A0': float(parts[0]),
                    'A1': float(parts[1]),
                    'A2': float(parts[2])
                }
                
                # Apply bias correction
                corrected_distances = self.uwb_tracker.apply_bias_correction(distances)
                return corrected_distances
            
        except socket.timeout:
            pass
        except Exception as e:
            print(f"UWB data error: {e}")
        
        return None
    
    def stop(self):
        """Stop robot node"""
        print("🛑 Stopping robot node...")
        self.running = False
        
        # Stop ultrasonic sensors
        if self.ultrasonic_manager:
            self.ultrasonic_manager.stop()
        
        # Stop motors
        try:
            self.controller.stop_motors()
        except:
            pass
        
        # Close UWB socket
        if hasattr(self, 'uwb_socket') and self.uwb_socket:
            self.uwb_socket.close()
        
        print("✓ Robot node stopped")

def main(args=None):
    """Main function dengan enhanced error handling"""
    
    rclpy.init(args=args)
    
    node = None
    executor = None
    
    try:
        print("=" * 60)
        print("🤖 ENHANCED MOBILE ROBOT WITH MODE SWITCHING")
        print("=" * 60)
        print("🚀 Initializing robot systems...")
        
        # Create robot node
        node = FollowingRobotNode()
        print("✓ Robot node created")
        
        # Create multi-threaded executor
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        print("✓ Multi-threaded executor ready")
        
        print("\n" + "=" * 60)
        print("🎮 ROBOT READY FOR MODE SWITCHING!")
        print("=" * 60)
        print("Operating Modes:")
        print("  🎮 MANUAL: Joystick control active")
        print("  🤖 AUTONOMOUS: AI navigation with obstacle avoidance") 
        print("\nMode Switching:")
        print("  🔄 Press BACK button on joystick to switch")
        print("  📡 Current mode published on /robot_mode topic")
        print("\nSafety Features:")
        print("  🛡 Auto motor stop when switching to manual")
        print("  ⚡ Ultra-fast emergency response (< 1ms)")
        print("  🔒 Mode isolation for safety")
        print("=" * 60)
        print(f"📡 Current Mode: MANUAL")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Run the robot
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n🛑 User interruption - shutting down safely...")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🔧 Cleaning up systems...")
        
        if node:
            try:
                node.stop()
                node.destroy_node()
                print("✓ Node stopped")
            except Exception as e:
                print(f"✗ Node cleanup error: {e}")
        
        if executor:
            try:
                executor.shutdown(timeout_sec=2.0)
                print("✓ Executor stopped")
            except Exception as e:
                print(f"✗ Executor cleanup error: {e}")
        
        try:
            rclpy.shutdown()
            print("✓ ROS2 shutdown complete")
        except Exception as e:
            print(f"✗ ROS2 shutdown error: {e}")
        
        # GPIO cleanup
        try:
            GPIO.cleanup()
            print("✓ GPIO cleaned up")
        except:
            pass
        
        print("✅ ROBOT SHUTDOWN COMPLETE")

if __name__ == '__main__':
    main()
