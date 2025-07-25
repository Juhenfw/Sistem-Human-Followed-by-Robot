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
from uwbSENSOR.ddsm115 import MotorControl
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Import gpiozero for controlling GPIO pins
from gpiozero import LED

# GPIO imports for ultrasonic sensors
import RPi.GPIO as GPIO

# ============================================================================
# CONFIGURATION GLOBALS - OPTIMIZED FOR UWB-FIRST NAVIGATION
# ============================================================================

# UWB Navigation Configuration (PRIMARY)
UWB_UPDATE_RATE = 200  # Hz - High frequency UWB updates
UWB_SMOOTHING_FACTOR = 0.15  # Aggressive smoothing for responsive movement
UWB_DEAD_ZONE = 80  # cm - Stop zone when target reached
UWB_SLOW_ZONE = 150  # cm - Slow down zone

# Collision Avoidance Configuration (SECONDARY)
COLLISION_THRESHOLD = 15  # cm - Emergency stop distance
WARNING_THRESHOLD = 25   # cm - Slow down distance
SAFE_THRESHOLD = 40      # cm - Normal operation distance

# Motor Control Configuration
MAX_SPEED = 85          # Maximum motor speed
MIN_SPEED = 20          # Minimum motor speed for movement
ACCELERATION_RATE = 8   # Speed change rate
TURN_SMOOTHING = 0.8    # Turn response smoothing

# Control Loop Configuration
MAIN_CONTROL_FREQUENCY = 100  # Hz - Main control loop
COLLISION_CHECK_FREQUENCY = 200  # Hz - Collision avoidance loop
SENSOR_TIMEOUT = 0.1  # Sensor data timeout

# GPIO Pins Initialization
gpio_pin_17 = LED(17)  # Program start indicator
gpio_pin_27 = LED(27)  # Error indicator

# Ultrasonic Sensor Configuration
SENSORS = {
    'front_left': {'trig': 18, 'echo': 24, 'angle': 315},
    'front_center': {'trig': 23, 'echo': 25, 'angle': 0}, 
    'front_right': {'trig': 12, 'echo': 16, 'angle': 45}
}

def get_ip_from_wifi(interface='wlan0'):
    """Get the IP address of the Raspberry Pi from the Wi-Fi interface"""
    try:
        ip = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
        return ip
    except (KeyError, ValueError):
        print(f"Failed to get IP address for interface: {interface}")
        return None

# ============================================================================
# UWB TRACKER - PRIMARY NAVIGATION SYSTEM
# ============================================================================

class UWBTracker:
    """Primary navigation system using UWB positioning"""
    
    def __init__(self):
        # Calibrated bias correction for accurate positioning
        self.bias = {
            'A0': 45.0,
            'A1': 48.0, 
            'A2': 47.0
        }
        
        self.scale_factor = {
            'A0': 1.0,
            'A1': 1.003,
            'A2': 1.006
        }
        
        # Smooth movement tracking
        self.target_direction = 0
        self.target_distance = 1000
        self.direction_history = deque(maxlen=5)
        self.distance_history = deque(maxlen=3)
        
        # Movement state
        self.last_update_time = time.time()
        self.movement_confidence = 0.0
        
    def apply_bias_correction(self, distances):
        """Apply calibrated bias correction"""
        corrected = {}
        for anchor in ['A0', 'A1', 'A2']:
            raw_distance = distances[anchor] * 100  # Convert to cm
            corrected[anchor] = max((raw_distance * self.scale_factor[anchor]) - self.bias[anchor], 0)
        return corrected
    
    def calculate_target_direction_smooth(self, distances):
        """Calculate smooth target direction with enhanced 360° coverage"""
        A0, A1, A2 = distances['A0'], distances['A1'], distances['A2']
        
        # Calculate steering error (primary navigation signal)
        steering_error = A2 - A1
        
        # Determine target quadrant based on anchor relationships
        if A0 < min(A1, A2) - 20:  # Target behind robot
            if abs(steering_error) < 10:
                raw_direction = 180  # Directly behind
            elif steering_error < 0:  # Behind-right
                raw_direction = 180 - min(30, abs(steering_error) * 0.8)
            else:  # Behind-left
                raw_direction = 180 + min(30, abs(steering_error) * 0.8)
                
        elif abs(steering_error) < 8:  # Target in front
            raw_direction = 0
            
        elif steering_error < 0:  # Target to the right
            if A0 > min(A1, A2) + 10:  # Far right side
                raw_direction = 90 + min(45, abs(steering_error) * 1.1)
            else:  # Front-right
                raw_direction = min(35, abs(steering_error) * 1.2)
                
        else:  # Target to the left
            if A0 > min(A1, A2) + 10:  # Far left side
                raw_direction = 270 - min(45, abs(steering_error) * 1.1)
            else:  # Front-left
                raw_direction = 360 - min(35, abs(steering_error) * 1.2)
        
        # Normalize angle
        raw_direction = raw_direction % 360
        
        # Apply smoothing filter
        self.direction_history.append(raw_direction)
        self.distance_history.append(A0)
        
        # Smooth direction calculation
        if len(self.direction_history) >= 3:
            # Handle angle wraparound for smoothing
            directions = list(self.direction_history)
            
            # Check for wraparound (e.g., 350° to 10°)
            angle_diffs = []
            for i in range(1, len(directions)):
                diff = directions[i] - directions[i-1]
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                angle_diffs.append(diff)
            
            # Apply smoothing
            smooth_direction = directions[0]
            for diff in angle_diffs:
                smooth_direction += diff * UWB_SMOOTHING_FACTOR
            
            smooth_direction = smooth_direction % 360
        else:
            smooth_direction = raw_direction
        
        # Update internal state
        self.target_direction = smooth_direction
        self.target_distance = A0
        self.movement_confidence = min(1.0, len(self.direction_history) / 5.0)
        self.last_update_time = time.time()
        
        return smooth_direction, A0

# ============================================================================
# COLLISION AVOIDANCE SYSTEM - SECONDARY SAFETY SYSTEM
# ============================================================================

class CollisionAvoidanceSystem:
    """Lightweight collision avoidance system"""
    
    def __init__(self):
        # Initialize ultrasonic sensors
        self.init_ultrasonic_sensors()
        
        # Collision state
        self.collision_detected = False
        self.avoidance_direction = None
        self.last_check_time = time.time()
        
        # Sensor threads
        self.sensor_threads = {}
        self.sensor_data = {
            'front_left': {'distance': 999, 'last_update': 0},
            'front_center': {'distance': 999, 'last_update': 0},
            'front_right': {'distance': 999, 'last_update': 0}
        }
        self.data_lock = threading.Lock()
        self.running = True
        
        # Start sensor threads
        self.start_sensor_threads()
    
    def init_ultrasonic_sensors(self):
        """Initialize ultrasonic sensors"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        for sensor_name, config in SENSORS.items():
            try:
                GPIO.setup(config['trig'], GPIO.OUT)
                GPIO.setup(config['echo'], GPIO.IN)
                GPIO.output(config['trig'], False)
                time.sleep(0.01)
                print(f"✓ {sensor_name} sensor initialized")
            except Exception as e:
                print(f"✗ Error initializing {sensor_name}: {e}")
    
    def measure_distance(self, sensor_name):
        """Measure distance for a specific sensor"""
        if sensor_name not in SENSORS:
            return -1
        
        config = SENSORS[sensor_name]
        trig_pin = config['trig']
        echo_pin = config['echo']
        
        try:
            # Send trigger pulse
            GPIO.output(trig_pin, True)
            time.sleep(0.00001)  # 10μs pulse
            GPIO.output(trig_pin, False)
            
            # Measure echo time with timeout
            timeout_start = time.time()
            
            # Wait for echo start
            while GPIO.input(echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > 0.02:  # 20ms timeout
                    return -1
            
            # Wait for echo end
            while GPIO.input(echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end - timeout_start > 0.02:
                    return -1
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # cm
            
            if 2 <= distance <= 400:  # Valid range
                return round(distance, 1)
            else:
                return -1
                
        except Exception as e:
            return -1
    
    def sensor_reading_loop(self, sensor_name):
        """Continuous sensor reading loop"""
        while self.running:
            try:
                distance = self.measure_distance(sensor_name)
                current_time = time.time()
                
                with self.data_lock:
                    if distance > 0:
                        self.sensor_data[sensor_name]['distance'] = distance
                    self.sensor_data[sensor_name]['last_update'] = current_time
                
                time.sleep(0.01)  # 100Hz per sensor
                
            except Exception as e:
                print(f"Sensor {sensor_name} error: {e}")
                time.sleep(0.05)
    
    def start_sensor_threads(self):
        """Start sensor reading threads"""
        for sensor_name in SENSORS.keys():
            thread = threading.Thread(
                target=self.sensor_reading_loop,
                args=(sensor_name,),
                daemon=True
            )
            thread.start()
            self.sensor_threads[sensor_name] = thread
            time.sleep(0.005)  # Stagger start times
        
        print(f"✓ Started {len(self.sensor_threads)} sensor threads")
    
    def get_collision_status(self):
        """Get current collision avoidance status"""
        current_time = time.time()
        
        with self.data_lock:
            status = {
                'collision_imminent': False,
                'warning_zone': False,
                'avoidance_needed': False,
                'safe_directions': [],
                'closest_obstacle': 999,
                'avoidance_action': None
            }
            
            # Check each sensor
            distances = {}
            for sensor_name, data in self.sensor_data.items():
                # Check if data is recent
                if current_time - data['last_update'] < SENSOR_TIMEOUT:
                    distances[sensor_name] = data['distance']
                else:
                    distances[sensor_name] = 999  # Assume clear if no recent data
            
            # Determine collision status
            front_distance = distances['front_center']
            left_distance = distances['front_left']
            right_distance = distances['front_right']
            
            status['closest_obstacle'] = min(front_distance, left_distance, right_distance)
            
            # Emergency collision detection
            if front_distance < COLLISION_THRESHOLD:
                status['collision_imminent'] = True
                status['avoidance_needed'] = True
                
                # Determine best avoidance direction
                if left_distance > right_distance and left_distance > 30:
                    status['avoidance_action'] = 'turn_left'
                elif right_distance > 30:
                    status['avoidance_action'] = 'turn_right'
                else:
                    status['avoidance_action'] = 'stop'
            
            # Warning zone detection
            elif front_distance < WARNING_THRESHOLD:
                status['warning_zone'] = True
                status['avoidance_needed'] = True
                
                if left_distance > right_distance:
                    status['avoidance_action'] = 'slight_left'
                else:
                    status['avoidance_action'] = 'slight_right'
            
            # Determine safe directions
            if left_distance > SAFE_THRESHOLD:
                status['safe_directions'].append('left')
            if right_distance > SAFE_THRESHOLD:
                status['safe_directions'].append('right')
            if front_distance > SAFE_THRESHOLD:
                status['safe_directions'].append('forward')
        
        return status
    
    def stop(self):
        """Stop collision avoidance system"""
        self.running = False
        try:
            GPIO.cleanup()
        except:
            pass

# ============================================================================
# SMOOTH MOTION CONTROLLER - OPTIMIZED FOR UWB FOLLOWING
# ============================================================================

class SmoothMotionController:
    """Advanced motion controller with smooth UWB following"""
    
    def __init__(self, right_port, left_port):
        # Motor initialization
        self.right_motor = MotorControl(device=right_port)
        self.left_motor = MotorControl(device=left_port)
        self.right_motor.set_drive_mode(1, 2)
        self.left_motor.set_drive_mode(1, 2)
        
        # Motion state
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.target_left_speed = 0
        self.target_right_speed = 0
        
        # Smooth acceleration
        self.acceleration_rate = ACCELERATION_RATE
        self.last_update_time = time.time()
        
        # Movement parameters
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        self.turn_gain = 1.5
        
        # UWB following state
        self.uwb_lost_time = 0
        self.last_valid_direction = 0
        
    def calculate_uwb_speeds(self, target_direction, target_distance):
        """Calculate motor speeds based on UWB target"""
        current_time = time.time()
        
        # Handle target distance
        if target_distance < UWB_DEAD_ZONE:
            return 0, 0  # Stop when target reached
        
        # Calculate base speed based on distance
        if target_distance < UWB_SLOW_ZONE:
            base_speed = self.min_speed + (self.max_speed - self.min_speed) * \
                        ((target_distance - UWB_DEAD_ZONE) / (UWB_SLOW_ZONE - UWB_DEAD_ZONE))
        else:
            base_speed = self.max_speed
        
        # Calculate turn angle (convert direction to turn angle)
        if target_direction <= 180:
            turn_angle = target_direction
        else:
            turn_angle = target_direction - 360
        
        # Apply turn angle limits for smooth movement
        turn_angle = max(-90, min(90, turn_angle))
        
        # Calculate differential steering
        turn_factor = abs(turn_angle) / 90.0  # Normalize to 0-1
        turn_intensity = turn_factor * self.turn_gain
        
        if abs(turn_angle) < 5:  # Nearly straight
            left_speed = base_speed
            right_speed = -base_speed
        elif turn_angle > 0:  # Turn left
            left_speed = base_speed * (1.0 - turn_intensity)
            right_speed = -base_speed * (1.0 + turn_intensity * 0.5)
        else:  # Turn right
            turn_angle = abs(turn_angle)
            left_speed = base_speed * (1.0 + turn_intensity * 0.5)
            right_speed = -base_speed * (1.0 - turn_intensity)
        
        # Ensure minimum speeds for movement
        if abs(left_speed) < self.min_speed and left_speed != 0:
            left_speed = self.min_speed if left_speed > 0 else -self.min_speed
        if abs(right_speed) < self.min_speed and right_speed != 0:
            right_speed = self.min_speed if right_speed > 0 else -self.min_speed
        
        return int(left_speed), int(right_speed)
    
    def apply_collision_avoidance(self, uwb_left, uwb_right, collision_status):
        """Apply collision avoidance adjustments to UWB speeds"""
        if not collision_status['avoidance_needed']:
            return uwb_left, uwb_right
        
        action = collision_status['avoidance_action']
        
        if action == 'stop':
            return 0, 0
        elif action == 'turn_left':
            # Emergency left turn
            return -abs(uwb_left), -abs(uwb_left)
        elif action == 'turn_right':
            # Emergency right turn
            return abs(uwb_right), abs(uwb_right)
        elif action == 'slight_left':
            # Slight left adjustment
            left_reduction = 0.6
            right_boost = 1.2
            return int(uwb_left * left_reduction), int(uwb_right * right_boost)
        elif action == 'slight_right':
            # Slight right adjustment
            left_boost = 1.2
            right_reduction = 0.6
            return int(uwb_left * left_boost), int(uwb_right * right_reduction)
        
        return uwb_left, uwb_right
    
    def smooth_speed_transition(self, target_left, target_right):
        """Apply smooth speed transitions"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Calculate maximum speed change for this update
        max_change = self.acceleration_rate * dt * 60  # Convert to RPM/s
        
        # Apply smooth transitions
        left_diff = target_left - self.current_left_speed
        right_diff = target_right - self.current_right_speed
        
        if abs(left_diff) > max_change:
            left_diff = max_change if left_diff > 0 else -max_change
        if abs(right_diff) > max_change:
            right_diff = max_change if right_diff > 0 else -max_change
        
        # Update current speeds
        self.current_left_speed += left_diff
        self.current_right_speed += right_diff
        
        # Send to motors
        self.left_motor.send_rpm(1, int(self.current_left_speed))
        self.right_motor.send_rpm(1, int(self.current_right_speed))
    
    def execute_motion(self, uwb_direction, uwb_distance, collision_status):
        """Execute smooth motion with UWB priority and collision avoidance"""
        
        # Calculate primary UWB-based motion
        uwb_left, uwb_right = self.calculate_uwb_speeds(uwb_direction, uwb_distance)
        
        # Apply collision avoidance modifications
        final_left, final_right = self.apply_collision_avoidance(
            uwb_left, uwb_right, collision_status
        )
        
        # Apply smooth transitions
        self.smooth_speed_transition(final_left, final_right)
        
        return final_left, final_right
    
    def emergency_stop(self):
        """Emergency stop with smooth deceleration"""
        self.smooth_speed_transition(0, 0)
    
    def stop(self):
        """Stop motors"""
        self.left_motor.send_rpm(1, 0)
        self.right_motor.send_rpm(1, 0)
        self.current_left_speed = 0
        self.current_right_speed = 0

# ============================================================================
# LIDAR PROCESSOR - SUPPLEMENTARY SENSOR
# ============================================================================

class LidarProcessor:
    """Simplified LIDAR processor for supplementary collision detection"""
    
    def __init__(self):
        self.scan_data = {}
        self.lock = threading.Lock()
        self.last_scan_time = 0
        
        # Simple obstacle detection
        self.front_clear = True
        self.left_clear = True
        self.right_clear = True
        
    def process_scan(self, scan_msg):
        """Process LIDAR scan for basic obstacle detection"""
        with self.lock:
            self.last_scan_time = time.time()
            self.scan_data.clear()
            
            ranges = scan_msg.ranges
            angle_increment = scan_msg.angle_increment
            angle_min = scan_msg.angle_min
            
            # Process every 3rd point for efficiency
            for i in range(0, len(ranges), 3):
                distance = ranges[i]
                
                if distance < 0.05 or distance > 5.0 or math.isinf(distance):
                    continue
                
                angle_rad = angle_min + (i * angle_increment)
                angle_deg = int(math.degrees(angle_rad) % 360)
                distance_cm = distance * 100
                
                self.scan_data[angle_deg] = distance_cm
            
            # Update obstacle status
            self._update_obstacle_status()
    
    def _update_obstacle_status(self):
        """Update basic obstacle status"""
        front_min = 999
        left_min = 999
        right_min = 999
        
        for angle, distance in self.scan_data.items():
            if 350 <= angle <= 360 or 0 <= angle <= 10:  # Front
                front_min = min(front_min, distance)
            elif 30 <= angle <= 60:  # Right
                right_min = min(right_min, distance)
            elif 300 <= angle <= 330:  # Left
                left_min = min(left_min, distance)
        
        self.front_clear = front_min > 30  # 30cm threshold
        self.left_clear = left_min > 25
        self.right_clear = right_min > 25
    
    def get_lidar_collision_status(self):
        """Get basic collision status from LIDAR"""
        current_time = time.time()
        data_valid = (current_time - self.last_scan_time) < 0.5
        
        if not data_valid:
            return {'valid': False, 'front_blocked': False, 'left_blocked': False, 'right_blocked': False}
        
        return {
            'valid': True,
            'front_blocked': not self.front_clear,
            'left_blocked': not self.left_clear,
            'right_blocked': not self.right_clear
        }

# ============================================================================
# MAIN ROBOT NODE - STREAMLINED FOR PERFORMANCE
# ============================================================================

class UWBFollowingRobot(Node):
    """Main robot node optimized for UWB following with collision avoidance"""
    
    def __init__(self):
        super().__init__('uwb_following_robot')
        
        # Core components
        self.running = True
        self.uwb_tracker = UWBTracker()
        self.collision_system = CollisionAvoidanceSystem()
        self.motion_controller = SmoothMotionController("/dev/ttyUSB1", "/dev/ttyUSB0")
        self.lidar_processor = LidarProcessor()
        
        # UWB communication
        self.setup_uwb_communication()
        
        # LIDAR subscription
        self.setup_lidar_subscription()
        
        # Control timers
        self.setup_control_timers()
        
        # Status indicators
        gpio_pin_17.on()  # Program running indicator
        
        print("✓ UWB Following Robot initialized successfully")
    
    def setup_uwb_communication(self):
        """Setup UWB communication"""
        try:
            robot_ip = get_ip_from_wifi()
            if not robot_ip:
                robot_ip = "192.168.1.100"  # Fallback IP
            
            self.uwb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.uwb_socket.bind((robot_ip, 8888))
            self.uwb_socket.settimeout(0.005)  # 5ms timeout
            
            print(f"✓ UWB communication setup on {robot_ip}:8888")
            
        except Exception as e:
            print(f"✗ UWB setup failed: {e}")
            self.uwb_socket = None
    
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
    
    def setup_control_timers(self):
        """Setup control timers"""
        # Main control loop - UWB following
        main_period = 1.0 / MAIN_CONTROL_FREQUENCY
        self.main_timer = self.create_timer(
            main_period,
            self.main_control_loop,
            callback_group=ReentrantCallbackGroup()
        )
        
        print(f"✓ Control system running at {MAIN_CONTROL_FREQUENCY}Hz")
    
    def lidar_callback(self, msg):
        """LIDAR data callback"""
        try:
            self.lidar_processor.process_scan(msg)
        except Exception as e:
            pass  # Ignore LIDAR errors for now
    
    def get_uwb_data(self):
        """Get UWB data from socket"""
        if not self.uwb_socket:
            return None
        
        try:
            data, addr = self.uwb_socket.recvfrom(1024)
            received_data = data.decode('utf-8').strip()
            
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
            pass
        
        return None
    
    def main_control_loop(self):
        """Main control loop - UWB following with collision avoidance"""
        try:
            # 1. Get UWB data (PRIMARY NAVIGATION)
            uwb_data = self.get_uwb_data()
            
            if uwb_data:
                # Calculate target direction and distance
                target_direction, target_distance = self.uwb_tracker.calculate_target_direction_smooth(uwb_data)
                
                # 2. Get collision avoidance status (SECONDARY SAFETY)
                collision_status = self.collision_system.get_collision_status()
                
                # 3. Execute smooth motion with UWB priority
                final_left, final_right = self.motion_controller.execute_motion(
                    target_direction, target_distance, collision_status
                )
                
                # Debug output
                print(f"UWB: Dir={target_direction:.1f}° Dist={target_distance:.1f}cm | "
                      f"Motors: L={final_left} R={final_right} | "
                      f"Collision: {collision_status.get('avoidance_action', 'none')}")
                
            else:
                # No UWB data - stop motion for safety
                self.motion_controller.emergency_stop()
                print("No UWB data - stopped")
                
        except Exception as e:
            print(f"Control loop error: {e}")
            self.motion_controller.emergency_stop()
    
    def stop(self):
        """Stop the robot"""
        print("Stopping robot...")
        self.running = False
        
        # Stop motion
        self.motion_controller.stop()
        
        # Stop collision system
        self.collision_system.stop()
        
        # Close UWB socket
        if hasattr(self, 'uwb_socket') and self.uwb_socket:
            self.uwb_socket.close()
        
        # Turn off indicators
        gpio_pin_17.off()
        
        print("✓ Robot stopped")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(args=None):
    """Main function with comprehensive error handling"""
    
    rclpy.init(args=args)
    node = None
    executor = None
    
    try:
        print("=" * 60)
        print("    UWB-PRIORITIZED MOBILE ROBOT WITH SMOOTH MOTION")
        print("=" * 60)
        print("Initializing robot systems...")
        
        # System checks
        print("\n🔧 System Requirements Check:")
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            print("✓ GPIO system available")
        except Exception as e:
            print(f"✗ GPIO system error: {e}")
        
        try:
            ip = get_ip_from_wifi()
            print(f"✓ Network interface: {ip}")
        except Exception as e:
            print(f"⚠ Network check failed: {e}")
        
        # Create robot node
        print("\n🤖 Creating robot node...")
        node = UWBFollowingRobot()
        
        # Create executor
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        
        print("\n" + "=" * 60)
        print(" 🎯 ROBOT SYSTEMS READY!")
        print("=" * 60)
        print("Navigation System:")
        print("  🎯 PRIMARY: UWB positioning & tracking")
        print("  🛡️ SECONDARY: Collision avoidance")
        print("  🔄 Smooth motion control")
        print("  ⚡ Real-time response (100Hz)")
        
        print("\nKey Features:")
        print("  • UWB-first navigation priority")
        print("  • Smooth acceleration & turning")
        print("  • Multi-sensor collision avoidance")
        print("  • 360° target tracking")
        print("  • Adaptive speed control")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop the robot")
        print("=" * 60)
        
        # Run the robot
        print("🚀 Starting robot operation...\n")
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n⏹️  User interruption detected")
        print("🛑 Initiating safe shutdown...")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        gpio_pin_27.on()  # Error indicator
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🔧 CLEANUP SEQUENCE")
        print("-" * 40)
        
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
                print("✓ Executor shutdown")
            except Exception as e:
                print(f"✗ Executor error: {e}")
        
        try:
            rclpy.shutdown()
            print("✓ ROS2 shutdown")
        except Exception as e:
            print(f"✗ ROS2 shutdown error: {e}")
        
        try:
            gpio_pin_17.off()
            gpio_pin_27.off()
            GPIO.cleanup()
            print("✓ GPIO cleanup")
        except Exception as e:
            print(f"✗ GPIO cleanup error: {e}")
        
        print("-" * 40)
        print("🏁 ROBOT SHUTDOWN COMPLETE")
        print("=" * 60)

if __name__ == '__main__':
    main()
