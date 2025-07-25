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
import queue
from typing import Dict, Tuple, Optional, List

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

# =============================================================================
# KONFIGURASI GLOBAL - OPTIMIZED FOR SMOOTH MOVEMENT
# =============================================================================

# UWB Configuration - PRIMARY TARGET FOLLOWING
UWB_UPDATE_RATE = 100  # Hz - High frequency for smooth tracking
UWB_TIMEOUT = 0.001    # 1ms timeout for real-time response
UWB_BUFFER_SIZE = 4096

# Movement Configuration - SMOOTH & RESPONSIVE
BASE_SPEED = 60           # Base speed for smooth movement
MAX_SPEED = 120           # Maximum speed
MIN_SPEED = 20            # Minimum speed to avoid stopping
ACCELERATION_RATE = 15    # Smooth acceleration/deceleration
TURN_SENSITIVITY = 0.8    # Turn response sensitivity
SPEED_SMOOTHING = 0.85    # Speed change smoothing factor

# Safety Configuration - COLLISION AVOIDANCE ONLY
EMERGENCY_STOP_DISTANCE = 80   # mm - Emergency brake distance
CRITICAL_DISTANCE = 150        # mm - Critical avoidance distance  
WARNING_DISTANCE = 300         # mm - Warning distance
SAFE_DISTANCE = 500           # mm - Safe following distance

# Control Frequencies
MAIN_CONTROL_FREQ = 200       # Hz - Main control loop
SAFETY_CHECK_FREQ = 500       # Hz - Safety monitoring
SENSOR_FUSION_FREQ = 100      # Hz - Sensor data fusion

# Network Configuration
def get_ip_from_wifi(interface='wlan0'):
    """Get IP address from WiFi interface"""
    try:
        ip = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
        return ip
    except (KeyError, ValueError):
        print(f"Failed to get IP for {interface}")
        return None

# GPIO Configuration
gpio_pin_17 = LED(17)  # Program status
gpio_pin_27 = LED(27)  # Error indicator

# Ultrasonic Sensor Configuration - SIMPLIFIED
ULTRASONIC_SENSORS = {
    'front_left': {'trig': 18, 'echo': 24, 'angle': 315},
    'front_center': {'trig': 23, 'echo': 25, 'angle': 0},
    'front_right': {'trig': 12, 'echo': 16, 'angle': 45}
}

# Safety thresholds (cm)
ULTRASONIC_EMERGENCY = 8    # Emergency stop
ULTRASONIC_CRITICAL = 15    # Critical avoidance
ULTRASONIC_WARNING = 25     # Warning zone

# =============================================================================
# UWB TARGET TRACKER - PRIMARY NAVIGATION SYSTEM
# =============================================================================

class UWBTargetTracker:
    """Advanced UWB target tracking with smooth movement generation"""
    
    def __init__(self):
        # Bias correction - Fine-tuned for accuracy
        self.bias_correction = {
            'A0': 45.0,
            'A1': 50.0, 
            'A2': 48.0
        }
        
        # Scale factors
        self.scale_factors = {
            'A0': 1.0,
            'A1': 1.005,
            'A2': 1.008
        }
        
        # Smooth tracking variables
        self.target_direction = 0.0
        self.target_distance = 0.0
        self.previous_direction = 0.0
        self.direction_history = deque(maxlen=5)
        self.distance_history = deque(maxlen=3)
        
        # Movement generation
        self.desired_left_speed = 0.0
        self.desired_right_speed = 0.0
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        
        # Target following parameters
        self.target_close_threshold = 80    # cm - Slow down when close
        self.target_stop_threshold = 40     # cm - Stop when very close
        self.max_turn_rate = 45            # degrees/second max turn rate
        
        print("✓ UWB Target Tracker initialized")
    
    def apply_corrections(self, raw_distances: Dict[str, float]) -> Dict[str, float]:
        """Apply bias correction and scaling to raw UWB distances"""
        corrected = {}
        for anchor, distance in raw_distances.items():
            if anchor in self.bias_correction:
                # Convert to cm and apply corrections
                corrected_dist = (distance * 100 * self.scale_factors[anchor]) - self.bias_correction[anchor]
                corrected[anchor] = max(corrected_dist, 0)  # Ensure non-negative
            else:
                corrected[anchor] = distance * 100
        
        return corrected
    
    def estimate_target_position(self, distances: Dict[str, float]) -> Tuple[float, float]:
        """Estimate target direction and distance from UWB data"""
        A0, A1, A2 = distances['A0'], distances['A1'], distances['A2']
        
        # Primary distance is from A0 (front anchor)
        target_distance = A0
        
        # Direction estimation based on A1/A2 differential
        # A1 is left anchor, A2 is right anchor
        differential = A2 - A1  # Positive = target is left, Negative = target is right
        
        # Calculate target direction (0° = front, 90° = right, 270° = left)
        if abs(differential) < 5:
            # Target is directly ahead
            target_direction = 0.0
        elif differential > 0:
            # Target is to the left
            angle_offset = min(45, abs(differential) * 1.2)
            target_direction = 360 - angle_offset  # 315° to 360°
        else:
            # Target is to the right  
            angle_offset = min(45, abs(differential) * 1.2)
            target_direction = angle_offset  # 0° to 45°
        
        # Check for target behind robot
        if (A1 < A0 - 30) and (A2 < A0 - 30):
            # Target is likely behind
            if abs(differential) < 10:
                target_direction = 180.0  # Directly behind
            elif differential > 0:
                target_direction = 180 + min(30, abs(differential) * 0.8)  # Behind-left
            else:
                target_direction = 180 - min(30, abs(differential) * 0.8)  # Behind-right
        
        # Smooth direction changes
        self.direction_history.append(target_direction)
        if len(self.direction_history) > 1:
            # Apply smoothing to prevent jittery movements
            smoothed_direction = sum(self.direction_history) / len(self.direction_history)
            target_direction = smoothed_direction
        
        # Store for next iteration
        self.target_direction = target_direction
        self.target_distance = target_distance
        
        return target_direction, target_distance
    
    def generate_movement_commands(self, target_direction: float, target_distance: float) -> Tuple[float, float]:
        """Generate smooth movement commands based on target position"""
        
        # Distance-based speed control
        if target_distance <= self.target_stop_threshold:
            # Very close to target - stop
            base_speed = 0
        elif target_distance <= self.target_close_threshold:
            # Close to target - slow down
            base_speed = MIN_SPEED + (target_distance - self.target_stop_threshold) * 0.8
        else:
            # Normal following speed
            distance_factor = min(1.0, target_distance / 200.0)  # Normalize to 2m
            base_speed = MIN_SPEED + (BASE_SPEED - MIN_SPEED) * distance_factor
        
        # Direction-based steering
        # Convert direction to steering angle (-180 to +180)
        steering_angle = target_direction
        if steering_angle > 180:
            steering_angle -= 360
        
        # Generate differential wheel speeds
        if abs(steering_angle) < 5:
            # Go straight
            left_speed = base_speed
            right_speed = -base_speed  # Negative for forward movement
        else:
            # Calculate turn intensities
            turn_intensity = abs(steering_angle) / 180.0  # 0 to 1
            turn_factor = turn_intensity * TURN_SENSITIVITY
            
            if steering_angle > 0:
                # Turn left
                left_speed = base_speed * (1.0 - turn_factor)
                right_speed = -base_speed * (1.0 + turn_factor * 0.5)
            else:
                # Turn right
                left_speed = base_speed * (1.0 + turn_factor * 0.5)
                right_speed = -base_speed * (1.0 - turn_factor)
        
        # Apply speed limits
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
        
        # Smooth speed transitions
        self.desired_left_speed = left_speed
        self.desired_right_speed = right_speed
        
        # Apply smoothing to actual speeds
        self.current_left_speed = (SPEED_SMOOTHING * self.current_left_speed + 
                                  (1 - SPEED_SMOOTHING) * self.desired_left_speed)
        self.current_right_speed = (SPEED_SMOOTHING * self.current_right_speed + 
                                   (1 - SPEED_SMOOTHING) * self.desired_right_speed)
        
        return self.current_left_speed, self.current_right_speed

# =============================================================================
# SAFETY COLLISION AVOIDANCE SYSTEM
# =============================================================================

class CollisionAvoidanceSystem:
    """Lightweight collision avoidance - only prevents crashes, doesn't interfere with UWB tracking"""
    
    def __init__(self):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Ultrasonic sensors
        self.sensors = {}
        self.sensor_data = {}
        self.sensor_threads = {}
        self.running = True
        
        # Safety state
        self.emergency_stop_active = False
        self.collision_threat_detected = False
        self.last_threat_time = 0
        
        # Initialize sensors
        self._initialize_sensors()
        self._start_sensor_threads()
        
        print("✓ Collision Avoidance System initialized")
    
    def _initialize_sensors(self):
        """Initialize ultrasonic sensors"""
        for name, config in ULTRASONIC_SENSORS.items():
            try:
                GPIO.setup(config['trig'], GPIO.OUT)
                GPIO.setup(config['echo'], GPIO.IN)
                GPIO.output(config['trig'], False)
                
                self.sensors[name] = config
                self.sensor_data[name] = {'distance': -1, 'timestamp': 0}
                
                print(f"✓ {name} sensor initialized (GPIO{config['trig']}/{config['echo']})")
                
            except Exception as e:
                print(f"✗ Failed to initialize {name}: {e}")
        
        time.sleep(0.1)  # Settle time
    
    def _measure_distance(self, sensor_name: str) -> float:
        """Measure distance from ultrasonic sensor"""
        if sensor_name not in self.sensors:
            return -1
        
        config = self.sensors[sensor_name]
        
        try:
            # Send trigger pulse
            GPIO.output(config['trig'], True)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(config['trig'], False)
            
            # Wait for echo
            timeout = time.time() + 0.03  # 30ms timeout
            
            # Wait for echo start
            while GPIO.input(config['echo']) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return -1
            
            # Wait for echo end
            while GPIO.input(config['echo']) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return -1
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # Speed of sound
            
            # Filter valid readings
            if 1 <= distance <= 400:  # 1cm to 4m range
                return round(distance, 1)
            else:
                return -1
                
        except Exception as e:
            return -1
    
    def _sensor_reading_thread(self, sensor_name: str):
        """Continuous sensor reading thread"""
        while self.running:
            try:
                distance = self._measure_distance(sensor_name)
                
                self.sensor_data[sensor_name] = {
                    'distance': distance,
                    'timestamp': time.time()
                }
                
                time.sleep(0.02)  # 50Hz per sensor
                
            except Exception as e:
                print(f"Sensor thread error {sensor_name}: {e}")
                time.sleep(0.1)
    
    def _start_sensor_threads(self):
        """Start sensor reading threads"""
        for sensor_name in self.sensors.keys():
            thread = threading.Thread(
                target=self._sensor_reading_thread,
                args=(sensor_name,),
                daemon=True
            )
            thread.start()
            self.sensor_threads[sensor_name] = thread
            time.sleep(0.01)  # Stagger starts
        
        print(f"✓ {len(self.sensor_threads)} sensor threads started")
    
    def check_immediate_collision_threat(self) -> Tuple[bool, str, float]:
        """Check for immediate collision threats"""
        current_time = time.time()
        min_distance = float('inf')
        threat_sensor = None
        
        for sensor_name, data in self.sensor_data.items():
            distance = data['distance']
            data_age = current_time - data['timestamp']
            
            # Skip invalid or old data
            if distance <= 0 or data_age > 0.1:  # 100ms timeout
                continue
            
            # Check for emergency collision threat
            if distance < ULTRASONIC_EMERGENCY:
                self.emergency_stop_active = True
                self.collision_threat_detected = True
                self.last_threat_time = current_time
                return True, sensor_name, distance
            
            # Track closest obstacle
            if distance < min_distance:
                min_distance = distance
                threat_sensor = sensor_name
        
        # Check for critical threat
        if min_distance < ULTRASONIC_CRITICAL:
            self.collision_threat_detected = True
            self.last_threat_time = current_time
            return True, threat_sensor, min_distance
        
        # Clear threat if no obstacles detected
        if current_time - self.last_threat_time > 0.5:  # 500ms clear time
            self.collision_threat_detected = False
            self.emergency_stop_active = False
        
        return False, None, min_distance
    
    def get_avoidance_adjustment(self, left_speed: float, right_speed: float) -> Tuple[float, float]:
        """Get minimal avoidance adjustment to prevent collision while maintaining UWB tracking"""
        
        threat_detected, threat_sensor, threat_distance = self.check_immediate_collision_threat()
        
        if not threat_detected:
            return left_speed, right_speed
        
        # Emergency stop for very close obstacles
        if threat_distance < ULTRASONIC_EMERGENCY:
            print(f"EMERGENCY STOP: {threat_sensor} at {threat_distance:.1f}cm")
            return 0, 0
        
        # Minimal avoidance adjustments
        if threat_sensor == 'front_center':
            # Front blocked - slight turn to clearer side
            left_dist = self.sensor_data.get('front_left', {}).get('distance', 100)
            right_dist = self.sensor_data.get('front_right', {}).get('distance', 100)
            
            if left_dist > right_dist and left_dist > ULTRASONIC_CRITICAL:
                # Turn slightly left
                return left_speed * 0.3, right_speed * 1.2
            elif right_dist > ULTRASONIC_CRITICAL:
                # Turn slightly right
                return left_speed * 1.2, right_speed * 0.3
            else:
                # Both sides blocked - slow down significantly
                return left_speed * 0.1, right_speed * 0.1
        
        elif threat_sensor == 'front_left':
            # Left side blocked - adjust right
            return left_speed * 1.1, right_speed * 0.7
        
        elif threat_sensor == 'front_right':
            # Right side blocked - adjust left
            return left_speed * 0.7, right_speed * 1.1
        
        return left_speed * 0.5, right_speed * 0.5  # Default slow down
    
    def get_sensor_status(self) -> Dict:
        """Get current sensor status for monitoring"""
        status = {}
        current_time = time.time()
        
        for sensor_name, data in self.sensor_data.items():
            data_age = current_time - data['timestamp']
            status[sensor_name] = {
                'distance': data['distance'],
                'age': data_age,
                'valid': data['distance'] > 0 and data_age < 0.1
            }
        
        return status
    
    def stop(self):
        """Stop collision avoidance system"""
        self.running = False
        try:
            GPIO.cleanup()
            print("✓ Collision avoidance system stopped")
        except:
            pass

# =============================================================================
# LIDAR SAFETY MONITOR  
# =============================================================================

class LidarSafetyMonitor:
    """Lightweight LIDAR monitoring for additional safety coverage"""
    
    def __init__(self):
        self.scan_data = {}
        self.last_scan_time = 0
        self.data_lock = threading.Lock()
        
        # Safety zones (in mm)
        self.emergency_zone = 80    # 8cm
        self.critical_zone = 150    # 15cm
        self.warning_zone = 300     # 30cm
        
        print("✓ LIDAR Safety Monitor initialized")
    
    def process_scan(self, scan_msg):
        """Process LIDAR scan for safety monitoring"""
        with self.data_lock:
            self.scan_data.clear()
            self.last_scan_time = time.time()
            
            ranges = scan_msg.ranges
            angle_increment = scan_msg.angle_increment
            angle_min = scan_msg.angle_min
            
            # Process every 3rd point for performance
            for i in range(0, len(ranges), 3):
                distance = ranges[i]
                
                if 0.05 < distance < 6.0:  # Valid range
                    angle_rad = angle_min + (i * angle_increment)
                    angle_deg = int(math.degrees(angle_rad) % 360)
                    distance_mm = int(distance * 1000)
                    
                    self.scan_data[angle_deg] = distance_mm
    
    def check_lidar_collision_threat(self) -> Tuple[bool, float]:
        """Check for collision threats from LIDAR"""
        with self.data_lock:
            current_time = time.time()
            
            # Check if data is recent
            if current_time - self.last_scan_time > 0.2:  # 200ms timeout
                return False, float('inf')
            
            min_front_distance = float('inf')
            
            # Check front sector (330° to 30°)
            front_angles = list(range(330, 360)) + list(range(0, 31))
            
            for angle in front_angles:
                if angle in self.scan_data:
                    distance = self.scan_data[angle]
                    min_front_distance = min(min_front_distance, distance)
                    
                    # Emergency threat
                    if distance < self.emergency_zone:
                        return True, distance
            
            return False, min_front_distance

# =============================================================================
# SMOOTH MOTOR CONTROLLER
# =============================================================================

class SmoothMotorController:
    """Smooth motor control with acceleration/deceleration"""
    
    def __init__(self, left_port: str, right_port: str):
        # Initialize motors
        self.left_motor = MotorControl(device=left_port)
        self.right_motor = MotorControl(device=right_port)
        
        # Set drive mode
        self.left_motor.set_drive_mode(1, 2)
        self.right_motor.set_drive_mode(1, 2)
        
        # Current speeds
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        
        # Target speeds
        self.target_left_speed = 0.0
        self.target_right_speed = 0.0
        
        # Acceleration limits
        self.max_acceleration = ACCELERATION_RATE  # RPM per update
        
        # Control thread
        self.control_active = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        print("✓ Smooth Motor Controller initialized")
    
    def set_target_speeds(self, left_speed: float, right_speed: float):
        """Set target speeds for smooth transition"""
        self.target_left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        self.target_right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
    
    def _control_loop(self):
        """Smooth motor control loop"""
        while self.control_active:
            try:
                # Calculate speed differences
                left_diff = self.target_left_speed - self.current_left_speed
                right_diff = self.target_right_speed - self.current_right_speed
                
                # Apply acceleration limits
                if abs(left_diff) > self.max_acceleration:
                    left_step = self.max_acceleration if left_diff > 0 else -self.max_acceleration
                else:
                    left_step = left_diff
                
                if abs(right_diff) > self.max_acceleration:
                    right_step = self.max_acceleration if right_diff > 0 else -self.max_acceleration
                else:
                    right_step = right_diff
                
                # Update current speeds
                self.current_left_speed += left_step
                self.current_right_speed += right_step
                
                # Send to motors
                self.left_motor.send_rpm(1, int(self.current_left_speed))
                self.right_motor.send_rpm(1, int(self.current_right_speed))
                
                time.sleep(1.0 / MAIN_CONTROL_FREQ)  # Control frequency
                
            except Exception as e:
                print(f"Motor control error: {e}")
                time.sleep(0.01)
    
    def emergency_stop(self):
        """Emergency stop - immediate"""
        self.target_left_speed = 0
        self.target_right_speed = 0
        self.current_left_speed = 0
        self.current_right_speed = 0
        
        try:
            self.left_motor.send_rpm(1, 0)
            self.right_motor.send_rpm(1, 0)
        except:
            pass
    
    def stop(self):
        """Stop motor controller"""
        self.control_active = False
        self.emergency_stop()
        print("✓ Motor controller stopped")

# =============================================================================
# MAIN ROBOT CONTROLLER
# =============================================================================

class EnhancedFollowingRobotNode(Node):
    """Enhanced robot node with UWB-priority control and smooth movement"""
    
    def __init__(self):
        super().__init__('enhanced_following_robot')
        
        # System status
        self.running = True
        
        # Initialize core components
        self.uwb_tracker = UWBTargetTracker()
        self.collision_avoidance = CollisionAvoidanceSystem()
        self.lidar_monitor = LidarSafetyMonitor()
        self.motor_controller = SmoothMotorController("/dev/ttyUSB1", "/dev/ttyUSB0")
        
        # UWB Communication
        self.setup_uwb_communication()
        
        # LIDAR Subscription
        self.setup_lidar_subscription()
        
        # Control Timer
        self.setup_control_timer()
        
        # Performance monitoring
        self.loop_count = 0
        self.last_performance_log = time.time()
        
        # Turn on status LED
        gpio_pin_17.on()
        
        print("=" * 60)
        print("🚀 ENHANCED UWB-PRIORITY FOLLOWING ROBOT READY")
        print("=" * 60)
        print("Primary System: UWB Target Tracking")
        print("Safety Systems: Ultrasonic + LIDAR Collision Avoidance")
        print("Movement: Smooth Acceleration with Real-time Response")
        print("=" * 60)
    
    def setup_uwb_communication(self):
        """Setup UWB communication"""
        try:
            robot_ip = get_ip_from_wifi()
            if not robot_ip:
                raise Exception("No IP address available")
            
            self.uwb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.uwb_socket.bind((robot_ip, 8888))
            self.uwb_socket.settimeout(UWB_TIMEOUT)
            
            print(f"✓ UWB communication ready on {robot_ip}:8888")
            
        except Exception as e:
            print(f"✗ UWB setup failed: {e}")
            self.uwb_socket = None
    
    def setup_lidar_subscription(self):
        """Setup LIDAR subscription"""
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos,
            callback_group=ReentrantCallbackGroup()
        )
        
        print("✓ LIDAR subscription created")
    
    def setup_control_timer(self):
        """Setup main control timer"""
        timer_period = 1.0 / MAIN_CONTROL_FREQ
        
        self.control_timer = self.create_timer(
            timer_period,
            self.main_control_loop,
            callback_group=ReentrantCallbackGroup()
        )
        
        print(f"✓ Control timer setup at {MAIN_CONTROL_FREQ}Hz")
    
    def lidar_callback(self, msg):
        """LIDAR data callback"""
        try:
            self.lidar_monitor.process_scan(msg)
        except Exception as e:
            print(f"LIDAR callback error: {e}")
    
    def get_uwb_data(self) -> Optional[Dict[str, float]]:
        """Get UWB data from socket"""
        if not self.uwb_socket:
            return None
        
        try:
            data, addr = self.uwb_socket.recvfrom(UWB_BUFFER_SIZE)
            received_data = data.decode('utf-8').strip()
            
            parts = received_data.split(',')
            if len(parts) >= 3:
                raw_distances = {
                    'A0': float(parts[0]),
                    'A1': float(parts[1]),
                    'A2': float(parts[2])
                }
                
                # Apply corrections
                corrected_distances = self.uwb_tracker.apply_corrections(raw_distances)
                return corrected_distances
            
        except socket.timeout:
            pass
        except Exception as e:
            print(f"UWB data error: {e}")
        
        return None
    
    def main_control_loop(self):
        """Main control loop - UWB priority with safety override"""
        loop_start = time.time()
        
        try:
            # Step 1: Get UWB target data (PRIMARY)
            uwb_data = self.get_uwb_data()
            
            if uwb_data is None:
                # No UWB data - stop motors for safety
                self.motor_controller.set_target_speeds(0, 0)
                return
            
            # Step 2: Generate primary movement based on UWB
            target_direction, target_distance = self.uwb_tracker.estimate_target_position(uwb_data)
            desired_left_speed, desired_right_speed = self.uwb_tracker.generate_movement_commands(
                target_direction, target_distance
            )
            
            # Step 3: Apply collision avoidance adjustments (SAFETY ONLY)
            # Check ultrasonic sensors first (fastest response)
            safe_left_speed, safe_right_speed = self.collision_avoidance.get_avoidance_adjustment(
                desired_left_speed, desired_right_speed
            )
            
            # Check LIDAR for additional safety
            lidar_threat, lidar_distance = self.lidar_monitor.check_lidar_collision_threat()
            if lidar_threat and lidar_distance < EMERGENCY_STOP_DISTANCE:
                print(f"LIDAR EMERGENCY STOP: {lidar_distance}mm")
                safe_left_speed = 0
                safe_right_speed = 0
            
            # Step 4: Send commands to motors
            self.motor_controller.set_target_speeds(safe_left_speed, safe_right_speed)
            
            # Step 5: Performance monitoring
            self.loop_count += 1
            if self.loop_count % 100 == 0:  # Every 100 loops
                self.log_performance(loop_start, uwb_data, target_direction, target_distance)
            
        except Exception as e:
            print(f"Control loop error: {e}")
            self.motor_controller.emergency_stop()
    
    def log_performance(self, loop_start: float, uwb_data: Dict, direction: float, distance: float):
        """Log performance metrics"""
        loop_time = (time.time() - loop_start) * 1000  # ms
        
        # Get sensor status
        ultrasonic_status = self.collision_avoidance.get_sensor_status()
        valid_sensors = sum(1 for s in ultrasonic_status.values() if s['valid'])
        
        print(f"📊 Target: {direction:.1f}° @ {distance:.1f}cm | "
              f"UWB: A0={uwb_data['A0']:.1f} A1={uwb_data['A1']:.1f} A2={uwb_data['A2']:.1f} | "
              f"Sensors: {valid_sensors}/3 | Loop: {loop_time:.1f}ms")
    
    def stop(self):
        """Stop the robot system"""
        print("\n🛑 Stopping Enhanced Following Robot...")
        
        self.running = False
        
        # Stop motors
        self.motor_controller.stop()
        
        # Stop collision avoidance
        self.collision_avoidance.stop()
        
        # Close UWB socket
        if hasattr(self, 'uwb_socket') and self.uwb_socket:
            self.uwb_socket.close()
        
        # Turn off status LED
        gpio_pin_17.off()
        
        print("✓ Robot stopped safely")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(args=None):
    """Main function with enhanced error handling"""
    
    rclpy.init(args=args)
    
    node = None
    executor = None
    
    try:
        print("🚀 Starting Enhanced UWB-Priority Following Robot")
        print("=" * 60)
        
        # Create robot node
        node = EnhancedFollowingRobotNode()
        
        # Create executor
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        
        print("✅ System Ready - Robot is following UWB target")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Run the robot
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n🛑 User stop requested")
        
    except Exception as e:
        print(f"\n❌ System error: {e}")
        gpio_pin_27.on()  # Error indicator
        
    finally:
        # Cleanup
        print("\n🔄 Cleaning up system...")
        
        if node:
            node.stop()
            node.destroy_node()
        
        if executor:
            executor.shutdown(timeout_sec=2.0)
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        try:
            GPIO.cleanup()
            gpio_pin_17.off()
            gpio_pin_27.off()
        except:
            pass
        
        print("✅ System shutdown complete")

if __name__ == '__main__':
    main()
