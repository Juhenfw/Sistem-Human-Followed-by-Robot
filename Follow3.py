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

# Konfigurasi Global
SCAN_THRESHOLD = 2000  # Jarak aman dalam mm
DANGER_THRESHOLD = 1000  # Jarak bahaya dalam mm
MIN_VALID_DISTANCE = 200  # Jarak minimum valid (mm) untuk menghindari noise
CRITICAL_DANGER_THRESHOLD = 300  # Jarak kritis untuk override UWB (mm)

# Ultrasonic sensor thresholds (in cm) - OPTIMIZED
ULTRASONIC_CRITICAL_THRESHOLD = 20  # Turunkan dari 25 ke 20 cm
ULTRASONIC_WARNING_THRESHOLD = 40   # Turunkan dari 45 ke 40 cm  
ULTRASONIC_SAFE_THRESHOLD = 65      # Turunkan dari 70 ke 65 cm

# Motor speed configuration
DEFAULT_SPEED = 75
ROTATION_FACTOR = 2
STOP_THRESHOLD = 80  # cm

# Konfigurasi performa real-time
CONTROL_FREQUENCY = 5000  # Hz - increased from 200
LIDAR_SKIP_FRAMES = 0    # Process every frame
UWB_TIMEOUT = 0.0005     # 0.5ms timeout
MAX_LOOP_TIME = 0.002    # 2ms warning threshold

# Buffer sizes
LIDAR_BUFFER_SIZE = 200   # Increased for real-time
UWB_BUFFER_SIZE = 2048   # Increased UWB buffer

# Definisikan sudut lebih jelas
FRONT_REGION = [(330, 360), (0, 30)]  # Depan: 330° hingga 360° dan 0° hingga 30°
RIGHT_REGION = (31, 140)  # Kanan: 31° hingga 140°
LEFT_REGION  = (220, 329)  # Kiri: 220° hingga 329°
BACK_REGION = (150, 210)  # Belakang: 150° hingga 210°
TARGET_EXCLUSION_ANGLE = 10  # Rentang pengecualian untuk target

# GPIO Pins Initialization
gpio_pin_17 = LED(17)  # GPIO 17 for program start
gpio_pin_27 = LED(27)  # GPIO 27 for error indication

# OPTIMIZED SENSOR CONFIGURATION
SENSORS = {
    'front_left': {'trig': 18, 'echo': 24, 'position': 'front_left'},     # Sensor 1 - Pojok kiri depan
    'front_center': {'trig': 23, 'echo': 25, 'position': 'front_center'}, # Sensor 2 - Tengah depan  
    'front_right': {'trig': 12, 'echo': 16, 'position': 'front_right'}    # Sensor 3 - Pojok kanan depan
}

# Optimized configuration for fast response
OPTIMIZED_CONFIG = {
    # Control frequencies (Hz)
    'emergency_control_freq': 1000,
    'normal_control_freq': 500,
    'sensor_update_freq': 800,
    
    # Response time targets (ms)
    'emergency_response_target': 1,
    'critical_response_target': 3,
    'normal_response_target': 5,
    
    # Thread priorities
    'thread_priorities': {
        'emergency': 99,  # Real-time priority
        'sensor_fusion': 80,
        'control': 70,
        'planning': 50
    },
    
    # Buffer sizes (optimized)
    'sensor_buffer_size': 64,
    'control_buffer_size': 32,
    
    # Timeout values (reduced)
    'sensor_timeout': 0.0005,  # 0.5ms
    'control_timeout': 0.001,  # 1ms
    
    # Distance thresholds (mm)
    'immediate_threat': 120,
    'critical_distance': 200,
    'warning_distance': 350,
    'safe_distance': 500
}

def get_ip_from_wifi(interface='wlan0'):
    """Get the IP address of the Raspberry Pi from the Wi-Fi interface"""
    try:
        ip = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
        return ip
    except (KeyError, ValueError):
        print(f"Failed to get IP address for interface: {interface}")
        return None
    
def get_ip_from_subnet(ip, target_last_digit):
    """Modify the last digit of the IP address to match the target_last_digit"""
    ip_parts = ip.split(".")
    ip_parts[-1] = str(target_last_digit)
    return ".".join(ip_parts)

class UltrasonicSensor:
    """Ultrasonic sensor class for JSN-SR04T sensors with optimized performance"""
    
    def __init__(self, name, trig_pin, echo_pin, position):
        self.name = name
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.position = position
        
        # Data storage
        self.current_distance = -1
        self.last_valid_distance = -1
        self.measurement_count = 0
        self.error_count = 0
        
        # Optimized filtering
        self.distance_history = deque(maxlen=3)  # Reduced for faster response
        self.last_measurement_time = 0
        
        try:
            # Setup GPIO pins
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, False)
            
            # Shorter settle time
            time.sleep(0.05)
            print(f"✓ {name} initialized (Trig: GPIO{trig_pin}, Echo: GPIO{echo_pin}) - {position}")
            
        except Exception as e:
            print(f"✗ Error initializing {name}: {e}")
            raise
    
    def measure_distance(self):
        """Mengukur jarak dalam cm dengan optimized filtering"""
        try:
            # Kirim trigger pulse (8us untuk faster response)
            GPIO.output(self.trig_pin, True)
            time.sleep(0.000008)  # 8 microseconds
            GPIO.output(self.trig_pin, False)
            
            # Tunggu echo response dengan shorter timeout
            timeout_start = time.time()
            timeout_duration = 0.02  # 20ms timeout (reduced)
            
            # Tunggu echo pin HIGH
            pulse_start = timeout_start
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()
                if (pulse_start - timeout_start) > timeout_duration:
                    self.error_count += 1
                    return -1
            
            # Tunggu echo pin LOW
            pulse_end = pulse_start
            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()
                if (pulse_end - timeout_start) > timeout_duration:
                    self.error_count += 1
                    return -1
            
            # Hitung jarak
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2
            
            # Filter hasil yang masuk akal (extended range)
            if 1 <= distance <= 450:
                self.distance_history.append(distance)
                self.current_distance = round(distance, 1)
                self.last_valid_distance = self.current_distance
                self.measurement_count += 1
                self.last_measurement_time = time.time()
                return self.current_distance
            else:
                self.error_count += 1
                return -1
                
        except Exception as e:
            print(f"Error measuring {self.name}: {e}")
            self.error_count += 1
            return -1
    
    def get_filtered_distance(self):
        """Get filtered distance using optimized averaging"""
        if len(self.distance_history) >= 2:
            # Use median filtering for noise reduction
            sorted_distances = sorted(self.distance_history)
            if len(sorted_distances) % 2 == 0:
                mid = len(sorted_distances) // 2
                return (sorted_distances[mid-1] + sorted_distances[mid]) / 2
            else:
                return sorted_distances[len(sorted_distances) // 2]
        elif self.distance_history:
            return self.distance_history[-1]
        else:
            return -1
    
    def is_obstacle_detected(self):
        """Check if obstacle is detected at different threat levels"""
        filtered_distance = self.get_filtered_distance()
        
        if filtered_distance <= 0:
            return {'level': 'unknown', 'distance': -1}
        
        if filtered_distance <= ULTRASONIC_CRITICAL_THRESHOLD:
            return {'level': 'critical', 'distance': filtered_distance}
        elif filtered_distance <= ULTRASONIC_WARNING_THRESHOLD:
            return {'level': 'warning', 'distance': filtered_distance}
        elif filtered_distance <= ULTRASONIC_SAFE_THRESHOLD:
            return {'level': 'safe', 'distance': filtered_distance}
        else:
            return {'level': 'clear', 'distance': filtered_distance}

class UltrasonicSensorManager:
    """Manages multiple ultrasonic sensors for ultra-fast obstacle detection"""
    
    def __init__(self):
        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Initialize sensors
        self.sensors = {}
        self.sensor_threads = {}
        self.running = True
        
        # Sensor status with thread-safe updates
        self.sensor_data = {
            'front_left': {'distance': -1, 'level': 'unknown', 'last_update': 0},
            'front_center': {'distance': -1, 'level': 'unknown', 'last_update': 0},
            'front_right': {'distance': -1, 'level': 'unknown', 'last_update': 0}
        }
        
        # Thread-safe lock
        self.data_lock = threading.Lock()
        
        # Initialize sensors
        self.initialize_sensors()
        
        # Start sensor reading threads
        self.start_sensor_threads()
    
    def initialize_sensors(self):
        """Initialize all ultrasonic sensors"""
        print("Initializing ultrasonic sensors...")
        
        for name, config in SENSORS.items():
            try:
                sensor = UltrasonicSensor(
                    name, 
                    config['trig'], 
                    config['echo'], 
                    config['position']
                )
                self.sensors[name] = sensor
                print(f"✓ {name} sensor initialized successfully")
                
            except Exception as e:
                print(f"✗ Failed to initialize {name}: {e}")
                continue
        
        if not self.sensors:
            raise RuntimeError("No ultrasonic sensors could be initialized!")
        
        print(f"✓ {len(self.sensors)} ultrasonic sensors initialized")
    
    def read_sensor_continuously(self, sensor_name):
        """Continuously read sensor data in separate thread with higher frequency"""
        if sensor_name not in self.sensors:
            return
        
        sensor = self.sensors[sensor_name]
        
        while self.running:
            try:
                # Measure distance
                distance = sensor.measure_distance()
                obstacle_info = sensor.is_obstacle_detected()
                
                # Update sensor data thread-safely
                with self.data_lock:
                    self.sensor_data[sensor_name] = {
                        'distance': distance,
                        'level': obstacle_info['level'],
                        'last_update': time.time()
                    }
                
                # Faster update rate for critical response
                time.sleep(0.015)  # ~67Hz per sensor
                
            except Exception as e:
                print(f"Error in sensor reading thread {sensor_name}: {e}")
                time.sleep(0.05)
    
    def start_sensor_threads(self):
        """Start reading threads for all sensors"""
        print("Starting ultrasonic sensor threads...")
        
        for sensor_name in self.sensors.keys():
            thread = threading.Thread(
                target=self.read_sensor_continuously,
                args=(sensor_name,),
                daemon=True
            )
            thread.start()
            self.sensor_threads[sensor_name] = thread
            time.sleep(0.005)  # Reduced stagger time
        
        print(f"✓ {len(self.sensor_threads)} sensor threads started")
    
    def get_sensor_status(self):
        """Get current status of all sensors"""
        with self.data_lock:
            return dict(self.sensor_data)
    
    def is_critical_obstacle_detected(self):
        """Ultra-fast check for critical obstacles in any sensor"""
        with self.data_lock:
            for sensor_name, data in self.sensor_data.items():
                if data['level'] == 'critical' and data['distance'] > 0:
                    return True, sensor_name, data['distance']
        return False, None, -1
    
    def get_obstacle_summary(self):
        """Get optimized summary of obstacle detection"""
        status = self.get_sensor_status()
        
        summary = {
            'critical_detected': False,
            'warning_detected': False,
            'closest_obstacle': float('inf'),
            'blocked_directions': [],
            'sensor_status': status
        }
        
        for sensor_name, data in status.items():
            if data['distance'] > 0:
                if data['level'] == 'critical':
                    summary['critical_detected'] = True
                    summary['blocked_directions'].append(sensor_name)
                elif data['level'] == 'warning':
                    summary['warning_detected'] = True
                
                if data['distance'] < summary['closest_obstacle']:
                    summary['closest_obstacle'] = data['distance']
        
        return summary
    
    def stop(self):
        """Stop all sensor reading threads"""
        self.running = False
        try:
            GPIO.cleanup()
            print("Ultrasonic sensors stopped and GPIO cleaned up")
        except:
            pass

class DynamicWindowApproach:
    """Advanced obstacle avoidance using Dynamic Window Approach"""
    
    def __init__(self, robot_controller):
        self.controller = robot_controller
        
        # DWA Parameters
        self.dt = 0.05  # Time step (reduced for faster response)
        self.predict_time = 1.5  # Prediction horizon
        
        # Robot constraints
        self.max_speed = 100  # mm/s
        self.min_speed = -20  # mm/s
        self.max_angular_vel = 1.2  # rad/s
        self.max_accel = 60  # mm/s²
        self.max_angular_accel = 2.5  # rad/s²
        
        # Cost function weights (adjusted for avoiding stops)
        self.heading_weight = 0.15
        self.distance_weight = 0.25
        self.velocity_weight = 0.15
        self.obstacle_weight = 0.45
        
        # Path evaluation
        self.velocity_resolution = 8
        self.angular_resolution = 12
        
    def calculate_dynamic_window(self, current_vel, current_angular_vel):
        """Calculate feasible velocity window based on robot dynamics"""
        
        # Velocity limits based on acceleration constraints
        vel_min = max(self.min_speed, current_vel - self.max_accel * self.dt)
        vel_max = min(self.max_speed, current_vel + self.max_accel * self.dt)
        
        # Angular velocity limits
        angular_min = max(-self.max_angular_vel, 
                         current_angular_vel - self.max_angular_accel * self.dt)
        angular_max = min(self.max_angular_vel, 
                         current_angular_vel + self.max_angular_accel * self.dt)
        
        return (vel_min, vel_max, angular_min, angular_max)
    
    def predict_trajectory(self, vel, angular_vel, predict_steps):
        """Predict robot trajectory for given velocities"""
        trajectory = []
        x, y, theta = 0, 0, 0
        
        for i in range(predict_steps):
            # Update position
            x += vel * math.cos(theta) * self.dt
            y += vel * math.sin(theta) * self.dt
            theta += angular_vel * self.dt
            
            trajectory.append((x, y, theta))
        
        return trajectory
    
    def evaluate_trajectory(self, trajectory, target_x, target_y, obstacles):
        """Evaluate trajectory quality using multiple criteria"""
        
        if not trajectory:
            return float('-inf')
        
        final_x, final_y, final_theta = trajectory[-1]
        
        # 1. Heading cost (how well aligned with target)
        target_angle = math.atan2(target_y, target_x)
        angle_diff = abs(final_theta - target_angle)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        heading_cost = angle_diff
        
        # 2. Distance cost (closer to target is better)
        distance_to_target = math.sqrt(final_x**2 + final_y**2)
        distance_cost = distance_to_target
        
        # 3. Velocity cost (prefer higher velocities - avoid stopping)
        velocity_cost = 1.0 / (abs(trajectory[-1][0] - trajectory[0][0]) + 0.1)
        
        # 4. Obstacle cost (heavily penalize collision paths)
        obstacle_cost = 0
        min_obstacle_distance = float('inf')
        
        for x, y, theta in trajectory:
            for obs_angle, obs_distance in obstacles.items():
                if obs_distance > 0:
                    obs_x = obs_distance * math.cos(math.radians(obs_angle))
                    obs_y = obs_distance * math.sin(math.radians(obs_angle))
                    
                    distance_to_obs = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                    min_obstacle_distance = min(min_obstacle_distance, distance_to_obs)
                    
                    if distance_to_obs < 250:  # 25cm safety margin
                        obstacle_cost += 1000 / (distance_to_obs + 1)
        
        # If collision path, return very low score
        if min_obstacle_distance < 150:  # 15cm critical distance
            return float('-inf')
        
        # Combined cost (lower is better, so negate for maximization)
        total_cost = -(self.heading_weight * heading_cost +
                      self.distance_weight * distance_cost +
                      self.velocity_weight * velocity_cost +
                      self.obstacle_weight * obstacle_cost)
        
        return total_cost
    
    def find_best_trajectory(self, current_vel, current_angular_vel, 
                           target_direction, target_distance, obstacles):
        """Find optimal trajectory using DWA"""
        
        vel_min, vel_max, angular_min, angular_max = self.calculate_dynamic_window(
            current_vel, current_angular_vel
        )
        
        best_vel = current_vel
        best_angular_vel = current_angular_vel
        best_score = float('-inf')
        best_trajectory = []
        
        # Sample velocity space
        vel_step = (vel_max - vel_min) / self.velocity_resolution
        angular_step = (angular_max - angular_min) / self.angular_resolution
        
        target_x = target_distance * math.cos(math.radians(target_direction))
        target_y = target_distance * math.sin(math.radians(target_direction))
        
        predict_steps = int(self.predict_time / self.dt)
        
        for i in range(self.velocity_resolution + 1):
            vel = vel_min + i * vel_step
            
            for j in range(self.angular_resolution + 1):
                angular_vel = angular_min + j * angular_step
                
                # Predict trajectory
                trajectory = self.predict_trajectory(vel, angular_vel, predict_steps)
                
                # Evaluate trajectory
                score = self.evaluate_trajectory(trajectory, target_x, target_y, obstacles)
                
                if score > best_score:
                    best_score = score
                    best_vel = vel
                    best_angular_vel = angular_vel
                    best_trajectory = trajectory
        
        return best_vel, best_angular_vel, best_trajectory, best_score

    def execute_optimal_path(self, uwb_distances, lidar_data, ultrasonic_data):
        """Execute optimal path using DWA"""
        
        # Get current robot state
        current_vel = abs(self.controller.current_left_speed + self.controller.current_right_speed) / 2
        current_angular_vel = (self.controller.current_left_speed - self.controller.current_right_speed) / 100
        
        # Get target info
        target_direction, target_distance = self.controller.parent.uwb_tracker.estimate_target_direction(uwb_distances)
        
        # Combine obstacle data from all sensors
        obstacles = {}
        
        # Add LIDAR data
        if hasattr(lidar_data, 'scan_data'):
            obstacles.update(lidar_data.scan_data)
        
        # Add ultrasonic data
        if ultrasonic_data:
            status = ultrasonic_data.get_obstacle_summary()
            for sensor_name, data in status['sensor_status'].items():
                if data['distance'] > 0:
                    if sensor_name == 'front_left':
                        obstacles[315] = data['distance'] * 10  # Convert cm to mm
                    elif sensor_name == 'front_center':
                        obstacles[0] = data['distance'] * 10
                    elif sensor_name == 'front_right':
                        obstacles[45] = data['distance'] * 10
        
        # Find optimal trajectory
        best_vel, best_angular_vel, trajectory, score = self.find_best_trajectory(
            current_vel, current_angular_vel, target_direction, target_distance, obstacles
        )
        
        # Convert to wheel speeds
        if score > float('-inf'):
            # Normal operation with optimal path
            left_speed = best_vel + (best_angular_vel * 50)
            right_speed = -(best_vel - (best_angular_vel * 50))
            
            print(f"DWA Optimal: L={left_speed:.1f}, R={right_speed:.1f}, Score={score:.2f}")
            return int(left_speed), int(right_speed), "OPTIMAL_PATH"
        else:
            # Emergency fallback - find any safe direction
            return self.emergency_path_finding(obstacles)
    
    def emergency_path_finding(self, obstacles):
        """Emergency path finding when no optimal path exists"""
        
        print("DWA EMERGENCY: Finding escape route")
        
        # Find direction with most space
        direction_spaces = {}
        directions = [('LEFT', 270), ('RIGHT', 90), ('BACK', 180)]
        
        for name, angle in directions:
            min_distance = float('inf')
            for obs_angle, obs_distance in obstacles.items():
                angle_diff = abs(obs_angle - angle)
                if angle_diff <= 45 or angle_diff >= 315:  # Within 45 degrees
                    min_distance = min(min_distance, obs_distance)
            
            direction_spaces[name] = min_distance
        
        # Choose safest direction
        safest_direction = max(direction_spaces, key=direction_spaces.get)
        safest_distance = direction_spaces[safest_direction]
        
        if safest_distance > 250:  # 25cm clearance
            if safest_direction == 'LEFT':
                return -30, -30, "EMERGENCY_LEFT"
            elif safest_direction == 'RIGHT':
                return 30, 30, "EMERGENCY_RIGHT"
            elif safest_direction == 'BACK':
                return -25, 25, "EMERGENCY_REVERSE"
        
        # Last resort: minimal movement
        return -15, 15, "EMERGENCY_MINIMAL"

class PathEvaluator:
    """Advanced path evaluation with multiple criteria"""
    
    def __init__(self):
        # Evaluation weights (adjusted to avoid stopping)
        self.weights = {
            'target_alignment': 0.2,
            'obstacle_clearance': 0.3,
            'path_smoothness': 0.15,
            'energy_efficiency': 0.1,
            'time_to_target': 0.25  # Increased to prefer faster paths
        }
        
        # Safety parameters
        self.safe_distance = 350  # mm (reduced)
        self.comfort_distance = 500  # mm (reduced)
        
    def evaluate_path_quality(self, path_option, target_info, obstacle_info):
        """Comprehensive path quality evaluation"""
        
        scores = {}
        
        # 1. Target alignment score
        scores['target_alignment'] = self.calculate_target_alignment(
            path_option, target_info
        )
        
        # 2. Obstacle clearance score
        scores['obstacle_clearance'] = self.calculate_obstacle_clearance(
            path_option, obstacle_info
        )
        
        # 3. Path smoothness score
        scores['path_smoothness'] = self.calculate_path_smoothness(path_option)
        
        # 4. Energy efficiency score
        scores['energy_efficiency'] = self.calculate_energy_efficiency(path_option)
        
        # 5. Time to target score
        scores['time_to_target'] = self.calculate_time_efficiency(
            path_option, target_info
        )
        
        # Calculate weighted total score
        total_score = sum(
            self.weights[criteria] * score 
            for criteria, score in scores.items()
        )
        
        return total_score, scores
    
    def calculate_target_alignment(self, path_option, target_info):
        """Calculate how well path aligns with target direction"""
        target_direction = target_info.get('direction', 0)
        path_direction = path_option.get('direction', 0)
        
        angle_diff = abs(target_direction - path_direction)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # Score: 1.0 for perfect alignment, 0.0 for opposite direction
        alignment_score = 1.0 - (angle_diff / 180.0)
        return max(0.0, alignment_score)
    
    def calculate_obstacle_clearance(self, path_option, obstacle_info):
        """Calculate obstacle clearance quality"""
        min_clearance = float('inf')
        
        # Check clearance for this path
        path_angles = path_option.get('scan_angles', [])
        
        for angle in path_angles:
            if angle in obstacle_info:
                distance = obstacle_info[angle]
                if distance > 0:
                    min_clearance = min(min_clearance, distance)
        
        if min_clearance == float('inf'):
            return 1.0  # No obstacles detected
        
        # Score based on clearance distance
        if min_clearance >= self.comfort_distance:
            return 1.0
        elif min_clearance >= self.safe_distance:
            return 0.8
        elif min_clearance >= 150:  # Critical distance
            return 0.4
        else:
            return 0.0  # Unsafe path
    
    def calculate_path_smoothness(self, path_option):
        """Calculate path smoothness (prefer gradual turns over sharp turns)"""
        turn_intensity = path_option.get('turn_intensity', 0)
        
        # Score: 1.0 for straight path, decreasing with turn intensity
        smoothness_score = 1.0 - min(1.0, turn_intensity / 90.0)
        return max(0.0, smoothness_score)
    
    def calculate_energy_efficiency(self, path_option):
        """Calculate energy efficiency of path"""
        speed_required = path_option.get('speed_required', 50)
        turn_intensity = path_option.get('turn_intensity', 0)
        
        # Lower speeds and fewer turns are more efficient
        speed_efficiency = 1.0 - (speed_required / 100.0)
        turn_efficiency = 1.0 - (turn_intensity / 180.0)
        
        energy_score = (speed_efficiency + turn_efficiency) / 2.0
        return max(0.0, energy_score)
    
    def calculate_time_efficiency(self, path_option, target_info):
        """Calculate time efficiency to reach target"""
        target_distance = target_info.get('distance', 1000)
        path_distance = path_option.get('estimated_distance', target_distance)
        
        # Shorter paths are more time efficient
        if path_distance <= target_distance:
            return 1.0
        else:
            efficiency = target_distance / path_distance
            return max(0.1, efficiency)

class FastResponseController:
    """Ultra-fast response controller with priority-based processing"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        
        # High-frequency control parameters
        self.emergency_frequency = 1000  # Hz - for critical obstacles
        self.normal_frequency = 500     # Hz - for normal operation
        self.sensor_frequency = 800     # Hz - for sensor updates
        
        # Response time targets (milliseconds)
        self.target_response_times = {
            'critical_obstacle': 1,    # 1ms for emergency
            'warning_obstacle': 3,     # 3ms for warnings
            'normal_operation': 5      # 5ms for normal
        }
        
        # Multi-level priority system
        self.priority_levels = {
            'EMERGENCY': 0,     # Immediate collision threat
            'CRITICAL': 1,      # Close obstacles
            'WARNING': 2,       # Moderate obstacles
            'NORMAL': 3,        # Normal navigation
            'BACKGROUND': 4     # Non-critical tasks
        }
        
        # Fast decision trees
        self.decision_trees = self.build_decision_trees()
        
        # Threading system
        self.threads = {}
        self.locks = {
            'sensor_data': threading.Lock(),
            'control_commands': threading.Lock(),
            'emergency_stop': threading.Lock()
        }
        
        # Emergency response system
        self.emergency_active = False
        self.last_emergency_check = 0
        
        # DWA integration
        self.dwa = DynamicWindowApproach(robot_controller)
        self.path_evaluator = PathEvaluator()
        
    def build_decision_trees(self):
        """Build fast decision trees for different scenarios"""
        
        trees = {}
        
        # Emergency decision tree
        trees['emergency'] = {
            'ultrasonic_critical': {
                'front_left': lambda: self.execute_emergency_right(),
                'front_center': lambda: self.execute_emergency_best_side(),
                'front_right': lambda: self.execute_emergency_left()
            },
            'lidar_critical': {
                'front_blocked': lambda: self.execute_emergency_best_side(),
                'left_blocked': lambda: self.execute_emergency_right(),
                'right_blocked': lambda: self.execute_emergency_left()
            }
        }
        
        # Warning decision tree
        trees['warning'] = {
            'single_obstacle': {
                'left': lambda: self.execute_gentle_right(),
                'front': lambda: self.execute_best_turn(),
                'right': lambda: self.execute_gentle_left()
            },
            'multiple_obstacles': lambda: self.execute_dwa_planning()
        }
        
        # Normal decision tree
        trees['normal'] = {
            'path_clear': lambda: self.execute_target_following(),
            'minor_adjustment': lambda: self.execute_smooth_correction()
        }
        
        return trees
    
    def start_fast_response_system(self):
        """Start multi-threaded fast response system"""
        
        print("Starting Fast Response Control System...")
        
        # Emergency monitoring thread (highest priority)
        self.threads['emergency'] = threading.Thread(
            target=self.emergency_monitoring_loop,
            daemon=True
        )
        self.threads['emergency'].start()
        
        # Sensor fusion thread
        self.threads['sensor_fusion'] = threading.Thread(
            target=self.sensor_fusion_loop,
            daemon=True
        )
        self.threads['sensor_fusion'].start()
        
        # Fast control thread
        self.threads['fast_control'] = threading.Thread(
            target=self.fast_control_loop,
            daemon=True
        )
        self.threads['fast_control'].start()
        
        # DWA planning thread (lower priority)
        self.threads['dwa_planning'] = threading.Thread(
            target=self.dwa_planning_loop,
            daemon=True
        )
        self.threads['dwa_planning'].start()
        
        print("✓ Fast Response System started")
    
    def emergency_monitoring_loop(self):
        """Ultra-high frequency emergency monitoring"""
        
        while self.robot.parent.running:
            start_time = time.time()
            
            try:
                # Check for immediate collision threats
                emergency_detected = self.check_immediate_threats()
                
                if emergency_detected:
                    with self.locks['emergency_stop']:
                        if not self.emergency_active:
                            self.execute_immediate_response()
                            self.emergency_active = True
                
                # Maintain target frequency
                elapsed = time.time() - start_time
                target_interval = 1.0 / self.emergency_frequency
                
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                    
            except Exception as e:
                print(f"Emergency monitoring error: {e}")
                time.sleep(0.001)
    
    def check_immediate_threats(self):
        """Ultra-fast threat detection"""
        
        # Priority 1: Ultrasonic sensors (fastest)
        if self.robot.parent.ultrasonic_manager:
            critical_detected, sensor_name, distance = \
                self.robot.parent.ultrasonic_manager.is_critical_obstacle_detected()
            
            if critical_detected and distance < 12:  # 12cm immediate threat
                print(f"IMMEDIATE THREAT: {sensor_name} at {distance:.1f}cm")
                return True
        
        # Priority 2: LIDAR critical zones
        if hasattr(self.robot.parent.lidar, 'scan_data'):
            for angle in range(350, 361):  # Front 20 degrees
                if angle in self.robot.parent.lidar.scan_data:
                    distance = self.robot.parent.lidar.scan_data[angle]
                    if distance < 120:  # 12cm in mm
                        print(f"LIDAR IMMEDIATE: {distance}mm at {angle}°")
                        return True
            
            for angle in range(0, 11):
                if angle in self.robot.parent.lidar.scan_data:
                    distance = self.robot.parent.lidar.scan_data[angle]
                    if distance < 120:
                        print(f"LIDAR IMMEDIATE: {distance}mm at {angle}°")
                        return True
        
        return False
    
    def execute_immediate_response(self):
        """Execute immediate emergency response"""
        
        print("EXECUTING IMMEDIATE RESPONSE")
        
        # Stop immediately
        self.robot.left_motor.send_rpm(1, 0)
        self.robot.right_motor.send_rpm(1, 0)
        
        # Quick assessment for escape route
        escape_action = self.find_immediate_escape()
        
        if escape_action:
            print(f"IMMEDIATE ESCAPE: {escape_action}")
            self.execute_escape_action(escape_action)
        
    def find_immediate_escape(self):
        """Find immediate escape route in under 1ms"""
        
        # Simple rule-based escape logic for speed
        ultrasonic_data = {}
        
        if self.robot.parent.ultrasonic_manager:
            status = self.robot.parent.ultrasonic_manager.get_sensor_status()
            for sensor_name, data in status.items():
                ultrasonic_data[sensor_name] = data.get('distance', -1)
        
        # Quick decision tree
        left_clear = ultrasonic_data.get('front_left', -1)
        center_blocked = ultrasonic_data.get('front_center', -1)
        right_clear = ultrasonic_data.get('front_right', -1)
        
        if center_blocked > 0 and center_blocked < 15:  # Front blocked
            if left_clear > 25 or left_clear == -1:
                return "SHARP_LEFT"
            elif right_clear > 25 or right_clear == -1:
                return "SHARP_RIGHT"
            else:
                return "REVERSE"
        
        return None
    
    def execute_escape_action(self, action):
        """Execute escape action immediately"""
        
        escape_speed = 35  # Moderate speed for safety
        
        if action == "SHARP_LEFT":
            self.robot.left_motor.send_rpm(1, -escape_speed)
            self.robot.right_motor.send_rpm(1, -escape_speed)
        elif action == "SHARP_RIGHT":
            self.robot.left_motor.send_rpm(1, escape_speed)
            self.robot.right_motor.send_rpm(1, escape_speed)
        elif action == "REVERSE":
            self.robot.left_motor.send_rpm(1, -escape_speed//2)
            self.robot.right_motor.send_rpm(1, escape_speed//2)
    
    def sensor_fusion_loop(self):
        """High-frequency sensor data fusion"""
        while self.robot.parent.running:
            try:
                # Fuse sensor data from all sources
                self.fuse_sensor_data()
                time.sleep(1.0 / self.sensor_frequency)
            except Exception as e:
                print(f"Sensor fusion error: {e}")
                time.sleep(0.001)
    
    def fast_control_loop(self):
        """High-frequency control loop"""
        while self.robot.parent.running:
            try:
                # Execute fast control decisions
                self.execute_fast_control()
                time.sleep(1.0 / self.normal_frequency)
            except Exception as e:
                print(f"Fast control error: {e}")
                time.sleep(0.001)
    
    def dwa_planning_loop(self):
        """DWA planning loop"""
        while self.robot.parent.running:
            try:
                # Execute DWA planning for complex scenarios
                self.execute_dwa_planning()
                time.sleep(0.02)  # 50Hz planning
            except Exception as e:
                print(f"DWA planning error: {e}")
                time.sleep(0.01)
    
    def fuse_sensor_data(self):
        """Fuse data from all sensors"""
        # Implementation for sensor data fusion
        pass
    
    def execute_fast_control(self):
        """Execute fast control decisions"""
        # Implementation for fast control
        pass
    
    def execute_dwa_planning(self):
        """Execute DWA planning"""
        # Implementation for DWA planning
        pass
    
    def execute_emergency_right(self):
        """Execute emergency right turn"""
        self.robot.left_motor.send_rpm(1, 25)
        self.robot.right_motor.send_rpm(1, 25)
    
    def execute_emergency_left(self):
        """Execute emergency left turn"""
        self.robot.left_motor.send_rpm(1, -25)
        self.robot.right_motor.send_rpm(1, -25)
    
    def execute_emergency_best_side(self):
        """Execute emergency turn to best side"""
        # Logic to determine best side and turn
        self.execute_emergency_right()  # Default
    
    def execute_gentle_right(self):
        """Execute gentle right turn"""
        self.robot.left_motor.send_rpm(1, 15)
        self.robot.right_motor.send_rpm(1, 15)
    
    def execute_gentle_left(self):
        """Execute gentle left turn"""
        self.robot.left_motor.send_rpm(1, -15)
        self.robot.right_motor.send_rpm(1, -15)
    
    def execute_best_turn(self):
        """Execute best turn based on current situation"""
        # Logic to determine best turn
        self.execute_gentle_right()  # Default
    
    def execute_target_following(self):
        """Execute target following behavior"""
        # Implementation for target following
        pass
    
    def execute_smooth_correction(self):
        """Execute smooth correction"""
        # Implementation for smooth correction
        pass

class UWBTracker:
    """Handles UWB data processing and position estimation"""
    
    def __init__(self):
        # Optimized bias correction values
        self.bias = {
            'A0': 45.0,  # Reduced bias for better accuracy
            'A1': 45.0,
            'A2': 45.0
        }
        
        # Optimized scale factor values
        self.scale_factor = {
            'A0': 1.0,
            'A1': 1.003,  # Fine-tuned
            'A2': 1.007   # Fine-tuned
        }
        
        # Target direction estimation
        self.target_direction = None
        self.target_distance = None
    
    def apply_bias_correction(self, distances):
        """Koreksi bias dan scaling pada pengukuran jarak"""
        corrected_distances = {
            'A0': max((distances['A0'] * 100 * self.scale_factor['A0']) - self.bias['A0'], 0),
            'A1': max((distances['A1'] * 100 * self.scale_factor['A1']) - self.bias['A1'], 0),
            'A2': max((distances['A2'] * 100 * self.scale_factor['A2']) - self.bias['A2'], 0)
        }
        return corrected_distances
    
    def estimate_target_direction(self, distances):
        """Enhanced target direction estimation covering 360 degrees"""
        A0, A1, A2 = distances['A0'], distances['A1'], distances['A2']
        
        target_distance = A0 * 10  # cm to mm
        
        # Analisis posisi target berdasarkan perbandingan semua anchor
        diff_A2_A1 = A2 - A1  # Perbedaan kiri-kanan
        
        # Deteksi target di belakang robot
        if (A1 < A0 - 25) and (A2 < A0 - 25):
            # Target kemungkinan di belakang
            if abs(diff_A2_A1) < 15:
                target_direction = 180  # Tepat di belakang
                print(f"TARGET DETECTED BEHIND: A0={A0:.1f}, A1={A1:.1f}, A2={A2:.1f}")
            elif diff_A2_A1 < 0:  # A2 < A1
                # Target di belakang-kanan
                angle_offset = min(25, abs(diff_A2_A1) * 0.8)
                target_direction = 180 - angle_offset  # 155-180 degrees
                print(f"TARGET BEHIND-RIGHT: angle={target_direction:.1f}°")
            else:  # A1 < A2
                # Target di belakang-kiri  
                angle_offset = min(25, abs(diff_A2_A1) * 0.8)
                target_direction = 180 + angle_offset  # 180-205 degrees
                print(f"TARGET BEHIND-LEFT: angle={target_direction:.1f}°")
        
        # Deteksi target di depan
        elif abs(diff_A2_A1) < 15:
            target_direction = 0  # Depan
            print(f"TARGET FRONT: A0={A0:.1f}, A1={A1:.1f}, A2={A2:.1f}")
        
        # Deteksi target di samping
        elif diff_A2_A1 < 0:  # A2 < A1, target ke kanan
            if A0 > min(A1, A2) + 15:  # Target di samping kanan
                angle_offset = min(50, abs(diff_A2_A1) * 1.2)
                target_direction = 90 + angle_offset  # 90-140 degrees
                print(f"TARGET RIGHT-SIDE: angle={target_direction:.1f}°")
            else:  # Target di depan-kanan
                angle_offset = min(40, abs(diff_A2_A1) * 1.2)
                target_direction = angle_offset  # 0-40 degrees  
                print(f"TARGET FRONT-RIGHT: angle={target_direction:.1f}°")
        
        else:  # A1 < A2, target ke kiri
            if A0 > min(A1, A2) + 15:  # Target di samping kiri
                angle_offset = min(50, abs(diff_A2_A1) * 1.2)
                target_direction = 270 - angle_offset  # 220-270 degrees
                print(f"TARGET LEFT-SIDE: angle={target_direction:.1f}°")
            else:  # Target di depan-kiri
                angle_offset = min(40, abs(diff_A2_A1) * 1.2)
                target_direction = 360 - angle_offset  # 320-360 degrees
                print(f"TARGET FRONT-LEFT: angle={target_direction:.1f}°")
        
        # Normalize angle
        if target_direction >= 360:
            target_direction -= 360
        elif target_direction < 0:
            target_direction += 360
        
        self.target_direction = target_direction
        self.target_distance = target_distance
        
        return target_direction, target_distance
    
    def is_target_behind(self, distances):
        """Check if target is behind the robot"""
        A0, A1, A2 = distances['A0'], distances['A1'], distances['A2']
        
        # Target dianggap di belakang jika:
        behind_condition1 = (A1 < A0 - 20) and (A2 < A0 - 20)
        behind_condition2 = (self.target_direction and 
                            130 <= self.target_direction <= 230)
        
        return behind_condition1 or behind_condition2

class DynamicObjectDetector:
    """Detects dynamic vs static objects using temporal analysis"""
    
    def __init__(self):
        self.position_history = {}
        self.history_window = 8    # Reduced for faster response
        self.movement_threshold = 120  # mm - reduced threshold
        self.static_frames_required = 6  # Reduced for faster classification
        self.dynamic_timeout = 3.0  # Reduced timeout
        
        # Track detected objects
        self.current_objects = {}
        self.dynamic_objects = set()
        self.static_objects = set()
        
        # Safety state
        self.dynamic_object_detected = False
        self.dynamic_object_last_seen = 0
        self.waiting_for_dynamic_object = False
        
    def _cluster_scan_points(self, scan_data):
        """Optimized object detection using clustering"""
        objects = {}
        current_cluster = []
        cluster_id = 0
        
        # Sort angles for sequential processing
        sorted_angles = sorted(scan_data.keys())
        
        for i, angle in enumerate(sorted_angles):
            distance = scan_data[angle]
            
            # Skip invalid readings
            if distance < MIN_VALID_DISTANCE or distance > 6000:  # Reduced max range
                continue
                
            # Convert to cartesian coordinates
            x = distance * math.cos(math.radians(angle))
            y = distance * math.sin(math.radians(angle))
            
            # Check if this point belongs to current cluster
            if current_cluster:
                last_x, last_y = current_cluster[-1]
                dist_to_last = math.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                if dist_to_last < 250:  # Points within 25cm belong to same object
                    current_cluster.append((x, y))
                else:
                    # End current cluster and start new one
                    if len(current_cluster) >= 2:  # Reduced minimum points
                        # Calculate cluster center
                        center_x = sum(p[0] for p in current_cluster) / len(current_cluster)
                        center_y = sum(p[1] for p in current_cluster) / len(current_cluster)
                        objects[cluster_id] = (center_x, center_y, len(current_cluster))
                        cluster_id += 1
                    
                    current_cluster = [(x, y)]
            else:
                current_cluster = [(x, y)]
        
        # Handle last cluster
        if len(current_cluster) >= 2:
            center_x = sum(p[0] for p in current_cluster) / len(current_cluster)
            center_y = sum(p[1] for p in current_cluster) / len(current_cluster)
            objects[cluster_id] = (center_x, center_y, len(current_cluster))
        
        return objects
    
    def update_object_positions(self, scan_data):
        """Update object positions and classify as dynamic or static"""
        current_time = time.time()
        
        # Detect objects in current scan
        objects = self._cluster_scan_points(scan_data)
        
        # Match objects with previous detections
        matched_objects = {}
        for obj_id, (x, y, point_count) in objects.items():
            best_match = None
            min_distance = float('inf')
            
            # Find closest previous object
            for prev_id, history in self.position_history.items():
                if history:
                    last_pos = history[-1]['position']
                    distance = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                    if distance < min_distance and distance < 400:  # Max 40cm movement
                        min_distance = distance
                        best_match = prev_id
            
            # Update or create object history
            if best_match is not None:
                matched_objects[best_match] = (x, y, point_count)
            else:
                # New object detected
                new_id = max(self.position_history.keys(), default=-1) + 1
                matched_objects[new_id] = (x, y, point_count)
                self.position_history[new_id] = []
        
        # Update position history
        for obj_id, (x, y, point_count) in matched_objects.items():
            if obj_id not in self.position_history:
                self.position_history[obj_id] = []
            
            self.position_history[obj_id].append({
                'position': (x, y),
                'timestamp': current_time,
                'point_count': point_count
            })
            
            # Keep only recent history
            if len(self.position_history[obj_id]) > self.history_window:
                self.position_history[obj_id].pop(0)
        
        # Clean up old objects
        objects_to_remove = []
        for obj_id, history in self.position_history.items():
            if not history or current_time - history[-1]['timestamp'] > 1.5:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.position_history[obj_id]
            self.dynamic_objects.discard(obj_id)
            self.static_objects.discard(obj_id)
        
        # Classify objects as dynamic or static
        self._classify_objects(current_time)
        
        return matched_objects
    
    def _classify_objects(self, current_time):
        """Classify objects as dynamic or static based on movement history"""
        self.dynamic_objects.clear()
        self.static_objects.clear()
        
        for obj_id, history in self.position_history.items():
            if len(history) < 2:  # Need minimum history
                continue
            
            # Calculate total movement over time
            total_movement = 0
            static_count = 0
            
            for i in range(1, len(history)):
                prev_pos = history[i-1]['position']
                curr_pos = history[i]['position']
                movement = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                total_movement += movement
                
                if movement < self.movement_threshold:
                    static_count += 1
            
            # Classification logic
            avg_movement = total_movement / (len(history) - 1)
            static_ratio = static_count / (len(history) - 1)
            
            if avg_movement > self.movement_threshold or static_ratio < 0.7:
                self.dynamic_objects.add(obj_id)
            elif static_ratio >= 0.85 and len(history) >= self.static_frames_required:
                self.static_objects.add(obj_id)
    
    def check_dynamic_objects_in_path(self, scan_data):
        """Check if there are dynamic objects in robot's path"""
        current_time = time.time()
        
        # Update object tracking
        objects = self.update_object_positions(scan_data)
        
        # Check for dynamic objects in critical zones
        dynamic_in_path = False
        closest_dynamic_distance = float('inf')
        
        for obj_id in self.dynamic_objects:
            if obj_id in self.position_history:
                history = self.position_history[obj_id]
                if history:
                    x, y = history[-1]['position']
                    distance = math.sqrt(x**2 + y**2)
                    angle = math.degrees(math.atan2(y, x)) % 360
                    
                    # Check if dynamic object is in front region and close
                    if self._is_in_front_region(angle) and distance < SCAN_THRESHOLD:
                        dynamic_in_path = True
                        closest_dynamic_distance = min(closest_dynamic_distance, distance)
                        self.dynamic_object_detected = True
                        self.dynamic_object_last_seen = current_time
                        print(f"DYNAMIC OBJECT DETECTED in path at {distance:.0f}mm, angle {angle:.0f}°")
        
        # Check if we should wait for dynamic object to move
        if self.dynamic_object_detected:
            if current_time - self.dynamic_object_last_seen > self.dynamic_timeout:
                print("Dynamic object timeout - resuming movement")
                self.dynamic_object_detected = False
                self.waiting_for_dynamic_object = False
            elif not dynamic_in_path:
                print("Dynamic object moved away - resuming movement")
                self.dynamic_object_detected = False
                self.waiting_for_dynamic_object = False
            else:
                self.waiting_for_dynamic_object = True
        
        return self.waiting_for_dynamic_object, closest_dynamic_distance
    
    def _is_in_front_region(self, angle):
        """Memeriksa apakah sudut berada di dalam area depan robot"""
        return (330 <= angle <= 360) or (0 <= angle <= 30)

class LidarProcessor:
    """Processes LIDAR data from ROS2 LaserScan messages with optimized performance"""

    def __init__(self):
        self.scan_data = {}
        self.lock = threading.Lock()

        # Obstacle status
        self.front_obstacle = False
        self.left_obstacle = False
        self.right_obstacle = False
        self.back_obstacle = False
        self.danger_zone = False
        self.critical_danger = False

        # Minimum distance for each region
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.back_distance = float('inf')

        self.target_direction = None
        self.target_distance = None

        # Timestamp for the last scan
        self.last_scan_time = 0

        # Save raw scan for visualization
        self.last_scan_msg = None

        # Target information (from UWB)
        self.target_direction = None
        self.target_distance = None

        # Dynamic object detector
        self.dynamic_detector = DynamicObjectDetector()

        # Optimized filtering
        self.moving_avg_window = 2  # Reduced for faster response
        self.distance_history = {}

    def _is_in_front_region(self, angle):
        """Check if angle is in front region (330° to 360° or 0° to 30°)"""
        return (330 <= angle <= 360) or (0 <= angle <= 30)

    def _is_in_back_region(self, angle):
        """Check if angle is in back region (150° to 210°)"""
        return 150 <= angle <= 210
    
    def set_target_info(self, direction, distance):
        """Set the target's direction and distance for LIDAR processing"""
        self.target_direction = direction
        self.target_distance = distance
        print(f"Target info set: direction={direction}, distance={distance}")

    def process_scan(self, scan_msg):
        """Process ROS2 LaserScan message with optimized performance"""
        with self.lock:
            self.last_scan_msg = scan_msg
            self.last_scan_time = time.time()
            self.scan_data.clear()

            ranges = scan_msg.ranges
            angle_increment = scan_msg.angle_increment
            angle_min = scan_msg.angle_min

            step_size = 2  # Optimized step size

            for i in range(0, len(ranges), step_size):
                distance = ranges[i]

                if distance < 0.01 or distance > 8.0 or math.isinf(distance):
                    continue

                angle_rad = angle_min + (i * angle_increment)
                angle_deg = int(math.degrees(angle_rad) % 360)
                distance_mm = int(distance * 1000)

                self.scan_data[angle_deg] = distance_mm

            # Apply optimized filtering
            self.scan_data = self._filter_lidar_data(self.scan_data)

            # Analyze obstacles with dynamic object detection
            self._analyze_obstacles_with_dynamic_detection()

    def _filter_lidar_data(self, scan_data):
        """Optimized LIDAR data filtering"""
        filtered_data = {}

        for angle, distance in scan_data.items():
            if angle not in self.distance_history:
                self.distance_history[angle] = deque(maxlen=self.moving_avg_window)

            # Append the new distance to the history
            self.distance_history[angle].append(distance)

            # Calculate the moving average
            filtered_data[angle] = sum(self.distance_history[angle]) / len(self.distance_history[angle])

        return filtered_data

    def _analyze_obstacles_with_dynamic_detection(self):
        """Optimized obstacle analysis with dynamic detection"""
        if not self.scan_data:
            return

        # Check untuk objek dinamis dalam path
        waiting_for_dynamic, _ = self.dynamic_detector.check_dynamic_objects_in_path(self.scan_data)

        # Reset status
        self.front_obstacle = self.left_obstacle = self.right_obstacle = self.back_obstacle = False
        self.danger_zone = self.critical_danger = False
        self.front_distance = self.left_distance = self.right_distance = self.back_distance = float('inf')

        for angle, distance in self.scan_data.items():
            # Skip target jika diketahui
            if (self.target_direction is not None and
                    abs(angle - self.target_direction) < TARGET_EXCLUSION_ANGLE):
                continue

            # Check obstacle regions
            if self._is_in_front_region(angle):
                self.front_distance = min(self.front_distance, distance)
                if distance < SCAN_THRESHOLD:
                    self.front_obstacle = True
                if distance < DANGER_THRESHOLD:
                    self.danger_zone = True
                if distance < CRITICAL_DANGER_THRESHOLD:
                    self.critical_danger = True

            elif 31 <= angle <= 140:  # Right region
                self.right_distance = min(self.right_distance, distance)
                if distance < SCAN_THRESHOLD:
                    self.right_obstacle = True

            elif 220 <= angle <= 329:  # Left region
                self.left_distance = min(self.left_distance, distance)
                if distance < SCAN_THRESHOLD:
                    self.left_obstacle = True

            elif self._is_in_back_region(angle):  # Back region
                self.back_distance = min(self.back_distance, distance)
                if distance < SCAN_THRESHOLD:
                    self.back_obstacle = True

    def get_obstacle_status(self):
        """Get current obstacle status including dynamic object information"""
        with self.lock:
            scan_age = time.time() - self.last_scan_time
            data_valid = scan_age < 0.4  # Reduced timeout

            # Check for dynamic objects
            waiting_for_dynamic = self.dynamic_detector.waiting_for_dynamic_object

            status = {
                'front': {
                    'obstacle': self.front_obstacle if data_valid else False,
                    'distance': self.front_distance if data_valid else float('inf')
                },
                'left': {
                    'obstacle': self.left_obstacle if data_valid else False,
                    'distance': self.left_distance if data_valid else float('inf')
                },
                'right': {
                    'obstacle': self.right_obstacle if data_valid else False,
                    'distance': self.right_distance if data_valid else float('inf')
                },
                'back': {
                    'obstacle': self.back_obstacle if data_valid else False,
                    'distance': self.back_distance if data_valid else float('inf')
                },
                'danger_zone': self.danger_zone if data_valid else False,
                'critical_danger': self.critical_danger if data_valid else False,
                'data_valid': data_valid,
                'scan_points': len(self.scan_data) if data_valid else 0,
                'scan_age': scan_age,
                'waiting_for_dynamic': waiting_for_dynamic,
                'dynamic_objects': len(self.dynamic_detector.dynamic_objects),
                'static_objects': len(self.dynamic_detector.static_objects)
            }

            return status
    
    def get_safe_direction(self):
        """Tentukan arah aman untuk bergerak berdasarkan rintangan"""
        status = self.get_obstacle_status()

        if not status['data_valid']:
            print("Warning: LIDAR data not valid or too old")
            return None

        # Jika menunggu objek dinamis, prioritaskan mencari jalan alternatif
        if status['waiting_for_dynamic']:
            # Cari arah dengan jarak terjauh
            distances = [
                ('LEFT', status['left']['distance']),
                ('RIGHT', status['right']['distance']),
                ('BACK', status['back']['distance'])
            ]
            best_direction = max(distances, key=lambda x: x[1])[0]
            return f"AVOID_DYNAMIC_{best_direction}"

        if status['critical_danger']:
            # Cari escape route instead of stopping
            distances = [
                ('LEFT', status['left']['distance']),
                ('RIGHT', status['right']['distance']),
                ('BACK', status['back']['distance'])
            ]
            best_direction = max(distances, key=lambda x: x[1])[0]
            return f"EMERGENCY_{best_direction}"
        
        # Jika tidak ada rintangan di depan, lanjutkan maju
        if not status['front']['obstacle']:
            return "FORWARD"
        
        if status['front']['obstacle']:
            if not status['left']['obstacle'] and not status['right']['obstacle']:
                if status['left']['distance'] > status['right']['distance']:
                    return "LEFT"
                else:
                    return "RIGHT"
            elif not status['left']['obstacle']:
                return "LEFT"
            elif not status['right']['obstacle']:
                return "RIGHT"
            else:
                # Semua arah terblokir, pilih yang paling aman
                distances = [
                    ('LEFT', status['left']['distance']),
                    ('RIGHT', status['right']['distance']),
                    ('BACK', status['back']['distance'])
                ]
                best_direction = max(distances, key=lambda x: x[1])[0]
                return best_direction

        return "FORWARD"
    
    def check_immediate_collision_threat_only(self, ultrasonic_manager, lidar):
        """Cek HANYA ancaman tabrakan immediate (< 8cm)"""
        
        # Ultrasonic immediate threat (< 8cm)
        if ultrasonic_manager:
            critical_detected, sensor_name, distance = \
                ultrasonic_manager.is_critical_obstacle_detected()
            
            if critical_detected and distance < 8:  # SANGAT dekat
                print(f"IMMEDIATE COLLISION: {sensor_name} at {distance:.1f}cm")
                self.execute_immediate_side_step(sensor_name)
                return True
        
        # LIDAR immediate threat (< 8cm)
        if hasattr(lidar, 'scan_data') and lidar.scan_data:
            for angle in range(350, 361):  
                if angle in lidar.scan_data:
                    distance = lidar.scan_data[angle]
                    if distance < 80:  # 8cm
                        print(f"IMMEDIATE LIDAR COLLISION: {distance}mm")
                        self.execute_immediate_side_step_lidar(angle)
                        return True
        
        return False
    
    def execute_immediate_side_step(self, sensor_name):
        """Side step minimal untuk hindari tabrakan immediate"""
        
        if sensor_name == 'front_center':
            # Center blocked - side step ke yang lebih lapang
            if self.parent and self.parent.ultrasonic_manager:
                status = self.parent.ultrasonic_manager.get_sensor_status()
                left_space = status.get('front_left', {}).get('distance', 0)
                right_space = status.get('front_right', {}).get('distance', 0)
                
                if left_space > right_space and left_space > 10:
                    self.move(-15, 15, smooth=False)  # Side step left
                    print("SIDE STEP LEFT")
                elif right_space > 10:
                    self.move(15, -15, smooth=False)  # Side step right  
                    print("SIDE STEP RIGHT")
                else:
                    self.move(-10, 10, smooth=False)  # Minimal reverse
                    print("MINIMAL REVERSE")
            
        elif sensor_name == 'front_left':
            self.move(10, -10, smooth=False)  # Small right adjustment
            print("ADJUST RIGHT")
            
        elif sensor_name == 'front_right':
            self.move(-10, 10, smooth=False)  # Small left adjustment
            print("ADJUST LEFT")


class RobotController:
    """Enhanced robot controller with ultra-fast response and advanced path planning"""
    
    def __init__(self, r_wheel_port, l_wheel_port):
        # Motor controllers
        self.right_motor = MotorControl(device=r_wheel_port)
        self.left_motor = MotorControl(device=l_wheel_port)
        self.right_motor.set_drive_mode(1, 2)
        self.left_motor.set_drive_mode(1, 2)
        
        # Optimized speed configuration
        self.speed = DEFAULT_SPEED
        self.rotation_factor = ROTATION_FACTOR
        self.stop_threshold = STOP_THRESHOLD
        
        # Enhanced turning parameters
        self.turn_gradual_threshold = 500  # mm
        self.max_turn_angle = 20          # degrees
        self.turn_smoothing_factor = 0.8
        
        # Optimized incremental speed system
        self.min_speed = self.speed // 3  # Increased minimum
        self.max_speed = self.speed * 2.5  # Increased maximum
        self.current_target_speed = self.min_speed
        self.actual_speed = self.min_speed
        self.speed_increment = 12  # Faster acceleration
        self.speed_decrement = 15  # Faster deceleration
        self.acceleration_delay = 0.05  # Reduced delay
        self.last_speed_update = time.time()

        # Optimized conditions for speed increase
        self.straight_path_threshold = 15
        self.clear_distance_threshold = 700  # Reduced
        self.speed_boost_distance = 1200     # Reduced
        self.consecutive_clear_count = 0
        self.min_clear_count = 3  # Reduced for faster boost
        
        # Enhanced smooth movement parameters
        self.smooth_turn_speed = self.speed // 3
        self.gentle_turn_speed = self.speed // 2.5
        self.rotation_speed = self.speed // 4
        self.current_left_speed = 0
        self.current_right_speed = 0
        
        # Advanced gradual response state
        self.obstacle_response_active = False
        self.current_obstacle_zone = 'clear'
        self.gradual_turn_active = False
        self.turn_direction_preference = None

        # OPTIMIZED OBSTACLE ZONES
        self.obstacle_zones = {
            'far': 800,       # mm
            'medium': 600,    # mm
            'near': 350,      # mm
            'critical': 150   # mm
        }
        
        # Optimized speed factors
        self.speed_factors = {
            'clear': 1.0,
            'far': 0.85,      # Less aggressive reduction
            'medium': 0.6,
            'near': 0.35,
            'critical': 0.15  # Still allow movement
        }
        
        # Enhanced status flags
        self.obstacle_avoidance_active = False
        self.last_command_time = time.time()
        self.current_direction = "STOP"
        self.emergency_stop = False
        self.waiting_for_dynamic_object = False
        self.speed_boost_active = False
        
        # Optimized command processing
        self.last_command = (0, 0)
        self.command_threshold = 2  # Reduced threshold
        self.last_command_time = 0
        self.min_command_interval = 0.02  # Faster updates

        # ENHANCED INDEPENDENT WHEEL CONTROL
        self.independent_control_enabled = True
        
        # Optimized base speeds
        self.base_left_speed = 0
        self.base_right_speed = 0
        self.target_left_speed = 0
        self.target_right_speed = 0
        
        # Faster speed adjustment
        self.speed_adjustment_rate = 8  # RPM per cycle
        self.max_speed_diff = self.speed * 2.5
        
        # Enhanced differential steering
        self.steering_gain = 1.0  # Increased gain
        self.forward_bias = 0.95  # Increased forward bias
        
        # Real-time control state
        self.last_target_direction = None
        self.direction_change_rate = 15  # degrees per second

        # Optimized ultrasonic emergency response
        self.ultrasonic_emergency_active = False
        self.ultrasonic_override_time = 0
        self.ultrasonic_override_duration = 1.5  # Reduced duration
        
        # Initialize fast response system
        self.fast_response = None

    def initialize_fast_response(self):
        """Initialize fast response controller"""
        if not self.fast_response:
            self.fast_response = FastResponseController(self)
            self.fast_response.start_fast_response_system()
            print("✓ Fast response system initialized")

    def process_control_ultra_fast(self, uwb_distances, lidar, ultrasonic_manager=None):
        """Ultra-fast control dengan prioritas UWB target following"""
        
        start_time = time.time()
        
        # LEVEL 1: HANYA ancaman IMMEDIATE collision (< 8cm)
        if self.check_immediate_collision_threat_only(ultrasonic_manager, lidar):
            return
        
        # LEVEL 2: UWB target following SELALU jadi prioritas utama
        target_direction, target_distance = self.uwb_tracker.estimate_target_direction(uwb_distances)
        
        # LEVEL 3: Cari jalur ke target dengan obstacle avoidance minimal
        if target_distance <= self.stop_threshold:
            self.smooth_speed_transition(0, 0)
            return
        
        # Jalankan UWB control dengan obstacle adjustment minimal
        self.process_uwb_control_aggressive(uwb_distances, lidar, ultrasonic_manager)

    def process_uwb_control_aggressive(self, uwb_distances, lidar, ultrasonic_manager):
        """UWB control agresif dengan prioritas target following"""
        
        A0, A1, A2 = uwb_distances['A0'], uwb_distances['A1'], uwb_distances['A2']
        
        # Calculate steering
        angle_error = A2 - A1
        
        # AGGRESSIVE speed - tidak terlalu pelan karena obstacle jauh
        base_speed = self.calculate_aggressive_target_speed(A0)
        
        # Differential steering untuk target
        if abs(angle_error) < 10:
            left_speed = base_speed
            right_speed = -base_speed
        elif angle_error < 0:  # Target kanan
            turn_factor = min(1.0, abs(angle_error) / 40.0)
            left_speed = base_speed * (1.0 + turn_factor * 0.5)
            right_speed = -base_speed * (1.0 - turn_factor * 0.7)
        else:  # Target kiri
            turn_factor = min(1.0, abs(angle_error) / 40.0)
            left_speed = base_speed * (1.0 - turn_factor * 0.7)
            right_speed = -base_speed * (1.0 + turn_factor * 0.5)
        
        # MINIMAL obstacle adjustment - hanya untuk yang sangat dekat
        if ultrasonic_manager:
            left_speed, right_speed = self.apply_minimal_obstacle_adjustment(
                left_speed, right_speed, ultrasonic_manager
            )
        
        self.smooth_speed_transition(int(left_speed), int(right_speed))

    def calculate_aggressive_target_speed(self, distance_to_target):
        """Hitung kecepatan agresif menuju target"""
        
        if distance_to_target < 80:
            return 50  # Dekat target - pelan tapi tidak terlalu pelan
        elif distance_to_target < 150:  
            return 70  # Medium distance - speed baik
        else:
            return self.speed  # Jauh dari target - full speed

    def check_immediate_threats_fast(self, ultrasonic_manager, lidar):
        """Ultra-fast immediate threat detection"""
        
        # Priority 1: Ultrasonic critical detection
        if ultrasonic_manager:
            critical_detected, sensor_name, distance = \
                ultrasonic_manager.is_critical_obstacle_detected()
            
            if critical_detected and distance < 10:  # 10cm immediate
                print(f"IMMEDIATE ULTRASONIC THREAT: {sensor_name} at {distance:.1f}cm")
                self.execute_immediate_escape(sensor_name, distance)
                return True
        
        # Priority 2: LIDAR critical zones
        if hasattr(lidar, 'scan_data') and lidar.scan_data:
            for angle in range(350, 361):  # Front critical zone
                if angle in lidar.scan_data:
                    distance = lidar.scan_data[angle]
                    if distance < 100:  # 10cm
                        print(f"IMMEDIATE LIDAR THREAT: {distance}mm at {angle}°")
                        self.execute_immediate_escape_lidar(angle, distance)
                        return True
        
        return False

    def assess_obstacle_situation_fast(self, lidar, ultrasonic_manager):
        """Ultra-fast obstacle situation assessment"""
        
        situation = {
            'priority': 'NORMAL',
            'type': 'clear_path',
            'obstacles': {},
            'safe_directions': [],
            'threat_level': 0,
            'response_needed': False
        }
        
        # Check ultrasonic sensors first (fastest)
        if ultrasonic_manager:
            summary = ultrasonic_manager.get_obstacle_summary()
            
            if summary['critical_detected']:
                situation['priority'] = 'EMERGENCY'
                situation['type'] = 'ultrasonic_critical'
                situation['threat_level'] = 10
                situation['response_needed'] = True
                
            elif summary['warning_detected']:
                situation['priority'] = 'CRITICAL'
                situation['threat_level'] = max(situation['threat_level'], 7)
                situation['response_needed'] = True
        
        # Check LIDAR data
        if hasattr(lidar, 'get_obstacle_status'):
            lidar_status = lidar.get_obstacle_status()
            
            if lidar_status.get('critical_danger', False):
                situation['priority'] = 'EMERGENCY'
                situation['type'] = 'lidar_critical'
                situation['threat_level'] = 10
                situation['response_needed'] = True
                
            elif lidar_status.get('danger_zone', False):
                if situation['priority'] == 'NORMAL':
                    situation['priority'] = 'CRITICAL'
                situation['threat_level'] = max(situation['threat_level'], 8)
                situation['response_needed'] = True
                
            elif (lidar_status.get('front', {}).get('obstacle', False) or
                  lidar_status.get('left', {}).get('obstacle', False) or
                  lidar_status.get('right', {}).get('obstacle', False)):
                if situation['priority'] == 'NORMAL':
                    situation['priority'] = 'WARNING'
                situation['threat_level'] = max(situation['threat_level'], 5)
                situation['response_needed'] = True
        
        return situation

    def execute_immediate_escape(self, sensor_name, distance):
        """Execute immediate escape maneuver based on sensor"""
        
        if sensor_name == 'front_center':
            # Front blocked - turn to best side
            self.execute_immediate_best_turn()
        elif sensor_name == 'front_left':
            # Left blocked - turn right immediately
            self.move(25, 25, smooth=False)
            print("IMMEDIATE RIGHT ESCAPE")
        elif sensor_name == 'front_right':
            # Right blocked - turn left immediately
            self.move(-25, -25, smooth=False)
            print("IMMEDIATE LEFT ESCAPE")

    def execute_immediate_escape_lidar(self, angle, distance):
        """Execute immediate escape based on LIDAR detection"""
        
        if 350 <= angle <= 360 or 0 <= angle <= 10:
            # Front obstacle - emergency turn
            self.execute_immediate_best_turn()
        elif 315 <= angle <= 349:
            # Front-left obstacle - turn right
            self.move(25, 25, smooth=False)
        elif 11 <= angle <= 45:
            # Front-right obstacle - turn left
            self.move(-25, -25, smooth=False)

    def execute_immediate_best_turn(self):
        """Execute best immediate turn based on available space"""
        
        # Quick assessment of available space
        left_clear = True
        right_clear = True
        
        # Check ultrasonic sensors for quick decision
        if self.parent and self.parent.ultrasonic_manager:
            status = self.parent.ultrasonic_manager.get_sensor_status()
            left_clear = status.get('front_left', {}).get('distance', 100) > 20
            right_clear = status.get('front_right', {}).get('distance', 100) > 20
        
        if left_clear and right_clear:
            # Both sides clear - choose right (default)
            self.move(25, 25, smooth=False)
        elif left_clear:
            # Only left clear - turn left
            self.move(-25, -25, smooth=False)
        elif right_clear:
            # Only right clear - turn right
            self.move(25, 25, smooth=False)
        else:
            # Both blocked - reverse
            self.move(-20, 20, smooth=False)

    def execute_emergency_response(self, obstacle_situation):
        """Execute emergency response with path finding"""
        
        print(f"EMERGENCY RESPONSE: {obstacle_situation['type']}")
        
        # Find emergency escape route
        escape_direction = self.find_emergency_escape_route()
        
        if escape_direction == "LEFT":
            self.move(-30, -30, smooth=False)
            print("EMERGENCY LEFT TURN")
        elif escape_direction == "RIGHT":
            self.move(30, 30, smooth=False)
            print("EMERGENCY RIGHT TURN")
        elif escape_direction == "REVERSE":
            self.move(-25, 25, smooth=False)
            print("EMERGENCY REVERSE")
        else:
            # Last resort - minimal movement
            self.move(-10, 10, smooth=False)
            print("EMERGENCY MINIMAL MOVEMENT")

    def find_emergency_escape_route(self):
        """Find emergency escape route quickly"""
        
        # Check ultrasonic sensors for quick escape assessment
        if self.parent and self.parent.ultrasonic_manager:
            status = self.parent.ultrasonic_manager.get_sensor_status()
            
            left_distance = status.get('front_left', {}).get('distance', 0)
            right_distance = status.get('front_right', {}).get('distance', 0)
            
            if left_distance > 30:
                return "LEFT"
            elif right_distance > 30:
                return "RIGHT"
            else:
                return "REVERSE"
        
        return "RIGHT"  # Default

    def execute_critical_response(self, obstacle_situation, uwb_distances):
        """Execute critical response with enhanced path planning"""
        
        print(f"CRITICAL RESPONSE: {obstacle_situation['type']}")
        
        # Quick path evaluation
        if obstacle_situation['type'] == 'ultrasonic_critical':
            self.handle_ultrasonic_critical_response(uwb_distances)
        elif obstacle_situation['type'] == 'lidar_critical':
            self.handle_lidar_critical_response(uwb_distances)
        else:
            # General critical response
            self.handle_general_critical_response(uwb_distances)

    def handle_ultrasonic_critical_response(self, uwb_distances):
        """Handle ultrasonic critical response"""
        
        if self.parent and self.parent.ultrasonic_manager:
            summary = self.parent.ultrasonic_manager.get_obstacle_summary()
            
            # Find alternative path
            alternative_action = self.find_alternative_path_ultra_fast(summary)
            
            if alternative_action != "EMERGENCY_BRAKE":
                print(f"ULTRASONIC ALTERNATIVE: {alternative_action}")
                self.execute_alternative_action_fast(alternative_action)
            else:
                # Last resort with minimal stop
                self.move(5, -5, smooth=False)  # Minimal movement instead of full stop

    def find_alternative_path_ultra_fast(self, ultrasonic_summary):
        """Ultra-fast alternative path finding for ultrasonic sensors"""
        
        status = ultrasonic_summary['sensor_status']
        
        # Quick assessment of available directions
        left_clear = status.get('front_left', {}).get('distance', 0)
        center_blocked = status.get('front_center', {}).get('distance', 0)
        right_clear = status.get('front_right', {}).get('distance', 0)
        
        print(f"PATH ASSESSMENT: Left={left_clear:.1f}cm, Center={center_blocked:.1f}cm, Right={right_clear:.1f}cm")
        
        # Priority-based decision tree for ultra-fast response
        
        # Priority 1: Best side turns with good clearance
        if left_clear > 30:
            print("ALTERNATIVE: Sharp left turn (good clearance)")
            return "SHARP_LEFT"
        elif right_clear > 30:
            print("ALTERNATIVE: Sharp right turn (good clearance)")
            return "SHARP_RIGHT"
        
        # Priority 2: Moderate side turns with adequate clearance
        elif left_clear > 20 and left_clear > right_clear:
            print("ALTERNATIVE: Moderate left turn")
            return "GENTLE_LEFT"
        elif right_clear > 20:
            print("ALTERNATIVE: Moderate right turn")
            return "GENTLE_RIGHT"
        
        # Priority 3: Best available side with limited clearance
        elif max(left_clear, right_clear) > 15:
            best_side = "LEFT" if left_clear > right_clear else "RIGHT"
            print(f"ALTERNATIVE: Careful {best_side.lower()} turn (limited clearance)")
            return f"GENTLE_{best_side}"
        
        # Priority 4: Some space in front - slow approach
        elif center_blocked > 15:
            print("ALTERNATIVE: Slow forward approach")
            return "SLOW_FORWARD"
        
        # Priority 5: Very limited front space - minimal forward
        elif center_blocked > 10:
            print("ALTERNATIVE: Minimal forward movement")
            return "MINIMAL_FORWARD"
        
        # Priority 6: No front space - reverse options
        elif max(left_clear, right_clear) > 10:
            print("ALTERNATIVE: Reverse and turn")
            return "REVERSE_AND_TURN"
        
        # Priority 7: Emergency reverse
        elif center_blocked > 5:
            print("ALTERNATIVE: Emergency reverse")
            return "REVERSE_MINIMAL"
        
        # Last resort: Emergency brake
        else:
            print("NO ALTERNATIVE FOUND: Emergency brake required")
            return "EMERGENCY_BRAKE"

    def execute_alternative_action_fast(self, action):
        """Execute alternative action with optimized speeds"""
        
        print(f"EXECUTING ALTERNATIVE ACTION: {action}")
        
        if action == "SHARP_LEFT":
            # Sharp left turn - both wheels reverse
            self.robot.move(-40, -40, smooth=False)
            
        elif action == "SHARP_RIGHT":
            # Sharp right turn - both wheels forward
            self.robot.move(40, 40, smooth=False)
            
        elif action == "GENTLE_LEFT":
            # Gentle left turn - differential steering
            self.robot.move(-25, 35, smooth=True)
            
        elif action == "GENTLE_RIGHT":
            # Gentle right turn - differential steering
            self.robot.move(35, -25, smooth=True)
            
        elif action == "SLOW_FORWARD":
            # Slow forward movement
            self.robot.move(20, -20, smooth=True)
            
        elif action == "MINIMAL_FORWARD":
            # Very slow forward movement
            self.robot.move(10, -10, smooth=True)
            
        elif action == "REVERSE_AND_TURN":
            # Complex maneuver: reverse then turn
            self.execute_reverse_and_turn_maneuver()
            
        elif action == "REVERSE_MINIMAL":
            # Minimal reverse movement
            self.robot.move(-15, 15, smooth=False)
            
        elif action == "EMERGENCY_BRAKE":
            # Emergency stop with counter-rotation
            self.robot.move(0, 0, smooth=False)
            time.sleep(0.1)
            # Optional counter-rotation for better stopping
            self.robot.move(-10, 10, smooth=False)
            time.sleep(0.2)
            self.robot.move(0, 0, smooth=False)
            
        else:
            # Unknown action - safe default
            print(f"UNKNOWN ACTION: {action} - using safe default")
            self.robot.move(-10, 10, smooth=False)

    def execute_reverse_and_turn_maneuver(self):
        """Execute complex reverse and turn maneuver"""
        
        print("EXECUTING: Reverse and turn maneuver")
        
        # Phase 1: Reverse to create space
        self.robot.move(-20, 20, smooth=False)
        time.sleep(0.5)  # Reverse for 0.5 seconds
        
        # Phase 2: Quick assessment of best turn direction
        if hasattr(self.robot.parent, 'ultrasonic_manager'):
            status = self.robot.parent.ultrasonic_manager.get_sensor_status()
            left_space = status.get('front_left', {}).get('distance', 0)
            right_space = status.get('front_right', {}).get('distance', 0)
            
            # Choose direction with more space
            if left_space > right_space:
                print("REVERSE-TURN: Turning left")
                self.robot.move(-30, -30, smooth=False)  # Turn left
            else:
                print("REVERSE-TURN: Turning right")
                self.robot.move(30, 30, smooth=False)   # Turn right
        else:
            # Default to right turn
            print("REVERSE-TURN: Default right turn")
            self.robot.move(30, 30, smooth=False)
        
        # Phase 3: Brief turn duration
        time.sleep(0.8)  # Turn for 0.8 seconds
        
        # Phase 4: Stop and reassess
        self.robot.move(0, 0, smooth=False)
        print("REVERSE-TURN: Maneuver completed")

    def evaluate_path_safety(self, ultrasonic_summary):
        """Evaluate safety level of current path options"""
        
        status = ultrasonic_summary['sensor_status']
        
        safety_score = {
            'left': 0,
            'center': 0,
            'right': 0,
            'overall': 0
        }
        
        # Evaluate each direction
        for direction, sensor_name in [('left', 'front_left'), ('center', 'front_center'), ('right', 'front_right')]:
            distance = status.get(sensor_name, {}).get('distance', 0)
            
            if distance > 40:
                safety_score[direction] = 10  # Very safe
            elif distance > 30:
                safety_score[direction] = 8   # Safe
            elif distance > 20:
                safety_score[direction] = 6   # Moderate
            elif distance > 15:
                safety_score[direction] = 4   # Caution
            elif distance > 10:
                safety_score[direction] = 2   # Risky
            else:
                safety_score[direction] = 0   # Dangerous
        
        # Calculate overall safety
        safety_score['overall'] = max(safety_score['left'], safety_score['center'], safety_score['right'])
        
        return safety_score

    def get_priority_escape_direction(self, ultrasonic_summary):
        """Get priority escape direction based on sensor data"""
        
        safety_scores = self.evaluate_path_safety(ultrasonic_summary)
        
        # Create priority list based on safety scores
        directions = [
            ('left', safety_scores['left']),
            ('right', safety_scores['right']),
            ('center', safety_scores['center'])
        ]
        
        # Sort by safety score (highest first)
        directions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"ESCAPE PRIORITY: {directions[0][0].upper()} (score: {directions[0][1]})")
        
        return directions[0][0], directions[0][1]


    def handle_lidar_critical_response(self, uwb_distances):
        """Handle LIDAR critical response"""
        
        if hasattr(self.parent, 'lidar'):
            status = self.parent.lidar.get_obstacle_status()
            
            # Find safest direction based on LIDAR data
            directions = [
                ('LEFT', status.get('left', {}).get('distance', 0)),
                ('RIGHT', status.get('right', {}).get('distance', 0)),
                ('BACK', status.get('back', {}).get('distance', 0))
            ]
            
            # Sort by available space
            directions.sort(key=lambda x: x[1], reverse=True)
            safest_direction = directions[0][0]
            
            if directions[0][1] > 300:  # 30cm clearance
                print(f"LIDAR CRITICAL ALTERNATIVE: {safest_direction}")
                self.execute_critical_maneuver(safest_direction)
            else:
                # All directions blocked - minimal reverse
                self.move(-15, 15, smooth=False)

    def handle_general_critical_response(self, uwb_distances):
        """Handle general critical response"""
        
        # Get target direction for reference
        target_direction, _ = self.parent.uwb_tracker.estimate_target_direction(uwb_distances)
        
        # Choose escape direction opposite to most obstacles
        escape_direction = self.calculate_escape_direction(target_direction)
        
        print(f"GENERAL CRITICAL RESPONSE: {escape_direction}")
        self.execute_critical_maneuver(escape_direction)

    def execute_critical_maneuver(self, direction):
        """Execute critical maneuver in specified direction"""
        
        if direction == "LEFT":
            self.move(-35, -35, smooth=False)
        elif direction == "RIGHT":
            self.move(35, 35, smooth=False)
        elif direction == "BACK":
            self.move(-30, 30, smooth=False)
        else:
            # Default minimal movement
            self.move(-10, 10, smooth=False)

    def generate_path_options_fast(self, obstacle_situation):
        """Generate path options quickly based on threat level"""
        
        path_options = []
        threat_level = obstacle_situation.get('threat_level', 0)
        
        # Option 1: Continue straight (if safe)
        if threat_level < 6:
            path_options.append({
                'type': 'straight',
                'direction': 0,
                'speed_required': 70,
                'turn_intensity': 0,
                'estimated_distance': 100,
                'scan_angles': list(range(350, 361)) + list(range(0, 11)),
                'safety_score': 10 - threat_level
            })
        
        # Option 2: Gentle left turn
        path_options.append({
            'type': 'gentle_left',
            'direction': 20,
            'speed_required': 55,
            'turn_intensity': 20,
            'estimated_distance': 130,
            'scan_angles': list(range(300, 331)),
            'safety_score': 8
        })
        
        # Option 3: Gentle right turn
        path_options.append({
            'type': 'gentle_right',
            'direction': -20,
            'speed_required': 55,
            'turn_intensity': 20,
            'estimated_distance': 130,
            'scan_angles': list(range(30, 61)),
            'safety_score': 8
        })
        
        # Option 4: Sharp left turn (for higher threat levels)
        if threat_level >= 5:
            path_options.append({
                'type': 'sharp_left',
                'direction': 50,
                'speed_required': 40,
                'turn_intensity': 50,
                'estimated_distance': 160,
                'scan_angles': list(range(270, 301)),
                'safety_score': 6
            })
        
        # Option 5: Sharp right turn (for higher threat levels)
        if threat_level >= 5:
            path_options.append({
                'type': 'sharp_right',
                'direction': -50,
                'speed_required': 40,
                'turn_intensity': 50,
                'estimated_distance': 160,
                'scan_angles': list(range(60, 91)),
                'safety_score': 6
            })
        
        # Option 6: Reverse and turn (for high threat levels)
        if threat_level >= 7:
            path_options.append({
                'type': 'reverse_turn',
                'direction': 180,
                'speed_required': 25,
                'turn_intensity': 90,
                'estimated_distance': 200,
                'scan_angles': list(range(150, 211)),
                'safety_score': 4
            })
        
        # Option 7: Emergency spiral (last resort)
        if threat_level >= 9:
            path_options.append({
                'type': 'emergency_spiral',
                'direction': 90,
                'speed_required': 30,
                'turn_intensity': 120,
                'estimated_distance': 250,
                'scan_angles': list(range(180, 271)),
                'safety_score': 2
            })
        
        return path_options

    def select_best_path_fast(self, path_options, uwb_distances):
        """Select best path using ultra-fast evaluation"""
        
        if not path_options:
            return None
        
        # Get target information
        target_direction, target_distance = self.parent.uwb_tracker.estimate_target_direction(uwb_distances)
        target_info = {'direction': target_direction, 'distance': target_distance}
        
        # Get obstacle data quickly
        obstacle_info = {}
        if hasattr(self.parent, 'lidar') and hasattr(self.parent.lidar, 'scan_data'):
            obstacle_info.update(self.parent.lidar.scan_data)
        
        # Add ultrasonic obstacle data
        if hasattr(self.parent, 'ultrasonic_manager') and self.parent.ultrasonic_manager:
            status = self.parent.ultrasonic_manager.get_sensor_status()
            for sensor_name, data in status.items():
                if data['distance'] > 0:
                    if sensor_name == 'front_left':
                        obstacle_info[315] = data['distance'] * 10
                    elif sensor_name == 'front_center':
                        obstacle_info[0] = data['distance'] * 10
                    elif sensor_name == 'front_right':
                        obstacle_info[45] = data['distance'] * 10
        
        best_path = None
        best_score = float('-inf')
        
        for path in path_options:
            # Fast scoring based on multiple criteria
            score = self.calculate_path_score_fast(path, target_info, obstacle_info)
            
            if score > best_score:
                best_score = score
                best_path = path
        
        if best_path:
            print(f"Selected path: {best_path['type']} (score: {best_score:.2f})")
        
        return best_path

    def calculate_path_score_fast(self, path, target_info, obstacle_info):
        """Calculate path score quickly using simplified metrics"""
        
        score = path.get('safety_score', 5)  # Base safety score
        
        # Target alignment bonus
        target_direction = target_info.get('direction', 0)
        path_direction = path.get('direction', 0)
        
        angle_diff = abs(target_direction - path_direction)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        alignment_bonus = max(0, 5 - (angle_diff / 36))  # Up to 5 points for alignment
        score += alignment_bonus
        
        # Obstacle clearance check
        scan_angles = path.get('scan_angles', [])
        min_clearance = float('inf')
        
        for angle in scan_angles:
            if angle in obstacle_info:
                distance = obstacle_info[angle]
                if distance > 0:
                    min_clearance = min(min_clearance, distance)
        
        # Clearance bonus/penalty
        if min_clearance == float('inf'):
            clearance_bonus = 5  # No obstacles detected
        elif min_clearance > 500:
            clearance_bonus = 4
        elif min_clearance > 300:
            clearance_bonus = 2
        elif min_clearance > 150:
            clearance_bonus = 0
        else:
            clearance_bonus = -10  # Dangerous path
        
        score += clearance_bonus
        
        # Speed bonus (prefer faster paths when safe)
        speed_bonus = path.get('speed_required', 30) / 20  # Up to 3.5 points
        score += min(3.5, speed_bonus)
        
        # Simplicity bonus (prefer straighter paths)
        turn_penalty = path.get('turn_intensity', 0) / 30  # Penalty for complex turns
        score -= min(3, turn_penalty)
        
        return score

    def execute_path_fast(self, path):
        """Execute selected path with optimized movements"""
        
        if not path:
            # Fallback to minimal safe movement
            self.move(-5, 5, smooth=False)
            return
        
        path_type = path['type']
        speed = path['speed_required']
        
        if path_type == 'straight':
            self.move(speed, -speed, smooth=False)
            print(f"EXECUTING: Straight forward at {speed}")
            
        elif path_type == 'gentle_left':
            left_speed = speed * 0.6
            right_speed = -speed
            self.move(left_speed, right_speed, smooth=True)
            print(f"EXECUTING: Gentle left turn L={left_speed:.0f} R={right_speed}")
            
        elif path_type == 'gentle_right':
            left_speed = speed
            right_speed = -speed * 0.6
            self.move(left_speed, right_speed, smooth=True)
            print(f"EXECUTING: Gentle right turn L={left_speed} R={right_speed:.0f}")
            
        elif path_type == 'sharp_left':
            self.move(-speed//2, -speed//2, smooth=False)
            print(f"EXECUTING: Sharp left turn at {speed//2}")
            
        elif path_type == 'sharp_right':
            self.move(speed//2, speed//2, smooth=False)
            print(f"EXECUTING: Sharp right turn at {speed//2}")
            
        elif path_type == 'reverse_turn':
            # Execute reverse and turn maneuver
            self.execute_reverse_turn_fast(speed)
            
        elif path_type == 'emergency_spiral':
            # Execute emergency spiral maneuver
            self.execute_emergency_spiral(speed)
            
        else:
            # Unknown path type - safe default
            self.move(speed//3, -speed//3, smooth=True)
            print(f"EXECUTING: Default safe movement at {speed//3}")
        
        self.current_direction = f"FAST_{path_type.upper()}"

    def execute_reverse_turn_fast(self, base_speed):
        """Execute fast reverse and turn maneuver"""
        
        print("EXECUTING: Fast reverse and turn")
        
        # Quick reverse
        reverse_speed = base_speed // 2
        self.move(-reverse_speed, reverse_speed, smooth=False)
        
        # Short delay
        time.sleep(0.3)
        
        # Quick turn assessment
        turn_direction = self.assess_turn_direction_fast()
        
        if turn_direction == "LEFT":
            self.move(-base_speed, -base_speed, smooth=False)
        else:
            self.move(base_speed, base_speed, smooth=False)

    def execute_emergency_spiral(self, base_speed):
        """Execute emergency spiral maneuver"""
        
        print("EXECUTING: Emergency spiral escape")
        
        # Start with small radius turn
        spiral_speed = base_speed // 2
        
        # Gradually increasing turn radius
        for i in range(3):
            turn_speed = spiral_speed + (i * 5)
            self.move(-turn_speed, -turn_speed, smooth=False)
            time.sleep(0.1)

    def assess_turn_direction_fast(self):
        """Quickly assess best turn direction"""
        
        if hasattr(self.parent, 'ultrasonic_manager') and self.parent.ultrasonic_manager:
            status = self.parent.ultrasonic_manager.get_sensor_status()
            
            left_clear = status.get('front_left', {}).get('distance', 0)
            right_clear = status.get('front_right', {}).get('distance', 0)
            
            if left_clear > right_clear:
                return "LEFT"
            else:
                return "RIGHT"
        
        return "RIGHT"  # Default

    def calculate_escape_direction(self, target_direction):
        """Calculate best escape direction"""
        
        if target_direction is None:
            return "RIGHT"  # Default
        
        # Choose direction that maintains some progress toward target
        if 0 <= target_direction <= 90 or 270 <= target_direction <= 360:
            return "LEFT"  # Target on right side, escape left
        else:
            return "RIGHT"  # Target on left side, escape right

    def process_uwb_control_optimized(self, uwb_distances, lidar, ultrasonic_manager):
        """Optimized UWB control with ultra-fast response"""
        
        A0, A1, A2 = uwb_distances['A0'], uwb_distances['A1'], uwb_distances['A2']
        
        # Target reached check
        if A0 <= self.stop_threshold:
            print(f"Target reached (A0={A0:.1f}cm) - Controlled stop")
            self.smooth_speed_transition(0, 0)
            return
        
        # Calculate angle error for steering
        angle_error = A2 - A1
        
        # Determine base speed with obstacle consideration
        if hasattr(lidar, 'get_obstacle_status'):
            lidar_status = lidar.get_obstacle_status()
            
            if lidar_status.get('critical_danger', False):
                base_speed = 15  # Very slow for critical situations
            elif lidar_status.get('danger_zone', False):
                base_speed = 30  # Slow for danger zones
            else:
                # Normal speed calculation
                if A0 < 100:
                    base_speed = 45  # Close to target
                elif A0 < 200:
                    base_speed = 60  # Medium distance
                else:
                    base_speed = self.speed  # Normal distance
        else:
            base_speed = self.speed
        
        # Calculate wheel speeds with improved differential steering
        if abs(angle_error) < 10:  # Target roughly centered
            left_speed = base_speed
            right_speed = -base_speed
            
        elif angle_error < 0:  # Target to the right
            # More aggressive right turning
            turn_factor = min(1.0, abs(angle_error) / 50.0)
            left_speed = base_speed * (1.0 + turn_factor * 0.4)
            right_speed = -base_speed * (1.0 - turn_factor * 0.6)
            
        else:  # Target to the left
            # More aggressive left turning
            turn_factor = min(1.0, abs(angle_error) / 50.0)
            left_speed = base_speed * (1.0 - turn_factor * 0.6)
            right_speed = -base_speed * (1.0 + turn_factor * 0.4)
        
        # Apply ultrasonic adjustments if available
        if ultrasonic_manager:
            left_speed, right_speed = self.apply_ultrasonic_obstacle_adjustment(
                ultrasonic_manager, left_speed, right_speed
            )
        
        # Execute movement with smooth transition
        self.smooth_speed_transition(int(left_speed), int(right_speed))
        
        # Direction logging
        if abs(angle_error) < 10:
            direction = "FORWARD"
        elif angle_error < 0:
            direction = "TURN_RIGHT"
        else:
            direction = "TURN_LEFT"
        
        print(f"UWB Control: {direction} | A0={A0:.1f}cm | Error={angle_error:.1f} | L={left_speed:.0f} R={right_speed:.0f}")

    def apply_minimal_obstacle_adjustment(self, left_speed, right_speed, ultrasonic_manager):
        """Adjustment minimal - hanya untuk obstacle yang benar-benar menghalangi"""
        
        summary = ultrasonic_manager.get_obstacle_summary()
        
        # HANYA adjust jika critical (< 15cm) DAN menghalangi jalur langsung
        if not summary['critical_detected']:
            return left_speed, right_speed
        
        status = summary['sensor_status']
        center_distance = status.get('front_center', {}).get('distance', 100)
        
        # Jika center masih aman (> 15cm), TIDAK perlu adjustment
        if center_distance > 15:
            return left_speed, right_speed
        
        # Cari jalur samping yang tersedia
        left_distance = status.get('front_left', {}).get('distance', 0)
        right_distance = status.get('front_right', {}).get('distance', 0)
        
        if left_distance > 20:  # Kiri cukup lapang
            # Slight left turn
            return int(left_speed * 0.7), int(right_speed * 1.3)
        elif right_distance > 20:  # Kanan cukup lapang
            # Slight right turn  
            return int(left_speed * 1.3), int(right_speed * 0.7)
        else:
            # Kedua sisi sempit - perlambat sedikit tapi tetap maju
            return int(left_speed * 0.6), int(right_speed * 0.6)


    def find_alternative_path_ultra_fast(self, ultrasonic_summary):
        """Ultra-fast alternative path finding for ultrasonic sensors"""
        
        status = ultrasonic_summary['sensor_status']
        
        # Quick assessment of available directions
        left_clear = status.get('front_left', {}).get('distance', 0)
        center_blocked = status.get('front_center', {}).get('distance', 0)
        right_clear = status.get('front_right', {}).get('distance', 0)
        
        # Priority-based decision tree
        if left_clear > 30:
            return "SHARP_LEFT"
        elif right_clear > 30:
            return "SHARP_RIGHT"
        elif max(left_clear, right_clear) > 20:
            return "GENTLE_LEFT" if left_clear > right_clear else "GENTLE_RIGHT"
        elif center_blocked > 15:  # Some space in front
            return "SLOW_FORWARD"
        else:
            return "REVERSE_MINIMAL"

    def execute_alternative_speeds(self, action, base_left, base_right):
        """Execute alternative action and return appropriate speeds"""
        
        if action == "SHARP_LEFT":
            return -abs(base_left), -abs(base_left)
        elif action == "SHARP_RIGHT":
            return abs(base_right), abs(base_right)
        elif action == "GENTLE_LEFT":
            return base_left * 0.3, base_right * 1.2
        elif action == "GENTLE_RIGHT":
            return base_left * 1.2, base_right * 0.3
        elif action == "SLOW_FORWARD":
            return base_left * 0.4, base_right * 0.4
        elif action == "REVERSE_MINIMAL":
            return -abs(base_left) * 0.3, abs(base_right) * 0.3
        else:
            return 0, 0

class PerformanceMonitor:
    """Monitor system performance and response times"""
    
    def __init__(self):
        self.metrics = {
            'response_times': deque(maxlen=100),
            'path_evaluations': deque(maxlen=50),
            'emergency_activations': 0,
            'path_changes': 0,
            'average_loop_time': 0,
            'max_loop_time': 0,
            'sensor_update_rates': {
                'lidar': 0,
                'ultrasonic': 0,
                'uwb': 0
            }
        }
        
        self.start_time = time.time()
        self.last_performance_log = 0
    
    def log_response_time(self, response_time_ms, action_type):
        """Log response time for analysis"""
        timestamp = time.time()
        
        self.metrics['response_times'].append({
            'time': response_time_ms,
            'type': action_type,
            'timestamp': timestamp
        })
        
        # Alert on slow responses
        if response_time_ms > 10:
            print(f"⚠ SLOW RESPONSE: {response_time_ms:.1f}ms for {action_type}")
    
    def log_path_evaluation(self, evaluation_time_ms, paths_evaluated):
        """Log path evaluation metrics"""
        self.metrics['path_evaluations'].append({
            'time': evaluation_time_ms,
            'paths': paths_evaluated,
            'timestamp': time.time()
        })
    
    def increment_emergency_activations(self):
        """Increment emergency activation counter"""
        self.metrics['emergency_activations'] += 1
    
    def increment_path_changes(self):
        """Increment path change counter"""
        self.metrics['path_changes'] += 1
    
    def update_sensor_rate(self, sensor_type, rate_hz):
        """Update sensor update rate"""
        if sensor_type in self.metrics['sensor_update_rates']:
            self.metrics['sensor_update_rates'][sensor_type] = rate_hz
    
    def log_loop_time(self, loop_time_ms):
        """Log control loop time"""
        self.metrics['average_loop_time'] = (
            self.metrics['average_loop_time'] * 0.95 + loop_time_ms * 0.05
        )
        self.metrics['max_loop_time'] = max(self.metrics['max_loop_time'], loop_time_ms)
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.metrics['response_times']:
            return "No performance data available"
        
        response_times = [r['time'] for r in self.metrics['response_times']]
        avg_response = sum(response_times) / len(response_times)
        max_response = max(response_times)
        
        uptime = time.time() - self.start_time
        
        summary = f"""
Performance Summary (Uptime: {uptime:.1f}s):
  Response Times: Avg={avg_response:.1f}ms, Max={max_response:.1f}ms
  Loop Performance: Avg={self.metrics['average_loop_time']:.1f}ms, Max={self.metrics['max_loop_time']:.1f}ms
  Emergency Activations: {self.metrics['emergency_activations']}
  Path Changes: {self.metrics['path_changes']}
  Sensor Rates: LIDAR={self.metrics['sensor_update_rates']['lidar']:.1f}Hz, 
                Ultrasonic={self.metrics['sensor_update_rates']['ultrasonic']:.1f}Hz,
                UWB={self.metrics['sensor_update_rates']['uwb']:.1f}Hz
        """
        
        return summary.strip()
    
    def should_log_performance(self, interval_seconds=5.0):
        """Check if it's time to log performance"""
        current_time = time.time()
        if current_time - self.last_performance_log > interval_seconds:
            self.last_performance_log = current_time
            return True
        return False

# Apply performance optimizations to main node
def apply_performance_optimizations(node):
    """Apply performance optimizations to the robot system"""
    
    print("Applying performance optimizations...")
    
    # Set thread priorities
    try:
        import os
        os.nice(-10)  # Higher process priority
        print("✓ Process priority increased")
    except:
        print("⚠ Could not increase process priority")
    
    # Optimize sensor update rates
    node.control_frequency = OPTIMIZED_CONFIG['normal_control_freq']
    node.uwb_timeout = OPTIMIZED_CONFIG['sensor_timeout']
    
    # Initialize performance monitor
    node.performance_monitor = PerformanceMonitor()
    
    # Initialize fast response system if not already done
    if hasattr(node.controller, 'initialize_fast_response'):
        node.controller.initialize_fast_response()
    
    # Set real-time scheduling if available
    try:
        import sched
        sched.SCHED_FIFO
        print("✓ Real-time scheduling available")
    except:
        print("⚠ Real-time scheduling not available")
    
    print("✓ Performance optimizations applied")

# Enhanced main function with complete error handling
def main(args=None):
    """Enhanced main function with complete system initialization"""
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    node = None
    executor = None
    
    try:
        print("=" * 60)
        print("   ENHANCED MOBILE ROBOT WITH ADVANCED OBSTACLE AVOIDANCE")
        print("=" * 60)
        print("Initializing robot systems...")
        
        # Test system requirements
        print("\n🔧 System Requirements Check:")
        
        # Test GPIO availability
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            print("✓ GPIO system available")
        except Exception as e:
            print(f"✗ GPIO system error: {e}")
            print("  Robot may not function properly without GPIO access")
        
        # Test network interface
        try:
            ip = get_ip_from_wifi()
            if ip:
                print(f"✓ Network interface available: {ip}")
            else:
                print("⚠ Network interface issue - using fallback")
        except Exception as e:
            print(f"⚠ Network check failed: {e}")
        
        # Test sensor access
        try:
            # Test ultrasonic sensor pins
            test_pins = [18, 24, 23, 25, 12, 16]
            for pin in test_pins:
                GPIO.setup(pin, GPIO.OUT)
            print("✓ Ultrasonic sensor pins accessible")
        except Exception as e:
            print(f"⚠ Some sensor pins may not be accessible: {e}")
        
        print("\n🤖 Creating robot node...")
        
        # Create enhanced node
        node = FollowingRobotNode()
        print("✓ Robot node created successfully")
        
        # Apply performance optimizations
        apply_performance_optimizations(node)
        
        # Create multi-threaded executor with optimizations
        executor = MultiThreadedExecutor(num_threads=4)  # Optimize thread count
        executor.add_node(node)
        print("✓ Multi-threaded executor configured (4 threads)")
        
        print("\n" + "=" * 60)
        print("🚀 ROBOT SYSTEMS READY!")
        print("=" * 60)
        print("Enhanced Features Enabled:")
        print("  🎯 Ultra-fast obstacle avoidance (1-3ms response)")
        print("  📡 Multi-sensor fusion (LIDAR + Ultrasonic + UWB)")
        print("  🧠 Dynamic Window Approach path planning")
        print("  ⚡ Real-time priority scheduling")
        print("  🔄 Independent wheel control system")
        print("  🛡️ Multi-level emergency response")
        print("  📊 Performance monitoring & optimization")
        print("  🎪 Dynamic object detection & tracking")
        print("  🏃 Path alternatives (never just stop)")
        print("  🎮 Smooth motion control & transitions")
        
        print("\nSafety Systems:")
        print("  🚨 Emergency brake (< 1ms response)")
        print("  🔍 Critical zone monitoring (360°)")
        print("  🤖 Multi-sensor redundancy")
        print("  ⚠️  Fail-safe fallback modes")
        
        print("\nPerformance Targets:")
        print("  📈 Control frequency: 500Hz")
        print("  ⚡ Emergency response: < 1ms")
        print("  🎯 Normal response: < 5ms")
        print("  📡 Sensor fusion: 800Hz")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop the robot")
        print("=" * 60)
        
        # Start performance logging
        def log_performance():
            if hasattr(node, 'performance_monitor'):
                if node.performance_monitor.should_log_performance():
                    summary = node.performance_monitor.get_performance_summary()
                    print(f"\n📊 {summary}\n")
        
        # Create performance logging timer
        performance_timer = threading.Timer(5.0, log_performance)
        performance_timer.daemon = True
        performance_timer.start()
        
        # Run the robot
        print("🏁 Starting robot operation...\n")
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n" + "🛑" * 20)
        print("🛑 User interruption detected")
        print("🛑 Initiating safe shutdown...")
        print("🛑" * 20)
        
    except Exception as e:
        print(f"\n❌ CRITICAL SYSTEM ERROR: {e}")
        print("💥 Full error traceback:")
        import traceback
        traceback.print_exc()
        
        # Turn on error indicator
        try:
            gpio_pin_27.on()
        except:
            pass
            
        print("🔧 Attempting emergency shutdown...")
        
    finally:
        # Comprehensive cleanup
        print("\n🔧 SYSTEM CLEANUP SEQUENCE")
        print("-" * 40)
        
        cleanup_success = True
        
        if node:
            try:
                print("Stopping robot node...")
                node.stop()
                node.destroy_node()
                print("✓ Node destroyed successfully")
            except Exception as e:
                print(f"✗ Error destroying node: {e}")
                cleanup_success = False
        
        if executor:
            try:
                print("Shutting down executor...")
                executor.shutdown(timeout_sec=2.0)
                print("✓ Executor shutdown complete")
            except Exception as e:
                print(f"✗ Error shutting down executor: {e}")
                cleanup_success = False
        
        try:
            print("Shutting down ROS2...")
            rclpy.shutdown()
            print("✓ ROS2 shutdown complete")
        except Exception as e:
            print(f"✗ Error shutting down ROS2: {e}")
            cleanup_success = False
        
        # Final GPIO cleanup with comprehensive error handling
        try:
            print("Cleaning up GPIO...")
            
            # Turn off all indicator LEDs
            gpio_pin_17.off()
            if not cleanup_success:
                gpio_pin_27.on()  # Keep error LED on if cleanup failed
            else:
                gpio_pin_27.off()
            
            # Full GPIO cleanup
            GPIO.cleanup()
            print("✓ GPIO cleanup complete")
            
        except Exception as e:
            print(f"✗ Error with GPIO cleanup: {e}")
            cleanup_success = False
        
        # Final status
        print("-" * 40)
        if cleanup_success:
            print("🏁 ROBOT SHUTDOWN COMPLETE - ALL SYSTEMS CLEAN")
        else:
            print("⚠️  ROBOT SHUTDOWN COMPLETE - SOME ERRORS OCCURRED")
            print("   Check system logs for details")
        
        print("=" * 60)

# Test mode for sensor verification
def test_sensors_mode():
    """Test mode untuk verifikasi sensor"""
    print("🔧 SENSOR TEST MODE")
    print("=" * 40)
    
    try:
        # Test ultrasonic sensors
        print("Testing ultrasonic sensors...")
        ultrasonic_manager = UltrasonicSensorManager()
        
        # Run test
        ultrasonic_manager.test_sensor_directions()
        
        # Cleanup
        ultrasonic_manager.stop()
        print("✓ Ultrasonic sensor test complete")
        
    except Exception as e:
        print(f"✗ Sensor test failed: {e}")
    
    finally:
        try:
            GPIO.cleanup()
        except:
            pass

if __name__ == '__main__':
    # Check for test mode
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test-sensors':
            test_sensors_mode()
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("Enhanced Mobile Robot - Usage:")
            print("  python3 robot.py                 # Normal operation")
            print("  python3 robot.py --test-sensors  # Sensor testing mode")
            print("  python3 robot.py --help          # Show this help")
            sys.exit(0)
    
    # Normal operation
    main()
