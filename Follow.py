#!/usr/bin/env python3

import socket
import numpy as np
import math
import time
import threading
import sys
import os
import select
import netifaces  # Module to get IP address from the Wi-Fi network interface
from collections import deque

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from ddsm115 import MotorControl
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Import gpiozero for controlling GPIO pins
from gpiozero import LED

# Konfigurasi Global
SCAN_THRESHOLD = 2000  # Jarak aman dalam mm
DANGER_THRESHOLD = 1000  # Jarak bahaya dalam mm
MIN_VALID_DISTANCE = 200  # Jarak minimum valid (mm) untuk menghindari noise
CRITICAL_DANGER_THRESHOLD = 300  # Jarak kritis untuk override UWB (mm)

# Motor speed configuration
DEFAULT_SPEED = 75
ROTATION_FACTOR = 2
STOP_THRESHOLD = 80  # cm

# Konfigurasi performa real-time
CONTROL_FREQUENCY = 150  # Hz - lebih realistis
LIDAR_SKIP_FRAMES = 1    # Process setiap 2 frame
UWB_TIMEOUT = 0.0001      # 1ms timeout
MAX_LOOP_TIME = 0.01     # 10ms warning threshold

# Buffer sizes
LIDAR_BUFFER_SIZE = 100   # Increase untuk real-time
UWB_BUFFER_SIZE = 1024   # Increase UWB buffer

# Definisikan sudut lebih jelas
FRONT_REGION = [(330, 360), (0, 30)]  # Depan: 330° hingga 360° dan 0° hingga 30°
RIGHT_REGION = (31, 140)  # Kanan: 31° hingga 140°
LEFT_REGION  = (220, 329)  # Kiri: 220° hingga 329°
BACK_REGION = (150, 210)  # Belakang: 150° hingga 210°
TARGET_EXCLUSION_ANGLE = 10  # Rentang pengecualian untuk target (menghindari tabrakan dengan target)

# GPIO Pins Initialization
gpio_pin_17 = LED(17)  # GPIO 17 for program start
gpio_pin_27 = LED(27)  # GPIO 27 for error indication

# Dynamic IP function to get Raspberry Pi IP from Wi-Fi interface (default is wlan0)
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
    ip_parts[-1] = str(target_last_digit)  # Replace the last digit
    return ".".join(ip_parts)

# Dynamic Object Detection Class
class DynamicObjectDetector:
    """Detects dynamic vs static objects using temporal analysis"""
    
    def __init__(self):
        self.position_history = {}  # Store position history for each detected object
        self.history_window = 10    # Number of frames to keep in history
        self.movement_threshold = 150  # mm - minimum movement to consider dynamic
        self.static_frames_required = 8  # Frames object must be static to be considered static
        self.dynamic_timeout = 5.0  # Seconds to wait for dynamic object to move away
        
        # Track detected objects
        self.current_objects = {}
        self.dynamic_objects = set()
        self.static_objects = set()
        
        # Safety state
        self.dynamic_object_detected = False
        self.dynamic_object_last_seen = 0
        self.waiting_for_dynamic_object = False
        
    def _cluster_scan_points(self, scan_data):
        """Simple object detection using clustering of nearby points"""
        objects = {}
        current_cluster = []
        cluster_id = 0
        
        # Sort angles for sequential processing
        sorted_angles = sorted(scan_data.keys())
        
        for i, angle in enumerate(sorted_angles):
            distance = scan_data[angle]
            
            # Skip invalid readings
            if distance < MIN_VALID_DISTANCE or distance > 8000:
                continue
                
            # Convert to cartesian coordinates
            x = distance * math.cos(math.radians(angle))
            y = distance * math.sin(math.radians(angle))
            
            # Check if this point belongs to current cluster
            if current_cluster:
                last_x, last_y = current_cluster[-1]
                dist_to_last = math.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                if dist_to_last < 300:  # Points within 30cm belong to same object
                    current_cluster.append((x, y))
                else:
                    # End current cluster and start new one
                    if len(current_cluster) >= 3:  # Minimum points for valid object
                        # Calculate cluster center
                        center_x = sum(p[0] for p in current_cluster) / len(current_cluster)
                        center_y = sum(p[1] for p in current_cluster) / len(current_cluster)
                        objects[cluster_id] = (center_x, center_y, len(current_cluster))
                        cluster_id += 1
                    
                    current_cluster = [(x, y)]
            else:
                current_cluster = [(x, y)]
        
        # Handle last cluster
        if len(current_cluster) >= 3:
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
                    if distance < min_distance and distance < 500:  # Max 50cm movement between frames
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
            if not history or current_time - history[-1]['timestamp'] > 2.0:
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
            if len(history) < 3:  # Need minimum history for classification
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
            
            if avg_movement > self.movement_threshold or static_ratio < 0.6:
                self.dynamic_objects.add(obj_id)
            elif static_ratio >= 0.8 and len(history) >= self.static_frames_required:
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

class UWBTracker:
    """Handles UWB data processing and position estimation"""
    
    def __init__(self):
        # Default bias correction values
        self.bias = {
            'A0': 50.0,  # Bias value in cm
            'A1': 50.0,  # Bias value in cm
            'A2': 50.0   # Bias value in cm
        }
        
        # Default scale factor values
        self.scale_factor = {
            'A0': 1.0,   # Scale factor
            'A1': 1.005,  # Scale factor
            'A2': 1.01   # Scale factor
        }
        
        # Target direction estimation
        self.target_direction = None  # Estimated angle to target (degrees)
        self.target_distance = None   # Estimated distance to target (mm)
    
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
        # Jika A1 dan A2 keduanya lebih kecil dari A0 dengan margin signifikan
        if (A1 < A0 - 30) and (A2 < A0 - 30):
            # Target kemungkinan di belakang
            if abs(diff_A2_A1) < 20:
                target_direction = 180  # Tepat di belakang
                print(f"TARGET DETECTED BEHIND: A0={A0:.1f}, A1={A1:.1f}, A2={A2:.1f}")
            elif diff_A2_A1 < 0:  # A2 < A1
                # Target di belakang-kanan
                angle_offset = min(30, abs(diff_A2_A1) * 1.0)
                target_direction = 180 - angle_offset  # 150-180 degrees
                print(f"TARGET BEHIND-RIGHT: angle={target_direction:.1f}°")
            else:  # A1 < A2
                # Target di belakang-kiri  
                angle_offset = min(30, abs(diff_A2_A1) * 1.0)
                target_direction = 180 + angle_offset  # 180-210 degrees
                print(f"TARGET BEHIND-LEFT: angle={target_direction:.1f}°")
        
        # Deteksi target di depan (logika existing)
        elif abs(diff_A2_A1) < 20:
            target_direction = 0  # Depan
            print(f"TARGET FRONT: A0={A0:.1f}, A1={A1:.1f}, A2={A2:.1f}")
        
        # Deteksi target di samping
        elif diff_A2_A1 < 0:  # A2 < A1, target ke kanan
            # Periksa apakah di samping atau depan-kanan
            if A0 > min(A1, A2) + 20:  # Target di samping kanan
                angle_offset = min(60, abs(diff_A2_A1) * 1.5)
                target_direction = 90 + angle_offset  # 90-150 degrees
                print(f"TARGET RIGHT-SIDE: angle={target_direction:.1f}°")
            else:  # Target di depan-kanan
                angle_offset = min(45, abs(diff_A2_A1) * 1.5)
                target_direction = angle_offset  # 0-45 degrees  
                print(f"TARGET FRONT-RIGHT: angle={target_direction:.1f}°")
        
        else:  # A1 < A2, target ke kiri
            # Periksa apakah di samping atau depan-kiri
            if A0 > min(A1, A2) + 20:  # Target di samping kiri
                angle_offset = min(60, abs(diff_A2_A1) * 1.5)
                target_direction = 270 - angle_offset  # 210-270 degrees
                print(f"TARGET LEFT-SIDE: angle={target_direction:.1f}°")
            else:  # Target di depan-kiri
                angle_offset = min(45, abs(diff_A2_A1) * 1.5)
                target_direction = 360 - angle_offset  # 315-360 degrees
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
        # 1. A1 dan A2 keduanya signifikan lebih kecil dari A0
        # 2. Atau jika estimated direction antara 135-225 degrees
        behind_condition1 = (A1 < A0 - 25) and (A2 < A0 - 25)
        behind_condition2 = (self.target_direction and 
                            135 <= self.target_direction <= 225)
        
        return behind_condition1 or behind_condition2

class LidarProcessor:
    """Processes LIDAR data from ROS2 LaserScan messages with dynamic object detection"""

    def __init__(self):
        self.scan_data = {}  # Dictionary to store scan data (angle -> distance)
        self.lock = threading.Lock()

        # Obstacle status
        self.front_obstacle = False
        self.left_obstacle = False
        self.right_obstacle = False
        self.back_obstacle = False  # For the back region
        self.danger_zone = False
        self.critical_danger = False

        # Minimum distance for each region
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.back_distance = float('inf')  # For the back region

        self.target_direction = None  # The direction of the target
        self.target_distance = None   # The distance to the target

        # Timestamp for the last scan
        self.last_scan_time = 0

        # Save raw scan for visualization
        self.last_scan_msg = None

        # Target information (from UWB)
        self.target_direction = None
        self.target_distance = None

        # Dynamic object detector
        self.dynamic_detector = DynamicObjectDetector()

        # Improved filtering
        self.moving_avg_window = 2
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
        """Process ROS2 LaserScan message with dynamic object detection"""
        with self.lock:
            self.last_scan_msg = scan_msg
            self.last_scan_time = time.time()
            self.scan_data.clear()

            ranges = scan_msg.ranges
            angle_increment = scan_msg.angle_increment
            angle_min = scan_msg.angle_min

            if min(self.front_distance, self.left_distance, self.right_distance) < 1000:
                step_size = 2  # Higher resolution near obstacles
            else:
                step_size = 5  # Lower resolution when clear

            for i in range(0, len(ranges), step_size):
                distance = ranges[i]

                if distance < 0.01 or distance > 10.0 or math.isinf(distance):
                    continue

                angle_rad = angle_min + (i * angle_increment)
                angle_deg = int(math.degrees(angle_rad) % 360)
                distance_mm = int(distance * 1000)

                self.scan_data[angle_deg] = distance_mm

            # Apply the moving average filter to smooth the scan data
            self.scan_data = self._filter_lidar_data(self.scan_data)

            # Analyze obstacles with dynamic object detection
            self._analyze_obstacles_with_dynamic_detection()

    def _filter_lidar_data(self, scan_data):
        """Filter LIDAR data using a moving average to smooth out readings"""
        filtered_data = {}

        for angle, distance in scan_data.items():
            if angle not in self.distance_history:
                self.distance_history[angle] = deque()

            # Append the new distance to the history
            self.distance_history[angle].append(distance)

            # Keep only the last `self.moving_avg_window` number of distances
            if len(self.distance_history[angle]) > self.moving_avg_window:
                self.distance_history[angle].popleft()

            # Calculate the moving average
            filtered_data[angle] = sum(self.distance_history[angle]) / len(self.distance_history[angle])

        return filtered_data

    def _analyze_obstacles_with_dynamic_detection(self):
        """Analisis rintangan dengan deteksi objek dinamis"""
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
            data_valid = scan_age < 0.6

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

        # Jika menunggu objek dinamis, berhenti dulu
        if status['waiting_for_dynamic']:
            return "STOP_DYNAMIC"

        if status['critical_danger']:
            return "STOP"
        
        # Jika tidak ada rintangan di depan, kiri, kanan, dan belakang, lanjutkan maju
        if not status['front']['obstacle'] and not status['left']['obstacle'] and not status['right']['obstacle'] and not status['back']['obstacle']:
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
                distances = [
                    ('FORWARD', status['front']['distance']),
                    ('LEFT', status['left']['distance']),
                    ('RIGHT', status['right']['distance']),
                    ('BACK', status['back']['distance'])
                ]
                best_direction = max(distances, key=lambda x: x[1])[0]
                return best_direction

        return "FORWARD"

# Robot Controller with Enhanced Safety and Dynamic Object Handling
class RobotController:
    """Controls robot movement with gradual obstacle response and smooth transitions"""
    
    def __init__(self, r_wheel_port, l_wheel_port):
        # Motor controllers
        self.right_motor = MotorControl(device=r_wheel_port)
        self.left_motor = MotorControl(device=l_wheel_port)
        self.right_motor.set_drive_mode(1, 2)
        self.left_motor.set_drive_mode(1, 2)
        
        # Speed configuration
        self.speed = DEFAULT_SPEED
        self.rotation_factor = ROTATION_FACTOR
        self.stop_threshold = STOP_THRESHOLD
        
        # Gradual turning parameters
        self.turn_gradual_threshold = 600  # mm - jarak untuk mulai gentle turn
        self.max_turn_angle = 15          # degrees - maksimal turn angle per step
        self.turn_smoothing_factor = 0.7  # faktor untuk smooth turning
        
        # Incremental speed system parameters
        self.min_speed = self.speed // 2
        self.max_speed = self.speed * 2
        self.current_target_speed = self.min_speed
        self.actual_speed = self.min_speed
        self.speed_increment = 8
        self.speed_decrement = 12
        self.acceleration_delay = 0.08
        self.last_speed_update = time.time()

        # Conditions for speed increase
        self.straight_path_threshold = 20
        self.clear_distance_threshold = 800
        self.speed_boost_distance = 1500
        self.consecutive_clear_count = 0
        self.min_clear_count = 4
        
        # Smooth movement parameters
        self.smooth_turn_speed = self.speed // 4
        self.gentle_turn_speed = self.speed // 3
        self.rotation_speed = self.speed // 5
        self.current_left_speed = 0
        self.current_right_speed = 0
        
        # Gradual response state
        self.obstacle_response_active = False
        self.current_obstacle_zone = 'clear'
        self.gradual_turn_active = False
        self.turn_direction_preference = None

        # KONSISTEN OBSTACLE ZONES DAN SPEED FACTORS
        self.obstacle_zones = {
            'far': 1000,      # mm - mulai perlambat
            'medium': 800,    # mm - perlambat lebih
            'near': 400,      # mm - perlambat drastis
            'critical': 200   # mm - stop atau turn minimal
        }
        
        # Speed factors yang KONSISTEN dengan obstacle_zones
        self.speed_factors = {
            'clear': 1.0,     # Normal speed
            'far': 0.8,       # 80% speed
            'medium': 0.5,    # 50% speed  
            'near': 0.3,      # 30% speed
            'critical': 0.1   # 10% speed
        }
        
        # Status flags
        self.obstacle_avoidance_active = False
        self.last_command_time = time.time()
        self.current_direction = "STOP"
        self.emergency_stop = False
        self.waiting_for_dynamic_object = False
        self.speed_boost_active = False
        
        # Command batching
        self.last_command = (0, 0)
        self.command_threshold = 3
        self.last_command_time = 0
        self.min_command_interval = 0.03

        # INDEPENDENT WHEEL CONTROL SYSTEM
        self.independent_control_enabled = True
        
        # Base speed untuk masing-masing roda
        self.base_left_speed = 0
        self.base_right_speed = 0
        
        # Target speed untuk masing-masing roda
        self.target_left_speed = 0
        self.target_right_speed = 0
        
        # Speed adjustment rates untuk smooth transition
        self.speed_adjustment_rate = 5  # RPM per cycle
        self.max_speed_diff = self.speed * 2  # Maksimal perbedaan speed antar roda
        
        # Differential steering parameters
        self.steering_gain = 0.8  # Seberapa aggressive steering response
        self.forward_bias = 0.9   # Bias untuk tetap bergerak maju
        
        # Real-time control state
        self.last_target_direction = None
        self.direction_change_rate = 10  # degrees per second

    def calculate_independent_wheel_speeds(self, target_distance, target_angle_error, base_speed):
        """Hitung kecepatan roda independen berdasarkan target dan error angle"""
        
        # Normalisasi angle error (-180 to 180)
        while target_angle_error > 180:
            target_angle_error -= 360
        while target_angle_error < -180:
            target_angle_error += 360
        
        # Base speed adjustment berdasarkan jarak
        if target_distance < 100:  # Sangat dekat
            speed_factor = 0.3
        elif target_distance < 200:  # Dekat
            speed_factor = 0.6
        else:  # Normal
            speed_factor = 1.0
        
        adjusted_base_speed = base_speed * speed_factor
        
        # Hitung steering correction berdasarkan angle error
        # Positive angle error = target di kiri, perlu belok kiri
        # Negative angle error = target di kanan, perlu belok kanan
        
        steering_correction = target_angle_error * self.steering_gain
        steering_correction = max(-self.max_speed_diff, min(self.max_speed_diff, steering_correction))
        
        # Hitung kecepatan target untuk masing-masing roda
        if target_angle_error > 5:  # Target di kiri
            # Perlambat roda kiri, percepat roda kanan
            left_speed = adjusted_base_speed - abs(steering_correction * 0.7)
            right_speed = -(adjusted_base_speed + abs(steering_correction * 0.3))
            
        elif target_angle_error < -5:  # Target di kanan
            # Percepat roda kiri, perlambat roda kanan
            left_speed = adjusted_base_speed + abs(steering_correction * 0.3)
            right_speed = -(adjusted_base_speed - abs(steering_correction * 0.7))
            
        else:  # Target di depan (centered)
            # Kedua roda sama, maju lurus
            left_speed = adjusted_base_speed
            right_speed = -adjusted_base_speed
        
        # Apply forward bias untuk tetap maju
        left_speed *= self.forward_bias
        right_speed *= self.forward_bias
        
        # Clamp ke batas maksimal
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))
        
        return int(left_speed), int(right_speed)
    
    def smooth_speed_transition(self, target_left, target_right):
        """Transisi smooth ke target speed tanpa stopping"""
        
        # Hitung perubahan yang diperlukan
        left_diff = target_left - self.current_left_speed
        right_diff = target_right - self.current_right_speed
        
        # Batasi rate perubahan untuk smooth transition
        if abs(left_diff) > self.speed_adjustment_rate:
            if left_diff > 0:
                new_left = self.current_left_speed + self.speed_adjustment_rate
            else:
                new_left = self.current_left_speed - self.speed_adjustment_rate
        else:
            new_left = target_left
            
        if abs(right_diff) > self.speed_adjustment_rate:
            if right_diff > 0:
                new_right = self.current_right_speed + self.speed_adjustment_rate
            else:
                new_right = self.current_right_speed - self.speed_adjustment_rate
        else:
            new_right = target_right
        
        # Update current speeds
        self.current_left_speed = new_left
        self.current_right_speed = new_right
        
        # Send ke motor langsung tanpa delay
        self.left_motor.send_rpm(1, int(new_left))
        self.right_motor.send_rpm(1, int(new_right))
        
        return new_left, new_right
    
    def analyze_obstacle_zones(self, lidar_status):
        """Analisis zona obstacle untuk gradual response dengan validasi output"""
        try:
            distances = {
                'front': lidar_status.get('front', {}).get('distance', float('inf')),
                'left': lidar_status.get('left', {}).get('distance', float('inf')),
                'right': lidar_status.get('right', {}).get('distance', float('inf'))
            }
            
            # Cari jarak terdekat
            min_distance = min(distances.values())
            closest_direction = min(distances, key=distances.get)
            
            # Tentukan zona berdasarkan jarak terdekat
            if min_distance <= self.obstacle_zones.get('critical', 200):
                zone = 'critical'
            elif min_distance <= self.obstacle_zones.get('near', 400):
                zone = 'near'
            elif min_distance <= self.obstacle_zones.get('medium', 800):
                zone = 'medium'
            elif min_distance <= self.obstacle_zones.get('far', 1000):
                zone = 'far'
            else:
                zone = 'clear'
            
            result = {
                'zone': zone,
                'min_distance': min_distance,
                'closest_direction': closest_direction,
                'distances': distances
            }
            
            # VALIDASI OUTPUT
            if result['zone'] not in self.speed_factors:
                print(f"WARNING: Invalid zone '{result['zone']}' detected, forcing to 'critical'")
                result['zone'] = 'critical'
            
            return result
            
        except Exception as e:
            print(f"Error in analyze_obstacle_zones: {e}")
            # Return safe default
            return {
                'zone': 'critical',
                'min_distance': 200,
                'closest_direction': 'front',
                'distances': {'front': 200, 'left': 200, 'right': 200}
            }

    
    def calculate_gradual_speed(self, base_speed, obstacle_analysis, uwb_distances):
        """Hitung kecepatan dengan gradual reduction dan error handling lengkap"""
        # PASTIKAN obstacle_analysis SELALU ADA
        if obstacle_analysis is None:
            print("WARNING: obstacle_analysis is None, creating default")
            obstacle_analysis = {
                'zone': 'critical',
                'min_distance': 200,
                'closest_direction': 'front',
                'distances': {'front': 200, 'left': 200, 'right': 200}
            }
        
        zone = obstacle_analysis.get('zone', 'critical')
        
        # Validasi zone
        if zone not in self.speed_factors:
            print(f"WARNING: Invalid zone '{zone}', using 'critical' as fallback")
            zone = 'critical'
        
        if zone == 'clear':
            # Gunakan incremental speed system normal
            try:
                lidar_status = {
                    'front': {'distance': 2000}, 
                    'left': {'distance': 2000}, 
                    'right': {'distance': 2000}, 
                    'critical_danger': False, 
                    'danger_zone': False, 
                    'waiting_for_dynamic': False
                }
                return self.calculate_target_speed(uwb_distances, lidar_status)
            except Exception as e:
                print(f"Error in calculate_target_speed: {e}")
                return self.speed

        # Gradual speed reduction berdasarkan zona dengan error handling
        try:
            speed_factor = self.speed_factors.get(zone, 0.1)
            gradual_speed = int(base_speed * speed_factor)
            
            # Reset speed boost jika ada obstacle
            if zone != 'clear':
                self.consecutive_clear_count = max(0, self.consecutive_clear_count - 1)
                if zone in ['near', 'critical']:
                    self.speed_boost_active = False
            
            return max(0, gradual_speed)
            
        except Exception as e:
            print(f"Error calculating gradual speed for zone '{zone}': {e}")
            return self.min_speed // 2

    def move_forward_gradual(self, uwb_distances, lidar_status):
        """Move forward dengan gradual obstacle response"""
        # PASTIKAN analisis obstacle selalu dilakukan
        try:
            obstacle_analysis = self.analyze_obstacle_zones(lidar_status)
        except Exception as e:
            print(f"Error in analyze_obstacle_zones: {e}")
            # Default fallback analysis
            obstacle_analysis = {
                'zone': 'critical',
                'min_distance': 200,
                'closest_direction': 'front',
                'distances': {'front': 200, 'left': 200, 'right': 200}
            }
        
        # Quick start jika robot baru mulai bergerak
        if self.actual_speed == 0 and obstacle_analysis['zone'] == 'clear':
            print("QUICK START MODE - Immediate acceleration")
            self.quick_start_forward(self.speed)
            return
        
        # Hitung kecepatan gradual
        target_speed = self.calculate_gradual_speed(self.speed, obstacle_analysis, uwb_distances)
        
        # Update actual speed
        current_speed = self.update_speed_gradually(target_speed)
        
        if current_speed > 0:
            self.move(current_speed, -current_speed, smooth=False)
            self.current_direction = "FORWARD"
            
            print(f"Moving FORWARD ({obstacle_analysis['zone'].upper()}): L={current_speed}, R={-current_speed}")
            print(f"Zone: {obstacle_analysis['zone']} | Distance: {obstacle_analysis['min_distance']:.0f}mm")
        else:
            self.emergency_brake()

    def update_speed_gradually(self, target_speed):
        """Update kecepatan secara bertahap menuju target speed"""
        current_time = time.time()
        
        # Check if enough time has passed for speed update
        if current_time - self.last_speed_update < self.acceleration_delay:
            return self.actual_speed
        
        # Calculate speed difference
        speed_diff = target_speed - self.actual_speed
        
        if abs(speed_diff) <= self.speed_increment:
            # Close enough, set to target
            self.actual_speed = target_speed
        elif speed_diff > 0:
            # Need to accelerate
            self.actual_speed += self.speed_increment
        else:
            # Need to decelerate (faster for safety)
            self.actual_speed -= self.speed_decrement
        
        # Clamp to limits
        self.actual_speed = max(0, min(self.actual_speed, self.max_speed))
        
        self.last_speed_update = current_time
        return self.actual_speed
    
    def move(self, left_speed, right_speed, smooth=True):
        """Basic move method with smooth transitions"""
        if smooth:
            self.smooth_transition(left_speed, right_speed, steps=3)
        else:
            self.left_motor.send_rpm(1, int(left_speed))
            self.right_motor.send_rpm(1, int(right_speed))
            self.current_left_speed = left_speed
            self.current_right_speed = right_speed
    
    def smooth_transition(self, target_left, target_right, steps=5):
        """Transisi smooth untuk gerakan halus"""
        left_diff = (target_left - self.current_left_speed) / steps
        right_diff = (target_right - self.current_right_speed) / steps
        
        for i in range(steps):
            new_left = self.current_left_speed + (left_diff * (i + 1))
            new_right = self.current_right_speed + (right_diff * (i + 1))
            
            self.left_motor.send_rpm(1, int(new_left))
            self.right_motor.send_rpm(1, int(new_right))
            
            time.sleep(0.02)
        
        self.current_left_speed = target_left
        self.current_right_speed = target_right

    def turn_left(self, speed=None, smooth=True):
        """Turn left dengan kecepatan lebih pelan dan smooth"""
        if speed is None:
            speed = self.rotation_speed
        
        # Reset incremental speed system
        self.consecutive_clear_count = 0
        self.speed_boost_active = False
        self.actual_speed = self.min_speed
        
        self.move(-speed, -speed, smooth=smooth)
        self.current_direction = "LEFT"
        print(f"Turning LEFT: L={-speed}, R={-speed}")
    
    def turn_right(self, speed=None, smooth=True):
        """Turn right dengan kecepatan lebih pelan dan smooth"""
        if speed is None:
            speed = self.rotation_speed
        
        # Reset incremental speed system
        self.consecutive_clear_count = 0
        self.speed_boost_active = False
        self.actual_speed = self.min_speed
        
        self.move(speed, speed, smooth=smooth)
        self.current_direction = "RIGHT"
        print(f"Turning RIGHT: L={speed}, R={speed}")
    
    def turn_left_forward(self, speed=None, turn_ratio=0.2, smooth=True):
        """Turn left forward dengan ratio untuk gerakan halus"""
        if speed is None:
            speed = self.gentle_turn_speed
        
        left_speed = speed * (1.0 - turn_ratio)
        right_speed = -speed
        self.move(left_speed, right_speed, smooth=smooth)
        self.current_direction = "LEFT_FORWARD"
        print(f"Turning LEFT FORWARD: L={left_speed}, R={right_speed}")
    
    def turn_right_forward(self, speed=None, turn_ratio=0.2, smooth=True):
        """Turn right forward dengan ratio untuk gerakan halus"""
        if speed is None:
            speed = self.gentle_turn_speed
        
        left_speed = speed
        right_speed = -speed * (1.0 - turn_ratio)
        self.move(left_speed, right_speed, smooth=smooth)
        self.current_direction = "RIGHT_FORWARD"
        print(f"Turning RIGHT FORWARD: L={left_speed}, R={right_speed}")

    def turn_around_to_target(self, target_direction, speed=None):
        """Execute 180-degree turn to face target behind"""
        if speed is None:
            speed = self.rotation_speed
        
        print(f"TARGET BEHIND - EXECUTING 180° TURN (target at {target_direction:.1f}°)")
        
        # Reset speed system
        self.consecutive_clear_count = 0
        self.speed_boost_active = False
        self.actual_speed = self.min_speed
        
        # Tentukan arah putar yang lebih efisien
        if 135 <= target_direction <= 180:
            print("Turning RIGHT to face target behind")
            self.move(speed, speed, smooth=True)
            self.current_direction = "TURN_AROUND_RIGHT"
        elif 180 < target_direction <= 225:
            print("Turning LEFT to face target behind")
            self.move(-speed, -speed, smooth=True)
            self.current_direction = "TURN_AROUND_LEFT"
        else:
            print("Turning RIGHT to face target behind (default)")
            self.move(speed, speed, smooth=True)
            self.current_direction = "TURN_AROUND_RIGHT"

    def is_facing_target(self, target_direction, tolerance=30):
        """Check if robot is roughly facing the target"""
        if target_direction is None:
            return False
        
        # Robot dianggap menghadap target jika target dalam rentang depan
        return (target_direction <= tolerance) or (target_direction >= 360 - tolerance)

    def handle_obstacle_avoidance_gradual(self, lidar, target_in_view=False):
        """Enhanced obstacle avoidance dengan gradual response"""
        status = lidar.get_obstacle_status()
        
        if not status['data_valid']:
            if status['scan_age'] > 0.01:
                print(f"LIDAR data too old: {status['scan_age']:.2f} seconds")
            return False
        
        # Target proximity override
        if target_in_view and lidar.target_distance and lidar.target_distance < STOP_THRESHOLD * 10:
            if not status['critical_danger'] and not status['waiting_for_dynamic']:
                print("Target in close proximity, ignoring non-critical static obstacles")
                return False

        # Dynamic objects - immediate response
        if status['waiting_for_dynamic']:
            print("DYNAMIC OBJECT IN PATH - GRADUAL STOP")
            self.stop(smooth=True)
            self.obstacle_avoidance_active = True
            return True

        # Critical danger - immediate but smooth response
        if status['critical_danger']:
            print("CRITICAL DANGER DETECTED - SMOOTH EMERGENCY STOP")
            self.stop(smooth=True)
            self.obstacle_avoidance_active = True
            return True

        # Gradual obstacle avoidance
        obstacle_analysis = self.analyze_obstacle_zones(status)
        zone = obstacle_analysis.get('zone', 'clear')
        
        if zone in ['near', 'critical']:
            safe_direction = lidar.get_safe_direction()
            print(f"Gradual avoidance needed - Zone: {zone}")
            
            if safe_direction == "STOP" or safe_direction == "STOP_DYNAMIC":
                print("GRADUAL STOP for obstacle clearance")
                self.stop(smooth=True)
                self.obstacle_avoidance_active = True
                return True

            elif safe_direction == "LEFT":
                print("Gradual avoidance: Gentle LEFT turn")
                speed_factor = self.speed_factors.get(zone, 0.1)
                turn_speed = max(10, int(self.rotation_speed * speed_factor))
                self.turn_left(turn_speed, smooth=True)
                self.obstacle_avoidance_active = True
                return True

            elif safe_direction == "RIGHT":
                print("Gradual avoidance: Gentle RIGHT turn")
                speed_factor = self.speed_factors.get(zone, 0.1)
                turn_speed = max(10, int(self.rotation_speed * speed_factor))
                self.turn_right(turn_speed, smooth=True)
                self.obstacle_avoidance_active = True
                return True
        
        # Side obstacle checks
        try:
            if status['left']['distance'] < DANGER_THRESHOLD:
                print("Left side blocked - Gradual right adjustment")
                turn_speed = int(self.gentle_turn_speed * 0.7)
                self.turn_right_forward(speed=turn_speed, turn_ratio=0.3, smooth=True)
                self.obstacle_avoidance_active = True
                return True
            
            elif status['right']['distance'] < DANGER_THRESHOLD:
                print("Right side blocked - Gradual left adjustment")
                turn_speed = int(self.gentle_turn_speed * 0.7)
                self.turn_left_forward(speed=turn_speed, turn_ratio=0.3, smooth=True)
                self.obstacle_avoidance_active = True
                return True
        except (KeyError, TypeError) as e:
            print(f"Error accessing obstacle distances: {e}")
            self.stop(smooth=True)
            return True
        
        self.obstacle_avoidance_active = False
        self.waiting_for_dynamic_object = False
        return False

    def process_uwb_control_gradual(self, uwb_distances, lidar_status):
        """Enhanced UWB control dengan deteksi target di belakang"""
        A0, A1, A2 = uwb_distances['A0'], uwb_distances['A1'], uwb_distances['A2']
        
        print("\n--- UWB Distances (cm) ---")
        print(f"A0: {A0:.2f} | A1: {A1:.2f} | A2: {A2:.2f}")
        
        # Target reached check
        if A0 <= self.stop_threshold:
            print(f"Target reached (A0 <= {self.stop_threshold} cm). Gradual stopping.")
            self.stop(smooth=True)
            return
        
        diff = A2 - A1
        
        if abs(diff) < 15:
            # Target centered - GRADUAL FORWARD
            print("Move Forward - Target centered (GRADUAL MODE)")
            self.move_forward_gradual(uwb_distances, lidar_status)
            
        elif A0 < 150:  # Close to target - precise movement
            self.consecutive_clear_count = 0
            self.speed_boost_active = False
            
            if diff < 0:  # Target to the right
                turn_ratio = min(0.4, abs(diff) / 80.0)
                print(f"Close target right turn (GRADUAL ratio: {turn_ratio:.2f})")
                self.turn_right_forward(speed=self.gentle_turn_speed, turn_ratio=turn_ratio, smooth=True)
            else:  # Target to the left
                turn_ratio = min(0.4, abs(diff) / 80.0)
                print(f"Close target left turn (GRADUAL ratio: {turn_ratio:.2f})")
                self.turn_left_forward(speed=self.gentle_turn_speed, turn_ratio=turn_ratio, smooth=True)
                
        else:  # Navigate towards target
            if diff < 0:  # Target to the right
                print("Turn right towards target")
                self.turn_right_forward(speed=self.gentle_turn_speed, turn_ratio=0.25, smooth=True)
            else:  # Target to the left
                print("Turn left towards target")
                self.turn_left_forward(speed=self.gentle_turn_speed, turn_ratio=0.25, smooth=True)

    def calculate_target_speed(self, uwb_distances, lidar_status):
        """Hitung kecepatan target berdasarkan kondisi path dan obstacle"""
        A0, A1, A2 = uwb_distances['A0'], uwb_distances['A1'], uwb_distances['A2']
        
        # Check if path is straight (target centered)
        angle_diff = abs(A2 - A1)
        is_straight_path = angle_diff < self.straight_path_threshold
        
        # Check obstacle distances
        min_obstacle_distance = min(
            lidar_status['front']['distance'],
            lidar_status['left']['distance'],
            lidar_status['right']['distance']
        )
        
        # Check if path is clear
        is_path_clear = (
            min_obstacle_distance > self.clear_distance_threshold and
            not lidar_status['critical_danger'] and
            not lidar_status['danger_zone'] and
            not lidar_status['waiting_for_dynamic']
        )
        
        # Determine target speed
        if not is_path_clear or not is_straight_path:
            self.consecutive_clear_count = 0
            self.speed_boost_active = False
            
            if lidar_status['critical_danger']:
                return 0
            elif lidar_status['danger_zone']:
                return self.min_speed // 2
            elif min_obstacle_distance < DANGER_THRESHOLD:
                return self.min_speed
            else:
                return self.speed
        else:
            # Path is straight and clear
            self.consecutive_clear_count += 1
            
            if self.consecutive_clear_count >= self.min_clear_count:
                self.speed_boost_active = True
                
                # Calculate speed based on distance and clear path duration
                distance_factor = min(1.0, min_obstacle_distance / self.speed_boost_distance)
                clear_factor = min(1.0, self.consecutive_clear_count / (self.min_clear_count * 3))
                
                # Progressive speed increase
                speed_multiplier = 1.0 + (distance_factor * clear_factor)
                target_speed = int(self.speed * speed_multiplier)
                
                # Clamp to max speed
                return min(target_speed, self.max_speed)
            else:
                return self.speed
    
    def emergency_brake(self, brake_intensity=0.8, brake_duration=0.2):
        """Emergency braking dengan counter-rotation untuk mengerem cepat"""
        print("EMERGENCY BRAKING ACTIVATED!")
        
        # Hitung kecepatan pengereman berdasarkan kecepatan saat ini
        current_left = self.current_left_speed
        current_right = self.current_right_speed
        
        # Counter-rotation untuk mengerem
        brake_left = -abs(current_left) * brake_intensity
        brake_right = -abs(current_right) * brake_intensity
        
        # Apply emergency brake
        self.left_motor.send_rpm(1, int(brake_left))
        self.right_motor.send_rpm(1, int(brake_right))
        
        # Tahan sebentar untuk efek pengereman
        time.sleep(brake_duration)
        
        # Kemudian stop
        self.left_motor.send_rpm(1, 0)
        self.right_motor.send_rpm(1, 0)
        
        # Update status
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.actual_speed = 0

    def quick_start_forward(self, target_speed=None):
        """Quick start untuk langsung bergerak tanpa gradual acceleration"""
        if target_speed is None:
            target_speed = self.speed
        
        print(f"QUICK START: Immediate forward at speed {target_speed}")
        
        # Set speed langsung tanpa gradual
        self.actual_speed = target_speed
        self.current_target_speed = target_speed
        
        # Move immediately
        self.move(target_speed, -target_speed, smooth=False)
        self.current_direction = "QUICK_FORWARD"
    
    def stop(self, smooth=True):
        """Stop dengan reset incremental speed system"""
        # Reset incremental speed system
        self.consecutive_clear_count = 0
        self.speed_boost_active = False
        self.actual_speed = 0
        self.gradual_turn_active = False
        
        if smooth:
            self.smooth_transition(0, 0, steps=5)
        else:
            self.move(0, 0, smooth=False)
        self.current_direction = "STOP"
        print("STOPPING (GRADUAL SYSTEM RESET)")
    
    def process_control(self, uwb_distances, lidar):
        """Main control - pilih mode independent atau gradual"""
        
        if self.independent_control_enabled:
            self.process_control_independent(uwb_distances, lidar)
        else:
            # Fallback ke control lama
            # Critical obstacle check
            if self.check_critical_obstacles(lidar):
                return
            
            # Get sensor status
            lidar_status = lidar.get_obstacle_status()
            
            # Target proximity check
            target_in_view = (
                uwb_distances['A0'] < 100 and
                abs(uwb_distances['A1'] - uwb_distances['A2']) < 30
            )
            
            # Gradual obstacle avoidance
            obstacle_action_needed = self.handle_obstacle_avoidance_gradual(lidar, target_in_view)
            
            # Use gradual control if no obstacles
            if not obstacle_action_needed and not self.emergency_stop and not self.waiting_for_dynamic_object:
                self.process_uwb_control_gradual(uwb_distances, lidar_status)
    
    def check_critical_obstacles(self, lidar):
        """Check critical obstacles dengan emergency braking"""
        status = lidar.get_obstacle_status()
        
        if status['waiting_for_dynamic']:
            print("DYNAMIC OBJECT - EMERGENCY BRAKE")
            self.emergency_brake(brake_intensity=0.6, brake_duration=0.15)
            self.waiting_for_dynamic_object = True
            return True
        
        if status['data_valid'] and status['critical_danger']:
            print("CRITICAL DANGER - EMERGENCY BRAKE") 
            self.emergency_brake(brake_intensity=0.8, brake_duration=0.2)
            self.emergency_stop = True
            return True
        
        self.emergency_stop = False
        self.waiting_for_dynamic_object = False
        return False

    def handle_obstacle_avoidance(self, lidar, target_in_view=False):
        """Wrapper untuk gradual obstacle avoidance"""
        return self.handle_obstacle_avoidance_gradual(lidar, target_in_view)
    
    def calculate_independent_wheel_speeds(self, target_distance, target_angle_error, base_speed):
        """Hitung kecepatan roda independen berdasarkan target dan error angle"""
        
        # Normalisasi angle error (-180 to 180)
        while target_angle_error > 180:
            target_angle_error -= 360
        while target_angle_error < -180:
            target_angle_error += 360
        
        # Base speed adjustment berdasarkan jarak
        if target_distance < 100:  # Sangat dekat
            speed_factor = 0.3
        elif target_distance < 200:  # Dekat
            speed_factor = 0.6
        else:  # Normal
            speed_factor = 1.0
        
        adjusted_base_speed = base_speed * speed_factor
        
        # Hitung steering correction berdasarkan angle error
        steering_correction = target_angle_error * self.steering_gain
        steering_correction = max(-self.max_speed_diff, min(self.max_speed_diff, steering_correction))
        
        # Hitung kecepatan target untuk masing-masing roda
        if target_angle_error > 5:  # Target di kiri
            # Perlambat roda kiri, percepat roda kanan
            left_speed = adjusted_base_speed - abs(steering_correction * 0.7)
            right_speed = -(adjusted_base_speed + abs(steering_correction * 0.3))
            
        elif target_angle_error < -5:  # Target di kanan
            # Percepat roda kiri, perlambat roda kanan
            left_speed = adjusted_base_speed + abs(steering_correction * 0.3)
            right_speed = -(adjusted_base_speed - abs(steering_correction * 0.7))
            
        else:  # Target di depan (centered)
            # Kedua roda sama, maju lurus
            left_speed = adjusted_base_speed
            right_speed = -adjusted_base_speed
        
        # Apply forward bias untuk tetap maju
        left_speed *= self.forward_bias
        right_speed *= self.forward_bias
        
        # Clamp ke batas maksimal
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))
        
        return int(left_speed), int(right_speed)

    def smooth_speed_transition(self, target_left, target_right):
        """Transisi smooth ke target speed tanpa stopping"""
        
        # Hitung perubahan yang diperlukan
        left_diff = target_left - self.current_left_speed
        right_diff = target_right - self.current_right_speed
        
        # Batasi rate perubahan untuk smooth transition
        if abs(left_diff) > self.speed_adjustment_rate:
            if left_diff > 0:
                new_left = self.current_left_speed + self.speed_adjustment_rate
            else:
                new_left = self.current_left_speed - self.speed_adjustment_rate
        else:
            new_left = target_left
            
        if abs(right_diff) > self.speed_adjustment_rate:
            if right_diff > 0:
                new_right = self.current_right_speed + self.speed_adjustment_rate
            else:
                new_right = self.current_right_speed - self.speed_adjustment_rate
        else:
            new_right = target_right
        
        # Update current speeds
        self.current_left_speed = new_left
        self.current_right_speed = new_right
        
        # Send ke motor langsung tanpa delay
        self.left_motor.send_rpm(1, int(new_left))
        self.right_motor.send_rpm(1, int(new_right))
        
        return new_left, new_right

    def process_uwb_control_independent(self, uwb_distances, lidar_status):
        """UWB control dengan independent wheel system - NO STOPPING"""
        A0, A1, A2 = uwb_distances['A0'], uwb_distances['A1'], uwb_distances['A2']
        
        print(f"\n--- Independent Control Mode ---")
        print(f"A0: {A0:.1f}cm | A1: {A1:.1f}cm | A2: {A2:.1f}cm")
        
        # PASTIKAN obstacle_analysis SELALU DI-ASSIGN
        obstacle_analysis = self.analyze_obstacle_zones(lidar_status)
        
        # Target reached check
        if A0 <= self.stop_threshold:
            print(f"Target reached - Gradual stop")
            target_left, target_right = 0, 0
        else:
            # Hitung angle error untuk steering
            angle_error = A2 - A1
            
            # Tentukan base speed berdasarkan obstacle
            base_speed = self.calculate_gradual_speed(self.speed, obstacle_analysis, uwb_distances)
            
            # Hitung independent wheel speeds
            target_left, target_right = self.calculate_independent_wheel_speeds(
                target_distance=A0 * 10,
                target_angle_error=angle_error,
                base_speed=base_speed
            )
            
            print(f"Angle Error: {angle_error:.1f} | Base Speed: {base_speed}")
        
        # Smooth transition ke target speeds
        actual_left, actual_right = self.smooth_speed_transition(target_left, target_right)
        
        # Determine direction for logging
        if abs(actual_left - abs(actual_right)) < 5:
            direction = "FORWARD"
        elif actual_left > abs(actual_right):
            direction = "TURN_LEFT"
        else:
            direction = "TURN_RIGHT"
        
        self.current_direction = f"INDEPENDENT_{direction}"
        
        print(f"Independent Control: L={actual_left:.0f}, R={actual_right:.0f} [{direction}]")
        print(f"Distance: {A0:.1f}cm | Zone: {obstacle_analysis['zone']}")

    def handle_obstacle_avoidance_independent(self, lidar, current_left_speed, current_right_speed):
        """Obstacle avoidance dengan independent wheel adjustment"""
        status = lidar.get_obstacle_status()
        
        if not status['data_valid']:
            return current_left_speed, current_right_speed
        
        # Critical danger - emergency adjustment
        if status['critical_danger']:
            print("CRITICAL DANGER - Independent emergency adjustment")
            return 0, 0  # Emergency stop
        
        # Dynamic object - pause tapi keep position
        if status['waiting_for_dynamic']:
            print("DYNAMIC OBJECT - Maintain position")
            return 0, 0
        
        # Gradual obstacle avoidance dengan independent adjustment
        obstacle_analysis = self.analyze_obstacle_zones(status)
        zone = obstacle_analysis.get('zone', 'clear')
        
        if zone in ['near', 'critical']:
            safe_direction = lidar.get_safe_direction()
            
            if safe_direction == "LEFT":
                # Adjust untuk belok kiri dengan mengurangi speed roda kiri
                print(f"Independent avoidance: Adjust LEFT (zone: {zone})")
                adjusted_left = current_left_speed * 0.3
                adjusted_right = current_right_speed * 1.2
                
            elif safe_direction == "RIGHT":
                # Adjust untuk belok kanan dengan mengurangi speed roda kanan
                print(f"Independent avoidance: Adjust RIGHT (zone: {zone})")
                adjusted_left = current_left_speed * 1.2
                adjusted_right = current_right_speed * 0.3
                
            else:  # STOP needed
                print(f"Independent avoidance: Stop required (zone: {zone})")
                adjusted_left = 0
                adjusted_right = 0
            
            # Clamp values
            adjusted_left = max(-self.max_speed, min(self.max_speed, adjusted_left))
            adjusted_right = max(-self.max_speed, min(self.max_speed, adjusted_right))
            
            return int(adjusted_left), int(adjusted_right)
        
        # No obstacle adjustment needed
        return current_left_speed, current_right_speed

    def process_control_independent(self, uwb_distances, lidar):
        """Main control dengan independent wheel system"""
        
        # Critical obstacle check
        if self.check_critical_obstacles(lidar):
            return
        
        # Get current sensor status
        try:
            lidar_status = lidar.get_obstacle_status()
        except Exception as e:
            print(f"Error getting lidar status: {e}")
            # Default safe status
            lidar_status = {
                'front': {'distance': 200},
                'left': {'distance': 200},
                'right': {'distance': 200},
                'critical_danger': True
            }
        
        # Process UWB control untuk base speeds
        self.process_uwb_control_independent(uwb_distances, lidar_status)
        
        # Apply obstacle avoidance adjustments
        adjusted_left, adjusted_right = self.handle_obstacle_avoidance_independent(
            lidar, self.target_left_speed, self.target_right_speed
        )
        
        # Final speed adjustment
        if adjusted_left != self.target_left_speed or adjusted_right != self.target_right_speed:
            self.target_left_speed = adjusted_left
            self.target_right_speed = adjusted_right
            
            # Apply immediately
            self.smooth_speed_transition(adjusted_left, adjusted_right)


# FollowingRobotNode Class
class FollowingRobotNode(Node):
    def __init__(self):
        super().__init__('following_robot_node')

        # Get the dynamic IP address from Wi-Fi interface
        self.udp_ip = get_ip_from_wifi()
        if not self.udp_ip:
            raise ValueError("Unable to retrieve IP address from Wi-Fi interface.")

        # Optionally, modify the last part of the IP address
        target_last_digit = 128
        self.udp_ip = get_ip_from_subnet(self.udp_ip, target_last_digit)

        print(f"Robot will use dynamic IP: {self.udp_ip}")
        
        self.udp_port = 5005
        
        # Configuration
        self.r_wheel_port = "/dev/ttyRS485-1"
        self.l_wheel_port = "/dev/ttyRS485-2"
        
        # Performance configuration
        self.control_frequency = CONTROL_FREQUENCY
        self.lidar_skip_frames = LIDAR_SKIP_FRAMES
        self.uwb_timeout = UWB_TIMEOUT
        self.max_loop_time = MAX_LOOP_TIME
        
        # Frame counter
        self.lidar_frame_counter = 0
        self.last_control_update = 0
        self.last_loop_time = 0
        self._processing_lidar = False
        
        # Initialize components
        self.uwb_tracker = UWBTracker()
        self.lidar = LidarProcessor()
        self.controller = RobotController(self.r_wheel_port, self.l_wheel_port)
        
        # Berikan referensi untuk parent access
        self.controller.parent = self
        
        # Callback group untuk LIDAR
        self.lidar_cb_group = ReentrantCallbackGroup()
        
        # LIDAR subscription
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=LIDAR_BUFFER_SIZE
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback_optimized,
            qos_profile=lidar_qos,
            callback_group=self.lidar_cb_group
        )
        
        # UWB setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UWB_BUFFER_SIZE)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.settimeout(self.uwb_timeout)
        
        # UWB data
        self.raw_uwb_distances = {'A0': 1000, 'A1': 1000, 'A2': 1000}
        self.corrected_uwb_distances = {'A0': 1000, 'A1': 1000, 'A2': 1000}
        
        # Status control
        self.running = True
        self.last_uwb_update = 0
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )

        # Quick start flag
        self.quick_start_enabled = True
        self.first_movement = True
        
        # Performance monitoring
        self.last_lidar_time = time.time()
        self.lidar_update_count = 0
        self.lidar_update_rate = 0
        
        # Log statistics periodically
        self.stats_timer = self.create_timer(2.0, self.log_statistics)
        
        self.get_logger().info(f'Enhanced Safety Following Robot Node started with {self.control_frequency}Hz control frequency')
        self.get_logger().info('Features: Dynamic object detection, Static object avoidance, Industrial safety protocols')

        # Turn on GPIO 17 to indicate the program is running
        gpio_pin_17.on()

    def scan_callback_optimized(self, msg):
        """Callback LIDAR yang dioptimasi dengan frame skipping"""
        if hasattr(self, '_processing_lidar') and self._processing_lidar:
            return
        
        self._processing_lidar = True
        try:
            if self.lidar_frame_counter % self.lidar_skip_frames == 0:
                self.lidar.process_scan(msg)
            self.lidar_frame_counter += 1
            self.lidar_update_count += 1
        finally:
            self._processing_lidar = False

    def check_critical_obstacles_fast(self):
        """Immediate critical check tanpa complex processing"""
        # Check langsung dari data terakhir
        if hasattr(self.lidar, 'scan_data') and self.lidar.scan_data:
            # Quick scan untuk critical zones (330-30 degrees)
            critical_angles = list(range(330, 360)) + list(range(0, 31))
            
            for angle in critical_angles:
                if angle in self.lidar.scan_data:
                    distance = self.lidar.scan_data[angle]
                    if distance < CRITICAL_DANGER_THRESHOLD:
                        self.controller.emergency_brake()
                        return True
        
        return False

    def process_uwb_data_fast(self):
        """Process UWB data dengan timeout minimal"""
        try:
            self.sock.setblocking(False)
            ready = select.select([self.sock], [], [], self.uwb_timeout)
            if ready[0]:
                data, addr = self.sock.recvfrom(UWB_BUFFER_SIZE)
                parts = data.decode().split(",")
                
                if len(parts) >= 3:
                    self.raw_uwb_distances = {
                        'A0': float(parts[0]),
                        'A1': float(parts[1]),
                        'A2': float(parts[2])
                    }
                    
                    self.corrected_uwb_distances = self.uwb_tracker.apply_bias_correction(
                        self.raw_uwb_distances
                    )
                    self.last_uwb_update = time.time()
                    return True
                    
        except (socket.error, ValueError, IndexError):
            pass
        
        return False

    def execute_control_decision(self):
        """Execute control dengan independent wheel support"""
        uwb_data_age = time.time() - self.last_uwb_update
        uwb_data_valid = uwb_data_age < 0.5
        
        if uwb_data_valid:
            # Set target info untuk lidar
            target_direction, target_distance = self.uwb_tracker.estimate_target_direction(
                self.corrected_uwb_distances
            )
            self.lidar.set_target_info(target_direction, target_distance)
            
            # Quick start untuk gerakan pertama (opsional)
            if self.first_movement and self.quick_start_enabled:
                A0 = self.corrected_uwb_distances['A0']
                if A0 > 100:
                    print("FIRST MOVEMENT - QUICK START ENABLED")
                    self.controller.quick_start_forward()
                    self.first_movement = False
            
            # Use independent/gradual control
            self.controller.process_control(self.corrected_uwb_distances, self.lidar)
        else:
            self.controller.handle_obstacle_avoidance(self.lidar)
    
    def log_statistics(self):
        """Log statistics about sensor update rates and safety status"""
        current_time = time.time()
        update_interval = current_time - self.last_lidar_time
        if update_interval > 0:
            self.lidar_update_rate = self.lidar_update_count / update_interval
            self.last_lidar_time = current_time
            self.lidar_update_count = 0
            
            # Enhanced logging with safety information
            status = self.lidar.get_obstacle_status()
            self.get_logger().info(
                f"LIDAR: {self.lidar_update_rate:.2f}Hz | "
                f"UWB age: {time.time() - self.last_uwb_update:.2f}s | "
                f"Dynamic objs: {status['dynamic_objects']} | "
                f"Static objs: {status['static_objects']} | "
                f"Waiting for dynamic: {status['waiting_for_dynamic']}"
            )
    
    def control_loop(self):
        """Control loop yang dioptimasi untuk real-time dengan enhanced safety"""
        if not self.running:
            return
        start_time = time.time()
        try:
            # PRIORITAS 1: Critical obstacle check
            if self.check_critical_obstacles_fast():
                return
            
            # PRIORITAS 2: UWB data processing
            uwb_updated = self.process_uwb_data_fast()
            
            # PRIORITAS 3: Control decision
            if uwb_updated or time.time() - self.last_control_update > 0.05:
                self.execute_control_decision()
                self.last_control_update = time.time()
        except Exception as e:
            gpio_pin_27.on()
            self.get_logger().error(f"Control loop error: {e}")
        
        # Monitor performa loop
        loop_time = time.time() - start_time
        if loop_time > 0.015:
            self.get_logger().warn(f"Slow control loop: {loop_time * 1000:.1f}ms")

    def stop(self):
        """Stop the robot and clean up"""
        self.running = False
        self.controller.stop()
        self.sock.close()
        gpio_pin_17.off()
        gpio_pin_27.off()
        self.get_logger().info("Enhanced Safety Robot systems shut down.")

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node
    node = FollowingRobotNode()
    
    # Create multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # Run until interrupted
        executor.spin()
    except KeyboardInterrupt:
        print("\nUser interrupted. Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
