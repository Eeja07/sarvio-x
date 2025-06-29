#!/usr/bin/env python3
"""
Tello Drone Control System - Full Integrated Version
5-Thread Architecture with Flask-SocketIO Web Interface

Features:
- 5-Thread architecture for optimal performance
- Human detection with YOLO + MediaPipe
- Red color detection with ROI
- Smart auto-screenshot system
- Time-based video recording
- Autonomous behavior
- Flask-SocketIO web interface
- Dual control: Keyboard/Joystick + Web interface
- Real-time video streaming
- Enhanced Media Gallery

Run: python tello_integrated.py
Web Interface: http://localhost:5173 (React frontend)
Backend API: http://localhost:5000
"""

import threading
import time
import cv2
import queue
import pygame
import numpy as np
import os
import sys
import base64
import mimetypes
from pathlib import Path
from datetime import datetime
from collections import deque

# Drone and AI imports
from djitellopy import Tello
from ultralytics import YOLO
import mediapipe as mp

# Flask and web interface imports
from flask import Flask, Response, render_template_string, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# Display settings
FPS = 60
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

# ROI parameters (Region of Interest)
ROI_X = WINDOW_WIDTH // 4
ROI_Y = 5
ROI_WIDTH = WINDOW_WIDTH // 2
ROI_HEIGHT = WINDOW_HEIGHT // 3

# Thread count
THREAD_COUNT = 5

# Drone control settings
SPEED = 50
ROTATE = 80

# Screenshot settings
SCREENSHOT_INTERVAL = 3
COUNTDOWN_DURATION = 3.0

# Recording settings
RECORDING_FPS = 30
RECORDING_BUFFER_SIZE = 100

# Directories
SCREENSHOTS_DIR = "screenshots"
RECORDINGS_DIR = "recordings"

# Detection settings
RED_DETECTION_THRESHOLD = 10000
YOLO_CONFIDENCE_THRESHOLD = 0.5
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.3
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.3

# Color detection ranges (HSV)
LOWER_RED1 = [0, 120, 130]
UPPER_RED1 = [30, 255, 255]
LOWER_RED2 = [230, 150, 130]
UPPER_RED2 = [255, 255, 255]

# Performance monitoring
FRAME_TIME_BUFFER_SIZE = 30
BATTERY_CHECK_INTERVAL = 10  # seconds

# Thread timing
VIDEO_THREAD_SLEEP = 0.01
DETECTION_THREAD_SLEEP = 0.03
CONTROL_THREAD_SLEEP = 1/30
RECORDING_THREAD_SLEEP = 0.01
AUTONOMOUS_THREAD_SLEEP = 0.1

# =============================================================================
# SHARED STATE MANAGEMENT
# =============================================================================

class SharedState:
    """Thread-safe shared state management"""
    
    def __init__(self):
        # Thread control
        self.running = True
        self.threads = []
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.command_queue = queue.Queue()
        self.screenshot_queue = queue.Queue()
        self.recording_frame_buffer = queue.Queue(maxsize=RECORDING_BUFFER_SIZE)
        
        # Shared data with locks
        self.data_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        # Video frames
        self.current_frame = None
        self.current_processed_frame = None
        self.last_frame = None
        
        # Detection results
        self.current_detection = None
        self.battery_level = 0
        self.fps = 0
        self.humans_count = 0
        self.human_detected = False
        self.screenshot_count = 0
        
        # Tello State Variables
        self.temp = 0
        self.baro = 0
        self.height = 0
        self.tof = 0
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.vgx = 0
        self.vgy = 0
        self.vgz = 0
        self.agx = 0
        self.agy = 0
        self.agz = 0
        
        # Performance monitoring
        self.frame_times = deque(maxlen=FRAME_TIME_BUFFER_SIZE)
        self.last_frame_time = time.time()
        
        # Recording variables
        self.recording = False
        self.video_writer = None
        self.recording_start_time = 0
        self.current_recording_file = None
        self.last_recording_frame_time = 0
        self.frame_skip_counter = 0
        
        # Drone control variables
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.send_rc_control = False
        self.is_connected = False
        self.is_flying = False
        self.disconnect_requested = False
        
        # Screenshot variables
        self.last_screenshot_time = 0
        self.countdown_active = False
        self.countdown_start_time = 0
        self.last_human_detected = False
        
        # Control settings
        self.set_autonomous_behavior = False
        self.detection_enabled = True
        self.auto_screenshot_enabled = True
        self.keyboard_mode = True
        self.joystick_mode = True
        self.toggle_keyboard = True
        self.toggle_joystick = True
        self.speed = SPEED
        self.current_speed_display = SPEED
        
        # Joystick state variables
        self.last_joystick_screenshot_button_state = False
        self.joystick_screenshot_requested = False
        self.last_joystick_recording_button_state = False
        self.last_joystick_detection_toggle_button_state = False
        self.last_joystick_emergency_button_state = False
        
        # Web interface variables
        self.ml_detection_enabled = True
        self.socket_streaming = False
        self.connected_clients = 0
        self.should_stop = False
        self.flight_start_time = None
        
        # Global objects
        self.tello = None
        self.screen = None
        self.joystick = None
        self.yolo_model = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.mp_pose = None
        self.mp_hands = None
        self.pose = None
        self.hands = None

# Global state instance
state = SharedState()

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_pygame():
    """Initialize pygame and create window"""
    try:
        pygame.init()
        pygame.display.set_caption("Tello 5-Thread Control System - SARVIO-X")
        state.screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
        
        # Initialize joystick
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            state.joystick = pygame.joystick.Joystick(0)
            state.joystick.init()
            print(f"Joystick initialized: {state.joystick.get_name()}")
        else:
            print("No joystick detected - using keyboard only")
        
        # Create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
        return True
    except Exception as e:
        print(f"Failed to initialize pygame: {e}")
        return False

def initialize_tello():
    """Initialize Tello drone connection"""
    try:
        state.tello = Tello()
        state.tello.connect()
        state.tello.set_speed(SPEED)
        state.battery_level = state.tello.get_battery()
        state.is_connected = True
        print(f"✅ Tello connected! Battery: {state.battery_level}%")
        
        # Start video stream
        state.tello.streamoff()
        time.sleep(0.5)
        state.tello.streamon()
        
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Tello: {e}")
        state.is_connected = False
        return False

def initialize_ai_models():
    """Initialize AI models for detection"""
    try:
        # Load YOLOv8 model for human detection
        print("Loading YOLOv8 model...")
        state.yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8 model loaded successfully")
        
        # Mediapipe modules for body part detection
        state.mp_drawing = mp.solutions.drawing_utils
        state.mp_drawing_styles = mp.solutions.drawing_styles
        state.mp_pose = mp.solutions.pose
        state.mp_hands = mp.solutions.hands
        
        # Create MediaPipe models
        state.pose = state.mp_pose.Pose(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        state.hands = state.mp_hands.Hands(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            max_num_hands=2
        )
        
        # Initialize current_detection with default values
        state.current_detection = {
            'red_detected': False,
            'mask': None,
            'result': None,
            'roi_mask': None,
            'full_roi_mask': None,
            'pixel_count': 0
        }
        
        return True
    except Exception as e:
        print(f"Failed to initialize AI models: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    try:
        for directory in [SCREENSHOTS_DIR, RECORDINGS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        return True
    except Exception as e:
        print(f"Failed to create directories: {e}")
        return False

def initialize_all_systems():
    """Initialize all systems in correct order"""
    print("Initializing systems...")
    
    if not create_directories():
        return False
        
    if not initialize_pygame():
        return False
        
    if not initialize_tello():
        return False
        
    if not initialize_ai_models():
        return False
    
    print("All systems initialized successfully!")
    return True

def cleanup_systems():
    """Cleanup all systems"""
    try:
        if state.tello and state.is_connected:
            state.tello.streamoff()
            state.tello.end()
        
        if state.pose:
            state.pose.close()
            
        if state.hands:
            state.hands.close()
            
        pygame.quit()
        print("Systems cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# =============================================================================
# DETECTION ALGORITHMS
# =============================================================================

def process_human_detection(frame):
    """Process human detection and return processed frame with detection info"""
    try:
        output_frame = frame.copy()
        
        # YOLOv8 Human Detection
        results = state.yolo_model(frame, verbose=False)

        detected = False
        human_boxes = []

        # Process YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Check if it's a person (class_id = 0 in COCO dataset)
                    if class_id == 0 and confidence > YOLO_CONFIDENCE_THRESHOLD:
                        detected = True

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))

                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Calculate center of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Draw center point
                        cv2.circle(output_frame, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

                        # Add label
                        confidence_percentage = confidence * 100
                        label = f"Human: {confidence_percentage:.0f}%"

                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Process detailed body part detection if human detected
        if detected:
            output_frame = process_body_parts(frame, output_frame)

        return output_frame, detected, len(human_boxes)
    
    except Exception as e:
        print(f"Error in human detection: {e}")
        return frame, False, 0

def process_body_parts(original_frame, output_frame):
    """Process MediaPipe body part detection"""
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        # Process with pose detection
        pose_results = state.pose.process(rgb_frame)

        # Process with hand detection
        hands_results = state.hands.process(rgb_frame)

        # Draw hands if detected
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                state.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    state.mp_hands.HAND_CONNECTIONS,
                    state.mp_drawing_styles.get_default_hand_landmarks_style(),
                    state.mp_drawing_styles.get_default_hand_connections_style()
                )

        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            state.mp_drawing.draw_landmarks(
                output_frame,
                pose_results.pose_landmarks,
                state.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=state.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return output_frame
    
    except Exception as e:
        print(f"Error in body part detection: {e}")
        return output_frame

def detect_red_in_roi(img):
    """Detect red color specifically in ROI area"""
    try:
        if img is None:
            return False, None, None, 0
        
        bgr_img = img
        
        # Extract ROI
        roi = bgr_img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
        
        # Detect red color in ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array(LOWER_RED1)
        upper_red1 = np.array(UPPER_RED1)
        lower_red2 = np.array(LOWER_RED2)
        upper_red2 = np.array(UPPER_RED2)

        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_roi = cv2.bitwise_or(mask1, mask2)
        
        # Check if red color is detected
        red_detected = np.sum(mask_roi) > RED_DETECTION_THRESHOLD 
        pixel_count = np.sum(mask_roi > 0)
        
        # Create full-size mask for visualization
        full_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        full_mask[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH] = mask_roi
        
        return red_detected, mask_roi, full_mask, pixel_count
    
    except Exception as e:
        print(f"Error in red detection ROI: {e}")
        return False, None, None, 0

def detect_red_color(img):
    """Detect red color using OpenCV for full image"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array(LOWER_RED1)
        upper_red1 = np.array(UPPER_RED1)
        lower_red2 = np.array(LOWER_RED2)
        upper_red2 = np.array(UPPER_RED2)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        result = cv2.bitwise_and(img, img, mask=mask)
        return mask, result
    
    except Exception as e:
        print(f"Error in red color detection: {e}")
        return None, None

# =============================================================================
# RECORDING AND SCREENSHOT SYSTEM
# =============================================================================

def start_recording():
    """Start video recording"""
    try:
        if not state.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.current_recording_file = f"tello_flight_{timestamp}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, state.current_recording_file)
            
            # Use consistent 30 FPS with time-based control
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            state.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            if state.video_writer.isOpened():
                state.recording = True
                state.recording_start_time = time.time()
                state.last_recording_frame_time = time.time()
                print(f"🔴 Recording started: {state.current_recording_file}")
                return True
            else:
                print("❌ Failed to start recording - could not open video writer")
                return False
    except Exception as e:
        print(f"Error starting recording: {e}")
        return False

def stop_recording():
    """Stop video recording"""
    try:
        if state.recording and state.video_writer:
            state.recording = False
            state.video_writer.release()
            state.video_writer = None
            
            recording_duration = time.time() - state.recording_start_time
            print(f"⏹️ Recording stopped: {state.current_recording_file}")
            print(f"Duration: {recording_duration:.1f} seconds")
            return True
    except Exception as e:
        print(f"Error stopping recording: {e}")
        return False

def toggle_recording():
    """Toggle recording on/off"""
    if state.recording:
        stop_recording()
    else:
        start_recording()

def save_screenshot(frame, humans_count_param, source="auto"):
    """Save screenshot with timestamp and human count"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_prefix = "manual" if source in ["joystick", "keyboard", "web"] else "auto"
        filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count_param}persons_{state.screenshot_count:04d}.jpg"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving

        # Save the frame
        success = cv2.imwrite(filepath, frame_bgr)
        
        if success:
            state.screenshot_count += 1
            print(f"Screenshot saved ({source}): {filename}")
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False
    
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

def handle_auto_screenshot(output_frame, human_detected_now, humans_count_now):
    """Handle auto screenshot countdown logic"""
    try:
        current_time = time.time()
        
        if human_detected_now and humans_count_now >= 1:
            # Human detected
            if not state.last_human_detected and not state.countdown_active:
                # First time detecting human, start countdown
                state.countdown_active = True
                state.countdown_start_time = current_time
                print(f"Human detected! Starting 3-second countdown...")
            
            # If countdown is active
            if state.countdown_active:
                elapsed_time = current_time - state.countdown_start_time
                
                if elapsed_time >= COUNTDOWN_DURATION:
                    # Countdown finished, take screenshot
                    save_screenshot(output_frame.copy(), humans_count_now, "auto")
                    state.countdown_active = False
                    print("Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if state.countdown_active:
                # Cancel countdown if human disappears
                state.countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        state.last_human_detected = human_detected_now
        
    except Exception as e:
        print(f"Error in auto screenshot handler: {e}")

def request_manual_screenshot(source):
    """Request a manual screenshot"""
    try:
        with state.data_lock:
            if state.current_processed_frame is not None:
                state.screenshot_queue.put((
                    state.current_processed_frame.copy(), 
                    state.humans_count, 
                    source
                ))
                print(f"Manual screenshot requested ({source})")
    except Exception as e:
        print(f"Screenshot request error: {e}")

def process_recording_frame(output_frame):
    """Process frame for recording with time-based control"""
    try:
        if state.recording and state.video_writer and state.video_writer.isOpened():
            current_time = time.time()
            time_since_last_record = current_time - state.last_recording_frame_time
            
            # Save frame exactly every 1/30 second (33.33ms) for consistent 30 FPS
            if time_since_last_record >= (1.0 / 30.0):
                frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                state.video_writer.write(frame_bgr)
                state.last_recording_frame_time = current_time
                
                # Add frame to buffer for recording thread
                try:
                    if not state.recording_frame_buffer.full():
                        state.recording_frame_buffer.put_nowait(frame_bgr.copy())
                except:
                    pass  # Skip if buffer is full
    except Exception as e:
        print(f"Recording frame error: {e}")

# =============================================================================
# INPUT CONTROLLER
# =============================================================================

def get_joystick_input():
    """Get joystick input and update drone control variables"""
    if not state.joystick or not state.toggle_joystick:
        return

    try:
        # Read joystick input
        axis_lr = state.joystick.get_axis(0)  # Left-right movement
        axis_fb = state.joystick.get_axis(1)  # Forward-backward movement
        axis_yv = state.joystick.get_axis(2)  # Yaw rotation
        axis_ud = state.joystick.get_axis(3)  # Up-down movement

        # Handle D-pad for flips
        axis_side = state.joystick.get_axis(6)
        axis_line = state.joystick.get_axis(7)

        if axis_side > 0.1:
            state.send_rc_control = False
            state.command_queue.put("flip r")
            state.send_rc_control = True
            print("Flip Right")
        elif axis_side < -0.1:
            state.send_rc_control = False
            state.command_queue.put("flip l")
            state.send_rc_control = True
            print("Flip Left")
        
        if axis_line > 0.1:
            state.send_rc_control = False
            state.command_queue.put("flip f")
            state.send_rc_control = True
            print("Flip Forward")
        elif axis_line < -0.1:
            state.send_rc_control = False
            state.command_queue.put("flip b")
            state.send_rc_control = True
            print("Flip Back")

        # Only update velocities if joystick has significant input (deadzone)
        deadzone = 0.1
        if abs(axis_lr) > deadzone or abs(axis_fb) > deadzone or abs(axis_ud) > deadzone or abs(axis_yv) > deadzone:
            # Set velocities based on joystick input
            if state.joystick_mode:
                state.left_right_velocity = int(axis_lr * state.speed)
                state.for_back_velocity = int(-axis_fb * state.speed)
                state.up_down_velocity = int(-axis_ud * state.speed)
                state.yaw_velocity = int(axis_yv * ROTATE)
            else:
                state.left_right_velocity = int(axis_yv * state.speed)
                state.for_back_velocity = int(-axis_ud * state.speed)
                state.up_down_velocity = int(-axis_fb * state.speed)
                state.yaw_velocity = int(axis_lr * ROTATE)

        # Handle buttons
        if state.joystick.get_button(0):  # Button A - takeoff
            if not state.send_rc_control:
                state.command_queue.put("takeoff")
                time.sleep(0.5)

        if state.joystick.get_button(1):  # Button B - land
            if state.send_rc_control:
                state.command_queue.put("land")
                time.sleep(0.5)

        if state.joystick.get_button(3):  # Screenshot button
            request_manual_screenshot("joystick")
            time.sleep(0.2)

        # Handle recording button
        current_recording_button_state = state.joystick.get_button(4)
        if current_recording_button_state and not state.last_joystick_recording_button_state:
            toggle_recording()
            time.sleep(0.3)
        state.last_joystick_recording_button_state = current_recording_button_state

        # Toggle auto screenshot
        if state.joystick.get_button(6):
            state.auto_screenshot_enabled = not state.auto_screenshot_enabled
            print(f"Auto Screenshot {'Enabled' if state.auto_screenshot_enabled else 'Disabled'}")

        # Toggle autonomous behavior
        if state.joystick.get_button(7):
            state.set_autonomous_behavior = not state.set_autonomous_behavior
            print(f"Autonomous behavior {'enabled' if state.set_autonomous_behavior else 'disabled'}")

        # Toggle detection
        current_detection_toggle_button_state = state.joystick.get_button(8)
        if current_detection_toggle_button_state and not state.last_joystick_detection_toggle_button_state:
            state.detection_enabled = not state.detection_enabled
            status = "ENABLED" if state.detection_enabled else "DISABLED"
            print(f"🔍 Detection & Auto-Screenshot: {status}")
            time.sleep(0.3)
        state.last_joystick_detection_toggle_button_state = current_detection_toggle_button_state

        # Emergency button
        current_emergency_button_state = state.joystick.get_button(11)
        if current_emergency_button_state and not state.last_joystick_emergency_button_state:
            state.command_queue.put("emergency")
            print("🚨 EMERGENCY LANDING ACTIVATED!")
            time.sleep(0.5)
        state.last_joystick_emergency_button_state = current_emergency_button_state

    except Exception as e:
        print(f"Joystick input error: {e}")

def handle_keyboard_input(key):
    """Handle keyboard input"""
    if not state.toggle_keyboard:
        return
        
    try:
        if key == 'o':  # Screenshot
            request_manual_screenshot("keyboard")
        elif key == 'p':  # Toggle recording
            toggle_recording()
        elif key == 't':  # Takeoff
            if not state.send_rc_control:
                state.command_queue.put("takeoff")
                print("🚁 Takeoff command sent")
        elif key == 'q':  # Land
            if state.send_rc_control:
                state.command_queue.put("land")
                print("🚁 Land command sent")
        elif key == 'e':  # Emergency
            state.command_queue.put("emergency")
            print("🚨 Emergency command sent")
        elif key == 'z':  # Toggle detection
            state.detection_enabled = not state.detection_enabled
            status = "ENABLED" if state.detection_enabled else "DISABLED"
            print(f"🔍 Detection & Auto-Screenshot: {status}")
        elif key == 'x':  # Toggle auto screenshot
            state.auto_screenshot_enabled = not state.auto_screenshot_enabled
            print(f"Auto Screenshot {'Enabled' if state.auto_screenshot_enabled else 'Disabled'}")
        elif key == 'c':  # Toggle autonomous
            state.set_autonomous_behavior = not state.set_autonomous_behavior
            print(f"Autonomous behavior {'enabled' if state.set_autonomous_behavior else 'disabled'}")
        elif key == 'f':  # Change keyboard mode
            state.keyboard_mode = not state.keyboard_mode
            print(f"Keyboard Mode {'1' if state.keyboard_mode else '2'}")
        elif key in ['i', 'j', 'k', 'l']:  # Flip commands
            flip_commands = {'i': 'flip f', 'j': 'flip l', 'k': 'flip b', 'l': 'flip r'}
            state.send_rc_control = False
            state.command_queue.put(flip_commands[key])
            state.send_rc_control = True
            print(f"Flip {key.upper()}")
        elif key == 'm':  # Speed increase
            if state.speed < 100:
                state.speed += 5
                state.current_speed_display = state.speed
            print(f"Speed increased: {state.speed}")
        elif key == 'n':  # Speed decrease
            if state.speed > 5:
                state.speed -= 5
                state.current_speed_display = state.speed
            print(f"Speed decreased: {state.speed}")
    except Exception as e:
        print(f"Keyboard input error: {e}")

def handle_arrow_keys(keys_pressed):
    """Handle arrow key movement"""
    if not state.toggle_keyboard:
        return
        
    try:
        # Check if we have joystick input first
        has_joystick_input = False
        if state.joystick and state.toggle_joystick:
            deadzone = 0.1
            axis_lr = state.joystick.get_axis(0)
            axis_fb = state.joystick.get_axis(1)
            axis_ud = state.joystick.get_axis(3)
            axis_yv = state.joystick.get_axis(2)
            
            if (abs(axis_lr) > deadzone or abs(axis_fb) > deadzone or 
                abs(axis_ud) > deadzone or abs(axis_yv) > deadzone):
                has_joystick_input = True

        # Only process keyboard if no joystick input
        if not has_joystick_input:
            # Reset all velocities first
            state.for_back_velocity = 0
            state.left_right_velocity = 0
            state.up_down_velocity = 0
            state.yaw_velocity = 0

            # Set velocities based on pressed keys
            if state.keyboard_mode:
                if keys_pressed['up']:
                    state.for_back_velocity = state.speed
                elif keys_pressed['down']:
                    state.for_back_velocity = -state.speed
                    
                if keys_pressed['left']:
                    state.left_right_velocity = -state.speed
                elif keys_pressed['right']:
                    state.left_right_velocity = state.speed
                    
                if keys_pressed['w']:
                    state.up_down_velocity = state.speed
                elif keys_pressed['s']:
                    state.up_down_velocity = -state.speed
                    
                if keys_pressed['a']:
                    state.yaw_velocity = -ROTATE
                elif keys_pressed['d']:
                    state.yaw_velocity = ROTATE
            else:
                if keys_pressed['w']:
                    state.for_back_velocity = state.speed
                elif keys_pressed['s']:
                    state.for_back_velocity = -state.speed
                    
                if keys_pressed['a']:
                    state.left_right_velocity = -state.speed
                elif keys_pressed['d']:
                    state.left_right_velocity = state.speed
                    
                if keys_pressed['up']:
                    state.up_down_velocity = state.speed
                elif keys_pressed['down']:
                    state.up_down_velocity = -state.speed
                    
                if keys_pressed['left']:
                    state.yaw_velocity = -ROTATE
                elif keys_pressed['right']:
                    state.yaw_velocity = ROTATE

    except Exception as e:
        print(f"Arrow key handling error: {e}")

def reset_keyboard_velocities():
    """Reset all keyboard-controlled velocities to zero"""
    # Only reset if no joystick input
    has_joystick_input = False
    if state.joystick and state.toggle_joystick:
        deadzone = 0.1
        axis_lr = state.joystick.get_axis(0)
        axis_fb = state.joystick.get_axis(1)
        axis_ud = state.joystick.get_axis(3)
        axis_yv = state.joystick.get_axis(2)
        
        if (abs(axis_lr) > deadzone or abs(axis_fb) > deadzone or 
            abs(axis_ud) > deadzone or abs(axis_yv) > deadzone):
            has_joystick_input = True

    if not has_joystick_input:
        state.for_back_velocity = 0
        state.left_right_velocity = 0
        state.up_down_velocity = 0
        state.yaw_velocity = 0

# =============================================================================
# FLASK WEB INTERFACE SYSTEM
# =============================================================================

class FlaskWebInterface:
    """Flask-SocketIO web interface for remote control"""
    
    def __init__(self):
        # Flask and SocketIO setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'tello_secret_key'
        
        # CORS for React app
        CORS(self.app, 
            origins=["http://localhost:5173", "http://localhost:3000"],
            allow_headers=["Content-Type"],
            methods=["GET", "POST"])
        
        # Socket.IO initialization
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins=["http://localhost:5173", "http://localhost:3000"],
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            transports=['polling', 'websocket']
        )
        
        self.setup_routes()
        self.setup_socket_events()

    def get_file_thumbnail(self, filepath):
        """Generate thumbnail URL for media files"""
        try:
            filename = os.path.basename(filepath)
            file_ext = filename.lower().split('.')[-1]
            
            if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                return f'http://localhost:5000/media/{filename}'
            elif file_ext in ['mp4', 'avi', 'mov', 'wmv', 'flv']:
                return f'http://localhost:5000/media/{filename}'
            
            return None
        except Exception as e:
            print(f"Error generating thumbnail for {filepath}: {e}")
            return None

    def scan_media_directory(self, file_type='all'):
        """Scan screenshots directory and return detailed file information"""
        try:
            files = []
            
            if not os.path.exists(SCREENSHOTS_DIR):
                os.makedirs(SCREENSHOTS_DIR)
                return files
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'}
            
            for filename in os.listdir(SCREENSHOTS_DIR):
                filepath = os.path.join(SCREENSHOTS_DIR, filename)
                
                if not os.path.isfile(filepath):
                    continue
                    
                file_ext = Path(filename).suffix.lower()
                
                is_image = file_ext in image_extensions
                is_video = file_ext in video_extensions
                
                if file_type == 'images' and not is_image:
                    continue
                elif file_type == 'videos' and not is_video:
                    continue
                elif file_type == 'all' and not (is_image or is_video):
                    continue
                
                try:
                    stat = os.stat(filepath)
                    
                    humans_detected = 0
                    if '_human_detected_' in filename:
                        try:
                            parts = filename.split('_')
                            for i, part in enumerate(parts):
                                if 'persons' in part and i > 0:
                                    humans_detected = int(parts[i-1])
                                    break
                        except (ValueError, IndexError):
                            humans_detected = 0
                    
                    media_type = 'image' if is_image else 'video'
                    mime_type, _ = mimetypes.guess_type(filename)
                    
                    file_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'size': stat.st_size,
                        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'humans_detected': humans_detected,
                        'url': f'http://localhost:5000/media/{filename}',
                        'thumbnail': self.get_file_thumbnail(filepath),
                        'type': media_type,
                        'mime_type': mime_type or f'{media_type}/unknown',
                        'extension': file_ext[1:] if file_ext else 'unknown'
                    }
                    files.append(file_info)
                    
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue
            
            files.sort(key=lambda x: x['modified_at'], reverse=True)
            return files
            
        except Exception as e:
            print(f"Error scanning media directory: {e}")
            return []

    def get_media_stats(self):
        """Get statistics about media files"""
        try:
            all_files = self.scan_media_directory('all')
            
            stats = {
                'total_files': len(all_files),
                'total_images': len([f for f in all_files if f['type'] == 'image']),
                'total_videos': len([f for f in all_files if f['type'] == 'video']),
                'total_size': sum(f['size'] for f in all_files),
                'files_with_humans': len([f for f in all_files if f['humans_detected'] > 0]),
                'total_humans_detected': sum(f['humans_detected'] for f in all_files)
            }
            
            return stats
        except Exception as e:
            print(f"Error getting media stats: {e}")
            return {}

    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
                <body>
                    <h1>🚁 SARVIO-X 5-Thread Backend Server</h1>
                    <h3>Live Video Stream:</h3>
                    <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Tello Live Stream">
                </body>
            </html>
            """)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.frame_generator(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/media/<filename>')
        def serve_media(filename):
            try:
                response = send_from_directory(
                    SCREENSHOTS_DIR, 
                    filename,
                    as_attachment=False
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
            except Exception as e:
                print(f"Error serving media file {filename}: {e}")
                return "File not found", 404

        @self.app.route('/download/<filename>')
        def download_media(filename):
            try:
                response = send_from_directory(
                    SCREENSHOTS_DIR, 
                    filename, 
                    as_attachment=True,
                    download_name=filename
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
            except Exception as e:
                print(f"Error downloading file {filename}: {e}")
                return "File not found", 404

        @self.app.route('/api/media/stats')
        def get_media_statistics():
            try:
                stats = self.get_media_stats()
                return {'success': True, 'stats': stats}
            except Exception as e:
                return {'success': False, 'error': str(e)}, 500

        @self.app.route('/api/media/list')
        def list_media_files():
            try:
                file_type = request.args.get('type', 'all')
                files = self.scan_media_directory(file_type)
                
                return {
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'files': []
                }, 500

    def setup_socket_events(self):
        """Setup Socket.IO events"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            state.connected_clients += 1
            print(f'✅ React client connected. Total clients: {state.connected_clients}')
            
            self.broadcast_status()
            emit('ml_detection_status', {'enabled': state.ml_detection_enabled})
            emit('auto_capture_status', {'enabled': state.auto_screenshot_enabled})
            emit('speed_update', {'speed': state.current_speed_display})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            state.connected_clients -= 1
            print(f'❌ React client disconnected. Total clients: {state.connected_clients}')
            
            if state.connected_clients <= 0:
                state.socket_streaming = False

        @self.socketio.on('connect_tello')
        def handle_connect_tello():
            print("🔗 Connect Tello command from React client")
            state.disconnect_requested = False
            
            emit('tello_connection_result', {
                'success': state.is_connected,
                'message': 'Tello already connected' if state.is_connected else 'Tello connection failed'
            })
            
            self.broadcast_status()

        @self.socketio.on('disconnect_tello')
        def handle_disconnect_tello():
            print("🔌 Disconnect Tello command from React client")
            state.disconnect_requested = True
            
            emit('tello_connection_result', {
                'success': True,
                'message': 'Disconnect requested'
            })
            
            self.broadcast_status()
        
        @self.socketio.on('start_stream')
        def handle_start_stream():
            print("🎥 React client requested video stream")
            state.socket_streaming = True
            emit('stream_status', {'streaming': True, 'message': 'Video stream started'})
        
        @self.socketio.on('stop_stream')
        def handle_stop_stream():
            print("⏹️ React client stopped video stream")
            state.socket_streaming = False
            emit('stream_status', {'streaming': False, 'message': 'Video stream stopped'})
        
        @self.socketio.on('takeoff')
        def handle_takeoff():
            print("🚁 Takeoff command from React client")
            if not state.send_rc_control:
                state.command_queue.put("takeoff")
        
        @self.socketio.on('land')
        def handle_land():
            print("🏠 Land command from React client")
            if state.send_rc_control:
                state.command_queue.put("land")
        
        @self.socketio.on('move_control')
        def handle_move_control(data):
            if state.send_rc_control and not state.disconnect_requested:
                state.left_right_velocity = int(data.get('left_right', 0))
                state.for_back_velocity = int(data.get('for_back', 0))
                state.up_down_velocity = int(data.get('up_down', 0))
                state.yaw_velocity = int(data.get('yaw', 0))
        
        @self.socketio.on('stop_movement')
        def handle_stop_movement():
            state.left_right_velocity = 0
            state.for_back_velocity = 0
            state.up_down_velocity = 0
            state.yaw_velocity = 0

        @self.socketio.on('set_speed')
        def handle_set_speed(data):
            new_speed = data.get('speed', 20)
            if 10 <= new_speed <= 100:
                state.speed = new_speed
                state.current_speed_display = new_speed
                
                if state.is_connected:
                    try:
                        state.tello.set_speed(new_speed)
                        print(f"⚡ Speed set to: {new_speed} cm/s")
                    except Exception as e:
                        print(f"❌ Error setting speed: {e}")
                
                self.socketio.emit('speed_update', {'speed': new_speed})

        @self.socketio.on('emergency_land')
        def handle_emergency():
            print("🚨 Emergency command from React client")
            state.command_queue.put("emergency")

        @self.socketio.on('enable_ml_detection')
        def handle_enable_ml_detection(data):
            state.ml_detection_enabled = data.get('enabled', False)
            state.detection_enabled = state.ml_detection_enabled
            print(f"🤖 ML Detection: {'Enabled' if state.ml_detection_enabled else 'Disabled'}")
            
            emit('ml_detection_status', {
                'enabled': state.ml_detection_enabled,
                'message': f"ML Detection {'enabled' if state.ml_detection_enabled else 'disabled'}"
            })

        @self.socketio.on('enable_auto_capture')
        def handle_enable_auto_capture(data):
            state.auto_screenshot_enabled = data.get('enabled', False)
            print(f"📸 Auto Capture: {'Enabled' if state.auto_screenshot_enabled else 'Disabled'}")
            
            emit('auto_capture_status', {
                'enabled': state.auto_screenshot_enabled,
                'message': f"Auto Capture {'enabled' if state.auto_screenshot_enabled else 'disabled'}"
            })

        @self.socketio.on('toggle_recording')
        def handle_toggle_recording(data):
            state.recording = data.get('recording', False)
            if state.recording:
                start_recording()
            else:
                stop_recording()
            
            emit('recording_status', {
                'recording': state.recording,
                'message': f"Recording {'started' if state.recording else 'stopped'}"
            })

        @self.socketio.on('get_media_files')
        def handle_get_media_files(data):
            try:
                file_type = data.get('type', 'images')
                files = self.scan_media_directory(file_type)
                
                emit('media_files_response', {
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                })
                
            except Exception as e:
                emit('media_files_response', {
                    'success': False,
                    'error': str(e),
                    'files': []
                })

        @self.socketio.on('manual_screenshot')
        def handle_manual_screenshot():
            print("📸 Manual screenshot request from React client")
            request_manual_screenshot("web")

    def broadcast_status(self):
        """Broadcast status to all web clients"""
        try:
            flight_time = 0
            if state.flight_start_time and state.is_flying:
                flight_time = int(time.time() - state.flight_start_time)
            
            status = {
                'connected': state.is_connected,
                'flying': state.is_flying,
                'battery': state.battery_level,
                'flight_time': flight_time,
                'speed': state.current_speed_display
            }
            self.socketio.emit('tello_status', status)
        except Exception as e:
            print(f"Broadcast status error: {e}")

    def frame_generator(self):
        """Generator for streaming frames to browser"""
        while not state.should_stop:
            with state.frame_lock:
                if state.last_frame is None:
                    time.sleep(0.1)
                    continue
                
                frame_to_send = state.last_frame.copy()
                frame_bgr = cv2.cvtColor(frame_to_send, cv2.COLOR_RGB2BGR)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                ret, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)
                
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                time.sleep(1 / 30)  # 30 FPS for web stream

    def send_frame_to_react(self, frame):
        """Send frame to React clients via Socket.IO"""
        if state.socket_streaming and state.connected_clients > 0:
            try:
                frame_resized = cv2.resize(frame, (640, 480))
                frame_bgr2 = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                ret, buffer = cv2.imencode('.jpg', frame_bgr2, encode_params)
                if ret:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.socketio.emit('video_frame', {'frame': frame_base64})
            except Exception as e:
                print(f"Error sending frame to React: {e}")

    def run_server(self):
        """Run Flask server with Socket.IO"""
        print("Starting Flask server with Socket.IO...")
        print("React app: http://localhost:5173")
        print("Backend: http://localhost:5000")
        
        self.socketio.run(
            self.app, 
            host='127.0.0.1',
            port=5000, 
            debug=False, 
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )

# Global web interface instance
web_interface = FlaskWebInterface()

# =============================================================================
# THREAD MANAGEMENT SYSTEM
# =============================================================================

def video_stream_thread():
    """Thread 1: Handle video capture, processing, and screenshot operations"""
    print("Video stream thread started")
    
    try:
        frame_read = state.tello.get_frame_read()
        
        while state.running:
            try:
                if frame_read.stopped:
                    break
                    
                frame = frame_read.frame
                if frame is not None:
                    # Resize frame
                    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    
                    # Process human detection only if detection is enabled
                    if state.detection_enabled:
                        output_frame, detected, count = process_human_detection(frame)
                        # Handle auto screenshot logic
                        if state.auto_screenshot_enabled:
                            handle_auto_screenshot(output_frame, detected, count)
                    else:
                        output_frame = frame.copy()
                        detected = False
                        count = 0
                    
                    # Process screenshot requests
                    while not state.screenshot_queue.empty():
                        try:
                            screenshot_data = state.screenshot_queue.get_nowait()
                            frame_to_save, humans_count_param, source = screenshot_data
                            save_screenshot(frame_to_save, humans_count_param, source)
                        except queue.Empty:
                            break
                    
                    # Handle recording
                    process_recording_frame(output_frame)
                    
                    # Update shared data
                    with state.data_lock:
                        state.current_frame = frame.copy()
                        state.current_processed_frame = output_frame.copy()
                        state.human_detected = detected
                        state.humans_count = count
                        
                        # Calculate FPS
                        current_time = time.time()
                        state.frame_times.append(current_time)
                        if len(state.frame_times) > 1:
                            time_diff = state.frame_times[-1] - state.frame_times[0]
                            state.fps = len(state.frame_times) / time_diff if time_diff > 0 else 0
                    
                    # Update frame for web interface
                    with state.frame_lock:
                        state.last_frame = output_frame.copy()
                    
                    # Send frame to web clients
                    web_interface.send_frame_to_react(output_frame)
                
                time.sleep(VIDEO_THREAD_SLEEP)
                
            except Exception as e:
                print(f"Video stream error: {e}")
                time.sleep(0.1)
        
    except Exception as e:
        print(f"Critical video stream error: {e}")
    
    print("Video stream thread ended")

def drone_control_thread():
    """Thread 2: Handle drone control commands and battery monitoring"""
    print("Drone control thread started")
    last_battery_check = time.time()
    last_telemetry_update = time.time()
    telemetry_update_interval = 1.0
    
    while state.running:
        try:
            # Process queued commands
            while not state.command_queue.empty():
                try:
                    command = state.command_queue.get_nowait()
                    execute_drone_command(command)
                except queue.Empty:
                    break
            
            # Send RC control if active
            if state.send_rc_control and state.is_connected and not state.disconnect_requested:
                try:
                    state.tello.send_rc_control(
                        state.left_right_velocity, 
                        state.for_back_velocity,
                        state.up_down_velocity, 
                        state.yaw_velocity
                    )
                except Exception as rc_error:
                    print(f"RC command error: {rc_error}")
            
            current_time = time.time()
            
            # Update battery level periodically
            if current_time - last_battery_check >= BATTERY_CHECK_INTERVAL:
                try:
                    if state.is_connected:
                        with state.data_lock:
                            state.battery_level = state.tello.get_battery()
                    last_battery_check = current_time
                except Exception as e:
                    print(f"Battery check error: {e}")
            
            # Update telemetry data periodically
            if current_time - last_telemetry_update >= telemetry_update_interval:
                try:
                    if state.is_connected:
                        update_telemetry_data()
                    last_telemetry_update = current_time
                except Exception as e:
                    print(f"Telemetry update error: {e}")
            
            time.sleep(CONTROL_THREAD_SLEEP)
            
        except Exception as e:
            print(f"Drone control error: {e}")
            time.sleep(0.1)
    
    print("Drone control thread ended")

def update_telemetry_data():
    """Update all telemetry variables from Tello state"""
    try:
        state_string = state.tello.get_current_state()
        
        if state_string:
            state_data = {}
            for item in state_string.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    try:
                        state_data[key] = float(value)
                    except ValueError:
                        state_data[key] = value
            
            with state.data_lock:
                state.temp = state_data.get('templ', state.temp)
                if 'temph' in state_data:
                    state.temp = state_data.get('temph', state.temp)
                
                state.baro = state_data.get('baro', state.baro)
                state.height = state_data.get('h', state.height)
                state.tof = state_data.get('tof', state.tof)
                state.pitch = state_data.get('pitch', state.pitch)
                state.roll = state_data.get('roll', state.roll)
                state.yaw = state_data.get('yaw', state.yaw)
                state.vgx = state_data.get('vgx', state.vgx)
                state.vgy = state_data.get('vgy', state.vgy)
                state.vgz = state_data.get('vgz', state.vgz)
                state.agx = state_data.get('agx', state.agx)
                state.agy = state_data.get('agy', state.agy)
                state.agz = state_data.get('agz', state.agz)
                
    except Exception as e:
        print(f"Error updating telemetry: {e}")

def detection_thread():
    """Thread 3: Handle red color detection"""
    print("Detection thread started")
    
    while state.running:
        try:
            if state.detection_enabled:
                with state.data_lock:
                    if state.current_processed_frame is not None:
                        frame_copy = state.current_processed_frame.copy()
                    else:
                        frame_copy = None
                
                if frame_copy is not None:
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    mask, result = detect_red_color(frame_copy)
                    red_in_roi, roi_mask, full_roi_mask, pixel_count = detect_red_in_roi(frame_copy)
                    
                    with state.data_lock:
                        state.current_detection = {
                            'red_detected': red_in_roi,
                            'mask': mask,
                            'result': result,
                            'roi_mask': roi_mask,
                            'full_roi_mask': full_roi_mask,
                            'pixel_count': pixel_count
                        }
            else:
                with state.data_lock:
                    state.current_detection = {
                        'red_detected': False,
                        'mask': None,
                        'result': None,
                        'roi_mask': None,
                        'full_roi_mask': None,
                        'pixel_count': 0
                    }
            
            time.sleep(DETECTION_THREAD_SLEEP)
            
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)
    
    print("Detection thread ended")

def autonomous_behavior_thread():
    """Thread 4: Handle autonomous behavior based on detection results"""
    print("Autonomous behavior thread started")
    
    while state.running:
        try:
            if (state.set_autonomous_behavior and state.detection_enabled and 
                state.current_detection is not None and state.is_connected and state.is_flying):
                
                with state.data_lock:
                    if state.current_detection:
                        red_detected = state.current_detection.get('red_detected', False)
                        pixel_count = state.current_detection.get('pixel_count', 0)
                    else:
                        red_detected = False
                        pixel_count = 0
                
                # Autonomous behavior logic
                if red_detected:
                    print("🔴 Red detected in ROI! Moving towards target...")
                    try:
                        state.tello.move_back(70)
                        time.sleep(2)
                        state.tello.rotate_clockwise(90)
                        time.sleep(2)
                    except Exception as e:
                        print(f"Autonomous movement error: {e}")
                else:
                    print("⚪ No red in ROI. Searching...")
                    try:
                        state.tello.move_forward(30)
                        time.sleep(1)
                    except Exception as e:
                        print(f"Autonomous search error: {e}")
            else:
                time.sleep(AUTONOMOUS_THREAD_SLEEP)
                
        except Exception as e:
            print(f"Autonomous behavior error: {e}")
            time.sleep(0.5)
    
    print("Autonomous behavior thread ended")

def recording_thread_func():
    """Thread 5: Dedicated thread for smooth recording"""
    print("Recording thread started")
    
    while state.running:
        try:
            if state.recording and state.video_writer and state.video_writer.isOpened():
                frames_written = 0
                while not state.recording_frame_buffer.empty() and frames_written < 5:
                    try:
                        frame_bgr = state.recording_frame_buffer.get_nowait()
                        state.video_writer.write(frame_bgr)
                        frames_written += 1
                    except queue.Empty:
                        break
                
                time.sleep(RECORDING_THREAD_SLEEP)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Recording thread error: {e}")
            time.sleep(0.1)
    
    print("Recording thread ended")

def execute_drone_command(command):
    """Execute drone commands safely"""
    try:
        print(f"🎯 Executing command: {command}")
        if command == "takeoff":
            state.tello.takeoff()
            state.send_rc_control = True
            state.is_flying = True
            state.flight_start_time = time.time()
            print("✅ Takeoff completed, RC control enabled")
            web_interface.broadcast_status()
        elif command == "land":
            state.tello.land()
            state.send_rc_control = False
            state.is_flying = False
            state.flight_start_time = None
            print("✅ Landing completed, RC control disabled")
            web_interface.broadcast_status()
        elif command == "emergency":
            state.tello.emergency()
            state.send_rc_control = False
            state.is_flying = False
            state.flight_start_time = None
            print("🚨 Emergency landing executed")
            web_interface.broadcast_status()
        elif command == "flip f":
            state.tello.flip_forward()
        elif command == "flip b":
            state.tello.flip_back()
        elif command == "flip l":
            state.tello.flip_left()
        elif command == "flip r":
            state.tello.flip_right()
    except Exception as e:
        print(f"Command execution error: {e}")

def start_all_threads():
    """Start all worker threads"""
    print("Starting all threads...")
    
    threads_config = [
        ("Video Stream", video_stream_thread),
        ("Drone Control", drone_control_thread),
        ("Detection", detection_thread),
        ("Autonomous Behavior", autonomous_behavior_thread),
        ("Recording", recording_thread_func)
    ]
    
    for name, target_func in threads_config:
        thread = threading.Thread(target=target_func, daemon=True, name=name)
        thread.start()
        state.threads.append(thread)
    
    print(f"Started {len(state.threads)} worker threads")

def stop_all_threads():
    """Stop all worker threads"""
    print("Stopping threads...")
    state.running = False
    
    # Stop recording if active
    if state.recording:
        stop_recording()
    
    # Wait for threads to finish
    for thread in state.threads:
        if thread.is_alive():
            thread.join(timeout=2)
    
    cleanup_systems()
    print("All threads stopped")

# =============================================================================
# UI CONTROLLER
# =============================================================================

def draw_roi_rectangle(surface):
    """Draw ROI rectangle on pygame surface only if detection is enabled"""
    try:
        if state.detection_enabled:
            roi_color = (0, 255, 0)
            roi_rect = pygame.Rect(ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
            pygame.draw.rect(surface, roi_color, roi_rect, 3)
    except Exception as e:
        print(f"ROI drawing error: {e}")

def add_frame_overlays(frame):
    """Add information overlays to the frame"""
    try:
        with state.data_lock:
            current_fps = state.fps
            current_battery = state.battery_level
            current_humans = state.humans_count
            current_screenshots = state.screenshot_count
            detection_status = state.detection_enabled
        
        # Add info overlays
        cv2.putText(frame, f"Battery: {current_battery}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show detection status
        detection_text = "DETECTION: ON" if detection_status else "DETECTION: OFF"
        detection_color = (0, 255, 0) if detection_status else (0, 0, 255)
        cv2.putText(frame, detection_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
        
        # Only show human count if detection is enabled
        if detection_status and current_humans > 0:
            cv2.putText(frame, f"Humans: {current_humans}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Screenshots: {current_screenshots}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, "5-THREAD MODE", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Show recording status
        if state.recording:
            recording_duration = time.time() - state.recording_start_time
            cv2.putText(frame, f"🔴 REC {recording_duration:.1f}s", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (WINDOW_WIDTH - 30, 30), 15, (0, 0, 255), -1)

        # Show countdown if active and detection enabled
        if detection_status and state.countdown_active:
            elapsed = time.time() - state.countdown_start_time
            remaining = max(0, COUNTDOWN_DURATION - elapsed)
            cv2.putText(frame, f"Screenshot in: {remaining:.1f}s", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show autonomous behavior status
        if state.set_autonomous_behavior:
            auto_status = "AUTO MODE: ON" if detection_status else "AUTO MODE: OFF (No Detection)"
            auto_color = (255, 0, 255) if detection_status else (128, 0, 128)
            cv2.putText(frame, auto_status, (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, auto_color, 2)

        # Show current velocities
        cv2.putText(frame, f"LR:{state.left_right_velocity} FB:{state.for_back_velocity}", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"UD:{state.up_down_velocity} YAW:{state.yaw_velocity}", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show web interface status
        cv2.putText(frame, f"Web Clients: {state.connected_clients}", (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return frame
    except Exception as e:
        print(f"Overlay error: {e}")
        return frame

def handle_pygame_events():
    """Handle pygame events and return whether to continue"""
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT + 1:
            pass
        elif event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_p:
                handle_keyboard_input('p')
            elif event.key == pygame.K_o:
                handle_keyboard_input('o')
            elif event.key == pygame.K_t:
                handle_keyboard_input('t')
            elif event.key == pygame.K_q:
                handle_keyboard_input('q')
            elif event.key == pygame.K_e:
                handle_keyboard_input('e')
            elif event.key == pygame.K_z:
                handle_keyboard_input('z')
            elif event.key == pygame.K_c:
                handle_keyboard_input('c')
            elif event.key == pygame.K_f:
                handle_keyboard_input('f')
            elif event.key == pygame.K_i:
                handle_keyboard_input('i')
            elif event.key == pygame.K_j:
                handle_keyboard_input('j')
            elif event.key == pygame.K_k:
                handle_keyboard_input('k')
            elif event.key == pygame.K_l:
                handle_keyboard_input('l')
            elif event.key == pygame.K_m:
                handle_keyboard_input('m')
            elif event.key == pygame.K_n:
                handle_keyboard_input('n')
            elif event.key == pygame.K_x:
                handle_keyboard_input('x')
            elif event.key == pygame.K_v:
                handle_keyboard_input('v')
    return True

def handle_continuous_keys():
    """Handle continuously pressed keys (like arrow keys)"""
    keys = pygame.key.get_pressed()
    
    any_movement_key_pressed = (
        keys[pygame.K_UP] or keys[pygame.K_DOWN] or 
        keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or
        keys[pygame.K_w] or keys[pygame.K_s] or 
        keys[pygame.K_a] or keys[pygame.K_d]
    )
    
    if any_movement_key_pressed:
        if not state.send_rc_control and state.is_connected:
            state.send_rc_control = True
            print("🎮 RC Control: AUTO-ENABLED for keyboard movement")
        
        keys_pressed = {
            'up': keys[pygame.K_UP],
            'down': keys[pygame.K_DOWN],
            'left': keys[pygame.K_LEFT],
            'right': keys[pygame.K_RIGHT],
            'w': keys[pygame.K_w],
            's': keys[pygame.K_s],
            'a': keys[pygame.K_a],
            'd': keys[pygame.K_d]
        }
        
        handle_arrow_keys(keys_pressed)
    else:
        reset_keyboard_velocities()

def main_loop():
    """Main UI loop - handles display and input"""
    should_stop = False
    
    try:
        while not should_stop:
            if not handle_pygame_events():
                should_stop = True
                break
            
            handle_continuous_keys()
            get_joystick_input()

            # Clear screen
            state.screen.fill([0, 0, 0])

            # Get latest processed frame
            display_frame = None
            with state.data_lock:
                if state.current_processed_frame is not None:
                    display_frame = state.current_processed_frame.copy()
            
            if display_frame is not None:
                display_frame = add_frame_overlays(display_frame)

                frame_rgb = np.rot90(display_frame)
                frame_rgb = np.flipud(frame_rgb)

                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                state.screen.blit(frame_surface, (0, 0))

                draw_roi_rectangle(state.screen)

            # Display control instructions
            font = pygame.font.SysFont("Arial", 16)
            instructions = [
                "5-THREAD TELLO CONTROL SYSTEM",
                "T=Takeoff, Q=Land, P=Recording, O=Screenshot",
                "Z=Detection, X=Auto Screenshot, C=Autonomous", 
                "Arrow Keys=Move, WASD=Move/Rotate",
                f"Web Interface: http://localhost:5000",
                f"React App: http://localhost:5173"
            ]
            
            for i, instruction in enumerate(instructions):
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                text = font.render(instruction, True, color)
                state.screen.blit(text, (10, WINDOW_HEIGHT - 120 + i * 20))

            pygame.display.update()
            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print(f"UI loop ended. Total screenshots taken: {state.screenshot_count}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main function - Entry point of the application"""
    print("=" * 60)
    print(f"{THREAD_COUNT}-Thread Tello Drone Control System with Web Interface")
    print("=" * 60)
    print("Architecture:")
    print("- Thread 1: Video stream and human detection")
    print("- Thread 2: Drone control commands and battery monitoring") 
    print("- Thread 3: Red color detection")
    print("- Thread 4: Autonomous behavior")
    print("- Thread 5: Dedicated smooth recording")
    print("- Flask-SocketIO: Web interface server")
    print("=" * 60)
    print("Features:")
    print("- Smart auto screenshot with 3-second countdown")
    print("- Manual screenshot with 'O' key or joystick buttons")
    print("- Time-based video recording (consistent speed)")
    print("- Human detection with pose and hand tracking")
    print("- Red color detection with ROI")
    print("- Flask-SocketIO web interface")
    print("- Real-time video streaming")
    print("- Enhanced Media Gallery")
    print("=" * 60)
    print("Controls:")
    print("- Keyboard: Arrow keys=move, W/S=up/down, A/D=rotate")
    print("- T=takeoff, Q=land, P=recording, O=screenshot")
    print("- Z=toggle detection, X=toggle auto screenshot")
    print("- C=toggle autonomous, E=emergency")
    print("- Joystick: Move drone, A=takeoff, B=land, X/Y=screenshot")
    print("- ESC or close window to quit")
    print("=" * 60)
    print("Web Interface:")
    print("- React Frontend: http://localhost:5173")
    print("- Backend API: http://localhost:5000")
    print("- Real-time video streaming with ML detection")
    print("- Remote control via web browser")
    print("=" * 60)
    
    try:
        # Initialize all systems
        if not initialize_all_systems():
            print("Failed to initialize systems. Exiting...")
            return
        
        # Start Flask web server in separate thread
        flask_thread = threading.Thread(target=web_interface.run_server, daemon=True)
        flask_thread.start()
        
        # Start all worker threads
        start_all_threads()
        
        # Wait for threads to initialize
        time.sleep(2)
        
        # Start web streaming
        state.socket_streaming = True
        
        print("=" * 60)
        print("🚀 All systems ready!")
        print("🎮 Pygame window: Local control and display")
        print("🌐 Web interface: Remote control and streaming")
        print("📱 Open http://localhost:5173 for React frontend")
        print("=" * 60)
        
        # Run main UI loop
        main_loop()
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Shutting down...")
        stop_all_threads()

if __name__ == '__main__':
    main()