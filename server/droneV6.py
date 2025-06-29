#!/usr/bin/env python3
"""
Tello Drone Control System - Headless Web Server Version
5-Thread Architecture with Pure Web Interface (No GUI)

Features:
- 5-Thread architecture optimized for server deployment
- Human detection with YOLO + MediaPipe
- Red color detection with ROI
- Smart auto-screenshot system
- Time-based video recording
- Autonomous behavior
- Pure Flask-SocketIO web interface
- Real-time video streaming
- Enhanced Media Gallery
- Docker/Server ready

Run: python tello_headless.py
Web Interface: http://localhost:5000
API Documentation: http://localhost:5000/api/docs
"""

import threading
import time
import cv2
import queue
import numpy as np
import os
import sys
import base64
import mimetypes
import signal
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import deque

# Drone and AI imports
from djitellopy import Tello
from ultralytics import YOLO
import mediapipe as mp

# Flask and web interface imports
from flask import Flask, Response, render_template_string, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# Video settings
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

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
LOGS_DIR = "logs"

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

# Thread timing (optimized for headless)
VIDEO_THREAD_SLEEP = 0.008  # Slightly faster for better streaming
DETECTION_THREAD_SLEEP = 0.025
CONTROL_THREAD_SLEEP = 1/40  # More responsive control
RECORDING_THREAD_SLEEP = 0.008
AUTONOMOUS_THREAD_SLEEP = 0.1

# Web server settings
WEB_HOST = '0.0.0.0'  # Allow external connections
WEB_PORT = 5000
WEB_DEBUG = False

# =============================================================================
# LOGGING SYSTEM
# =============================================================================

import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup logging system for headless operation"""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    # Create logger
    logger = logging.getLogger('TelloHeadless')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, 'tello_headless.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# =============================================================================
# SHARED STATE MANAGEMENT
# =============================================================================

class SharedState:
    """Thread-safe shared state management for headless operation"""
    
    def __init__(self):
        # Thread control
        self.running = True
        self.threads = []
        self.shutdown_requested = False
        
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
        self.speed = SPEED
        self.current_speed_display = SPEED
        
        # Web interface variables
        self.ml_detection_enabled = True
        self.socket_streaming = False
        self.connected_clients = 0
        self.should_stop = False
        self.flight_start_time = None
        
        # System stats
        self.system_start_time = time.time()
        self.total_frames_processed = 0
        self.total_humans_detected = 0
        self.total_red_detections = 0
        
        # Global objects
        self.tello = None
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

def initialize_tello():
    """Initialize Tello drone connection"""
    try:
        logger.info("Initializing Tello connection...")
        state.tello = Tello()
        state.tello.connect()
        state.tello.set_speed(SPEED)
        state.battery_level = state.tello.get_battery()
        state.is_connected = True
        logger.info(f"✅ Tello connected! Battery: {state.battery_level}%")
        
        # Start video stream
        state.tello.streamoff()
        time.sleep(0.5)
        state.tello.streamon()
        logger.info("📹 Video stream started")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Tello: {e}")
        state.is_connected = False
        return False

def initialize_ai_models():
    """Initialize AI models for detection"""
    try:
        logger.info("Loading AI models...")
        
        # Load YOLOv8 model for human detection
        logger.info("Loading YOLOv8 model...")
        state.yolo_model = YOLO('yolov8n.pt')
        logger.info("✅ YOLOv8 model loaded successfully")
        
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
        
        logger.info("✅ AI models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI models: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    try:
        for directory in [SCREENSHOTS_DIR, RECORDINGS_DIR, LOGS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False

def initialize_all_systems():
    """Initialize all systems in correct order"""
    logger.info("Initializing headless drone control system...")
    
    if not create_directories():
        return False
        
    if not initialize_tello():
        return False
        
    if not initialize_ai_models():
        return False
    
    logger.info("✅ All systems initialized successfully!")
    return True

def cleanup_systems():
    """Cleanup all systems"""
    try:
        logger.info("Cleaning up systems...")
        
        if state.tello and state.is_connected:
            try:
                # Land drone if flying
                if state.is_flying:
                    logger.info("Landing drone before shutdown...")
                    state.tello.land()
                    time.sleep(2)
                
                state.tello.streamoff()
                state.tello.end()
                logger.info("Tello disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Tello: {e}")
        
        if state.pose:
            state.pose.close()
            
        if state.hands:
            state.hands.close()
            
        # Stop recording if active
        if state.recording:
            stop_recording()
            
        logger.info("✅ Systems cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

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
            state.total_humans_detected += 1

        return output_frame, detected, len(human_boxes)
    
    except Exception as e:
        logger.error(f"Error in human detection: {e}")
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
        logger.error(f"Error in body part detection: {e}")
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
        
        if red_detected:
            state.total_red_detections += 1
        
        # Create full-size mask for visualization
        full_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        full_mask[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH] = mask_roi
        
        return red_detected, mask_roi, full_mask, pixel_count
    
    except Exception as e:
        logger.error(f"Error in red detection ROI: {e}")
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
        logger.error(f"Error in red color detection: {e}")
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
                logger.info(f"🔴 Recording started: {state.current_recording_file}")
                return True
            else:
                logger.error("❌ Failed to start recording - could not open video writer")
                return False
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return False

def stop_recording():
    """Stop video recording"""
    try:
        if state.recording and state.video_writer:
            state.recording = False
            state.video_writer.release()
            state.video_writer = None
            
            recording_duration = time.time() - state.recording_start_time
            logger.info(f"⏹️ Recording stopped: {state.current_recording_file}")
            logger.info(f"Duration: {recording_duration:.1f} seconds")
            return True
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
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
        source_prefix = "manual" if source in ["web", "api"] else "auto"
        filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count_param}persons_{state.screenshot_count:04d}.jpg"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving

        # Save the frame
        success = cv2.imwrite(filepath, frame_bgr)
        
        if success:
            state.screenshot_count += 1
            logger.info(f"📸 Screenshot saved ({source}): {filename}")
            return True
        else:
            logger.error(f"Failed to save screenshot: {filename}")
            return False
    
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
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
                logger.info(f"👤 Human detected! Starting {COUNTDOWN_DURATION}-second countdown...")
            
            # If countdown is active
            if state.countdown_active:
                elapsed_time = current_time - state.countdown_start_time
                
                if elapsed_time >= COUNTDOWN_DURATION:
                    # Countdown finished, take screenshot
                    save_screenshot(output_frame.copy(), humans_count_now, "auto")
                    state.countdown_active = False
                    logger.info("✅ Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if state.countdown_active:
                # Cancel countdown if human disappears
                state.countdown_active = False
                logger.info("❌ Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        state.last_human_detected = human_detected_now
        
    except Exception as e:
        logger.error(f"Error in auto screenshot handler: {e}")

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
                logger.info(f"📸 Manual screenshot requested ({source})")
                return True
    except Exception as e:
        logger.error(f"Screenshot request error: {e}")
        return False

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
        logger.error(f"Recording frame error: {e}")

# =============================================================================
# FLASK WEB INTERFACE SYSTEM (ENHANCED FOR HEADLESS)
# =============================================================================

class FlaskWebInterface:
    """Enhanced Flask-SocketIO web interface for headless operation"""
    
    def __init__(self):
        # Flask and SocketIO setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'tello_headless_secret_key'
        
        # CORS for any frontend
        CORS(self.app, 
            origins=["*"],  # Allow all origins for headless deployment
            allow_headers=["Content-Type"],
            methods=["GET", "POST", "PUT", "DELETE"])
        
        # Socket.IO initialization
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",  # Allow all origins
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            transports=['polling', 'websocket']
        )
        
        self.setup_routes()
        self.setup_socket_events()
        
        logger.info("🌐 Flask-SocketIO web interface initialized")

    def get_system_stats(self):
        """Get comprehensive system statistics"""
        uptime = time.time() - state.system_start_time
        return {
            'system': {
                'uptime_seconds': uptime,
                'uptime_formatted': f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
                'total_frames_processed': state.total_frames_processed,
                'total_humans_detected': state.total_humans_detected,
                'total_red_detections': state.total_red_detections,
                'current_fps': state.fps,
                'thread_count': len(state.threads),
                'threads_running': state.running
            },
            'drone': {
                'connected': state.is_connected,
                'flying': state.is_flying,
                'battery': state.battery_level,
                'speed': state.speed,
                'temperature': state.temp,
                'height': state.height,
                'barometer': state.baro,
                'time_of_flight': state.tof
            },
            'detection': {
                'enabled': state.detection_enabled,
                'auto_screenshot': state.auto_screenshot_enabled,
                'humans_current': state.humans_count,
                'countdown_active': state.countdown_active
            },
            'recording': {
                'active': state.recording,
                'current_file': state.current_recording_file,
                'duration': time.time() - state.recording_start_time if state.recording else 0
            },
            'web': {
                'connected_clients': state.connected_clients,
                'streaming': state.socket_streaming
            }
        }

    def scan_media_directory(self, file_type='all'):
        """Scan directories and return detailed file information"""
        try:
            files = []
            
            # Scan both screenshots and recordings
            directories = {
                'screenshots': SCREENSHOTS_DIR,
                'recordings': RECORDINGS_DIR
            }
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'}
            
            for dir_type, directory in directories.items():
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    continue
                
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    
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
                            'directory': dir_type,
                            'size': stat.st_size,
                            'size_formatted': self.format_file_size(stat.st_size),
                            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'humans_detected': humans_detected,
                            'url': f'http://{WEB_HOST}:{WEB_PORT}/media/{dir_type}/{filename}',
                            'download_url': f'http://{WEB_HOST}:{WEB_PORT}/download/{dir_type}/{filename}',
                            'type': media_type,
                            'mime_type': mime_type or f'{media_type}/unknown',
                            'extension': file_ext[1:] if file_ext else 'unknown'
                        }
                        files.append(file_info)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {filename}: {e}")
                        continue
            
            # Sort by modified time (newest first)
            files.sort(key=lambda x: x['modified_at'], reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Error scanning media directory: {e}")
            return []

    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"

    def setup_routes(self):
        """Setup Flask routes with comprehensive API"""
        
        @self.app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tello Headless Control</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
                    .status.online { background: #d4edda; color: #155724; }
                    .status.offline { background: #f8d7da; color: #721c24; }
                    .video-container { text-align: center; }
                    .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                    button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
                    .btn-primary { background: #007bff; color: white; }
                    .btn-danger { background: #dc3545; color: white; }
                    .btn-success { background: #28a745; color: white; }
                    .btn-warning { background: #ffc107; color: #212529; }
                    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
                    .stat-item { padding: 10px; background: #f8f9fa; border-radius: 4px; }
                    .links { margin: 20px 0; }
                    .links a { display: inline-block; margin: 5px 10px 5px 0; padding: 8px 16px; background: #6c757d; color: white; text-decoration: none; border-radius: 4px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <h1>🚁 Tello Headless Control System</h1>
                        <p>5-Thread Architecture • Web-Only Interface • Server Ready</p>
                        
                        <div class="links">
                            <a href="/api/status">📊 System Status</a>
                            <a href="/api/stats">📈 Statistics</a>
                            <a href="/api/media/list">📁 Media Files</a>
                            <a href="/api/docs">📚 API Documentation</a>
                            <a href="/video_feed">📹 Raw Video Feed</a>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>📹 Live Video Stream</h3>
                        <div class="video-container">
                            <img src="/video_feed" style="max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px;" alt="Tello Live Stream">
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>🎮 Basic Controls</h3>
                        <div class="controls">
                            <button class="btn-success" onclick="sendCommand('takeoff')">🚁 Takeoff</button>
                            <button class="btn-warning" onclick="sendCommand('land')">🏠 Land</button>
                            <button class="btn-primary" onclick="sendCommand('screenshot')">📸 Screenshot</button>
                            <button class="btn-primary" onclick="sendCommand('toggle_recording')">🔴 Toggle Recording</button>
                            <button class="btn-danger" onclick="sendCommand('emergency')">🚨 Emergency</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>📊 System Information</h3>
                        <div id="system-stats">Loading...</div>
                    </div>
                </div>
                
                <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
                <script>
                    const socket = io();
                    
                    function sendCommand(command) {
                        fetch('/api/command', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({command: command})
                        }).then(response => response.json())
                          .then(data => alert(data.message || 'Command sent'));
                    }
                    
                    function updateStats() {
                        fetch('/api/stats')
                            .then(response => response.json())
                            .then(data => {
                                const statsDiv = document.getElementById('system-stats');
                                const stats = data.stats;
                                statsDiv.innerHTML = `
                                    <div class="stats-grid">
                                        <div class="stat-item">
                                            <strong>Drone Status:</strong><br>
                                            <span class="status ${stats.drone.connected ? 'online' : 'offline'}">
                                                ${stats.drone.connected ? 'Connected' : 'Disconnected'}
                                            </span><br>
                                            Battery: ${stats.drone.battery}%<br>
                                            Flying: ${stats.drone.flying ? 'Yes' : 'No'}
                                        </div>
                                        <div class="stat-item">
                                            <strong>Detection:</strong><br>
                                            Enabled: ${stats.detection.enabled ? 'Yes' : 'No'}<br>
                                            Humans Current: ${stats.detection.humans_current}<br>
                                            Total Detected: ${stats.system.total_humans_detected}
                                        </div>
                                        <div class="stat-item">
                                            <strong>System:</strong><br>
                                            Uptime: ${stats.system.uptime_formatted}<br>
                                            FPS: ${stats.system.current_fps.toFixed(1)}<br>
                                            Frames: ${stats.system.total_frames_processed}
                                        </div>
                                        <div class="stat-item">
                                            <strong>Web Interface:</strong><br>
                                            Clients: ${stats.web.connected_clients}<br>
                                            Streaming: ${stats.web.streaming ? 'Active' : 'Inactive'}<br>
                                            Recording: ${stats.recording.active ? 'Active' : 'Inactive'}
                                        </div>
                                    </div>
                                `;
                            });
                    }
                    
                    // Update stats every 2 seconds
                    setInterval(updateStats, 2000);
                    updateStats();
                    
                    socket.on('connect', function() {
                        console.log('Connected to server');
                    });
                </script>
            </body>
            </html>
            """)

        @self.app.route('/api/docs')
        def api_docs():
            return jsonify({
                'title': 'Tello Headless Control API',
                'version': '1.0.0',
                'endpoints': {
                    'GET /': 'Web interface',
                    'GET /api/status': 'Current system status',
                    'GET /api/stats': 'Comprehensive statistics',
                    'GET /api/media/list': 'List media files',
                    'GET /api/media/stats': 'Media statistics',
                    'POST /api/command': 'Send drone commands',
                    'POST /api/screenshot': 'Take manual screenshot',
                    'GET /video_feed': 'Live video stream',
                    'GET /media/<type>/<filename>': 'Serve media files',
                    'GET /download/<type>/<filename>': 'Download media files',
                    'DELETE /api/media/<type>/<filename>': 'Delete media file'
                },
                'socket_events': {
                    'connect': 'Client connected',
                    'disconnect': 'Client disconnected',
                    'video_frame': 'Real-time video frame',
                    'tello_status': 'Drone status updates',
                    'system_stats': 'System statistics'
                },
                'commands': [
                    'takeoff', 'land', 'emergency', 'screenshot', 
                    'toggle_recording', 'toggle_detection', 
                    'toggle_autonomous', 'flip_forward', 'flip_back',
                    'flip_left', 'flip_right'
                ]
            })

        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'status': {
                    'drone_connected': state.is_connected,
                    'drone_flying': state.is_flying,
                    'battery': state.battery_level,
                    'detection_enabled': state.detection_enabled,
                    'recording': state.recording,
                    'connected_clients': state.connected_clients,
                    'system_running': state.running
                }
            })

        @self.app.route('/api/stats')
        def get_stats():
            return jsonify({
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'stats': self.get_system_stats()
            })

        @self.app.route('/api/command', methods=['POST'])
        def handle_command():
            try:
                data = request.get_json()
                command = data.get('command')
                
                if not command:
                    return jsonify({'success': False, 'error': 'No command provided'}), 400
                
                # Handle different commands
                if command == 'takeoff':
                    if not state.send_rc_control and state.is_connected:
                        state.command_queue.put("takeoff")
                        return jsonify({'success': True, 'message': 'Takeoff command sent'})
                    else:
                        return jsonify({'success': False, 'error': 'Cannot takeoff: drone not ready'})
                
                elif command == 'land':
                    if state.send_rc_control and state.is_connected:
                        state.command_queue.put("land")
                        return jsonify({'success': True, 'message': 'Land command sent'})
                    else:
                        return jsonify({'success': False, 'error': 'Cannot land: drone not flying'})
                
                elif command == 'emergency':
                    state.command_queue.put("emergency")
                    return jsonify({'success': True, 'message': 'Emergency command sent'})
                
                elif command == 'screenshot':
                    success = request_manual_screenshot("api")
                    return jsonify({
                        'success': success, 
                        'message': 'Screenshot requested' if success else 'Screenshot failed'
                    })
                
                elif command == 'toggle_recording':
                    toggle_recording()
                    return jsonify({
                        'success': True, 
                        'message': f"Recording {'started' if state.recording else 'stopped'}"
                    })
                
                elif command == 'toggle_detection':
                    state.detection_enabled = not state.detection_enabled
                    return jsonify({
                        'success': True, 
                        'message': f"Detection {'enabled' if state.detection_enabled else 'disabled'}"
                    })
                
                elif command == 'toggle_autonomous':
                    state.set_autonomous_behavior = not state.set_autonomous_behavior
                    return jsonify({
                        'success': True, 
                        'message': f"Autonomous behavior {'enabled' if state.set_autonomous_behavior else 'disabled'}"
                    })
                
                elif command.startswith('flip_'):
                    flip_direction = command.split('_')[1]
                    state.command_queue.put(f"flip {flip_direction[0]}")
                    return jsonify({'success': True, 'message': f'Flip {flip_direction} command sent'})
                
                else:
                    return jsonify({'success': False, 'error': f'Unknown command: {command}'}), 400
                    
            except Exception as e:
                logger.error(f"Error handling command: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/screenshot', methods=['POST'])
        def take_screenshot():
            try:
                success = request_manual_screenshot("api")
                return jsonify({
                    'success': success,
                    'message': 'Screenshot captured' if success else 'Screenshot failed',
                    'count': state.screenshot_count
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.frame_generator(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/media/<media_type>/<filename>')
        def serve_media(media_type, filename):
            try:
                if media_type == 'screenshots':
                    directory = SCREENSHOTS_DIR
                elif media_type == 'recordings':
                    directory = RECORDINGS_DIR
                else:
                    return "Invalid media type", 400
                
                response = send_from_directory(
                    directory, 
                    filename,
                    as_attachment=False
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
            except Exception as e:
                logger.error(f"Error serving media file {filename}: {e}")
                return "File not found", 404

        @self.app.route('/download/<media_type>/<filename>')
        def download_media(media_type, filename):
            try:
                if media_type == 'screenshots':
                    directory = SCREENSHOTS_DIR
                elif media_type == 'recordings':
                    directory = RECORDINGS_DIR
                else:
                    return "Invalid media type", 400
                
                response = send_from_directory(
                    directory, 
                    filename, 
                    as_attachment=True,
                    download_name=filename
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
            except Exception as e:
                logger.error(f"Error downloading file {filename}: {e}")
                return "File not found", 404

        @self.app.route('/api/media/list')
        def list_media_files():
            try:
                file_type = request.args.get('type', 'all')
                files = self.scan_media_directory(file_type)
                
                return jsonify({
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'files': []
                }), 500

        @self.app.route('/api/media/stats')
        def get_media_statistics():
            try:
                all_files = self.scan_media_directory('all')
                
                stats = {
                    'total_files': len(all_files),
                    'total_images': len([f for f in all_files if f['type'] == 'image']),
                    'total_videos': len([f for f in all_files if f['type'] == 'video']),
                    'total_size': sum(f['size'] for f in all_files),
                    'total_size_formatted': self.format_file_size(sum(f['size'] for f in all_files)),
                    'files_with_humans': len([f for f in all_files if f['humans_detected'] > 0]),
                    'total_humans_detected': sum(f['humans_detected'] for f in all_files),
                    'screenshots': len([f for f in all_files if f['directory'] == 'screenshots']),
                    'recordings': len([f for f in all_files if f['directory'] == 'recordings'])
                }
                
                return jsonify({'success': True, 'stats': stats})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/media/<media_type>/<filename>', methods=['DELETE'])
        def delete_media(media_type, filename):
            try:
                if media_type == 'screenshots':
                    directory = SCREENSHOTS_DIR
                elif media_type == 'recordings':
                    directory = RECORDINGS_DIR
                else:
                    return jsonify({'success': False, 'error': 'Invalid media type'}), 400
                
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"🗑️ Deleted file: {filename}")
                    return jsonify({'success': True, 'message': f'File {filename} deleted'})
                else:
                    return jsonify({'success': False, 'error': 'File not found'}), 404
            except Exception as e:
                logger.error(f"Delete error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

    def setup_socket_events(self):
        """Setup Socket.IO events for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            state.connected_clients += 1
            logger.info(f'✅ Client connected. Total clients: {state.connected_clients}')
            
            self.broadcast_status()
            emit('system_stats', self.get_system_stats())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            state.connected_clients -= 1
            logger.info(f'❌ Client disconnected. Total clients: {state.connected_clients}')
            
            if state.connected_clients <= 0:
                state.socket_streaming = False

        @self.socketio.on('start_stream')
        def handle_start_stream():
            logger.info("🎥 Client requested video stream")
            state.socket_streaming = True
            emit('stream_status', {'streaming': True, 'message': 'Video stream started'})
        
        @self.socketio.on('stop_stream')
        def handle_stop_stream():
            logger.info("⏹️ Client stopped video stream")
            state.socket_streaming = False
            emit('stream_status', {'streaming': False, 'message': 'Video stream stopped'})
        
        @self.socketio.on('drone_command')
        def handle_drone_command(data):
            command = data.get('command')
            logger.info(f"🎮 Drone command from client: {command}")
            
            # Handle movement commands
            if command == 'move':
                if state.send_rc_control and not state.disconnect_requested:
                    state.left_right_velocity = int(data.get('left_right', 0))
                    state.for_back_velocity = int(data.get('for_back', 0))
                    state.up_down_velocity = int(data.get('up_down', 0))
                    state.yaw_velocity = int(data.get('yaw', 0))
            
            elif command == 'stop':
                state.left_right_velocity = 0
                state.for_back_velocity = 0
                state.up_down_velocity = 0
                state.yaw_velocity = 0
            
            elif command in ['takeoff', 'land', 'emergency']:
                state.command_queue.put(command)
            
            elif command == 'screenshot':
                request_manual_screenshot("web")
            
            elif command == 'toggle_recording':
                toggle_recording()

        @self.socketio.on('get_stats')
        def handle_get_stats():
            emit('system_stats', self.get_system_stats())

    def broadcast_status(self):
        """Broadcast status to all connected clients"""
        try:
            flight_time = 0
            if state.flight_start_time and state.is_flying:
                flight_time = int(time.time() - state.flight_start_time)
            
            status = {
                'connected': state.is_connected,
                'flying': state.is_flying,
                'battery': state.battery_level,
                'flight_time': flight_time,
                'speed': state.current_speed_display,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('tello_status', status)
        except Exception as e:
            logger.error(f"Broadcast status error: {e}")

    def broadcast_stats(self):
        """Broadcast comprehensive stats to all clients"""
        try:
            stats = self.get_system_stats()
            self.socketio.emit('system_stats', stats)
        except Exception as e:
            logger.error(f"Broadcast stats error: {e}")

    def frame_generator(self):
        """Generator for streaming frames to browser"""
        while not state.should_stop:
            with state.frame_lock:
                if state.last_frame is None:
                    time.sleep(0.1)
                    continue
                
                frame_to_send = state.last_frame.copy()
                
                # Add system info overlay for web stream
                self.add_web_overlays(frame_to_send)
                
                frame_bgr = cv2.cvtColor(frame_to_send, cv2.COLOR_RGB2BGR)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
                ret, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)
                
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                time.sleep(1 / 30)  # 30 FPS for web stream

    def add_web_overlays(self, frame):
        """Add information overlays optimized for web viewing"""
        try:
            # Add comprehensive overlay information
            overlay_data = [
                f"🔋 Battery: {state.battery_level}%",
                f"📊 FPS: {state.fps:.1f}",
                f"🔍 Detection: {'ON' if state.detection_enabled else 'OFF'}",
                f"👥 Humans: {state.humans_count}",
                f"📸 Screenshots: {state.screenshot_count}",
                f"🌐 Clients: {state.connected_clients}",
                f"📡 Headless Mode Active"
            ]
            
            if state.recording:
                duration = time.time() - state.recording_start_time
                overlay_data.append(f"🔴 REC {duration:.1f}s")
            
            if state.countdown_active:
                elapsed = time.time() - state.countdown_start_time
                remaining = max(0, COUNTDOWN_DURATION - elapsed)
                overlay_data.append(f"⏱️ Screenshot: {remaining:.1f}s")
            
            # Draw overlays
            y_offset = 25
            for i, text in enumerate(overlay_data):
                color = (0, 255, 0) if not text.startswith('🔴') else (0, 0, 255)
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw ROI rectangle if detection enabled
            if state.detection_enabled:
                cv2.rectangle(frame, (ROI_X, ROI_Y), 
                             (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), 
                             (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (ROI_X, ROI_Y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        except Exception as e:
            logger.error(f"Web overlay error: {e}")

    def send_frame_to_clients(self, frame):
        """Send frame to connected clients via Socket.IO"""
        if state.socket_streaming and state.connected_clients > 0:
            try:
                # Resize for web streaming
                frame_resized = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
                self.add_web_overlays(frame_resized)
                frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                ret, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)
                if ret:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.socketio.emit('video_frame', {'frame': frame_base64})
            except Exception as e:
                logger.error(f"Error sending frame to clients: {e}")

    def run_server(self):
        """Run Flask server with Socket.IO"""
        logger.info(f"🌐 Starting Flask server on {WEB_HOST}:{WEB_PORT}")
        logger.info(f"🔗 Web Interface: http://{WEB_HOST}:{WEB_PORT}")
        logger.info(f"📖 API Documentation: http://{WEB_HOST}:{WEB_PORT}/api/docs")
        
        self.socketio.run(
            self.app, 
            host=WEB_HOST,
            port=WEB_PORT, 
            debug=WEB_DEBUG, 
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )

# Global web interface instance
web_interface = FlaskWebInterface()

# =============================================================================
# THREAD MANAGEMENT SYSTEM (OPTIMIZED FOR HEADLESS)
# =============================================================================

def video_stream_thread():
    """Thread 1: Handle video capture, processing, and screenshot operations"""
    logger.info("📹 Video stream thread started")
    
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
                    state.total_frames_processed += 1
                    
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
                    web_interface.send_frame_to_clients(output_frame)
                
                time.sleep(VIDEO_THREAD_SLEEP)
                
            except Exception as e:
                logger.error(f"Video stream error: {e}")
                time.sleep(0.1)
        
    except Exception as e:
        logger.error(f"Critical video stream error: {e}")
    
    logger.info("📹 Video stream thread ended")

def drone_control_thread():
    """Thread 2: Handle drone control commands and battery monitoring"""
    logger.info("🎮 Drone control thread started")
    last_battery_check = time.time()
    last_telemetry_update = time.time()
    last_stats_broadcast = time.time()
    telemetry_update_interval = 1.0
    stats_broadcast_interval = 5.0
    
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
                    logger.error(f"RC command error: {rc_error}")
            
            current_time = time.time()
            
            # Update battery level periodically
            if current_time - last_battery_check >= BATTERY_CHECK_INTERVAL:
                try:
                    if state.is_connected:
                        with state.data_lock:
                            state.battery_level = state.tello.get_battery()
                    last_battery_check = current_time
                except Exception as e:
                    logger.error(f"Battery check error: {e}")
            
            # Update telemetry data periodically
            if current_time - last_telemetry_update >= telemetry_update_interval:
                try:
                    if state.is_connected:
                        update_telemetry_data()
                    last_telemetry_update = current_time
                except Exception as e:
                    logger.error(f"Telemetry update error: {e}")
            
            # Broadcast stats to web clients periodically
            if current_time - last_stats_broadcast >= stats_broadcast_interval:
                try:
                    web_interface.broadcast_stats()
                    web_interface.broadcast_status()
                    last_stats_broadcast = current_time
                except Exception as e:
                    logger.error(f"Stats broadcast error: {e}")
            
            time.sleep(CONTROL_THREAD_SLEEP)
            
        except Exception as e:
            logger.error(f"Drone control error: {e}")
            time.sleep(0.1)
    
    logger.info("🎮 Drone control thread ended")

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
        logger.error(f"Error updating telemetry: {e}")

def detection_thread():
    """Thread 3: Handle red color detection"""
    logger.info("🔍 Detection thread started")
    
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
            logger.error(f"Detection thread error: {e}")
            time.sleep(0.1)
    
    logger.info("🔍 Detection thread ended")

def autonomous_behavior_thread():
    """Thread 4: Handle autonomous behavior based on detection results"""
    logger.info("🤖 Autonomous behavior thread started")
    
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
                    logger.info("🔴 Red detected in ROI! Executing autonomous behavior...")
                    try:
                        state.tello.move_back(70)
                        time.sleep(2)
                        state.tello.rotate_clockwise(90)
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Autonomous movement error: {e}")
                else:
                    logger.info("⚪ No red in ROI. Searching...")
                    try:
                        state.tello.move_forward(30)
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Autonomous search error: {e}")
            else:
                time.sleep(AUTONOMOUS_THREAD_SLEEP)
                
        except Exception as e:
            logger.error(f"Autonomous behavior error: {e}")
            time.sleep(0.5)
    
    logger.info("🤖 Autonomous behavior thread ended")

def recording_thread_func():
    """Thread 5: Dedicated thread for smooth recording"""
    logger.info("🎥 Recording thread started")
    
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
            logger.error(f"Recording thread error: {e}")
            time.sleep(0.1)
    
    logger.info("🎥 Recording thread ended")

def execute_drone_command(command):
    """Execute drone commands safely"""
    try:
        logger.info(f"🎯 Executing command: {command}")
        if command == "takeoff":
            state.tello.takeoff()
            state.send_rc_control = True
            state.is_flying = True
            state.flight_start_time = time.time()
            logger.info("✅ Takeoff completed, RC control enabled")
            web_interface.broadcast_status()
        elif command == "land":
            state.tello.land()
            state.send_rc_control = False
            state.is_flying = False
            state.flight_start_time = None
            logger.info("✅ Landing completed, RC control disabled")
            web_interface.broadcast_status()
        elif command == "emergency":
            state.tello.emergency()
            state.send_rc_control = False
            state.is_flying = False
            state.flight_start_time = None
            logger.info("🚨 Emergency landing executed")
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
        logger.error(f"Command execution error: {e}")

def start_all_threads():
    """Start all worker threads"""
    logger.info("🚀 Starting all threads...")
    
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
        logger.info(f"✅ Started thread: {name}")
    
    logger.info(f"🎯 All {len(state.threads)} worker threads started successfully")

def stop_all_threads():
    """Stop all worker threads"""
    logger.info("⏹️ Stopping all threads...")
    state.running = False
    
    # Stop recording if active
    if state.recording:
        stop_recording()
    
    # Wait for threads to finish
    for thread in state.threads:
        if thread.is_alive():
            logger.info(f"Waiting for thread: {thread.name}")
            thread.join(timeout=3)
    
    cleanup_systems()
    logger.info("✅ All threads stopped successfully")

# =============================================================================
# SIGNAL HANDLERS FOR GRACEFUL SHUTDOWN
# =============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
    state.shutdown_requested = True
    state.should_stop = True
    stop_all_threads()
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tello Headless Control System')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Web server port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-tello', action='store_true', help='Run without Tello connection (testing)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    return parser.parse_args()

def main():
    """Main function - Entry point of the headless application"""
    # Parse arguments
    args = parse_arguments()
    
    # Update configuration based on arguments
    global WEB_HOST, WEB_PORT, WEB_DEBUG
    WEB_HOST = args.host
    WEB_PORT = args.port
    WEB_DEBUG = args.debug
    
    # Setup logging level
    logging.getLogger('TelloHeadless').setLevel(getattr(logging, args.log_level))
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info("=" * 80)
    logger.info("🚁 TELLO HEADLESS CONTROL SYSTEM")
    logger.info("=" * 80)
    logger.info("🎯 5-Thread Architecture • Pure Web Interface • Server Ready")
    logger.info("=" * 80)
    logger.info("📋 System Information:")
    logger.info(f"   • Host: {WEB_HOST}")
    logger.info(f"   • Port: {WEB_PORT}")
    logger.info(f"   • Debug: {WEB_DEBUG}")
    logger.info(f"   • Log Level: {args.log_level}")
    logger.info(f"   • Tello Connection: {'Disabled' if args.no_tello else 'Enabled'}")
    logger.info("=" * 80)
    logger.info("🔧 Architecture:")
    logger.info("   • Thread 1: Video stream and human detection")
    logger.info("   • Thread 2: Drone control commands and battery monitoring") 
    logger.info("   • Thread 3: Red color detection")
    logger.info("   • Thread 4: Autonomous behavior")
    logger.info("   • Thread 5: Dedicated smooth recording")
    logger.info("   • Flask-SocketIO: Web interface server")
    logger.info("=" * 80)
    logger.info("✨ Features:")
    logger.info("   • Smart auto screenshot with countdown")
    logger.info("   • Time-based video recording")
    logger.info("   • Human detection with pose tracking")
    logger.info("   • Red color detection with ROI")
    logger.info("   • Real-time video streaming")
    logger.info("   • RESTful API endpoints")
    logger.info("   • Enhanced media gallery")
    logger.info("   • Docker/Server deployment ready")
    logger.info("=" * 80)
    logger.info("🌐 Web Interface:")
    logger.info(f"   • Main Interface: http://{WEB_HOST}:{WEB_PORT}")
    logger.info(f"   • API Documentation: http://{WEB_HOST}:{WEB_PORT}/api/docs")
    logger.info(f"   • Live Video Feed: http://{WEB_HOST}:{WEB_PORT}/video_feed")
    logger.info(f"   • System Status: http://{WEB_HOST}:{WEB_PORT}/api/status")
    logger.info("=" * 80)
    
    try:
        # Initialize all systems
        if not args.no_tello:
            if not initialize_all_systems():
                logger.error("❌ Failed to initialize systems. Exiting...")
                return 1
        else:
            logger.warning("⚠️ Running in NO-TELLO mode for testing")
            if not create_directories() or not initialize_ai_models():
                logger.error("❌ Failed to initialize test systems. Exiting...")
                return 1
        
        # Start Flask web server in separate thread
        logger.info("🌐 Starting web server...")
        flask_thread = threading.Thread(target=web_interface.run_server, daemon=True)
        flask_thread.start()
        
        # Start all worker threads
        if not args.no_tello:
            start_all_threads()
            
            # Wait for threads to initialize
            time.sleep(3)
            
            # Start web streaming
            state.socket_streaming = True
        
        logger.info("=" * 80)
        logger.info("🚀 SYSTEM READY!")
        logger.info("=" * 80)
        logger.info("📱 Access the web interface to control your drone")
        logger.info("🔗 All controls available via HTTP API and WebSocket")
        logger.info("📊 Real-time monitoring and statistics available")
        logger.info("🐳 Docker-ready for containerized deployment")
        logger.info("=" * 80)
        logger.info("💡 TIP: Use Ctrl+C for graceful shutdown")
        logger.info("=" * 80)
        
        # Keep main thread alive
        try:
            while not state.shutdown_requested:
                time.sleep(1)
                
                # Periodic health check
                if state.running and not args.no_tello:
                    # Check if critical threads are still alive
                    alive_threads = [t for t in state.threads if t.is_alive()]
                    if len(alive_threads) < len(state.threads):
                        logger.warning(f"⚠️ Thread health check: {len(alive_threads)}/{len(state.threads)} threads alive")
                        
        except KeyboardInterrupt:
            logger.info("⌨️ Keyboard interrupt received")
        
    except Exception as e:
        logger.error(f"💥 Critical error occurred: {e}")
        return 1
    finally:
        logger.info("🛑 Initiating shutdown sequence...")
        stop_all_threads()
        logger.info("👋 Shutdown completed successfully")
        return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)