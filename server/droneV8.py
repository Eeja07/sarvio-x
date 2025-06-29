#!/usr/bin/env python3
"""
Drone Web Bridge Server - Single File Integration
Mengintegrasikan droneV7.py dengan React Frontend tanpa mengubah struktur asli

Menjalankan:
1. Backend only: python drone_web_bridge.py --backend-only
2. Drone only: python drone_web_bridge.py --drone-only  
3. Full integration: python drone_web_bridge.py --integrated

Dependencies:
pip install flask flask-socketio flask-cors djitellopy ultralytics mediapipe pygame opencv-python numpy
"""

import sys
import os
import threading
import queue
import time
import json
import base64
import cv2
import numpy as np
from datetime import datetime
from collections import deque
import subprocess
import signal
import atexit

# Web server imports
try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    WEB_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️ Web server dependencies not available. Install: pip install flask flask-socketio flask-cors")
    WEB_IMPORTS_AVAILABLE = False

# Drone imports (from droneV7.py)
try:
    import pygame
    import cv2
    import numpy as np
    from djitellopy import Tello
    from ultralytics import YOLO
    import mediapipe as mp
    DRONE_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️ Drone dependencies not available. Install required packages.")
    DRONE_IMPORTS_AVAILABLE = False

# ==================== CONFIGURATION ====================
class Config:
    # Web server config
    WEB_HOST = '127.0.0.1'
    WEB_PORT = 5000
    WEB_DEBUG = False
    
    # Drone config (from droneV7.py)
    FPS = 120
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    SPEED = 50
    THREAD_COUNT = 5
    
    # Integration config
    FRAME_SHARE_FILE = "shared_frame.jpg"
    STATUS_SHARE_FILE = "shared_status.json"
    COMMAND_SHARE_FILE = "shared_commands.json"
    
    # Directories
    SCREENSHOTS_DIR = "screenshots"
    RECORDINGS_DIR = "recordings"
    MEDIA_API_DIR = "media_files"

# ==================== SHARED DATA MANAGEMENT ====================
class SharedDataManager:
    """Manages communication between drone system and web server"""
    
    def __init__(self):
        self.status_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        # Shared status data
        self.status_data = {
            'connected': False,
            'flying': False,
            'battery': 0,
            'speed': 50,
            'temperature': 0,
            'height': 0,
            'humans_detected': 0,
            'fps': 0,
            'recording': False,
            'ml_detection_enabled': False,
            'auto_capture_enabled': False,
            'screenshot_count': 0,
            'flight_time': 0,
            'telemetry': {
                'pitch': 0, 'roll': 0, 'yaw': 0,
                'speed_x': 0, 'speed_y': 0, 'speed_z': 0,
                'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
                'barometer': 0, 'tof': 0
            }
        }
        
        # Command queue
        self.command_queue = queue.Queue()
        
        # Current frame
        self.current_frame = None
        self.current_frame_base64 = None
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR, Config.MEDIA_API_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def update_status(self, status_update):
        """Update status data thread-safely"""
        with self.status_lock:
            self.status_data.update(status_update)
    
    def get_status(self):
        """Get current status data"""
        with self.status_lock:
            return self.status_data.copy()
    
    def add_command(self, command):
        """Add command to queue"""
        self.command_queue.put(command)
    
    def get_command(self):
        """Get command from queue (non-blocking)"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def update_frame(self, frame):
        """Update current frame"""
        with self.frame_lock:
            if frame is not None:
                self.current_frame = frame.copy()
                # Convert to base64 for web transmission
                _, buffer = cv2.imencode('.jpg', frame)
                self.current_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    def get_frame_base64(self):
        """Get current frame as base64"""
        with self.frame_lock:
            return self.current_frame_base64

# Global shared data manager
shared_data = SharedDataManager()

# ==================== DRONE SYSTEM (Modified droneV7.py) ====================
class DroneSystem:
    """
    Modified drone system from droneV7.py with web integration hooks
    Struktur asli dipertahankan, hanya ditambahkan web integration points
    """
    
    def __init__(self):
        # Original droneV7.py variables
        self.running = True
        self.threads = []
        
        # Tello objects
        self.tello = None
        self.screen = None
        self.joystick = None
        
        # AI models
        self.yolo_model = None
        self.pose = None
        self.hands = None
        
        # Drone state
        self.current_frame = None
        self.current_processed_frame = None
        self.battery_level = 0
        self.human_detected = False
        self.humans_count = 0
        self.fps = 0
        
        # Control variables
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.send_rc_control = False
        self.speed = 50
        
        # Recording
        self.recording = False
        self.video_writer = None
        
        # Detection
        self.detection_enabled = True
        
        # Locks
        self.data_lock = threading.Lock()
        
        # Web integration flag
        self.web_integration_enabled = True
        
        print("🚁 Drone system initialized with web integration")
    
    def initialize_all_systems(self):
        """Initialize all drone systems"""
        try:
            if not self._initialize_pygame():
                return False
            if not self._initialize_tello():
                return False
            if not self._initialize_ai_models():
                return False
            
            print("✅ All drone systems initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize drone systems: {e}")
            return False
    
    def _initialize_pygame(self):
        """Initialize pygame (headless mode for web)"""
        try:
            if self.web_integration_enabled:
                # Headless mode - no display
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
            pygame.init()
            pygame.display.set_caption("Tello video stream")
            
            if not self.web_integration_enabled:
                self.screen = pygame.display.set_mode([Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT])
            
            # Initialize joystick
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"🎮 Joystick initialized: {self.joystick.get_name()}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize pygame: {e}")
            return False
    
    def _initialize_tello(self):
        """Initialize Tello drone connection"""
        try:
            self.tello = Tello()
            self.tello.connect()
            self.tello.set_speed(self.speed)
            self.battery_level = self.tello.get_battery()
            print(f"🔋 Battery: {self.battery_level}%")
            
            # Start video stream
            self.tello.streamoff()
            time.sleep(0.5)
            self.tello.streamon()
            
            # Update shared data
            shared_data.update_status({
                'connected': True,
                'battery': self.battery_level,
                'speed': self.speed
            })
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Tello: {e}")
            shared_data.update_status({'connected': False})
            return False
    
    def _initialize_ai_models(self):
        """Initialize AI models"""
        try:
            print("🤖 Loading AI models...")
            self.yolo_model = YOLO('yolov8n.pt')
            
            # MediaPipe
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                model_complexity=1
            )
            
            self.hands = mp_hands.Hands(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                max_num_hands=2
            )
            
            print("✅ AI models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AI models: {e}")
            return False
    
    def start_drone_threads(self):
        """Start all drone threads"""
        print("🚀 Starting drone threads...")
        
        thread_configs = [
            ("Video Stream", self._video_stream_thread),
            ("Drone Control", self._drone_control_thread),
            ("Detection", self._detection_thread),
            ("Web Integration", self._web_integration_thread)
        ]
        
        for name, target_func in thread_configs:
            thread = threading.Thread(target=target_func, daemon=True, name=name)
            thread.start()
            self.threads.append(thread)
        
        print(f"✅ Started {len(self.threads)} drone threads")
    
    def _video_stream_thread(self):
        """Handle video capture and processing"""
        print("📹 Video stream thread started")
        
        try:
            frame_read = self.tello.get_frame_read()
            frame_times = deque(maxlen=30)
            
            while self.running:
                try:
                    if frame_read.stopped:
                        break
                    
                    frame = frame_read.frame
                    if frame is not None:
                        # Resize frame
                        frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                        
                        # Process detection if enabled
                        if self.detection_enabled:
                            output_frame, detected, count = self._process_human_detection(frame)
                        else:
                            output_frame = frame.copy()
                            detected = False
                            count = 0
                        
                        # Update shared data
                        with self.data_lock:
                            self.current_frame = frame.copy()
                            self.current_processed_frame = output_frame.copy()
                            self.human_detected = detected
                            self.humans_count = count
                            
                            # Calculate FPS
                            current_time = time.time()
                            frame_times.append(current_time)
                            if len(frame_times) > 1:
                                time_diff = frame_times[-1] - frame_times[0]
                                self.fps = len(frame_times) / time_diff if time_diff > 0 else 0
                        
                        # Update shared frame for web
                        shared_data.update_frame(output_frame)
                        
                        # Update status
                        shared_data.update_status({
                            'fps': self.fps,
                            'humans_detected': count,
                            'ml_detection_enabled': self.detection_enabled
                        })
                    
                    time.sleep(0.01)  # Control processing speed
                    
                except Exception as e:
                    print(f"❌ Video stream error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Critical video stream error: {e}")
        
        print("📹 Video stream thread ended")
    
    def _drone_control_thread(self):
        """Handle drone control commands"""
        print("🎮 Drone control thread started")
        
        last_battery_check = time.time()
        
        while self.running:
            try:
                # Check for web commands
                command = shared_data.get_command()
                if command:
                    self._execute_web_command(command)
                
                # Send RC control if active
                if self.send_rc_control and self.tello:
                    try:
                        self.tello.send_rc_control(
                            self.left_right_velocity,
                            self.for_back_velocity, 
                            self.up_down_velocity,
                            self.yaw_velocity
                        )
                    except Exception as e:
                        print(f"❌ RC command error: {e}")
                
                # Update battery periodically
                current_time = time.time()
                if current_time - last_battery_check >= 10:  # Every 10 seconds
                    try:
                        if self.tello:
                            self.battery_level = self.tello.get_battery()
                            shared_data.update_status({'battery': self.battery_level})
                        last_battery_check = current_time
                    except Exception as e:
                        print(f"❌ Battery check error: {e}")
                
                time.sleep(1/30)  # 30 FPS control loop
                
            except Exception as e:
                print(f"❌ Drone control error: {e}")
                time.sleep(0.1)
        
        print("🎮 Drone control thread ended")
    
    def _detection_thread(self):
        """Handle AI detection"""
        print("🤖 Detection thread started")
        
        while self.running:
            try:
                if self.detection_enabled and self.current_processed_frame is not None:
                    # Detection already handled in video thread
                    pass
                
                time.sleep(0.03)  # Control detection frequency
                
            except Exception as e:
                print(f"❌ Detection thread error: {e}")
                time.sleep(0.1)
        
        print("🤖 Detection thread ended")
    
    def _web_integration_thread(self):
        """Handle web integration updates"""
        print("🌐 Web integration thread started")
        
        while self.running:
            try:
                # Update telemetry data if available
                if self.tello:
                    try:
                        # Get telemetry data (simplified)
                        state_str = self.tello.get_current_state()
                        telemetry = self._parse_telemetry(state_str)
                        
                        shared_data.update_status({
                            'telemetry': telemetry,
                            'flying': self.send_rc_control
                        })
                    except:
                        pass
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"❌ Web integration error: {e}")
                time.sleep(1)
        
        print("🌐 Web integration thread ended")
    
    def _parse_telemetry(self, state_str):
        """Parse telemetry string"""
        telemetry = {
            'pitch': 0, 'roll': 0, 'yaw': 0,
            'speed_x': 0, 'speed_y': 0, 'speed_z': 0,
            'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
            'barometer': 0, 'tof': 0
        }
        
        try:
            if state_str:
                for item in state_str.split(';'):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        if key in ['pitch', 'roll', 'yaw']:
                            telemetry[key] = float(value)
                        elif key in ['vgx', 'vgy', 'vgz']:
                            telemetry[f'speed_{key[-1]}'] = float(value)
                        elif key in ['agx', 'agy', 'agz']:
                            telemetry[f'accel_{key[-1]}'] = float(value)
                        elif key == 'baro':
                            telemetry['barometer'] = float(value)
                        elif key == 'tof':
                            telemetry['tof'] = float(value)
        except:
            pass
        
        return telemetry
    
    def _process_human_detection(self, frame):
        """Process human detection using YOLO"""
        try:
            output_frame = frame.copy()
            
            # YOLO Human Detection
            results = self.yolo_model(frame, verbose=False)
            
            detected = False
            human_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a person (class_id = 0 in COCO dataset)
                        if class_id == 0 and confidence > 0.5:
                            detected = True
                            human_count += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw bounding box
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label
                            label = f"Human: {confidence*100:.0f}%"
                            cv2.putText(output_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return output_frame, detected, human_count
            
        except Exception as e:
            print(f"❌ Human detection error: {e}")
            return frame, False, 0
    
    def _execute_web_command(self, command):
        """Execute command from web interface"""
        try:
            cmd_type = command.get('type')
            cmd_data = command.get('data', {})
            
            print(f"🎯 Executing web command: {cmd_type}")
            
            if cmd_type == 'takeoff':
                if self.tello and not self.send_rc_control:
                    self.tello.takeoff()
                    self.send_rc_control = True
                    shared_data.update_status({'flying': True})
                    print("✅ Takeoff completed")
            
            elif cmd_type == 'land':
                if self.tello and self.send_rc_control:
                    self.tello.land()
                    self.send_rc_control = False
                    shared_data.update_status({'flying': False})
                    print("✅ Landing completed")
            
            elif cmd_type == 'emergency':
                if self.tello:
                    self.tello.emergency()
                    self.send_rc_control = False
                    shared_data.update_status({'flying': False})
                    print("🚨 Emergency executed")
            
            elif cmd_type == 'move_control':
                controls = cmd_data
                self.left_right_velocity = controls.get('left_right', 0)
                self.for_back_velocity = controls.get('for_back', 0)
                self.up_down_velocity = controls.get('up_down', 0)
                self.yaw_velocity = controls.get('yaw', 0)
            
            elif cmd_type == 'stop_movement':
                self.left_right_velocity = 0
                self.for_back_velocity = 0
                self.up_down_velocity = 0
                self.yaw_velocity = 0
            
            elif cmd_type == 'set_speed':
                self.speed = max(10, min(100, cmd_data.get('speed', 50)))
                if self.tello:
                    self.tello.set_speed(self.speed)
                shared_data.update_status({'speed': self.speed})
                print(f"⚡ Speed set to: {self.speed}")
            
            elif cmd_type == 'flip':
                direction = cmd_data.get('direction', 'f')
                if self.tello and self.send_rc_control:
                    if direction == 'f':
                        self.tello.flip_forward()
                    elif direction == 'b':
                        self.tello.flip_back()
                    elif direction == 'l':
                        self.tello.flip_left()
                    elif direction == 'r':
                        self.tello.flip_right()
                    print(f"🔄 Flip {direction} executed")
            
            elif cmd_type == 'enable_ml_detection':
                self.detection_enabled = cmd_data.get('enabled', True)
                shared_data.update_status({'ml_detection_enabled': self.detection_enabled})
                print(f"🤖 ML Detection: {'ON' if self.detection_enabled else 'OFF'}")
            
            elif cmd_type == 'manual_screenshot':
                self._take_screenshot()
            
            elif cmd_type == 'toggle_recording':
                self._toggle_recording()
            
        except Exception as e:
            print(f"❌ Command execution error: {e}")
    
    def _take_screenshot(self):
        """Take a screenshot"""
        try:
            if self.current_processed_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"web_screenshot_{timestamp}.jpg"
                filepath = os.path.join(Config.SCREENSHOTS_DIR, filename)
                
                frame_bgr = cv2.cvtColor(self.current_processed_frame, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(filepath, frame_bgr)
                
                if success:
                    current_count = shared_data.get_status().get('screenshot_count', 0) + 1
                    shared_data.update_status({'screenshot_count': current_count})
                    print(f"📸 Screenshot saved: {filename}")
                    return True
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
        return False
    
    def _toggle_recording(self):
        """Toggle video recording"""
        try:
            if not self.recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"web_recording_{timestamp}.mp4"
                filepath = os.path.join(Config.RECORDINGS_DIR, filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, 
                                                   (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                
                if self.video_writer.isOpened():
                    self.recording = True
                    shared_data.update_status({'recording': True})
                    print(f"🔴 Recording started: {filename}")
                    return True
            else:
                # Stop recording
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                
                self.recording = False
                shared_data.update_status({'recording': False})
                print("⏹️ Recording stopped")
                return True
        except Exception as e:
            print(f"❌ Recording error: {e}")
        return False
    
    def stop_all_systems(self):
        """Stop all drone systems"""
        print("🛑 Stopping drone systems...")
        self.running = False
        
        # Stop recording if active
        if self.recording:
            self._toggle_recording()
        
        # Cleanup Tello
        if self.tello:
            try:
                self.tello.streamoff()
                self.tello.end()
            except:
                pass
        
        # Cleanup AI models
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
        
        pygame.quit()
        print("✅ Drone systems stopped")

# ==================== WEB SERVER ====================
class WebServer:
    """Flask + Socket.IO web server for React frontend"""
    
    def __init__(self):
        if not WEB_IMPORTS_AVAILABLE:
            raise ImportError("Web server dependencies not available")
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'drone_web_bridge_secret'
        
        # Enable CORS
        CORS(self.app, origins="*")
        
        # Initialize Socket.IO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )
        
        self.setup_routes()
        self.setup_socket_events()
        
        print("🌐 Web server initialized")
    
    def setup_routes(self):
        """Setup REST API routes"""
        
        @self.app.route('/')
        def index():
            return jsonify({
                'message': 'Drone Web Bridge Server',
                'status': 'running',
                'endpoints': {
                    'status': '/api/status',
                    'media': '/api/media/list',
                    'socket': '/socket.io'
                }
            })
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'success': True,
                'data': shared_data.get_status()
            })
        
        @self.app.route('/api/media/list')
        def list_media_files():
            try:
                media_type = request.args.get('type', 'images').lower()
                
                if media_type == 'images':
                    directory = Config.SCREENSHOTS_DIR
                    extensions = ['.jpg', '.jpeg', '.png']
                elif media_type == 'videos':
                    directory = Config.RECORDINGS_DIR
                    extensions = ['.mp4', '.avi', '.mov']
                else:
                    return jsonify({'success': False, 'error': 'Invalid media type'})
                
                files = []
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if any(filename.lower().endswith(ext) for ext in extensions):
                            filepath = os.path.join(directory, filename)
                            stat = os.stat(filepath)
                            
                            files.append({
                                'filename': filename,
                                'size': stat.st_size,
                                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'url': f'/media/{filename}',
                                'humans_detected': 0  # Could be parsed from filename
                            })
                
                # Sort by creation time (newest first)
                files.sort(key=lambda x: x['created_at'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'files': files,
                    'count': len(files)
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/media/<filename>')
        def serve_media(filename):
            """Serve media files"""
            try:
                # Check both directories
                for directory in [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR]:
                    filepath = os.path.join(directory, filename)
                    if os.path.exists(filepath):
                        return send_file(filepath)
                
                return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/download/<filename>')
        def download_media(filename):
            """Download media files"""
            try:
                # Check both directories
                for directory in [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR]:
                    filepath = os.path.join(directory, filename)
                    if os.path.exists(filepath):
                        return send_file(filepath, as_attachment=True)
                
                return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def setup_socket_events(self):
        """Setup Socket.IO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔗 Client connected: {request.sid}")
            # Send current status
            emit('tello_status', shared_data.get_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🔌 Client disconnected: {request.sid}")
        
        @self.socketio.on('connect_tello')
        def handle_connect_tello():
            """Handle Tello connection request"""
            # Tello should already be connected in integrated mode
            status = shared_data.get_status()
            emit('tello_status', status)
        
        @self.socketio.on('disconnect_tello')
        def handle_disconnect_tello():
            """Handle Tello disconnection request"""
            shared_data.add_command({'type': 'emergency'})
            emit('tello_status', {'connected': False, 'flying': False})
        
        @self.socketio.on('takeoff')
        def handle_takeoff():
            """Handle takeoff command"""
            shared_data.add_command({'type': 'takeoff'})
            emit('drone_action', {'action': 'takeoff', 'success': True})
        
        @self.socketio.on('land')
        def handle_land():
            """Handle land command"""
            shared_data.add_command({'type': 'land'})
            emit('drone_action', {'action': 'land', 'success': True})
        
        @self.socketio.on('emergency_land')
        def handle_emergency():
            """Handle emergency command"""
            shared_data.add_command({'type': 'emergency'})
            emit('drone_action', {'action': 'emergency', 'success': True})
        
        @self.socketio.on('move_control')
        def handle_move_control(data):
            """Handle movement control"""
            shared_data.add_command({
                'type': 'move_control',
                'data': {
                    'left_right': data.get('left_right', 0),
                    'for_back': data.get('for_back', 0),
                    'up_down': data.get('up_down', 0),
                    'yaw': data.get('yaw', 0)
                }
            })
        
        @self.socketio.on('stop_movement')
        def handle_stop_movement():
            """Handle stop movement command"""
            shared_data.add_command({'type': 'stop_movement'})
        
        @self.socketio.on('set_speed')
        def handle_set_speed(data):
            """Handle speed setting"""
            speed = data.get('speed', 50)
            shared_data.add_command({
                'type': 'set_speed',
                'data': {'speed': speed}
            })
            emit('speed_update', {'speed': speed})
        
        @self.socketio.on('flip_command')
        def handle_flip(data):
            """Handle flip command"""
            direction = data.get('direction', 'f')
            shared_data.add_command({
                'type': 'flip',
                'data': {'direction': direction}
            })
            emit('drone_action', {'action': f'flip_{direction}', 'success': True})
        
        @self.socketio.on('enable_ml_detection')
        def handle_enable_ml_detection(data):
            """Handle ML detection toggle"""
            enabled = data.get('enabled', True)
            shared_data.add_command({
                'type': 'enable_ml_detection',
                'data': {'enabled': enabled}
            })
            emit('ml_detection_status', {'enabled': enabled})
        
        @self.socketio.on('enable_auto_capture')
        def handle_enable_auto_capture(data):
            """Handle auto capture toggle"""
            enabled = data.get('enabled', True)
            shared_data.update_status({'auto_capture_enabled': enabled})
            emit('auto_capture_status', {'enabled': enabled})
        
        @self.socketio.on('manual_screenshot')
        def handle_manual_screenshot():
            """Handle manual screenshot request"""
            shared_data.add_command({'type': 'manual_screenshot'})
            
            # Simulate screenshot result
            current_count = shared_data.get_status().get('screenshot_count', 0)
            emit('screenshot_result', {
                'success': True,
                'count': current_count + 1,
                'filename': f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            })
        
        @self.socketio.on('toggle_recording')
        def handle_toggle_recording(data):
            """Handle recording toggle"""
            recording = data.get('recording', False)
            shared_data.add_command({'type': 'toggle_recording'})
            emit('recording_status', {'recording': recording})
        
        @self.socketio.on('start_stream')
        def handle_start_stream():
            """Handle video stream start"""
            emit('stream_status', {'streaming': True})
            # Start sending video frames
            self.start_video_stream()
        
        @self.socketio.on('start_autonomous_mode')
        def handle_start_autonomous():
            """Handle autonomous mode start"""
            # For now, just acknowledge
            emit('drone_action', {'action': 'autonomous_start', 'success': True})
        
        @self.socketio.on('get_media_files')
        def handle_get_media_files(data):
            """Handle media files request"""
            try:
                media_type = data.get('type', 'images').lower()
                
                if media_type == 'images':
                    directory = Config.SCREENSHOTS_DIR
                    extensions = ['.jpg', '.jpeg', '.png']
                elif media_type == 'videos':
                    directory = Config.RECORDINGS_DIR
                    extensions = ['.mp4', '.avi', '.mov']
                else:
                    emit('media_files_response', {
                        'success': False,
                        'error': 'Invalid media type'
                    })
                    return
                
                files = []
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if any(filename.lower().endswith(ext) for ext in extensions):
                            filepath = os.path.join(directory, filename)
                            stat = os.stat(filepath)
                            
                            files.append({
                                'filename': filename,
                                'size': stat.st_size,
                                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'url': f'/media/{filename}',
                                'humans_detected': 0
                            })
                
                files.sort(key=lambda x: x['created_at'], reverse=True)
                
                emit('media_files_response', {
                    'success': True,
                    'files': files,
                    'count': len(files)
                })
                
            except Exception as e:
                emit('media_files_response', {
                    'success': False,
                    'error': str(e)
                })
        
        @self.socketio.on('download_media')
        def handle_download_media(data):
            """Handle media download request"""
            try:
                filename = data.get('filename')
                if not filename:
                    emit('download_ready', {
                        'success': False,
                        'error': 'No filename provided'
                    })
                    return
                
                # Check both directories
                for directory in [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR]:
                    filepath = os.path.join(directory, filename)
                    if os.path.exists(filepath):
                        emit('download_ready', {
                            'success': True,
                            'url': f'/download/{filename}',
                            'filename': filename
                        })
                        return
                
                emit('download_ready', {
                    'success': False,
                    'error': 'File not found'
                })
                
            except Exception as e:
                emit('download_ready', {
                    'success': False,
                    'error': str(e)
                })
        
        @self.socketio.on('delete_media')
        def handle_delete_media(data):
            """Handle media deletion request"""
            try:
                filename = data.get('filename')
                if not filename:
                    emit('media_deleted', {
                        'success': False,
                        'error': 'No filename provided'
                    })
                    return
                
                # Check both directories
                deleted = False
                for directory in [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR]:
                    filepath = os.path.join(directory, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deleted = True
                        break
                
                if deleted:
                    emit('media_deleted', {
                        'success': True,
                        'filename': filename
                    })
                else:
                    emit('media_deleted', {
                        'success': False,
                        'error': 'File not found'
                    })
                
            except Exception as e:
                emit('media_deleted', {
                    'success': False,
                    'error': str(e)
                })
    
    def start_video_stream(self):
        """Start video streaming to clients"""
        def video_stream_worker():
            while True:
                try:
                    frame_base64 = shared_data.get_frame_base64()
                    if frame_base64:
                        self.socketio.emit('video_frame', {
                            'frame': frame_base64,
                            'timestamp': time.time()
                        })
                    
                    time.sleep(1/30)  # 30 FPS
                except Exception as e:
                    print(f"❌ Video stream error: {e}")
                    time.sleep(0.1)
        
        # Start video streaming in separate thread
        video_thread = threading.Thread(target=video_stream_worker, daemon=True)
        video_thread.start()
    
    def start_status_updates(self):
        """Start periodic status updates"""
        def status_update_worker():
            while True:
                try:
                    status = shared_data.get_status()
                    self.socketio.emit('tello_status', status)
                    self.socketio.emit('telemetry_update', status.get('telemetry', {}))
                    
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"❌ Status update error: {e}")
                    time.sleep(1)
        
        # Start status updates in separate thread
        status_thread = threading.Thread(target=status_update_worker, daemon=True)
        status_thread.start()
    
    def run(self):
        """Run the web server"""
        print(f"🚀 Starting web server on {Config.WEB_HOST}:{Config.WEB_PORT}")
        
        # Start video streaming and status updates
        self.start_video_stream()
        self.start_status_updates()
        
        # Run the server
        self.socketio.run(
            self.app,
            host=Config.WEB_HOST,
            port=Config.WEB_PORT,
            debug=Config.WEB_DEBUG,
            allow_unsafe_werkzeug=True
        )

# ==================== MAIN APPLICATION ====================
class DroneWebBridge:
    """Main application class that orchestrates everything"""
    
    def __init__(self, mode='integrated'):
        self.mode = mode
        self.drone_system = None
        self.web_server = None
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
        print(f"🚀 Drone Web Bridge starting in {mode} mode")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def start_backend_only(self):
        """Start only web server (for testing without drone)"""
        if not WEB_IMPORTS_AVAILABLE:
            print("❌ Web server dependencies not available")
            return False
        
        try:
            # Initialize with mock data
            shared_data.update_status({
                'connected': False,
                'flying': False,
                'battery': 0,
                'speed': 50
            })
            
            # Start web server
            self.web_server = WebServer()
            print("🌐 Backend-only mode: Web server ready")
            print(f"📍 Access at: http://{Config.WEB_HOST}:{Config.WEB_PORT}")
            print("⚠️ Drone system not initialized - limited functionality")
            
            self.web_server.run()
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_drone_only(self):
        """Start only drone system (original droneV7.py behavior)"""
        if not DRONE_IMPORTS_AVAILABLE:
            print("❌ Drone dependencies not available")
            return False
        
        try:
            # Initialize drone system in standalone mode
            self.drone_system = DroneSystem()
            self.drone_system.web_integration_enabled = False
            
            if not self.drone_system.initialize_all_systems():
                print("❌ Failed to initialize drone systems")
                return False
            
            # Start drone threads
            self.drone_system.start_drone_threads()
            
            print("🚁 Drone-only mode: Running original droneV7.py behavior")
            print("🎮 Use pygame interface for control")
            
            # Run original main loop (simplified)
            try:
                while self.running:
                    # Handle pygame events if not in headless mode
                    if self.drone_system.screen:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.running = False
                                break
                    
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt received")
            
        except Exception as e:
            print(f"❌ Failed to start drone system: {e}")
            return False
    
    def start_integrated(self):
        """Start both drone system and web server (full integration)"""
        if not WEB_IMPORTS_AVAILABLE or not DRONE_IMPORTS_AVAILABLE:
            print("❌ Missing dependencies for integrated mode")
            return False
        
        try:
            # Initialize drone system
            print("🚁 Initializing drone system...")
            self.drone_system = DroneSystem()
            self.drone_system.web_integration_enabled = True
            
            if not self.drone_system.initialize_all_systems():
                print("❌ Failed to initialize drone systems")
                return False
            
            # Start drone threads
            self.drone_system.start_drone_threads()
            
            # Initialize web server
            print("🌐 Initializing web server...")
            self.web_server = WebServer()
            
            print("✅ Integrated mode: Both systems ready")
            print(f"📍 Web interface: http://{Config.WEB_HOST}:{Config.WEB_PORT}")
            print("🚁 Drone system: Running with web integration")
            
            # Run web server (this will block)
            self.web_server.run()
            
        except Exception as e:
            print(f"❌ Failed to start integrated mode: {e}")
            return False
    
    def cleanup(self):
        """Cleanup all systems"""
        print("🧹 Cleaning up systems...")
        
        if self.drone_system:
            self.drone_system.stop_all_systems()
        
        # Additional cleanup
        try:
            # Remove shared files
            for filename in [Config.FRAME_SHARE_FILE, Config.STATUS_SHARE_FILE, Config.COMMAND_SHARE_FILE]:
                if os.path.exists(filename):
                    os.remove(filename)
        except:
            pass
        
        print("✅ Cleanup completed")

# ==================== CLI INTERFACE ====================
def print_usage():
    """Print usage information"""
    print("""
🚁 Drone Web Bridge - Single File Integration

Usage:
    python drone_web_bridge.py [MODE]

Modes:
    --backend-only    Start only web server (for frontend development)
    --drone-only      Start only drone system (original droneV7.py behavior)
    --integrated      Start both systems (full integration) [DEFAULT]
    --help           Show this help message

Examples:
    python drone_web_bridge.py --integrated
    python drone_web_bridge.py --backend-only
    python drone_web_bridge.py --drone-only

Dependencies:
    Backend: pip install flask flask-socketio flask-cors
    Drone: pip install djitellopy ultralytics mediapipe pygame opencv-python numpy

Web Interface:
    Access at http://127.0.0.1:5000 when web server is running
    """)

def main():
    """Main entry point"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h']:
            print_usage()
            return
        elif arg == '--backend-only':
            mode = 'backend-only'
        elif arg == '--drone-only':
            mode = 'drone-only'
        elif arg == '--integrated':
            mode = 'integrated'
        else:
            print(f"❌ Unknown mode: {arg}")
            print_usage()
            return
    else:
        mode = 'integrated'  # Default mode
    
    # Create and start application
    app = DroneWebBridge(mode)
    
    try:
        if mode == 'backend-only':
            success = app.start_backend_only()
        elif mode == 'drone-only':
            success = app.start_drone_only()
        elif mode == 'integrated':
            success = app.start_integrated()
        
        if not success:
            print("❌ Failed to start application")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        app.cleanup()

if __name__ == '__main__':
    main()