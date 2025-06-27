from djitellopy import Tello
import cv2
import numpy as np
import time
import sys
import mediapipe as mp
from ultralytics import YOLO
import os
from datetime import datetime
import threading
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import signal

# Speed of the drone
S = 60
# Frames per second for video processing
FPS = 30

class TelloHeadlessController:
    """
    Headless Tello drone controller with Flask-SocketIO web interface.
    No pygame window - pure web-based control via React frontend.
    """

    def __init__(self):
        print("🚁 Initializing SARVIO-X Headless Controller...")
        
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False
        self.is_connected = False
        self.is_flying = False

        # Create screenshots directory
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            print(f"📁 Created directory: {self.screenshot_dir}")

        # Load YOLOv8 model for human detection
        print("🤖 Loading YOLOv8 model...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.yolo_model = None

        # Mediapipe modules for body part detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        # Create MediaPipe models
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=2,
            enable_segmentation=False,
            smooth_landmarks=True
        )

        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            max_num_hands=2
        )

        # Screenshot variables
        self.last_screenshot_time = 0
        self.screenshot_interval = 3
        self.screenshot_count = 0

        # Auto screenshot countdown variables
        self.countdown_active = False
        self.countdown_start_time = 0
        self.countdown_duration = 3.0
        self.last_human_detected = False

        # FPS variables
        self.prev_time = time.time()
        self.fps = 0

        # Web interface control flags
        self.ml_detection_enabled = True
        self.auto_capture_enabled = True
        self.socket_streaming = False
        self.connected_clients = 0
        self.should_stop = False

        # Flight time tracking
        self.flight_start_time = None

        # Video processing variables
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.video_thread = None
        self.frame_read = None

        # Flask and SocketIO setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'tello_headless_secret_key'
        
        # CORS for React app (Vite default port 5173)
        CORS(self.app, 
            origins=["http://localhost:5173", "http://localhost:3000"],
            allow_headers=["Content-Type"],
            methods=["GET", "POST"])
        
        # Socket.IO initialization with CORS
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins=["http://localhost:5173", "http://localhost:3000"],
            async_mode='threading',
            logger=False,  # Disable untuk mengurangi noise
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            transports=['polling', 'websocket']
        )

        # Setup Flask routes and Socket events
        self._setup_flask_routes()
        self._setup_socket_events()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("✅ Headless controller initialized successfully")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.should_stop = True
        self._cleanup()
        sys.exit(0)

    def _setup_flask_routes(self):
        """Setup Flask routes for web interface"""
        @self.app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
                <head>
                    <title>SARVIO-X Headless Backend</title>
                    <meta charset="UTF-8">
                    <style>
                        body { 
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                            margin: 0; 
                            padding: 40px; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; 
                            min-height: 100vh;
                        }
                        .container { max-width: 1200px; margin: 0 auto; }
                        .header { text-align: center; margin-bottom: 40px; }
                        .status { 
                            padding: 15px; 
                            margin: 15px 0; 
                            border-radius: 10px; 
                            backdrop-filter: blur(10px);
                        }
                        .connected { background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; }
                        .disconnected { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; }
                        .info { background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; }
                        img { 
                            border: 3px solid #06b6d4; 
                            border-radius: 12px; 
                            max-width: 100%;
                            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                        }
                        .features { 
                            display: grid; 
                            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                            gap: 20px; 
                            margin: 30px 0; 
                        }
                        .feature-card {
                            background: rgba(255,255,255,0.1);
                            padding: 20px;
                            border-radius: 10px;
                            backdrop-filter: blur(10px);
                        }
                        a { color: #06b6d4; text-decoration: none; font-weight: bold; }
                        a:hover { color: #0891b2; }
                        .logo { font-size: 3em; margin-bottom: 10px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <div class="logo">🚁</div>
                            <h1>SARVIO-X Headless Backend</h1>
                            <p>Tello Drone Controller - Web-Only Mode</p>
                        </div>
                        
                        <div class="status connected">
                            ✅ <strong>Headless Mode Active</strong> - No pygame window required
                        </div>
                        
                        <div class="status connected">
                            🌐 <strong>Flask-SocketIO Server:</strong> Running on port 5000
                        </div>
                        
                        <div class="status info">
                            🎮 <strong>React Web Interface:</strong> <a href="http://localhost:5173" target="_blank">http://localhost:5173</a>
                        </div>
                        
                        <div class="status info">
                            🎥 <strong>Direct Video Feed:</strong> <a href="{{ url_for('video_feed') }}" target="_blank">Live Stream</a>
                        </div>
                        
                        <h3>📺 Live Video Stream:</h3>
                        <img src="{{ url_for('video_feed') }}" alt="Tello Live Stream">
                        
                        <div class="features">
                            <div class="feature-card">
                                <h3>🤖 AI Features</h3>
                                <ul>
                                    <li>YOLOv8 Human Detection</li>
                                    <li>MediaPipe Body/Hand Tracking</li>
                                    <li>Smart Auto Screenshot</li>
                                    <li>Real-time ML Processing</li>
                                </ul>
                            </div>
                            
                            <div class="feature-card">
                                <h3>🎮 Control Features</h3>
                                <ul>
                                    <li>Web-based Joystick Control</li>
                                    <li>Real-time Video Streaming</li>
                                    <li>Socket.IO Communication</li>
                                    <li>Battery & Sensor Monitoring</li>
                                </ul>
                            </div>
                            
                            <div class="feature-card">
                                <h3>⚡ Headless Benefits</h3>
                                <ul>
                                    <li>No GUI Dependencies</li>
                                    <li>Lower Resource Usage</li>
                                    <li>Server-ready Deployment</li>
                                    <li>Remote Access Ready</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div class="status info">
                            <strong>💡 Tip:</strong> Use the React interface at 
                            <a href="http://localhost:5173">localhost:5173</a> for full drone control
                        </div>
                    </div>
                    
                    <script>
                        // Auto-refresh page every 30 seconds to show latest status
                        setTimeout(() => location.reload(), 30000);
                    </script>
                </body>
            </html>
            """)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._frame_generator(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/status')
        def status():
            """API endpoint for system status"""
            return {
                'status': 'running',
                'mode': 'headless',
                'connected_clients': self.connected_clients,
                'socket_streaming': self.socket_streaming,
                'tello_connected': self.is_connected,
                'tello_flying': self.is_flying,
                'battery': self.get_battery(),
                'screenshots_taken': self.screenshot_count,
                'ml_detection_enabled': self.ml_detection_enabled,
                'auto_capture_enabled': self.auto_capture_enabled
            }

    def _setup_socket_events(self):
        """Setup Socket.IO event handlers for React frontend"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            self.connected_clients += 1
            print(f'✅ React client connected. Total clients: {self.connected_clients}')
            
            # Send Tello status when client connects
            self.broadcast_status()
            
            # Send welcome message
            self.socketio.emit('connection_info', {
                'message': 'Successfully connected to SARVIO-X Headless Backend',
                'server_version': '2.0-headless',
                'mode': 'headless',
                'features': ['video_streaming', 'ml_detection', 'auto_screenshot', 'web_control']
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients -= 1
            print(f'❌ React client disconnected. Total clients: {self.connected_clients}')
            
            # Stop socket streaming if no clients
            if self.connected_clients <= 0:
                self.socket_streaming = False
                print('📺 Video streaming stopped (no clients)')

        @self.socketio.on('connect_tello')
        def handle_connect_tello():
            """Connect to Tello drone"""
            print("🔗 Connect Tello command from React client")
            success = self.connect_tello()
            
            self.socketio.emit('tello_connection_result', {
                'success': success,
                'message': 'Tello connected successfully' if success else 'Failed to connect to Tello'
            })
            
            self.broadcast_status()

        @self.socketio.on('disconnect_tello')
        def handle_disconnect_tello():
            """Disconnect from Tello drone"""
            print("🔌 Disconnect Tello command from React client")
            success = self.disconnect_tello()
            
            self.socketio.emit('tello_connection_result', {
                'success': success,
                'message': 'Tello disconnected successfully' if success else 'Failed to disconnect from Tello'
            })
            
            self.broadcast_status()
        
        @self.socketio.on('start_stream')
        def handle_start_stream():
            print("🎥 React client requested video stream")
            self.socket_streaming = True
            
            self.socketio.emit('stream_status', {
                'streaming': True,
                'message': 'Video stream started'
            })
        
        @self.socketio.on('stop_stream')
        def handle_stop_stream():
            print("⏹️ React client stopped video stream")
            self.socket_streaming = False
            
            self.socketio.emit('stream_status', {
                'streaming': False,
                'message': 'Video stream stopped'
            })
        
        @self.socketio.on('takeoff')
        def handle_takeoff():
            print("🚁 Takeoff command from React client")
            success = self.takeoff_drone()
        
        @self.socketio.on('land')
        def handle_land():
            print("🏠 Land command from React client")
            success = self.land_drone()
        
        @self.socketio.on('move_control')
        def handle_move_control(data):
            """Handle movement control from React client"""
            if self.send_rc_control:
                self.left_right_velocity = int(data.get('left_right', 0))
                self.for_back_velocity = int(data.get('for_back', 0))
                self.up_down_velocity = int(data.get('up_down', 0))
                self.yaw_velocity = int(data.get('yaw', 0))
                
                # Send to Tello immediately
                if self.is_connected:
                    try:
                        self.tello.send_rc_control(
                            self.left_right_velocity,
                            self.for_back_velocity,
                            self.up_down_velocity,
                            self.yaw_velocity
                        )
                    except Exception as e:
                        print(f"❌ Movement control error: {e}")
        
        @self.socketio.on('stop_movement')
        def handle_stop_movement():
            """Stop all movement from React client"""
            print("⏹️ Stop movement command from React client")
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.up_down_velocity = 0
            self.yaw_velocity = 0
            
            if self.is_connected:
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except Exception as e:
                    print(f"❌ Stop movement error: {e}")

        @self.socketio.on('set_speed')
        def handle_set_speed(data):
            """Set drone speed"""
            try:
                speed = int(data)
                if 10 <= speed <= 100:
                    self.speed = speed
                    if self.is_connected:
                        self.tello.set_speed(speed)
                    print(f"🏃 Speed set to: {speed}")
                    self.socketio.emit('speed_updated', {'speed': speed})
            except Exception as e:
                print(f"❌ Set speed error: {e}")

        @self.socketio.on('enable_ml_detection')
        def handle_enable_ml_detection(data):
            """Enable/disable ML detection"""
            self.ml_detection_enabled = data.get('enabled', False)
            print(f"🤖 ML Detection: {'Enabled' if self.ml_detection_enabled else 'Disabled'}")
            
            self.socketio.emit('ml_detection_status', {
                'enabled': self.ml_detection_enabled,
                'message': f"ML Detection {'enabled' if self.ml_detection_enabled else 'disabled'}"
            })

        @self.socketio.on('enable_auto_capture')
        def handle_enable_auto_capture(data):
            """Enable/disable auto capture"""
            self.auto_capture_enabled = data.get('enabled', False)
            print(f"📸 Auto Capture: {'Enabled' if self.auto_capture_enabled else 'Disabled'}")
            
            self.socketio.emit('auto_capture_status', {
                'enabled': self.auto_capture_enabled,
                'message': f"Auto Capture {'enabled' if self.auto_capture_enabled else 'disabled'}"
            })

        @self.socketio.on('manual_screenshot')
        def handle_manual_screenshot():
            """Take manual screenshot from React client"""
            print("📸 Manual screenshot request from React client")
            
            if self.last_frame is not None:
                with self.frame_lock:
                    frame_copy = self.last_frame.copy()
                processed_frame, _, humans_count = self.process_human_detection(frame_copy)
                success = self.save_screenshot(processed_frame, humans_count, "web")
                
                self.socketio.emit('screenshot_taken', {
                    'success': success,
                    'count': self.screenshot_count,
                    'humans_detected': humans_count
                })
            else:
                self.socketio.emit('screenshot_taken', {
                    'success': False,
                    'message': 'No video frame available'
                })

    def connect_tello(self):
        """Connect to Tello drone"""
        try:
            if not self.is_connected:
                print("🔗 Connecting to Tello...")
                self.tello.connect()
                
                # Test connection
                battery = self.get_battery()
                print(f"✅ Connected! Battery: {battery}%")
                
                self.is_connected = True
                self.tello.set_speed(self.speed)
                
                # Start video streaming
                self.start_video_stream()
                
                return True
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Tello: {e}")
            self.is_connected = False
            return False

    def disconnect_tello(self):
        """Disconnect from Tello drone"""
        try:
            if self.is_connected:
                if self.is_flying:
                    self.tello.land()
                    self.is_flying = False
                    self.send_rc_control = False
                
                self.stop_video_stream()
                self.tello.end()
                self.is_connected = False
                print("🔌 Tello disconnected")
            return True
        except Exception as e:
            print(f"❌ Error disconnecting Tello: {e}")
            return False

    def start_video_stream(self):
        """Start video streaming from Tello"""
        if not self.is_connected:
            return
            
        try:
            self.tello.streamoff()
            time.sleep(0.5)
            self.tello.streamon()
            time.sleep(1)
            
            self.frame_read = self.tello.get_frame_read()
            
            # Start video processing thread
            if self.video_thread is None or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self._video_processing_loop, daemon=True)
                self.video_thread.start()
                print("🎥 Video stream started")
                
        except Exception as e:
            print(f"❌ Error starting video stream: {e}")

    def stop_video_stream(self):
        """Stop video streaming"""
        try:
            if self.is_connected:
                self.tello.streamoff()
            
            # Video thread will stop when should_stop is True
            print("⏹️ Video stream stopped")
            
        except Exception as e:
            print(f"❌ Error stopping video stream: {e}")

    def _video_processing_loop(self):
        """Main video processing loop running in separate thread"""
        print("🎬 Video processing loop started")
        battery_counter = 0
        
        while not self.should_stop and self.is_connected:
            try:
                if self.frame_read is None or self.frame_read.stopped:
                    time.sleep(0.1)
                    continue

                frame = self.frame_read.frame
                if frame is None:
                    time.sleep(0.1)
                    continue

                # Calculate FPS
                curr_time = time.time()
                self.fps = 1 / (curr_time - self.prev_time) if curr_time != self.prev_time else 0
                self.prev_time = curr_time

                # Resize frame for processing
                frame = cv2.resize(frame, (960, 720))

                # Process human detection
                output_frame, human_detected, humans_count = self.process_human_detection(frame)

                # Handle auto screenshot logic  
                self.handle_auto_screenshot(output_frame, human_detected, humans_count)

                # Add info overlays
                battery = self.get_battery()
                cv2.putText(output_frame, f"Battery: {battery}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_frame, "HEADLESS MODE", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if humans_count > 0:
                    cv2.putText(output_frame, f"Humans Detected: {humans_count}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(output_frame, f"Screenshots: {self.screenshot_count}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Store frame for Flask streaming and React clients
                with self.frame_lock:
                    self.last_frame = output_frame.copy()
                
                # Send frame to React clients
                self._send_frame_to_react(output_frame)

                # Send battery update to React clients periodically
                battery_counter += 1
                if battery_counter % 30 == 0 and self.connected_clients > 0:  # Every second at 30 FPS
                    self.socketio.emit('battery_update', {'battery': battery})
                    self.broadcast_status()

                time.sleep(1 / FPS)
                
            except Exception as e:
                print(f"❌ Video processing error: {e}")
                time.sleep(0.1)
        
        print("🎬 Video processing loop ended")

    def takeoff_drone(self):
        """Takeoff command"""
        if self.is_connected and not self.is_flying:
            try:
                self.tello.takeoff()
                self.is_flying = True
                self.send_rc_control = True
                self.flight_start_time = time.time()
                print("🚁 Takeoff successful")
                
                # Broadcast to web clients
                self.socketio.emit('drone_action', {
                    'action': 'takeoff',
                    'success': True
                })
                self.broadcast_status()
                return True
            except Exception as e:
                print(f"❌ Takeoff failed: {e}")
                self.socketio.emit('drone_action', {
                    'action': 'takeoff',
                    'success': False,
                    'error': str(e)
                })
                return False
        return False

    def land_drone(self):
        """Land command"""
        if self.is_connected and self.is_flying:
            try:
                self.tello.land()
                self.is_flying = False
                self.send_rc_control = False
                self.flight_start_time = None
                print("🏠 Landing successful")
                
                # Broadcast to web clients
                self.socketio.emit('drone_action', {
                    'action': 'land',
                    'success': True
                })
                self.broadcast_status()
                return True
            except Exception as e:
                print(f"❌ Landing failed: {e}")
                self.socketio.emit('drone_action', {
                    'action': 'land',
                    'success': False,
                    'error': str(e)
                })
                return False
        return False

    def get_battery(self):
        """Get battery level"""
        if self.is_connected:
            try:
                return self.tello.get_battery()
            except:
                return 0
        return 0

    def get_flight_time(self):
        """Get flight time in seconds"""
        if self.flight_start_time and self.is_flying:
            return int(time.time() - self.flight_start_time)
        return 0

    def broadcast_status(self):
        """Broadcast status to all web clients"""
        try:
            status = {
                'connected': self.is_connected,
                'flying': self.is_flying,
                'battery': self.get_battery(),
                'flight_time': self.get_flight_time()
            }
            self.socketio.emit('tello_status', status)
        except Exception as e:
            print(f"❌ Broadcast status error: {e}")

    def _frame_generator(self):
        """Generator for streaming frames to browser (for Flask route)"""
        while not self.should_stop:
            with self.frame_lock:
                if self.last_frame is None:
                    time.sleep(0.1)
                    continue
                frame_to_send = self.last_frame.copy()
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame_to_send)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1 / FPS)

    def _send_frame_to_react(self, frame):
        """Send frame to React clients via Socket.IO"""
        if self.socket_streaming and self.connected_clients > 0:
            try:
                # Resize frame for bandwidth efficiency
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Encode to JPEG with medium quality
                ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    # Convert to base64
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit to all connected React clients
                    self.socketio.emit('video_frame', {'frame': frame_base64})
            except Exception as e:
                print(f"❌ Error sending frame to React: {e}")

    def save_screenshot(self, frame, humans_count, source="auto"):
        """Save screenshot with timestamp and human count"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_prefix = "manual" if source in ["web", "api"] else "auto"
            filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count}persons_{self.screenshot_count:04d}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Save the frame
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.screenshot_count += 1
                print(f"📸 Screenshot saved ({source}): {filename}")
                
                # Notify web clients
                self.socketio.emit('screenshot_result', {
                    'success': True,
                    'count': self.screenshot_count,
                    'filename': filename
                })
                return True
            else:
                print(f"❌ Failed to save screenshot: {filename}")
                self.socketio.emit('screenshot_result', {
                    'success': False,
                    'count': self.screenshot_count
                })
                return False
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
            return False

    def process_human_detection(self, frame):
        """Process human detection and return processed frame with detection info"""
        # Make a copy for processing
        output_frame = frame.copy()
        
        human_detected = False
        human_boxes = []

        if self.yolo_model and self.ml_detection_enabled:
            try:
                # YOLOv8 Human Detection
                results = self.yolo_model(frame, verbose=False)

                # Process YOLO results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            # Check if it's a person (class_id = 0 in COCO dataset)
                            if class_id == 0 and confidence > 0.5:
                                human_detected = True

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
                                confidence_percentage = confidence * 100  # Convert to percentage
                                label = f"Human: {confidence_percentage:.0f}%"  # Format as an integer percentage

                                cv2.putText(output_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Process detailed body part detection if human detected
                if human_detected:
                    # Process with pose detection
                    pose_results = self.pose.process(frame)

                    # Process with hand detection
                    hands_results = self.hands.process(frame)

                    # Draw hands if detected
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                output_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                    # Draw pose landmarks if detected
                    if pose_results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            output_frame,
                            pose_results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
            except Exception as e:
                print(f"❌ Human detection error: {e}")

        return output_frame, human_detected, len(human_boxes)

    def handle_auto_screenshot(self, output_frame, human_detected, humans_count):
        """Handle auto screenshot countdown logic"""
        if not self.auto_capture_enabled:
            return
            
        current_time = time.time()
        
        if human_detected and humans_count >= 1:
            # Human detected
            if not self.last_human_detected and not self.countdown_active:
                # First time detecting human, start countdown
                self.countdown_active = True
                self.countdown_start_time = current_time
                print(f"👤 Human detected! Starting 3-second countdown...")
            
            # If countdown is active
            if self.countdown_active:
                elapsed_time = current_time - self.countdown_start_time
                
                if elapsed_time >= self.countdown_duration:
                    # Countdown finished, take screenshot
                    self.save_screenshot(output_frame, humans_count, "auto")
                    self.last_screenshot_time = current_time
                    self.countdown_active = False
                    print("📸 Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if self.countdown_active:
                # Cancel countdown if human disappears
                self.countdown_active = False
                print("❌ Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        self.last_human_detected = human_detected

    def run_web_server(self):
        """Run Flask server with Socket.IO"""
        print("🌐 Starting Flask server with Socket.IO...")
        print("📱 React app URL: http://localhost:5173")
        print("🖥️  Backend server URL: http://localhost:5000")
        print("📺 Direct video stream: http://localhost:5000/video_feed")
        
        # Use port 5000 for backend, Vite will use 5173 for frontend
        self.socketio.run(
            self.app, 
            host='127.0.0.1',
            port=5000, 
            debug=False, 
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )

    def _cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up resources...")
        self.should_stop = True
        
        try:
            if self.is_connected:
                if self.is_flying:
                    self.tello.land()
                self.stop_video_stream()
                self.tello.end()
            
            if self.pose:
                self.pose.close()
            if self.hands:
                self.hands.close()
                
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

    def run(self):
        """Main run method - headless mode"""
        print("=" * 60)
        print("🚁 SARVIO-X HEADLESS MODE STARTING")
        print("=" * 60)
        print("✅ Features:")
        print("   - No pygame window (headless)")
        print("   - React web interface control")
        print("   - YOLOv8 + MediaPipe ML detection")
        print("   - Smart auto screenshot")
        print("   - Real-time video streaming")
        print("   - Flask-SocketIO backend")
        print("=" * 60)
        print("🌐 Web Interface:")
        print("   - React app: http://localhost:5173")
        print("   - Backend: http://localhost:5000")
        print("   - Video feed: http://localhost:5000/video_feed")
        print("   - Status API: http://localhost:5000/status")
        print("=" * 60)
        print("💡 Usage:")
        print("   1. Start this backend: python drone_headless.py")
        print("   2. Start React app: npm run dev (in client folder)")
        print("   3. Open browser: http://localhost:5173")
        print("   4. Connect drone and start flying!")
        print("=" * 60)
        
        try:
            # Auto-connect to Tello
            print("🔗 Auto-connecting to Tello...")
            self.connect_tello()
            print(f"🔋 Battery: {self.get_battery()}%")

            # Auto-start socket streaming
            self.socket_streaming = True
            print("📡 Socket streaming enabled")

            # Run Flask server (this will block)
            self.run_web_server()
            
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt detected")
        except Exception as e:
            print(f"❌ Error occurred: {e}")
        finally:
            self._cleanup()
            print("👋 SARVIO-X headless mode stopped")


def main():
    """Main entry point"""
    controller = TelloHeadlessController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        controller._cleanup()


if __name__ == '__main__':
    main()