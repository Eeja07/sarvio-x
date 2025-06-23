from djitellopy import Tello
import cv2
import pygame
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

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 30  # Reduced from 120 for better video processing & streaming

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys and joystick.
        Now also includes Flask-SocketIO server for React web interface integration.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
            - P: Manual screenshot
            - Joystick: A=takeoff, B=land, X/Y=screenshot
        
        Web Interface:
            - All controls available via React web interface
            - Real-time video streaming with ML detection
            - Flask-SocketIO communication on port 5000
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window
        pygame.display.set_caption("Tello video stream - SARVIO-X")
        self.screen = pygame.display.set_mode([960, 720])
        self.font = pygame.font.SysFont("Arial", 20)

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

        # Initialize joystick
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick initialized: {self.joystick.get_name()}")
        else:
            print("No joystick detected - using keyboard only")

        # Create screenshots directory
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            print(f"Created directory: {self.screenshot_dir}")

        # Load YOLOv8 model for human detection
        print("Loading YOLOv8 model...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
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

        # Joystick screenshot variables
        self.last_joystick_screenshot_button_state = False
        self.joystick_screenshot_requested = False

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

        # Flask and SocketIO setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'tello_secret_key'
        
        # CORS for React app (Vite default port 5173)
        CORS(self.app, 
            origins=["http://localhost:5173", "http://localhost:3000"],
            allow_headers=["Content-Type"],
            methods=["GET", "POST"])
        
        # Socket.IO initialization with CORS
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins=["http://localhost:5173", "http://localhost:3000"],
            async_mode='threading',  # Gunakan threading mode
            logger=True,  # Enable logging untuk debugging
            engineio_logger=True,  # Enable engine.io logging
            ping_timeout=60,  # Timeout untuk ping
            ping_interval=25,  # Interval ping
            transports=['polling', 'websocket']  # Fallback ke polling jika websocket gagal
        )
        
        self.last_frame = None
        self.frame_lock = threading.Lock()

        # Setup Flask routes and Socket events
        self._setup_flask_routes()
        self._setup_socket_events()

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def _setup_flask_routes(self):
        """Setup Flask routes for web interface"""
        @self.app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Tello Live Stream - SARVIO-X Backend</title>
                    <meta charset="UTF-8">
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; }
                        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                        .connected { background: #065f46; }
                        .disconnected { background: #7f1d1d; }
                        img { border: 2px solid #06b6d4; border-radius: 8px; }
                    </style>
                </head>
                <body>
                    <h1>🚁 SARVIO-X Backend Server</h1>
                    <h2>Tello Live Video Feed with ML Detection</h2>
                    
                    <div class="status connected">
                        ✅ Flask-SocketIO Server: Active on port 5000
                    </div>
                    
                    <div class="status connected">
                        🎥 Video Stream: <a href="{{ url_for('video_feed') }}" target="_blank">Direct Feed</a>
                    </div>
                    
                    <div class="status connected">
                        🌐 React App: <a href="http://localhost:5173" target="_blank">Web Interface</a>
                    </div>
                    
                    <h3>Live Video Stream:</h3>
                    <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Tello Live Stream">
                    
                    <h3>Features:</h3>
                    <ul>
                        <li>✅ YOLOv8 Human Detection</li>
                        <li>✅ MediaPipe Body/Hand Tracking</li>
                        <li>✅ Auto Screenshot (3s countdown)</li>
                        <li>✅ Manual Screenshot via Web/Pygame</li>
                        <li>✅ Real-time Socket.IO streaming</li>
                        <li>✅ Dual Interface (Pygame + Web)</li>
                    </ul>
                    
                    <p><strong>Note:</strong> Use React interface at <a href="http://localhost:5173">localhost:5173</a> for full control</p>
                </body>
            </html>
            """)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._frame_generator(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/test_socket')
        def test_socket():
            """Test endpoint to check Socket.IO status"""
            return {
                'status': 'ok',
                'connected_clients': self.connected_clients,
                'socket_streaming': self.socket_streaming,
                'tello_connected': self.is_connected,
                'tello_flying': self.is_flying
            }
    def _setup_socket_events(self):
        """Setup Socket.IO event handlers for React frontend"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):  # Tambahkan parameter auth
            self.connected_clients += 1
            print(f'✅ React client connected. Total clients: {self.connected_clients}')
            # HAPUS baris yang error ini:
            # print(f'   Client ID: {request.sid}')  # HAPUS - request tidak tersedia
            
            # Send Tello status when client connects
            self.broadcast_status()
            
            # Send welcome message - HAPUS room=request.sid
            self.socketio.emit('connection_info', {
                'message': 'Successfully connected to SARVIO-X backend',
                'server_version': '2.0',
                'features': ['video_streaming', 'ml_detection', 'auto_screenshot']
            })  # Hapus room parameter
        
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
                print("Connecting to Tello...")
                self.tello.connect()
                
                # Test connection
                battery = self.get_battery()
                print(f"Connected! Battery: {battery}%")
                
                self.is_connected = True
                self.tello.set_speed(self.speed)
                return True
            return True
        except Exception as e:
            print(f"Failed to connect to Tello: {e}")
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
                
                self.tello.streamoff()
                self.tello.end()
                self.is_connected = False
                print("Tello disconnected")
            return True
        except Exception as e:
            print(f"Error disconnecting Tello: {e}")
            return False

    def takeoff_drone(self):
        """Takeoff command"""
        if self.is_connected and not self.is_flying:
            try:
                self.tello.takeoff()
                self.is_flying = True
                self.send_rc_control = True
                self.flight_start_time = time.time()
                print("Takeoff successful")
                
                # Broadcast to web clients
                self.socketio.emit('drone_action', {
                    'action': 'takeoff',
                    'success': True
                })
                self.broadcast_status()
                return True
            except Exception as e:
                print(f"Takeoff failed: {e}")
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
                print("Landing successful")
                
                # Broadcast to web clients
                self.socketio.emit('drone_action', {
                    'action': 'land',
                    'success': True
                })
                self.broadcast_status()
                return True
            except Exception as e:
                print(f"Landing failed: {e}")
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
            print(f"Broadcast status error: {e}")

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
                print(f"Error sending frame to React: {e}")

    def save_screenshot(self, frame, humans_count, source="auto"):
        """Save screenshot with timestamp and human count"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_prefix = "manual" if source in ["joystick", "keyboard", "web"] else "auto"
            filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count}persons_{self.screenshot_count:04d}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Save the frame
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.screenshot_count += 1
                print(f"Screenshot saved ({source}): {filename}")
                
                # Notify web clients
                self.socketio.emit('screenshot_result', {
                    'success': True,
                    'count': self.screenshot_count,
                    'filename': filename
                })
                return True
            else:
                print(f"Failed to save screenshot: {filename}")
                self.socketio.emit('screenshot_result', {
                    'success': False,
                    'count': self.screenshot_count
                })
                return False
        except Exception as e:
            print(f"Screenshot error: {e}")
            return False

    def get_joystick_input(self):
        if not self.joystick:
            return

        speed = 50
        rotate = 80

        # Read joystick input
        axis_lr = self.joystick.get_axis(0)  # Left-right movement
        axis_fb = self.joystick.get_axis(1)  # Forward-backward movement
        axis_yv = self.joystick.get_axis(2)  # Up-down movement  
        axis_ud = self.joystick.get_axis(3)  # Yaw rotation

        # Set velocities based on joystick input
        self.left_right_velocity = int(axis_lr * speed)
        self.for_back_velocity = int(-axis_fb * speed)
        self.up_down_velocity = int(-axis_ud * speed)
        self.yaw_velocity = int(axis_yv * rotate)

        # Handle buttons
        if self.joystick.get_button(0):  # Button A - takeoff
            if not self.send_rc_control:
                self.takeoff_drone()
                time.sleep(0.5)

        if self.joystick.get_button(1):  # Button B - land
            if self.send_rc_control:
                self.land_drone()
                time.sleep(0.5)

        # Screenshot buttons
        current_screenshot_button_state = self.joystick.get_button(2)
        if current_screenshot_button_state and not self.last_joystick_screenshot_button_state:
            self.joystick_screenshot_requested = True
            print("Joystick screenshot button pressed!")
        
        self.last_joystick_screenshot_button_state = current_screenshot_button_state

        if self.joystick.get_button(3):  # Alternative screenshot button
            self.joystick_screenshot_requested = True
            print("Alternative joystick screenshot button pressed!")
            time.sleep(0.2)

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
                print(f"Human detection error: {e}")

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
                print(f"Human detected! Starting 3-second countdown...")
            
            # If countdown is active
            if self.countdown_active:
                elapsed_time = current_time - self.countdown_start_time
                
                if elapsed_time >= self.countdown_duration:
                    # Countdown finished, take screenshot
                    self.save_screenshot(output_frame, humans_count, "auto")
                    self.last_screenshot_time = current_time
                    self.countdown_active = False
                    print("Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if self.countdown_active:
                # Cancel countdown if human disappears
                self.countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        self.last_human_detected = human_detected

    def run_web_server(self):
        """Run Flask server with Socket.IO in separate thread"""
        print("Starting Flask server with Socket.IO...")
        print("React app will run on: http://localhost:5173")
        print("Backend Socket.IO server on: http://localhost:5000")
        print("Browser HTML stream: http://localhost:5000")
        
        # Use port 5000 for backend, Vite will use 5173 for frontend
        self.socketio.run(
            self.app, 
            host='127.0.0.1',  # Gunakan 127.0.0.1 instead of 0.0.0.0
            port=5000, 
            debug=False, 
            use_reloader=False,
            allow_unsafe_werkzeug=True  # Untuk menghindari warning Werkzeug
        )

    def run(self):
        # Start Flask server in separate thread
        flask_thread = threading.Thread(target=self.run_web_server, daemon=True)
        flask_thread.start()

        # Connect to Tello
        self.connect_tello()
        print(f"Battery: {self.get_battery()}%")

        # In case streaming is on
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        # Auto-start socket streaming
        self.socket_streaming = True

        should_stop = False
        battery_counter = 0

        while not should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            # Get joystick input
            self.get_joystick_input()

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            if frame is None:
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

            # Handle joystick screenshot request
            if self.joystick_screenshot_requested:
                self.save_screenshot(output_frame, humans_count, "joystick")
                cv2.putText(output_frame, "JOYSTICK SCREENSHOT SAVED!", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                self.joystick_screenshot_requested = False

            # Add info overlays
            battery = self.get_battery()
            cv2.putText(output_frame, f"Battery: {battery}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if humans_count > 0:
                cv2.putText(output_frame, f"Humans Detected: {humans_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(output_frame, f"Screenshots: {self.screenshot_count}", (10, 120),
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

            # Convert frame for pygame display
            frame_rgb = np.rot90(output_frame)
            frame_rgb = np.flipud(frame_rgb)

            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            self.screen.blit(frame_surface, (0, 0))

            # Add pygame control instructions
            status_text = self.font.render("T=Takeoff, L=Land, P=Screenshot, ESC=Quit", True, (255, 255, 255))
            self.screen.blit(status_text, (10, 10))
            
            # Add React clients info
            react_text = self.font.render(f"React Clients: {self.connected_clients}", True, (255, 255, 0))
            self.screen.blit(react_text, (10, 40))

            pygame.display.update()

            time.sleep(1 / FPS)

        # Cleanup
        self.should_stop = True
        self.tello.streamoff()
        self.tello.end()
        self.pose.close()
        self.hands.close()
        print(f"Done! Total screenshots taken: {self.screenshot_count}")

    def keydown(self, key):
        """ Update velocities based on key pressed """
        if key == pygame.K_w:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_s:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_a:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_d:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_UP:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_p:  # Manual screenshot
            # Take screenshot
            if self.last_frame is not None:
                with self.frame_lock:
                    frame_copy = self.last_frame.copy()
                output_frame, _, humans_count = self.process_human_detection(frame_copy)
                self.save_screenshot(output_frame, humans_count, "keyboard")
                print("Manual keyboard screenshot taken!")
        elif key == pygame.K_DOWN:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_LEFT:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_RIGHT:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_q:
            """ Quit the program """
            pygame.quit()
            sys.exit()
            self.tello.streamoff()
            self.tello.end()
            self.pose.close()
            self.hands.close()
            print(f"Done! Total screenshots taken: {self.screenshot_count}")
            sys.exit()

    def keyup(self, key):
        """ Update velocities based on key released """
        if key == pygame.K_w or key == pygame.K_s:
            self.for_back_velocity = 0
        elif key == pygame.K_d or key == pygame.K_a:
            self.left_right_velocity = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:
            self.up_down_velocity = 0
        elif key == pygame.K_RIGHT or key == pygame.K_LEFT:
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.takeoff_drone()
        elif key == pygame.K_l:  # land
            self.land_drone()

    def update(self):
        """ Update routine. Send velocities to Tello. """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = TelloControllerStreamer()
    
    print("=" * 60)
    print("SARVIO-X - Tello Drone Control with Dual Interface")
    print("=" * 60)
    print("Features:")
    print("- Pygame window for local control and display")
    print("- React web interface with real-time streaming")
    print("- YOLOv8 + MediaPipe human detection")
    print("- Smart auto screenshot with 3-second countdown")
    print("- Dual control: Keyboard/Joystick + Web interface")
    print("- Flask-SocketIO backend on port 5000")
    print("=" * 60)
    print("Pygame Controls:")
    print("- Keyboard: Arrow keys=move, W/S=up/down, A/D=rotate")
    print("- T=takeoff, L=land, P=screenshot, ESC=quit")
    print("- Joystick: Move drone, A=takeoff, B=land, X/Y=screenshot")
    print("=" * 60)
    print("Web Interface:")
    print("- React app: http://localhost:5173")
    print("- Backend API: http://localhost:5000")
    print("- Real-time video streaming with ML detection")
    print("- Remote control via web browser")
    print("=" * 60)
    
    try:
        frontend.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    except Exception as e:
        print(f"Error occurred: {e}")


class TelloControllerStreamer(FrontEnd):
    """
    Alias class to match the reference naming convention
    This extends FrontEnd with all the Flask-SocketIO functionality
    """
    pass


if __name__ == '__main__':
    main()