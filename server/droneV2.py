from djitellopy import Tello
import cv2, queue, threading, time, pygame, os, sys
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import base64
import json

# Flask dan SocketIO imports
from flask import Flask, Response, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Speed and FPS settings
FPS = 30  # Reduced for better performance with web streaming
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# ROI parameters (Region of Interest)
ROI_X = WINDOW_WIDTH // 4
ROI_Y = 5
ROI_WIDTH = WINDOW_WIDTH // 2
ROI_HEIGHT = WINDOW_HEIGHT // 3

# Thread control
running = True
threads = []

# Thread-safe queues
frame_queue = queue.Queue(maxsize=5)
command_queue = queue.Queue()
screenshot_queue = queue.Queue()
web_command_queue = queue.Queue()  # New: Web commands queue

# Shared data with locks
data_lock = threading.Lock()
current_frame = None
current_processed_frame = None
current_detection = None
battery_level = 0
fps = 0
humans_count = 0
human_detected = False
screenshot_count = 0

# Performance monitoring
frame_times = deque(maxlen=30)
last_frame_time = time.time()

# Drone control variables
for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
speed = 10
send_rc_control = False
is_flying = False
is_connected = False

# Screenshot variables
last_screenshot_time = 0
screenshot_interval = 3
countdown_active = False
countdown_start_time = 0
countdown_duration = 3.0
last_human_detected = False

set_autonomous_behavior = False

# Joystick screenshot variables
last_joystick_screenshot_button_state = False
joystick_screenshot_requested = False

# Web interface control flags
ml_detection_enabled = True
auto_capture_enabled = True
socket_streaming = False
connected_clients = 0

# Global objects
tello = None
screen = None
joystick = None
screenshot_dir = "screenshots"
yolo_model = None
mp_drawing = None
mp_drawing_styles = None
mp_pose = None
mp_hands = None
pose = None
hands = None

# Flask dan SocketIO objects
app = None
socketio = None

def initialize_flask_app():
    """Initialize Flask app and SocketIO"""
    global app, socketio
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'tello_secret_key_2024'
    
    # CORS for React app
    CORS(app, 
         origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
         allow_headers=["Content-Type"],
         methods=["GET", "POST"])
    
    # Socket.IO initialization
    socketio = SocketIO(
        app, 
        cors_allowed_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
        async_mode='threading',
        logger=False,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25,
        transports=['polling', 'websocket']
    )
    
    setup_flask_routes()
    setup_socket_events()

def setup_flask_routes():
    """Setup Flask routes"""
    
    @app.route('/')
    def index():
        return render_template_string("""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Tello Control Backend - SARVIO-X</title>
                <meta charset="UTF-8">
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        margin: 40px; 
                        background: #1a1a1a; 
                        color: white; 
                    }
                    .status { 
                        padding: 10px; 
                        margin: 10px 0; 
                        border-radius: 5px; 
                    }
                    .connected { background: #065f46; }
                    .disconnected { background: #7f1d1d; }
                    img { 
                        border: 2px solid #06b6d4; 
                        border-radius: 8px; 
                        max-width: 100%;
                    }
                    .info { background: #1e40af; }
                </style>
            </head>
            <body>
                <h1>🚁 SARVIO-X Backend Server</h1>
                <h2>Tello 4-Thread Control System dengan React Integration</h2>
                
                <div class="status connected">
                    ✅ Flask-SocketIO Server: Active pada port 5000
                </div>
                
                <div class="status info">
                    🎥 Video Stream: <a href="{{ url_for('video_feed') }}" target="_blank">Direct Feed</a>
                </div>
                
                <div class="status info">
                    🌐 React App: <a href="http://localhost:5173" target="_blank">Web Interface (Vite)</a>
                </div>
                
                <div class="status info">
                    📊 API Status: <a href="{{ url_for('api_status') }}" target="_blank">JSON Status</a>
                </div>
                
                <h3>Live Video Stream dengan ML Detection:</h3>
                <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Tello Live Stream">
                
                <h3>System Features:</h3>
                <ul>
                    <li>✅ 4-Thread Architecture (Video, Control, Detection, Autonomous)</li>
                    <li>✅ YOLOv8 Human Detection + MediaPipe Body Tracking</li>
                    <li>✅ Red Color Detection dengan ROI</li>
                    <li>✅ Auto Screenshot (3s countdown)</li>
                    <li>✅ Manual Screenshot (Keyboard/Joystick/Web)</li>
                    <li>✅ Real-time Socket.IO Communication</li>
                    <li>✅ Dual Interface (Pygame + React Web)</li>
                    <li>✅ Autonomous Behavior Mode</li>
                </ul>
                
                <h3>Controls:</h3>
                <ul>
                    <li><strong>Pygame:</strong> Arrow keys, WASD, T=takeoff, L=land, P=screenshot</li>
                    <li><strong>Joystick:</strong> A=takeoff, B=land, X/Y=screenshot, Shoulder=RC control</li>
                    <li><strong>React Web:</strong> Virtual controls, real-time monitoring, settings</li>
                </ul>
                
                <p><strong>React Frontend:</strong> 
                   Gunakan <a href="http://localhost:5173">localhost:5173</a> untuk kontrol penuh via web browser
                </p>
            </body>
        </html>
        """)

    @app.route('/video_feed')
    def video_feed():
        return Response(frame_generator(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/status')
    def api_status():
        """API endpoint untuk status JSON"""
        with data_lock:
            status = {
                'tello': {
                    'connected': is_connected,
                    'flying': is_flying,
                    'battery': battery_level,
                    'send_rc_control': send_rc_control
                },
                'detection': {
                    'ml_enabled': ml_detection_enabled,
                    'auto_capture_enabled': auto_capture_enabled,
                    'humans_detected': humans_count,
                    'human_detected': human_detected,
                    'red_detected': current_detection.get('red_detected', False) if current_detection else False
                },
                'system': {
                    'fps': fps,
                    'screenshot_count': screenshot_count,
                    'connected_clients': connected_clients,
                    'autonomous_mode': set_autonomous_behavior
                },
                'threads': {
                    'running': running,
                    'active_threads': len([t for t in threads if t.is_alive()])
                }
            }
        return jsonify(status)

def setup_socket_events():
    """Setup Socket.IO event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        global connected_clients, socket_streaming
        connected_clients += 1
        socket_streaming = True
        print(f'✅ React client connected. Total: {connected_clients}')
        
        # Send welcome and status
        socketio.emit('connection_info', {
            'message': 'Connected to SARVIO-X 4-Thread Backend',
            'version': '2.0',
            'features': ['4_thread_architecture', 'ml_detection', 'autonomous_mode', 'dual_control']
        })
        
        broadcast_status()
    
    @socketio.on('disconnect')
    def handle_disconnect():
        global connected_clients, socket_streaming
        connected_clients -= 1
        if connected_clients <= 0:
            socket_streaming = False
        print(f'❌ Client disconnected. Total: {connected_clients}')

    @socketio.on('connect_tello')
    def handle_connect_tello():
        """Connect to Tello via web command"""
        web_command_queue.put(('connect', None))
        
    @socketio.on('disconnect_tello')  
    def handle_disconnect_tello():
        """Disconnect Tello via web command"""
        web_command_queue.put(('disconnect', None))

    @socketio.on('takeoff')
    def handle_takeoff():
        """Takeoff command from web"""
        web_command_queue.put(('takeoff', None))
        print("🚁 Web takeoff command queued")

    @socketio.on('land')
    def handle_land():
        """Land command from web"""
        web_command_queue.put(('land', None))
        print("🏠 Web land command queued")

    @socketio.on('emergency')
    def handle_emergency():
        """Emergency stop from web"""
        web_command_queue.put(('emergency', None))
        print("🚨 Web emergency command queued")

    @socketio.on('move_control')
    def handle_move_control(data):
        """Handle movement from web interface"""
        move_data = {
            'left_right': int(data.get('left_right', 0)),
            'for_back': int(data.get('for_back', 0)),
            'up_down': int(data.get('up_down', 0)),
            'yaw': int(data.get('yaw', 0))
        }
        web_command_queue.put(('move', move_data))

    @socketio.on('stop_movement')
    def handle_stop_movement():
        """Stop all movement"""
        web_command_queue.put(('stop_move', None))

    @socketio.on('manual_screenshot')
    def handle_manual_screenshot():
        """Manual screenshot from web"""
        request_manual_screenshot("web")
        socketio.emit('screenshot_requested', {'success': True})

    @socketio.on('toggle_ml_detection')
    def handle_toggle_ml_detection(data):
        """Toggle ML detection"""
        global ml_detection_enabled
        ml_detection_enabled = data.get('enabled', True)
        socketio.emit('ml_detection_status', {'enabled': ml_detection_enabled})
        print(f"🤖 ML Detection: {'Enabled' if ml_detection_enabled else 'Disabled'}")

    @socketio.on('toggle_auto_capture')
    def handle_toggle_auto_capture(data):
        """Toggle auto capture"""
        global auto_capture_enabled
        auto_capture_enabled = data.get('enabled', True)
        socketio.emit('auto_capture_status', {'enabled': auto_capture_enabled})
        print(f"📸 Auto Capture: {'Enabled' if auto_capture_enabled else 'Disabled'}")

    @socketio.on('toggle_autonomous')
    def handle_toggle_autonomous(data):
        """Toggle autonomous behavior"""
        global set_autonomous_behavior
        set_autonomous_behavior = data.get('enabled', False)
        socketio.emit('autonomous_status', {'enabled': set_autonomous_behavior})
        print(f"🤖 Autonomous Mode: {'Enabled' if set_autonomous_behavior else 'Disabled'}")

def broadcast_status():
    """Broadcast current status to all clients"""
    if connected_clients > 0:
        with data_lock:
            status = {
                'tello': {
                    'connected': is_connected,
                    'flying': is_flying,
                    'battery': battery_level
                },
                'detection': {
                    'humans_count': humans_count,
                    'human_detected': human_detected,
                    'red_detected': current_detection.get('red_detected', False) if current_detection else False,
                    'ml_enabled': ml_detection_enabled,
                    'auto_capture_enabled': auto_capture_enabled
                },
                'system': {
                    'fps': fps,
                    'screenshot_count': screenshot_count,
                    'autonomous_mode': set_autonomous_behavior
                }
            }
        socketio.emit('status_update', status)

def frame_generator():
    """Generator for HTTP video streaming"""
    while running:
        with data_lock:
            if current_processed_frame is None:
                time.sleep(0.1)
                continue
            frame_to_send = current_processed_frame.copy()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1 / 30)  # 30 FPS untuk HTTP stream

def send_frame_to_react():
    """Send frame to React via SocketIO"""
    if socket_streaming and connected_clients > 0:
        try:
            with data_lock:
                if current_processed_frame is None:
                    return
                frame = current_processed_frame.copy()
            
            # Resize untuk bandwidth efficiency
            frame_resized = cv2.resize(frame, (480, 360))
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_base64})
        except Exception as e:
            print(f"Error sending frame to React: {e}")

def initialize_pygame():
    """Initialize pygame and create window"""
    global screen, joystick
    
    pygame.init()
    pygame.display.set_caption("Tello 4-Thread + React Control")
    screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    
    # Initialize joystick
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick initialized: {joystick.get_name()}")
    else:
        print("No joystick detected - using keyboard only")
    
    # Create update timer
    pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

def initialize_tello():
    """Initialize Tello drone connection"""
    global tello, battery_level, is_connected
    
    try:
        tello = Tello()
        tello.connect()
        tello.set_speed(speed)
        battery_level = tello.get_battery()
        is_connected = True
        print(f"Tello connected! Battery: {battery_level}%")
        return True
    except Exception as e:
        print(f"Failed to connect to Tello: {e}")
        is_connected = False
        return False

def initialize_ai_models():
    """Initialize AI models for detection"""
    global yolo_model, mp_drawing, mp_drawing_styles, mp_pose, mp_hands, pose, hands, current_detection
    
    try:
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        yolo_model = YOLO('yolov8n.pt')
        
        # Mediapipe modules
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        
        # Create MediaPipe models
        pose = mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        hands = mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            max_num_hands=2
        )
        
        # Initialize detection data
        current_detection = {
            'red_detected': False,
            'mask': None,
            'result': None,
            'roi_mask': None,
            'full_roi_mask': None,
            'pixel_count': 0
        }
        
        print("AI models initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize AI models: {e}")
        return False

def create_screenshot_directory():
    """Create screenshots directory"""
    try:
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
            print(f"Created directory: {screenshot_dir}")
    except Exception as e:
        print(f"Failed to create screenshot directory: {e}")

def process_human_detection(frame):
    """Process human detection with YOLOv8 + MediaPipe"""
    if not ml_detection_enabled:
        return frame, False, 0
        
    try:
        output_frame = frame.copy()
        
        # YOLOv8 Human Detection
        results = yolo_model(frame, verbose=False)
        detected = False
        human_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == 0 and confidence > 0.5:  # Person class
                        detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))

                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        cv2.circle(output_frame, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

                        # Label
                        confidence_percentage = confidence * 100
                        label = f"Human: {confidence_percentage:.0f}%"
                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # MediaPipe body tracking if human detected
        if detected:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            pose_results = pose.process(rgb_frame)
            hands_results = hands.process(rgb_frame)

            # Draw hands
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Draw pose
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    output_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

        return output_frame, detected, len(human_boxes)
    
    except Exception as e:
        print(f"Error in human detection: {e}")
        return frame, False, 0

def detect_red_in_roi(img):
    """Detect red color in ROI"""
    try:
        if img is None:
            return False, None, None, 0
        
        # Extract ROI
        roi = img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
        
        # HSV conversion and red detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 130])
        upper_red1 = np.array([30, 255, 255])
        lower_red2 = np.array([230, 150, 130])
        upper_red2 = np.array([255, 255, 255])

        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_roi = cv2.bitwise_or(mask1, mask2)
        
        red_detected = np.sum(mask_roi) > 10000 
        pixel_count = np.sum(mask_roi > 0)
        
        # Full-size mask
        full_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        full_mask[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH] = mask_roi
        
        return red_detected, mask_roi, full_mask, pixel_count
    
    except Exception as e:
        print(f"Error in red detection ROI: {e}")
        return False, None, None, 0

def detect_red_color(img):
    """Detect red color in full image"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 130])
        upper_red1 = np.array([30, 255, 255])
        lower_red2 = np.array([230, 150, 130])
        upper_red2 = np.array([255, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return mask, result
    
    except Exception as e:
        print(f"Error in red color detection: {e}")
        return None, None

def handle_auto_screenshot(output_frame, human_detected_now, humans_count_now):
    """Handle auto screenshot with countdown"""
    if not auto_capture_enabled:
        return
        
    global countdown_active, countdown_start_time, last_human_detected
    
    try:
        current_time = time.time()
        
        if human_detected_now and humans_count_now >= 1:
            if not last_human_detected and not countdown_active:
                countdown_active = True
                countdown_start_time = current_time
                print("Human detected! Starting 3-second countdown...")
            
            if countdown_active:
                elapsed_time = current_time - countdown_start_time
                
                if elapsed_time >= countdown_duration:
                    save_screenshot(output_frame.copy(), humans_count_now, "auto")
                    countdown_active = False
                    print("Auto screenshot taken!")
                    
                    # Notify web clients
                    if connected_clients > 0:
                        socketio.emit('auto_screenshot_taken', {
                            'count': screenshot_count,
                            'humans': humans_count_now
                        })
        else:
            if countdown_active:
                countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        last_human_detected = human_detected_now
        
    except Exception as e:
        print(f"Error in auto screenshot: {e}")

def save_screenshot(frame, humans_count_param, source="auto"):
    """Save screenshot with metadata"""
    global screenshot_count
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_prefix = "manual" if source in ["joystick", "keyboard", "web"] else "auto"
        filename = f"{source_prefix}_humans_{timestamp}_{humans_count_param}persons_{screenshot_count:04d}.jpg"
        filepath = os.path.join(screenshot_dir, filename)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(filepath, frame_bgr)
        
        if success:
            screenshot_count += 1
            print(f"Screenshot saved ({source}): {filename}")
            
            # Notify web clients
            if connected_clients > 0:
                socketio.emit('screenshot_saved', {
                    'filename': filename,
                    'source': source,
                    'count': screenshot_count,
                    'humans': humans_count_param
                })
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False
    
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

def request_manual_screenshot(source):
    """Request manual screenshot"""
    try:
        with data_lock:
            if current_processed_frame is not None:
                screenshot_queue.put((
                    current_processed_frame.copy(), 
                    humans_count, 
                    source
                ))
                print(f"Manual screenshot requested ({source})")
    except Exception as e:
        print(f"Screenshot request error: {e}")

# THREAD FUNCTIONS

def video_stream_thread():
    """Thread 1: Video capture, processing, screenshots"""
    global current_frame, current_processed_frame, human_detected, humans_count, fps, frame_times
    
    print("🎥 Video stream thread started")
    
    try:
        frame_read = tello.get_frame_read()
        
        while running:
            try:
                if frame_read.stopped:
                    break
                    
                frame = frame_read.frame
                if frame is not None:
                    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    
                    # Process human detection
                    output_frame, detected, count = process_human_detection(frame)
                    
                    # Handle auto screenshot
                    handle_auto_screenshot(output_frame, detected, count)
                    
                    # Process screenshot queue
                    while not screenshot_queue.empty():
                        try:
                            screenshot_data = screenshot_queue.get_nowait()
                            frame_to_save, humans_count_param, source = screenshot_data
                            save_screenshot(frame_to_save, humans_count_param, source)
                        except queue.Empty:
                            break
                    
                    # Update shared data
                    with data_lock:
                        current_frame = frame.copy()
                        current_processed_frame = output_frame.copy()
                        human_detected = detected
                        humans_count = count
                        
                        # Calculate FPS
                        current_time = time.time()
                        frame_times.append(current_time)
                        if len(frame_times) > 1:
                            time_diff = frame_times[-1] - frame_times[0]
                            fps = len(frame_times) / time_diff if time_diff > 0 else 0
                    
                    # Send frame to React clients
                    send_frame_to_react()
                        
                time.sleep(1/60)  # 60 FPS capture limit
                
            except Exception as e:
                print(f"Video stream error: {e}")
                time.sleep(0.1)
        
    except Exception as e:
        print(f"Critical video stream error: {e}")
    
    print("🎥 Video stream thread ended")

def drone_control_thread():
    """Thread 2: Drone control and battery monitoring"""
    global battery_level, is_flying, send_rc_control
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    
    print("🎮 Drone control thread started")
    last_battery_check = time.time()
    
    while running:
        try:
            # Process local commands
            while not command_queue.empty():
                try:
                    command = command_queue.get_nowait()
                    execute_drone_command(command)
                except queue.Empty:
                    break
            
            # Process web commands
            while not web_command_queue.empty():
                try:
                    web_command, data = web_command_queue.get_nowait()
                    execute_web_command(web_command, data)
                except queue.Empty:
                    break
            
            # Send RC control
            if send_rc_control and is_connected:
                try:
                    tello.send_rc_control(
                        left_right_velocity, 
                        for_back_velocity,
                        up_down_velocity, 
                        yaw_velocity
                    )
                except Exception as e:
                    print(f"RC control error: {e}")
            
            # Battery monitoring
            current_time = time.time()
            if current_time - last_battery_check >= 10:
                try:
                    if is_connected:
                        with data_lock:
                            battery_level = tello.get_battery()
                        
                        # Broadcast to web clients
                        if connected_clients > 0:
                            socketio.emit('battery_update', {'battery': battery_level})
                        
                        last_battery_check = current_time
                except Exception as e:
                    print(f"Battery check error: {e}")
            
            time.sleep(1/30)  # 30 Hz control rate
            
        except Exception as e:
            print(f"Drone control error: {e}")
            time.sleep(0.1)
    
    print("🎮 Drone control thread ended")

def detection_thread():
    """Thread 3: Red color detection"""
    global current_detection
    
    print("🔍 Detection thread started")
    
    while running:
        try:
            with data_lock:
                if current_processed_frame is not None:
                    frame_copy = current_processed_frame.copy()
                else:
                    frame_copy = None
            
            if frame_copy is not None:
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                
                # Red color detection
                mask, result = detect_red_color(frame_copy)
                red_in_roi, roi_mask, full_roi_mask, pixel_count = detect_red_in_roi(frame_copy)
                
                # Update detection results
                with data_lock:
                    current_detection = {
                        'red_detected': red_in_roi,
                        'mask': mask,
                        'result': result,
                        'roi_mask': roi_mask,
                        'full_roi_mask': full_roi_mask,
                        'pixel_count': pixel_count
                    }
                
                # Broadcast red detection to web clients
                if connected_clients > 0 and red_in_roi:
                    socketio.emit('red_detected', {
                        'detected': red_in_roi,
                        'pixel_count': pixel_count,
                        'roi_position': {'x': ROI_X, 'y': ROI_Y, 'width': ROI_WIDTH, 'height': ROI_HEIGHT}
                    })
            
            time.sleep(0.03)  # ~30 FPS detection rate
            
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)
    
    print("🔍 Detection thread ended")

def autonomous_behavior_thread():
    """Thread 4: Autonomous behavior"""
    print("🤖 Autonomous behavior thread started")
    
    while running:
        try:
            if set_autonomous_behavior and is_connected and current_detection is not None:
                with data_lock:
                    if current_detection:
                        red_detected = current_detection.get('red_detected', False)
                        pixel_count = current_detection.get('pixel_count', 0)
                    else:
                        red_detected = False
                        pixel_count = 0
                
                # Autonomous behavior logic
                if red_detected:
                    print("🔴 Red detected in ROI! Executing autonomous maneuver...")
                    
                    # Notify web clients
                    if connected_clients > 0:
                        socketio.emit('autonomous_action', {
                            'action': 'red_target_found',
                            'pixel_count': pixel_count
                        })
                    
                    # Execute maneuver
                    try:
                        tello.move_back(70)
                        time.sleep(2)
                        tello.rotate_clockwise(90)
                        time.sleep(2)
                    except Exception as e:
                        print(f"Autonomous maneuver error: {e}")
                else:
                    print("⚪ No red in ROI. Searching...")
                    try:
                        tello.move_forward(30)
                        time.sleep(1)
                    except Exception as e:
                        print(f"Autonomous search error: {e}")
            else:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Autonomous behavior error: {e}")
            time.sleep(0.5)
    
    print("🤖 Autonomous behavior thread ended")

def web_server_thread():
    """Thread 5: Flask-SocketIO web server"""
    print("🌐 Web server thread started on port 5000")
    
    try:
        socketio.run(
            app, 
            host='127.0.0.1',
            port=5000, 
            debug=False, 
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"Web server error: {e}")
    
    print("🌐 Web server thread ended")

# COMMAND EXECUTION FUNCTIONS

def execute_drone_command(command):
    """Execute local drone commands"""
    global send_rc_control, is_flying
    
    try:
        if command == "takeoff" and is_connected and not is_flying:
            tello.takeoff()
            send_rc_control = True
            is_flying = True
            print("✈️ Takeoff successful")
            
        elif command == "land" and is_connected and is_flying:
            tello.land()
            send_rc_control = False
            is_flying = False
            print("🏠 Landing successful")
            
        elif command == "emergency":
            tello.emergency()
            send_rc_control = False
            is_flying = False
            print("🚨 Emergency stop executed")
            
    except Exception as e:
        print(f"Command execution error: {e}")

def execute_web_command(web_command, data):
    """Execute commands from web interface"""
    global send_rc_control, is_flying, is_connected
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    
    try:
        if web_command == "connect":
            success = initialize_tello()
            socketio.emit('tello_connection_result', {
                'success': success,
                'message': 'Tello connected' if success else 'Connection failed'
            })
            broadcast_status()
            
        elif web_command == "disconnect":
            if is_connected:
                if is_flying:
                    tello.land()
                    is_flying = False
                tello.streamoff()
                tello.end()
                is_connected = False
                send_rc_control = False
                print("Tello disconnected via web")
                socketio.emit('tello_connection_result', {
                    'success': True,
                    'message': 'Tello disconnected'
                })
                broadcast_status()
            
        elif web_command == "takeoff":
            execute_drone_command("takeoff")
            socketio.emit('drone_action_result', {
                'action': 'takeoff',
                'success': is_flying
            })
            broadcast_status()
            
        elif web_command == "land":
            execute_drone_command("land")
            socketio.emit('drone_action_result', {
                'action': 'land', 
                'success': not is_flying
            })
            broadcast_status()
            
        elif web_command == "emergency":
            execute_drone_command("emergency")
            socketio.emit('drone_action_result', {
                'action': 'emergency',
                'success': True
            })
            broadcast_status()
            
        elif web_command == "move" and data:
            left_right_velocity = data['left_right']
            for_back_velocity = data['for_back']
            up_down_velocity = data['up_down']
            yaw_velocity = data['yaw']
            
        elif web_command == "stop_move":
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
            
    except Exception as e:
        print(f"Web command execution error: {e}")

def get_joystick_input():
    """Handle joystick input"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    global last_joystick_screenshot_button_state, send_rc_control, set_autonomous_behavior
    
    if not joystick:
        return

    try:
        speed_joy = 50
        rotate = 80

        # Joystick axes
        axis_lr = joystick.get_axis(0)
        axis_fb = joystick.get_axis(1)
        axis_yv = joystick.get_axis(2)
        axis_ud = joystick.get_axis(3)

        # Set velocities
        left_right_velocity = int(axis_lr * speed_joy)
        for_back_velocity = int(-axis_fb * speed_joy)
        up_down_velocity = int(-axis_ud * speed_joy)
        yaw_velocity = int(axis_yv * rotate)

        # Button controls
        if joystick.get_button(0):  # A - takeoff
            if not send_rc_control:
                command_queue.put("takeoff")
                time.sleep(0.5)

        if joystick.get_button(1):  # B - land
            if send_rc_control:
                command_queue.put("land")
                time.sleep(0.5)

        # Screenshot buttons
        current_screenshot_button_state = joystick.get_button(2)
        if current_screenshot_button_state and not last_joystick_screenshot_button_state:
            request_manual_screenshot("joystick")
        last_joystick_screenshot_button_state = current_screenshot_button_state

        if joystick.get_button(3):
            request_manual_screenshot("joystick")
            time.sleep(0.2)
        
        # Autonomous control
        if joystick.get_button(6):
            set_autonomous_behavior = True
            print("🤖 Autonomous mode enabled via joystick")
            if connected_clients > 0:
                socketio.emit('autonomous_status', {'enabled': True})

        elif joystick.get_button(7):
            set_autonomous_behavior = False
            print("🤖 Autonomous mode disabled via joystick")
            if connected_clients > 0:
                socketio.emit('autonomous_status', {'enabled': False})
        
        # RC control toggle
        if joystick.get_button(8):  # Left shoulder
            send_rc_control = True
            print("🎮 RC Control: ENABLED")

        if joystick.get_button(9):  # Right shoulder
            send_rc_control = False
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
            print("🎮 RC Control: DISABLED")

    except Exception as e:
        print(f"Joystick input error: {e}")

def draw_roi_rectangle(surface):
    """Draw ROI rectangle on pygame surface"""
    try:
        roi_color = (0, 255, 0)
        roi_rect = pygame.Rect(ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
        pygame.draw.rect(surface, roi_color, roi_rect, 3)
    except Exception as e:
        print(f"ROI drawing error: {e}")

def start_threads():
    """Start all worker threads"""
    global threads
    
    # Thread 1: Video stream
    video_thread = threading.Thread(target=video_stream_thread, daemon=True)
    video_thread.start()
    threads.append(video_thread)
    
    # Thread 2: Drone control
    control_thread = threading.Thread(target=drone_control_thread, daemon=True)
    control_thread.start()
    threads.append(control_thread)

    # Thread 3: Detection
    detection_thread_instance = threading.Thread(target=detection_thread, daemon=True)
    detection_thread_instance.start()
    threads.append(detection_thread_instance)

    # Thread 4: Autonomous behavior
    autonomous_thread = threading.Thread(target=autonomous_behavior_thread, daemon=True)
    autonomous_thread.start()
    threads.append(autonomous_thread)
    
    # Thread 5: Web server
    web_thread = threading.Thread(target=web_server_thread, daemon=True)
    web_thread.start()
    threads.append(web_thread)
    
    print(f"🚀 Started {len(threads)} worker threads")

def stop_threads():
    """Stop all worker threads"""
    global running
    
    print("🛑 Stopping threads...")
    running = False
    
    # Wait for threads to finish
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=2)
    
    print("✅ All threads stopped")

def handle_pygame_keys(key):
    """Handle pygame keyboard input"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    
    S = 60  # Speed
    
    if key == pygame.K_w:
        for_back_velocity = S
    elif key == pygame.K_s:
        for_back_velocity = -S
    elif key == pygame.K_a:
        left_right_velocity = -S
    elif key == pygame.K_d:
        left_right_velocity = S
    elif key == pygame.K_UP:
        up_down_velocity = S
    elif key == pygame.K_DOWN:
        up_down_velocity = -S
    elif key == pygame.K_LEFT:
        yaw_velocity = -S
    elif key == pygame.K_RIGHT:
        yaw_velocity = S
    elif key == pygame.K_t:
        command_queue.put("takeoff")
    elif key == pygame.K_l:
        command_queue.put("land")
    elif key == pygame.K_p:
        request_manual_screenshot("keyboard")

def handle_pygame_keyup(key):
    """Handle pygame key release"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    
    if key in [pygame.K_w, pygame.K_s]:
        for_back_velocity = 0
    elif key in [pygame.K_a, pygame.K_d]:
        left_right_velocity = 0
    elif key in [pygame.K_UP, pygame.K_DOWN]:
        up_down_velocity = 0
    elif key in [pygame.K_LEFT, pygame.K_RIGHT]:
        yaw_velocity = 0

def main_loop():
    """Main pygame loop with UI"""
    global running
    
    should_stop = False
    status_broadcast_counter = 0
    
    try:
        while not should_stop:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    pass  # Update timer
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        handle_pygame_keys(event.key)
                elif event.type == pygame.KEYUP:
                    handle_pygame_keyup(event.key)

            # Get joystick input
            get_joystick_input()

            # Clear screen
            screen.fill([0, 0, 0])

            # Get latest frame for display
            display_frame = None
            with data_lock:
                if current_processed_frame is not None:
                    display_frame = current_processed_frame.copy()
                    current_fps = fps
                    current_battery = battery_level
                    current_humans = humans_count
                    current_screenshots = screenshot_count
                    current_red_detected = current_detection.get('red_detected', False) if current_detection else False
            
            if display_frame is not None:
                # Add overlays
                cv2.putText(display_frame, f"Battery: {current_battery}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_humans > 0:
                    cv2.putText(display_frame, f"Humans: {current_humans}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(display_frame, f"Screenshots: {current_screenshots}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(display_frame, "5-THREAD + REACT MODE", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                cv2.putText(display_frame, f"Web Clients: {connected_clients}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if current_red_detected:
                    cv2.putText(display_frame, "RED DETECTED IN ROI!", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if set_autonomous_behavior:
                    cv2.putText(display_frame, "AUTONOMOUS MODE", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Show countdown if active
                if countdown_active:
                    elapsed = time.time() - countdown_start_time
                    remaining = max(0, countdown_duration - elapsed)
                    cv2.putText(display_frame, f"Screenshot: {remaining:.1f}s", (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw ROI rectangle
                cv2.rectangle(display_frame, (ROI_X, ROI_Y), 
                             (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (0, 255, 0), 2)
                cv2.putText(display_frame, "ROI", (ROI_X, ROI_Y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert for pygame display
                frame_rgb = np.rot90(display_frame)
                frame_rgb = np.flipud(frame_rgb)
                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                screen.blit(frame_surface, (0, 0))

            # Add pygame status text
            font = pygame.font.Font(None, 36)
            status_text = font.render("T=Takeoff L=Land P=Screenshot ESC=Quit", True, (255, 255, 255))
            screen.blit(status_text, (10, 10))
            
            react_text = font.render(f"React Interface: http://localhost:5173", True, (255, 255, 0))
            screen.blit(react_text, (10, 50))

            pygame.display.update()
            
            # Broadcast status to web clients periodically
            status_broadcast_counter += 1
            if status_broadcast_counter % 90 == 0:  # Every ~1.5 seconds at 60 FPS
                broadcast_status()

            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        # Cleanup
        stop_threads()
        if tello and is_connected:
            try:
                if is_flying:
                    tello.land()
                tello.streamoff()
                tello.end()
            except:
                pass
        if pose:
            pose.close()
        if hands:
            hands.close()
        pygame.quit()
        print(f"✅ Shutdown complete! Screenshots taken: {screenshot_count}")

def main():
    """Main function"""
    print("=" * 80)
    print("🚁 SARVIO-X: 5-Thread Tello Control dengan React Integration")
    print("=" * 80)
    print("Architecture:")
    print("- Thread 1: 🎥 Video stream + Human detection + Screenshot processing")
    print("- Thread 2: 🎮 Drone control + Battery monitoring + Web commands")
    print("- Thread 3: 🔍 Red color detection + ROI analysis")
    print("- Thread 4: 🤖 Autonomous behavior + Target tracking")
    print("- Thread 5: 🌐 Flask-SocketIO web server + React communication")
    print()
    print("Features:")
    print("✅ Dual Control: Pygame (lokal) + React Web Interface")
    print("✅ Real-time video streaming dengan ML overlay")
    print("✅ YOLOv8 + MediaPipe human detection")
    print("✅ Red color detection dengan ROI targeting")
    print("✅ Smart auto screenshot (3s countdown)")
    print("✅ Manual screenshot (keyboard/joystick/web)")
    print("✅ Autonomous behavior mode")
    print("✅ Multi-client web interface support")
    print()
    print("Controls:")
    print("🖥️  Pygame: WASD/Arrow=move, T=takeoff, L=land, P=screenshot, ESC=quit")
    print("🎮 Joystick: A=takeoff, B=land, X/Y=screenshot, Shoulder=RC toggle")
    print("🌐 React Web: http://localhost:5173 (virtual controls + real-time monitoring)")
    print()
    print("Web Interfaces:")
    print("📱 React Frontend: http://localhost:5173")
    print("🖥️  Backend Status: http://localhost:5000")
    print("📊 API Status: http://localhost:5000/api/status")
    print("🎥 Video Stream: http://localhost:5000/video_feed")
    print("=" * 80)
    
    try:
        # Initialize all components
        print("🔧 Initializing components...")
        
        initialize_pygame()
        print("✅ Pygame initialized")
        
        create_screenshot_directory()
        print("✅ Screenshot directory ready")
        
        initialize_flask_app()
        print("✅ Flask-SocketIO app initialized")
        
        if not initialize_tello():
            print("❌ Failed to initialize Tello. Exiting...")
            return
        print("✅ Tello connected")
            
        if not initialize_ai_models():
            print("❌ Failed to initialize AI models. Exiting...")
            return
        print("✅ AI models loaded")
        
        # Start video stream
        tello.streamoff()
        time.sleep(0.5)
        tello.streamon()
        print("✅ Video stream started")
        
        # Start all worker threads
        start_threads()
        print("✅ All threads started")
        
        # Wait for threads to initialize
        time.sleep(3)
        print("🚀 System ready!")
        print()
        print("📱 Open your React app at: http://localhost:5173")
        print("🎮 Use this pygame window for local control")
        print("🌐 Backend server running at: http://localhost:5000")
        print()
        
        # Run main pygame loop
        main_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        print("🔄 Shutting down system...")

if __name__ == '__main__':
    main()