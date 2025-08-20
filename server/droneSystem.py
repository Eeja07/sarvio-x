import threading
import time
import os
import cv2
import numpy as np
import pygame
from config import Config
from shared_data import shared_data

try:
    from djitellopy import Tello
    from ultralytics import YOLO
    DRONE_IMPORTS_AVAILABLE = True
except ImportError:
    DRONE_IMPORTS_AVAILABLE = False

__all__ = ["DroneSystem", "DRONE_IMPORTS_AVAILABLE"]

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
        self.height = 0
        self.temperature = 0
        
        # Control variables
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.send_rc_control = False
        self.speed = 20
        
        # Recording
        self.recording = False
        self.video_writer = None
        
        # Detection
        self.detection_enabled = True
        self.current_detection = None
        
        # Autonomous behavior
        self.set_autonomous_behavior = False
        self.emergency_stop = False  # Tambahkan ini
        
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
            self.height = self.tello.get_height()
            self.temperature = self.tello.get_temperature()
            print(f"🔋 Battery: {self.battery_level}%")
            
            # Start video stream
            self.tello.streamoff()
            time.sleep(0.5)
            self.tello.streamon()
            
            # Update shared data
            shared_data.update_status({
                'connected': True,
                'battery': self.battery_level,
                'speed': self.speed,
                'temperature': self.temperature,
                'height': self.height
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
            ("Autonomous Behavior", self._autonomous_behavior_thread),
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
                        
                        if self.recording and self.video_writer:
                            frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                            self.video_writer.write(frame_bgr)
                        
                        # ✅ PERBAIKAN: Bulatkan FPS menjadi integer
                        fps_int = int(round(self.fps))
                        
                        # Update status dengan FPS yang sudah dibulatkan
                        shared_data.update_status({
                            'fps': fps_int,
                            'humans_detected': int(count),
                            'Height': 0,
                            'temperature': 0,
                            'wifiSignal': 100,
                            'humanDetection': 'ON' if detected else 'OFF',
                            'ml_detection_enabled': self.detection_enabled
                        })
                    
                    time.sleep(0.01)
                    
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
                            self.height = self.tello.get_height()
                            self.temperature = self.tello.get_temperature()
                            shared_data.update_status({'battery': self.battery_level, 
                                                       'height': self.height,
                                                       'temperature': self.temperature})
                        last_battery_check = current_time
                    except Exception as e:
                        print(f"❌ Battery check error: {e}")
                
                time.sleep(1/30)  # 30 FPS control loop
                
            except Exception as e:
                print(f"❌ Drone control error: {e}")
                time.sleep(0.1)
        
        print("🎮 Drone control thread ended")

    def _detect_red_in_roi(self, frame):
        """Detect red color in ROI for autonomous behavior - Fixed with type conversion"""
        try:
            if frame is None:
                return False, 0
            
            # Define ROI parameters
            roi_x = Config.WINDOW_WIDTH // 4
            roi_y = 5
            roi_width = Config.WINDOW_WIDTH // 2
            roi_height = Config.WINDOW_HEIGHT // 3
            
            # Extract ROI
            roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            
            # Handle color format
            if len(frame.shape) == 3:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            else:
                roi_bgr = roi
            
            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            # Red color ranges
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([179, 255, 255])
            
            # Create masks
            mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            mask_roi = cv2.bitwise_or(mask1, mask2)
            
            # Count red pixels
            pixel_count = np.sum(mask_roi > 0)
            
            # PERBAIKAN: Convert NumPy types to Python native types
            pixel_count = int(pixel_count)  # numpy.int64 -> int
            red_detected = pixel_count > 5000
            
            # if pixel_count > 0:
            #     print(f"🔍 Red detection: pixels={pixel_count}, detected={red_detected}")
            
            return red_detected, pixel_count
            
        except Exception as e:
            print(f"❌ Red detection error: {e}")
            return False, 0  # Python native types
        
    def _detection_thread(self):
        """Handle AI detection - Fixed version"""
        print("🤖 Detection thread started")
        
        while self.running:
            try:
                if self.current_processed_frame is not None:
                    # Get current frame safely
                    with self.data_lock:
                        if self.current_processed_frame is not None:
                            frame_copy = self.current_processed_frame.copy()
                        else:
                            frame_copy = None
                    
                    if frame_copy is not None:
                        # PERBAIKAN: Handle format warna dengan benar
                        # Jika frame dari YOLO adalah BGR, tidak perlu convert
                        # Jika frame dari YOLO adalah RGB, handle di _detect_red_in_roi
                        
                        # Perform red color detection for autonomous behavior
                        red_detected, pixel_count = self._detect_red_in_roi(frame_copy)
                        
                        # Update detection results safely
                        with self.data_lock:
                            self.current_detection = {
                                'red_detected': red_detected,
                                'pixel_count': pixel_count,
                                'timestamp': time.time()  # Tambahkan timestamp
                            }
                        
                        # Update shared data untuk web interface
                        shared_data.update_status({
                            'red_detected': red_detected,
                            'pixel_count': pixel_count
                        })
                else:
                    # Clear detection when disabled
                    with self.data_lock:
                        self.current_detection = {
                            'red_detected': False,
                            'pixel_count': 0,
                            'timestamp': time.time()
                        }
                
                time.sleep(0.03)  # Control detection frequency
                
            except Exception as e:
                print(f"❌ Detection thread error: {e}")
                time.sleep(0.1)
        
        print("🤖 Detection thread ended")

    def _autonomous_behavior_thread(self):
        """Handle autonomous behavior - Fixed version with emergency check"""
        print("🤖 Autonomous behavior thread started")
        
        last_action_time = 0
        action_cooldown = 3.0
        
        while self.running:
            try:
                # PERBAIKAN: Cek emergency flag PERTAMA
                if self.emergency_stop:
                    print("🚨 Emergency stop detected, exiting autonomous behavior")
                    self.set_autonomous_behavior = False
                    shared_data.update_status({
                        'autonomous_mode': False,
                        'autonomous_action': 'emergency_stopped'
                    })
                    break  # Exit loop immediately
                
                # Cek apakah autonomous mode masih aktif
                # if not self.set_autonomous_behavior:
                #     print("🤖 Autonomous behavior disabled, thread pausing...")
                #     time.sleep(0.5)
                #     continue
                
                # Only run autonomous behavior if conditions are met
                if (self.set_autonomous_behavior and 
                    self.tello and
                    not self.emergency_stop):  # TAMBAHAN: Cek emergency
                    
                    current_time = time.time()
                    
                    # Get current detection results safely
                    with self.data_lock:
                        if self.current_detection:
                            red_detected = self.current_detection.get('red_detected', False)
                            pixel_count = self.current_detection.get('pixel_count', 0)
                            detection_time = self.current_detection.get('timestamp', 0)
                        else:
                            red_detected = False
                            pixel_count = 0
                            detection_time = 0
                    
                    # PERBAIKAN: Cek emergency lagi sebelum movement
                    if self.emergency_stop:
                        print("🚨 Emergency detected during movement preparation")
                        break
                    
                    # Hanya ambil action jika ada deteksi baru dan cooldown selesai
                    if (current_time - last_action_time > action_cooldown and
                        current_time - detection_time < 1.0):
                        
                        if red_detected and not self.emergency_stop:  # Cek emergency
                            print(f"🔴 Red detected in ROI! Pixel count: {pixel_count}")
                            print("🎯 Moving towards target...")
                            
                            try:
                                # PERBAIKAN: Cek emergency sebelum setiap movement
                                if self.emergency_stop:
                                    print("🚨 Emergency detected, stopping target movement")
                                    break
                                    
                                self.tello.move_back(20)
                                time.sleep(2)
                                
                                if self.emergency_stop:
                                    print("🚨 Emergency detected, stopping rotation")
                                    break
                                    
                                self.tello.rotate_clockwise(90)
                                time.sleep(2)
                                
                                last_action_time = current_time
                                
                                # Update status
                                shared_data.update_status({
                                    'autonomous_action': 'target_detected',
                                    'red_detected': True,
                                    'pixel_count': pixel_count
                                })

                            except Exception as move_error:
                                print(f"❌ Autonomous movement error: {move_error}")
                        
                        elif (current_time - last_action_time > action_cooldown * 2 and 
                            not self.emergency_stop):  # Cek emergency
                            # Search behavior
                            print("⚪ No red in ROI. Searching...")
                            
                            try:
                                if self.emergency_stop:
                                    print("🚨 Emergency detected, stopping search")
                                    break
                                    
                                self.tello.move_forward(35)
                                time.sleep(0.5)
                                
                                last_action_time = current_time
                                
                                # Update status
                                shared_data.update_status({
                                    'autonomous_action': 'searching',
                                    'red_detected': False,
                                    'pixel_count': 0
                                })
                                
                            except Exception as search_error:
                                print(f"❌ Autonomous search error: {search_error}")
                
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Autonomous behavior error: {e}")
                time.sleep(0.5)
        
        print("🤖 Autonomous behavior thread ended")

    def _web_integration_thread(self):
        """Handle web integration updates"""
        print("🌐 Web integration thread started")
        
        while self.running:
            try:
                # Update telemetry data if available
                if self.tello:
                    try:
                        # Get telemetry data (handle both string and dict)
                        state_data = self.tello.get_current_state()
                        telemetry = self._parse_telemetry(state_data)  # Pass raw data
                        
                        shared_data.update_status({
                            'telemetry': telemetry,
                            'flying': self.send_rc_control
                        })
                    except Exception as e:
                        print(f"⚠️ Telemetry update error: {e}")
                        pass
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"❌ Web integration error: {e}")
                time.sleep(1)
        
        print("🌐 Web integration thread ended")


    def _parse_telemetry(self, state_data):
        """Parse telemetry data with proper type conversion - Fixed version"""
        telemetry = {
            'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
            'speed_x': 0.0, 'speed_y': 0.0, 'speed_z': 0.0,
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
            'barometer': 0.0, 'tof': 0.0
        }
        
        try:
            if not state_data:
                return telemetry
                
            # Handle different return types from get_current_state()
            if isinstance(state_data, dict):
                # Handle dictionary format
                for key, value in state_data.items():
                    try:
                        value = float(value)  # Convert to Python float
                        
                        if key in ['pitch', 'roll', 'yaw']:
                            telemetry[key] = value
                        elif key in ['vgx', 'vgy', 'vgz']:
                            telemetry[f'speed_{key[-1]}'] = value
                        elif key in ['agx', 'agy', 'agz']:
                            telemetry[f'accel_{key[-1]}'] = value
                        elif key == 'baro':
                            telemetry['barometer'] = value
                        elif key == 'tof':
                            telemetry['tof'] = value
                    except (ValueError, TypeError):
                        pass  # Skip invalid values
                        
            elif isinstance(state_data, str):
                # Handle string format
                for item in state_data.split(';'):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        try:
                            value = float(value)  # Convert to Python float
                            
                            if key in ['pitch', 'roll', 'yaw']:
                                telemetry[key] = value
                            elif key in ['vgx', 'vgy', 'vgz']:
                                telemetry[f'speed_{key[-1]}'] = value
                            elif key in ['agx', 'agy', 'agz']:
                                telemetry[f'accel_{key[-1]}'] = value
                            elif key == 'baro':
                                telemetry['barometer'] = value
                            elif key == 'tof':
                                telemetry['tof'] = value
                        except (ValueError, TypeError):
                            pass  # Skip invalid values
            else:
                print(f"⚠️ Unexpected telemetry data type: {type(state_data)}")
                
        except Exception as e:
            print(f"❌ Telemetry parsing error: {e}")
        
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
                # PERBAIKAN: Enhanced emergency handler
                print("🚨 Emergency command received")
                
                # Set emergency flag if autonomous is running
                if self.set_autonomous_behavior:
                    self.emergency_stop = True
                    self.set_autonomous_behavior = False
                    print("🚨 Emergency stop set for autonomous mode")
                
                if self.tello:
                    self.tello.emergency()
                    self.send_rc_control = False
                    print("🚨 Emergency executed")
            elif cmd_type == 'emergency_auto':
                # PERBAIKAN: Handler khusus untuk emergency di autonomous mode
                print("🚨 Emergency AUTO command received - stopping autonomous mode")
                self.set_emergency_stop = True
                # 1. STOP autonomous behavior immediately
                self.set_autonomous_behavior = False
                shared_data.update_status({'autonomous_mode': False})
                
                # 2. STOP all movement
                self.left_right_velocity = 0
                self.for_back_velocity = 0
                self.up_down_velocity = 0
                self.yaw_velocity = 0
                
                # 4. Send immediate stop command
                if self.tello and self.send_rc_control:
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                        print("✅ RC control stopped")
                    except Exception as e:
                        print(f"❌ RC stop error: {e}")
                
                # 5. Emergency landing
                if self.tello:
                    try:
                        self.tello.emergency()
                        print("✅ Emergency executed - drone stopped")
                    except Exception as e:
                        print(f"❌ Emergency execution error: {e}")                
                self.send_rc_control = False
                shared_data.update_status({
                    'flying': False,
                    'autonomous_mode': False,
                    'autonomous_action': 'emergency_stopped'
                })
                print("🚨 Emergency AUTO executed")
            
            elif cmd_type == 'enable_change_keyboard':
                enabled = cmd_data.get('enabled', False)
                shared_data.update_status({'keyboard_enabled': enabled})
                print(f"🎮 Keyboard mode updated: {'Mode 2 (Arrow Movement)' if enabled else 'Mode 1 (WASD Movement)'}")

            elif cmd_type == 'move_control':
                controls = cmd_data
                
                # Get current keyboard mode from shared data
                current_status = shared_data.get_status()
                keyboard_enabled = current_status.get('keyboard_enabled', False)
                
                # Process movement commands based on keyboard mode
                if keyboard_enabled:
                    # Mode 2: Frontend sends based on Arrow keys setup
                    # Frontend sudah mengirim nilai yang benar untuk Mode 2
                    self.left_right_velocity = controls.get('left_right', 0)
                    self.for_back_velocity = controls.get('for_back', 0)
                    self.up_down_velocity = controls.get('up_down', 0)
                    self.yaw_velocity = controls.get('yaw', 0)
                else:
                    # Mode 1: Frontend sends based on WASD setup  
                    # Frontend sudah mengirim nilai yang benar untuk Mode 1
                    self.left_right_velocity = controls.get('left_right', 0)
                    self.for_back_velocity = controls.get('for_back', 0)
                    self.up_down_velocity = controls.get('up_down', 0)
                    self.yaw_velocity = controls.get('yaw', 0)
                
                # Debug log untuk memastikan values diterima
                if any([self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity]):
                    mode_text = "Mode 2" if keyboard_enabled else "Mode 1"
                    print(f"🎮 Movement {mode_text}: LR={self.left_right_velocity}, FB={self.for_back_velocity}, UD={self.up_down_velocity}, YAW={self.yaw_velocity}")
                        
            elif cmd_type == 'stop_movement':
                self.left_right_velocity = 0
                self.for_back_velocity = 0
                self.up_down_velocity = 0
                self.yaw_velocity = 0
                print("movement stop")
            
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
 
            elif cmd_type == 'start_autonomous_mode':
                # PERBAIKAN: Reset emergency flag
                self.emergency_stop = False
                print("🛫 Auto-takeoff for autonomous mode")
                self.tello.takeoff()
                self.send_rc_control = False
                self.set_autonomous_behavior = True
                self.detection_enabled = True
                shared_data.update_status({'autonomous_mode': True, 'ml_detection_enabled': True })
                shared_data.update_status({'flying': True})            
                print("🤖 Autonomous mode: STARTED")
            elif cmd_type == 'stop_autonomous_mode':
                # PERBAIKAN: Enhanced stop autonomous
                print("🤖 Stopping autonomous mode...")
                
                # 1. Set flags to stop autonomous behavior
                self.set_autonomous_behavior = False
                self.emergency_stop = False  # Reset emergency flag
                self.send_rc_control = True
                self.tello.emergency()
                self.send_rc_control = False
                # 4. Update status
                shared_data.update_status({
                    'autonomous_mode': False,
                    'autonomous_action': 'stopped'
                })
                
                print("🤖 Autonomous mode: STOPPED")
                        
            elif cmd_type == 'land_autonomous_mode':
                # PERBAIKAN: Enhanced stop autonomous
                print("🤖 Stopping autonomous mode...")
                
                # 1. Set flags to stop autonomous behavior
                self.set_autonomous_behavior = False
                self.emergency_stop = False  # Reset emergency flag
                self.send_rc_control = True
                self.tello.land()
                self.send_rc_control = False
                # 4. Update status
                shared_data.update_status({
                    'autonomous_mode': False,
                    'autonomous_action': 'stopped'
                })
                
                print("🤖 Autonomous mode: STOPPED")
            
            
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
                
                fourcc = cv2.VideoWriter_fourcc(*'h264')
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
