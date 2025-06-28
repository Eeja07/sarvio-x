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
from flask import Flask, Response, render_template_string, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import mimetypes
from pathlib import Path

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
        self.speed = 20  # Default speed
        self.current_speed_display = 20  # For real-time display

        self.send_rc_control = False
        self.is_connected = False
        self.is_flying = False
        self.disconnect_requested = False  # Flag untuk handle disconnect

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

        # Recording state
        self.is_recording = False

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
            async_mode='threading',
            logger=False,  # Disable logging untuk mengurangi noise
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            transports=['polling', 'websocket']
        )
        
        self.last_frame = None
        self.frame_lock = threading.Lock()

        # Setup Flask routes and Socket events
        self._setup_flask_routes()
        self._setup_socket_events()
        
        # Setup enhanced media handling
        self._setup_additional_flask_routes()
        self._enhance_socket_events()

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def get_file_thumbnail(self, filepath):
        """Generate thumbnail URL for media files"""
        try:
            filename = os.path.basename(filepath)
            file_ext = filename.lower().split('.')[-1]
            
            # Untuk gambar, gunakan file asli sebagai thumbnail
            if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                return f'http://localhost:5000/media/{filename}'
            
            # Untuk video, bisa return placeholder atau generate thumbnail
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
            
            if not os.path.exists(self.screenshot_dir):
                os.makedirs(self.screenshot_dir)
                return files
            
            # Supported file extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'}
            
            for filename in os.listdir(self.screenshot_dir):
                filepath = os.path.join(self.screenshot_dir, filename)
                
                if not os.path.isfile(filepath):
                    continue
                    
                file_ext = Path(filename).suffix.lower()
                
                # Filter berdasarkan type
                is_image = file_ext in image_extensions
                is_video = file_ext in video_extensions
                
                if file_type == 'images' and not is_image:
                    continue
                elif file_type == 'videos' and not is_video:
                    continue
                elif file_type == 'all' and not (is_image or is_video):
                    continue
                
                try:
                    # Get file statistics
                    stat = os.stat(filepath)
                    
                    # Parse humans detected from filename
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
                    
                    # Determine file type
                    media_type = 'image' if is_image else 'video'
                    
                    # Get MIME type
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
            
            # Sort by modified time (newest first)
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

    def _setup_flask_routes(self):
        """Setup Flask routes for web interface"""
        @self.app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
                <body>
                    <h1>🚁 SARVIO-X Backend Server</h1>
                    <h3>Live Video Stream:</h3>
                    <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Tello Live Stream">
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

        @self.app.route('/media/<filename>')
        def serve_media(filename):
            """Serve media files from screenshots directory with proper headers"""
            try:
                response = send_from_directory(
                    self.screenshot_dir, 
                    filename,
                    as_attachment=False
                )
                # Add CORS headers for media files
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
            except Exception as e:
                print(f"Error serving media file {filename}: {e}")
                return "File not found", 404

        @self.app.route('/download/<filename>')
        def download_media(filename):
            """Download media files"""
            try:
                response = send_from_directory(
                    self.screenshot_dir, 
                    filename, 
                    as_attachment=True,
                    download_name=filename
                )
                # Add CORS headers
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
            except Exception as e:
                print(f"Error downloading file {filename}: {e}")
                return "File not found", 404

    def _setup_additional_flask_routes(self):
        """Setup additional Flask routes for enhanced media handling"""
        
        @self.app.route('/api/media/stats')
        def get_media_statistics():
            """Get media directory statistics"""
            try:
                stats = self.get_media_stats()
                return {
                    'success': True,
                    'stats': stats
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }, 500
        
        @self.app.route('/api/media/list')
        def list_media_files():
            """List all media files with detailed information"""
            try:
                file_type = request.args.get('type', 'all')  # all, images, videos
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
    
    def _setup_socket_events(self):
        """Setup Socket.IO event handlers for React frontend"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            self.connected_clients += 1
            print(f'✅ React client connected. Total clients: {self.connected_clients}')
            
            # Send Tello status when client connects
            self.broadcast_status()
            
            # Send current settings
            emit('ml_detection_status', {'enabled': self.ml_detection_enabled})
            emit('auto_capture_status', {'enabled': self.auto_capture_enabled})
            emit('speed_update', {'speed': self.current_speed_display})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients -= 1
            print(f'❌ React client disconnected. Total clients: {self.connected_clients}')
            
            if self.connected_clients <= 0:
                self.socket_streaming = False

        @self.socketio.on('connect_tello')
        def handle_connect_tello():
            """Connect to Tello drone"""
            print("🔗 Connect Tello command from React client")
            self.disconnect_requested = False  # Reset flag
            success = self.connect_tello()
            
            emit('tello_connection_result', {
                'success': success,
                'message': 'Tello connected successfully' if success else 'Failed to connect to Tello'
            })
            
            self.broadcast_status()

        @self.socketio.on('disconnect_tello')
        def handle_disconnect_tello():
            """Disconnect from Tello drone"""
            print("🔌 Disconnect Tello command from React client")
            self.disconnect_requested = True  # Set flag
            success = self.disconnect_tello()
            
            emit('tello_connection_result', {
                'success': success,
                'message': 'Tello disconnected successfully' if success else 'Failed to disconnect from Tello'
            })
            
            self.broadcast_status()
            emit('clear_video_frame')
        
        @self.socketio.on('start_stream')
        def handle_start_stream():
            print("🎥 React client requested video stream")
            self.socket_streaming = True
            emit('stream_status', {'streaming': True, 'message': 'Video stream started'})
        
        @self.socketio.on('stop_stream')
        def handle_stop_stream():
            print("⏹️ React client stopped video stream")
            self.socket_streaming = False
            emit('stream_status', {'streaming': False, 'message': 'Video stream stopped'})
        
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
            if self.send_rc_control and not self.disconnect_requested:
                self.left_right_velocity = int(data.get('left_right', 0))
                self.for_back_velocity = int(data.get('for_back', 0))
                self.up_down_velocity = int(data.get('up_down', 0))
                self.yaw_velocity = int(data.get('yaw', 0))
                
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
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.up_down_velocity = 0
            self.yaw_velocity = 0
            
            if self.is_connected and not self.disconnect_requested:
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except Exception as e:
                    print(f"❌ Stop movement error: {e}")

        @self.socketio.on('set_speed')
        def handle_set_speed(data):
            """Set drone speed"""
            new_speed = data.get('speed', 20)
            if 10 <= new_speed <= 100:
                self.speed = new_speed
                self.current_speed_display = new_speed
                
                if self.is_connected:
                    try:
                        self.tello.set_speed(new_speed)
                        print(f"⚡ Speed set to: {new_speed} cm/s")
                    except Exception as e:
                        print(f"❌ Error setting speed: {e}")
                
                # Broadcast speed update to all clients
                self.socketio.emit('speed_update', {'speed': new_speed})
            else:
                emit('speed_update', {'speed': self.current_speed_display, 'error': 'Invalid speed range'})

        @self.socketio.on('flip_command')
        def handle_flip_command(data):
            """Handle flip commands"""
            if self.is_connected and self.is_flying:
                direction = data.get('direction', 'f')
                try:
                    if direction == 'f':
                        self.tello.flip_forward()
                    elif direction == 'b':
                        self.tello.flip_back()
                    elif direction == 'l':
                        self.tello.flip_left()
                    elif direction == 'r':
                        self.tello.flip_right()
                    print(f"🔄 Flip {direction} executed")
                except Exception as e:
                    print(f"❌ Flip error: {e}")

        @self.socketio.on('emergency_land')
        def handle_emergency():
            """Emergency stop and land"""
            print("🚨 Emergency command from React client")
            try:
                if self.is_connected:
                    self.tello.send_rc_control(0, 0, 0, 0)  # Stop movement
                    if self.is_flying:
                        self.tello.emergency()  # Emergency land
                        self.is_flying = False
                        self.send_rc_control = False
                        self.flight_start_time = None
                self.broadcast_status()
            except Exception as e:
                print(f"❌ Emergency error: {e}")

        @self.socketio.on('enable_ml_detection')
        def handle_enable_ml_detection(data):
            """Enable/disable ML detection"""
            self.ml_detection_enabled = data.get('enabled', False)
            print(f"🤖 ML Detection: {'Enabled' if self.ml_detection_enabled else 'Disabled'}")
            
            emit('ml_detection_status', {
                'enabled': self.ml_detection_enabled,
                'message': f"ML Detection {'enabled' if self.ml_detection_enabled else 'disabled'}"
            })

        @self.socketio.on('enable_auto_capture')
        def handle_enable_auto_capture(data):
            """Enable/disable auto capture"""
            self.auto_capture_enabled = data.get('enabled', False)
            print(f"📸 Auto Capture: {'Enabled' if self.auto_capture_enabled else 'Disabled'}")
            
            emit('auto_capture_status', {
                'enabled': self.auto_capture_enabled,
                'message': f"Auto Capture {'enabled' if self.auto_capture_enabled else 'disabled'}"
            })

        @self.socketio.on('toggle_recording')
        def handle_toggle_recording(data):
            """Toggle video recording"""
            self.is_recording = data.get('recording', False)
            print(f"🎥 Recording: {'Started' if self.is_recording else 'Stopped'}")
            
            emit('recording_status', {
                'recording': self.is_recording,
                'message': f"Recording {'started' if self.is_recording else 'stopped'}"
            })

        @self.socketio.on('get_media_files')
        def handle_get_media_files(data):
            """Get media files from screenshots directory - ALWAYS WORKS"""
            try:
                file_type = data.get('type', 'images')
                
                # Always scan directory regardless of drone connection
                if file_type == 'images':
                    files = self.scan_media_directory('images')
                elif file_type == 'videos':
                    files = self.scan_media_directory('videos')
                else:
                    files = self.scan_media_directory('all')
                
                print(f"📁 Scanned {len(files)} {file_type} files")
                
                emit('media_files_response', {
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                })
                
            except Exception as e:
                print(f"Error getting media files: {e}")
                emit('media_files_response', {
                    'success': False,
                    'error': str(e),
                    'files': []
                })

        @self.socketio.on('download_media')
        def handle_download_media(data):
            """Handle media file download"""
            try:
                filename = data.get('filename')
                if not filename:
                    return
                
                filepath = os.path.join(self.screenshot_dir, filename)
                if os.path.exists(filepath):
                    print(f"📥 Download requested: {filename}")
                    emit('download_ready', {
                        'success': True,
                        'filename': filename,
                        'url': f'http://localhost:5000/download/{filename}'
                    })
                else:
                    emit('download_ready', {
                        'success': False,
                        'error': 'File not found'
                    })
            except Exception as e:
                print(f"Download error: {e}")

        @self.socketio.on('delete_media')
        def handle_delete_media(data):
            """Handle media file deletion"""
            try:
                filename = data.get('filename')
                if not filename:
                    return
                
                filepath = os.path.join(self.screenshot_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"🗑️ Deleted file: {filename}")
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
                print(f"Delete error: {e}")
                emit('media_deleted', {
                    'success': False,
                    'error': str(e)
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
                
                emit('screenshot_result', {
                    'success': success,
                    'count': self.screenshot_count,
                    'humans_detected': humans_count
                })
            else:
                emit('screenshot_result', {
                    'success': False,
                    'message': 'No video frame available'
                })

    def _enhance_socket_events(self):
        """Enhanced socket events"""
        pass  # Additional enhancements can be added here

    def connect_tello(self):
        """Connect to Tello drone"""
        try:
            if not self.is_connected and not self.disconnect_requested:
                print("🔗 Connecting to Tello...")
                self.tello.connect()
                
                battery = self.get_battery()
                print(f"✅ Connected! Battery: {battery}%")
                
                self.tello.streamoff()
                self.tello.streamon()
                
                self.is_connected = True
                self.tello.set_speed(self.speed)
                
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
                print("🔌 Disconnecting from Tello...")
                
                if self.is_flying:
                    self.tello.land()
                    self.is_flying = False
                    self.send_rc_control = False
                    self.flight_start_time = None
                
                # Stop streaming
                self.tello.streamoff()
                self.tello.end()
                self.is_connected = False
                
                # Clear frame
                with self.frame_lock:
                    self.last_frame = None
                
                self.socketio.emit('clear_video_frame')
                print("🔌 Tello disconnected successfully")
            return True
        except Exception as e:
            print(f"❌ Error disconnecting Tello: {e}")
            return False

    def takeoff_drone(self):
        """Takeoff command"""
        if self.is_connected and not self.is_flying and not self.disconnect_requested:
            try:
                self.tello.takeoff()
                self.is_flying = True
                self.send_rc_control = True
                self.flight_start_time = time.time()
                print("✅ Takeoff successful")
                
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
                print("✅ Landing successful")
                
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
                'flight_time': self.get_flight_time(),
                'speed': self.current_speed_display
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
                frame_bgr = cv2.cvtColor(frame_to_send, cv2.COLOR_RGB2BGR)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                ret, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)
                
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                time.sleep(1 / FPS)

    def _send_frame_to_react(self, frame):
        """Send frame to React clients via Socket.IO"""
        if self.socket_streaming and self.connected_clients > 0:
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

    def save_screenshot(self, frame, humans_count, source="auto"):
        """Save screenshot with timestamp and human count"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_prefix = "manual" if source in ["joystick", "keyboard", "web"] else "auto"
            filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count}persons_{self.screenshot_count:04d}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.screenshot_count += 1
                print(f"Screenshot saved ({source}): {filename}")
                
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

        axis_lr = self.joystick.get_axis(0)
        axis_fb = self.joystick.get_axis(1)
        axis_yv = self.joystick.get_axis(2)
        axis_ud = self.joystick.get_axis(3)

        self.left_right_velocity = int(axis_lr * speed)
        self.for_back_velocity = int(-axis_fb * speed)
        self.up_down_velocity = int(-axis_ud * speed)
        self.yaw_velocity = int(axis_yv * rotate)

        if self.joystick.get_button(0):
            if not self.send_rc_control:
                self.takeoff_drone()
                time.sleep(0.5)

        if self.joystick.get_button(1):
            if self.send_rc_control:
                self.land_drone()
                time.sleep(0.5)

        current_screenshot_button_state = self.joystick.get_button(2)
        if current_screenshot_button_state and not self.last_joystick_screenshot_button_state:
            self.joystick_screenshot_requested = True
            print("Joystick screenshot button pressed!")
        
        self.last_joystick_screenshot_button_state = current_screenshot_button_state

        if self.joystick.get_button(3):
            self.joystick_screenshot_requested = True
            print("Alternative joystick screenshot button pressed!")
            time.sleep(0.2)

    def process_human_detection(self, frame):
        """Process human detection and return processed frame with detection info"""
        output_frame = frame.copy()
        human_detected = False
        human_boxes = []

        if self.yolo_model and self.ml_detection_enabled:
            try:
                results = self.yolo_model(frame, verbose=False)

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            if class_id == 0 and confidence > 0.5:
                                human_detected = True

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                human_boxes.append((x1, y1, x2, y2, confidence))

                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                cv2.circle(output_frame, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

                                confidence_percentage = confidence * 100
                                label = f"Human: {confidence_percentage:.0f}%"
                                cv2.putText(output_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if human_detected:
                    pose_results = self.pose.process(frame)
                    hands_results = self.hands.process(frame)

                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                output_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

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
            if not self.last_human_detected and not self.countdown_active:
                self.countdown_active = True
                self.countdown_start_time = current_time
                print(f"Human detected! Starting 3-second countdown...")
            
            if self.countdown_active:
                elapsed_time = current_time - self.countdown_start_time
                
                if elapsed_time >= self.countdown_duration:
                    self.save_screenshot(output_frame, humans_count, "auto")
                    self.last_screenshot_time = current_time
                    self.countdown_active = False
                    print("Countdown completed! Screenshot taken.")
        else:
            if self.countdown_active:
                self.countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        self.last_human_detected = human_detected

    def run_web_server(self):
        """Run Flask server with Socket.IO in separate thread"""
        print("Starting Flask server with Socket.IO...")
        print("React app will run on: http://localhost:5173")
        print("Backend Socket.IO server on: http://localhost:5000")
        print("Browser HTML stream: http://localhost:5000")
        
        self.socketio.run(
            self.app, 
            host='127.0.0.1',
            port=5000, 
            debug=False, 
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )

    def display_waiting_screen(self):
        """Display waiting for connection screen"""
        self.screen.fill([20, 20, 40])
        
        font_large = pygame.font.SysFont("Arial", 36)
        font_medium = pygame.font.SysFont("Arial", 24)
        
        title_text = font_large.render("SARVIO-X", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(480, 200))
        self.screen.blit(title_text, title_rect)
        
        waiting_text = font_medium.render("Waiting for Tello Connection...", True, (200, 200, 200))
        waiting_rect = waiting_text.get_rect(center=(480, 280))
        self.screen.blit(waiting_text, waiting_rect)
        
        instruction_text = font_medium.render("Open http://localhost:5173 and click 'Connect'", True, (150, 150, 150))
        instruction_rect = instruction_text.get_rect(center=(480, 320))
        self.screen.blit(instruction_text, instruction_rect)
        
        server_status = "✅ Backend Server Running" if self.connected_clients >= 0 else "❌ Backend Server Error"
        server_text = self.font.render(server_status, True, (0, 255, 0) if self.connected_clients >= 0 else (255, 0, 0))
        self.screen.blit(server_text, (10, 650))
        
        if self.connected_clients > 0:
            clients_text = self.font.render(f"📱 Web Clients Connected: {self.connected_clients}", True, (0, 255, 255))
            self.screen.blit(clients_text, (10, 680))

    def run(self):
        flask_thread = threading.Thread(target=self.run_web_server, daemon=True)
        flask_thread.start()
        
        print("🔌 Waiting for connection command from web interface...")
        print("📱 Open http://localhost:5173 and click 'Connect' button")
        
        self.socket_streaming = True

        frame_read = None
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

            if not self.is_connected:
                self.display_waiting_screen()
                pygame.display.update()
                time.sleep(0.1)
                continue
                
            if self.is_connected and frame_read is None:
                frame_read = self.tello.get_frame_read()
                print("📹 Video stream initialized")

            if frame_read and frame_read.stopped:
                break

            self.get_joystick_input()
            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            if frame is None:
                continue

            curr_time = time.time()
            self.fps = 1 / (curr_time - self.prev_time) if curr_time != self.prev_time else 0
            self.prev_time = curr_time

            frame = cv2.resize(frame, (960, 720))
            output_frame, human_detected, humans_count = self.process_human_detection(frame)
            self.handle_auto_screenshot(output_frame, human_detected, humans_count)

            if self.joystick_screenshot_requested:
                self.save_screenshot(output_frame, humans_count, "joystick")
                cv2.putText(output_frame, "JOYSTICK SCREENSHOT SAVED!", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                self.joystick_screenshot_requested = False

            battery = self.get_battery()
            cv2.putText(output_frame, f"Battery: {battery}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Speed: {self.current_speed_display} cm/s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if humans_count > 0:
                cv2.putText(output_frame, f"Humans Detected: {humans_count}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(output_frame, f"Screenshots: {self.screenshot_count}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            with self.frame_lock:
                self.last_frame = output_frame.copy()
            
            self._send_frame_to_react(output_frame)

            battery_counter += 1
            if battery_counter % 30 == 0 and self.connected_clients > 0:
                self.socketio.emit('battery_update', {'battery': battery})
                self.broadcast_status()

            frame_rgb = np.rot90(output_frame)
            frame_rgb = np.flipud(frame_rgb)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            self.screen.blit(frame_surface, (0, 0))

            status_text = self.font.render("T=Takeoff, L=Land, P=Screenshot, ESC=Quit", True, (255, 255, 255))
            self.screen.blit(status_text, (10, 10))
            
            react_text = self.font.render(f"React Clients: {self.connected_clients}", True, (255, 255, 0))
            self.screen.blit(react_text, (10, 40))

            pygame.display.update()
            time.sleep(1 / FPS)

        self.should_stop = True
        if self.is_connected:
            self.tello.streamoff()
            self.tello.end()
        self.pose.close()
        self.hands.close()
        print(f"Done! Total screenshots taken: {self.screenshot_count}")

    def keydown(self, key):
        """ Update velocities based on key pressed """
        if key == pygame.K_w:
            self.for_back_velocity = S
        elif key == pygame.K_s:
            self.for_back_velocity = -S
        elif key == pygame.K_a:
            self.left_right_velocity = -S
        elif key == pygame.K_d:
            self.left_right_velocity = S
        elif key == pygame.K_UP:
            self.up_down_velocity = S
        elif key == pygame.K_p:
            if self.last_frame is not None:
                with self.frame_lock:
                    frame_copy = self.last_frame.copy()
                output_frame, _, humans_count = self.process_human_detection(frame_copy)
                self.save_screenshot(output_frame, humans_count, "keyboard")
                print("Manual keyboard screenshot taken!")
        elif key == pygame.K_DOWN:
            self.up_down_velocity = -S
        elif key == pygame.K_LEFT:
            self.yaw_velocity = -S
        elif key == pygame.K_RIGHT:
            self.yaw_velocity = S
        elif key == pygame.K_q:
            pygame.quit()
            sys.exit()
            if self.is_connected:
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
        elif key == pygame.K_t:
            self.takeoff_drone()
        elif key == pygame.K_l:
            self.land_drone()

    def update(self):
        """ Update routine. Send velocities to Tello. """
        if self.send_rc_control and not self.disconnect_requested:
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
    print("- Enhanced Media Gallery with file management")
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
    print("- Enhanced Media Gallery with search & stats")
    print("=" * 60)
    print("API Endpoints:")
    print("- GET /api/media/list?type=images|videos|all")
    print("- GET /api/media/stats")
    print("- GET /media/<filename> - Serve media files")
    print("- GET /download/<filename> - Download media files")
    print("=" * 60)
    print("⚠️  IMPORTANT: Drone will NOT auto-connect!")
    print("   Use web interface to connect manually")
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