try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    WEB_IMPORTS_AVAILABLE = True
except ImportError:
    WEB_IMPORTS_AVAILABLE = False
import os
import threading
import time
from datetime import datetime
from config import Config
from sharedData import shared_data

class WebServer:
    """Flask + Socket.IO web server for React frontend"""
    def __init__(self):
        if not WEB_IMPORTS_AVAILABLE:
            raise ImportError("Web server dependencies not available")
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'drone_web_bridge_secret'
        
        # Enable CORS
        CORS(self.app, 
            origins="*",
            methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allow_headers=['Content-Type', 'Authorization', 'Access-Control-Allow-Credentials'],
            supports_credentials=True)

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
                    'socket': '/socket.io'
                }
            })

        @self.app.route('/api/media/list')
        def list_media_files():
            """List media files with pagination and filtering"""
            try:
                file_type = request.args.get('type', 'images').lower()
                
                if file_type == 'images':
                    directory = Config.SCREENSHOTS_DIR
                    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                elif file_type == 'videos':
                    directory = Config.RECORDINGS_DIR  
                    extensions = ['.mp4', '.avi', '.mov', '.mkv']
                else:
                    return jsonify({'success': False, 'error': 'Invalid file type'})
                
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
                                'url': f"{Config.MEDIA_BASE_URL}/{filename}",
                                'file_path': filepath,
                                'type': file_type[:-1]  # Remove 's' from images/videos
                            })
                
                # Sort by creation time (newest first)
                files.sort(key=lambda x: x['created_at'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/media/<filename>')
        def serve_media_file(filename):
            """Serve media files from screenshots or recordings directory"""
            try:
                # Try screenshots directory first
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    return send_file(screenshots_path, as_attachment=False)
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename) 
                if os.path.exists(recordings_path):
                    return send_file(recordings_path, as_attachment=False)
                
                return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/download/<filename>')
        def download_media_file(filename):
            """Download media files"""
            try:
                # Try screenshots directory first
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    return send_file(screenshots_path, as_attachment=True, download_name=filename)
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                if os.path.exists(recordings_path):
                    return send_file(recordings_path, as_attachment=True, download_name=filename)
                
                return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/media/delete/<filename>', methods=['DELETE'])
        def delete_media_file(filename):
            """Delete media file"""
            try:
                # Try screenshots directory first  
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    os.remove(screenshots_path)
                    return jsonify({'success': True, 'message': f'File {filename} deleted'})
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                if os.path.exists(recordings_path):
                    os.remove(recordings_path)
                    return jsonify({'success': True, 'message': f'File {filename} deleted'})
                
                return jsonify({'success': False, 'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'success': True,
                'data': shared_data.get_status()
            })

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

        @self.socketio.on('get_media_files')
        def handle_get_media_files(data):
            """Handle request for media files list"""
            try:
                file_type = data.get('type', 'images').lower()
                
                if file_type == 'images':
                    directory = Config.SCREENSHOTS_DIR
                    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                elif file_type == 'videos':
                    directory = Config.RECORDINGS_DIR
                    extensions = ['.mp4', '.avi', '.mov', '.mkv']
                else:
                    emit('media_files_response', {'success': False, 'error': 'Invalid file type'})
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
                                'url': f"{Config.MEDIA_BASE_URL}/{filename}",
                                'file_path': filepath,
                                'type': file_type[:-1]
                            })
                
                # Sort by creation time (newest first)
                files.sort(key=lambda x: x['created_at'], reverse=True)
                
                emit('media_files_response', {
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                })
                
            except Exception as e:
                emit('media_files_response', {'success': False, 'error': str(e)})

        @self.socketio.on('delete_media')
        def handle_delete_media(data):
            """Handle media file deletion"""
            try:
                filename = data.get('filename')
                if not filename:
                    emit('media_deleted', {'success': False, 'error': 'Filename required'})
                    return
                
                # Try screenshots directory first
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    os.remove(screenshots_path)
                    emit('media_deleted', {'success': True, 'filename': filename})
                    return
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                if os.path.exists(recordings_path):
                    os.remove(recordings_path)
                    emit('media_deleted', {'success': True, 'filename': filename})
                    return
                
                emit('media_deleted', {'success': False, 'error': 'File not found'})
                
            except Exception as e:
                emit('media_deleted', {'success': False, 'error': str(e)})

        @self.socketio.on('download_media')
        def handle_download_media(data):
            """Handle media file download preparation"""
            try:
                filename = data.get('filename')
                if not filename:
                    emit('download_ready', {'success': False, 'error': 'Filename required'})
                    return
                
                # Check if file exists and prepare download URL
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                
                if os.path.exists(screenshots_path) or os.path.exists(recordings_path):
                    download_url = f"http://{Config.WEB_HOST}:{Config.WEB_PORT}/download/{filename}"
                    emit('download_ready', {
                        'success': True,
                        'url': download_url,
                        'filename': filename
                    })
                else:
                    emit('download_ready', {'success': False, 'error': 'File not found'})
                    
            except Exception as e:
                emit('download_ready', {'success': False, 'error': str(e)})

        @self.socketio.on('debug_media_system')
        def handle_debug_media():
            """Handle debug request for media system"""
            try:
                debug_info = {
                    'directories': {
                        'screenshots': Config.SCREENSHOTS_DIR,
                        'recordings': Config.RECORDINGS_DIR,
                        'screenshots_exists': os.path.exists(Config.SCREENSHOTS_DIR),
                        'recordings_exists': os.path.exists(Config.RECORDINGS_DIR)
                    },
                    'file_counts': {
                        'screenshots': len([f for f in os.listdir(Config.SCREENSHOTS_DIR) 
                                        if os.path.isfile(os.path.join(Config.SCREENSHOTS_DIR, f))]) 
                                    if os.path.exists(Config.SCREENSHOTS_DIR) else 0,
                        'recordings': len([f for f in os.listdir(Config.RECORDINGS_DIR) 
                                        if os.path.isfile(os.path.join(Config.RECORDINGS_DIR, f))]) 
                                    if os.path.exists(Config.RECORDINGS_DIR) else 0
                    },
                    'permissions': {
                        'screenshots_readable': os.access(Config.SCREENSHOTS_DIR, os.R_OK) if os.path.exists(Config.SCREENSHOTS_DIR) else False,
                        'recordings_readable': os.access(Config.RECORDINGS_DIR, os.R_OK) if os.path.exists(Config.RECORDINGS_DIR) else False
                    }
                }
                
                emit('debug_media_response', {'debug_info': debug_info})
                
            except Exception as e:
                emit('debug_media_response', {'debug_info': {'error': str(e)}})

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

        @self.socketio.on('enable_change_keyboard')
        def handle_enable_change_keyboard(data):
            """Handle keyboard mode change"""
            enabled = data.get('enabled', False)
            
            # Update shared data status
            shared_data.update_status({'keyboard_enabled': enabled})
            
            # Add command to drone system
            shared_data.add_command({
                'type': 'enable_change_keyboard',
                'data': {'enabled': enabled}
            })
            
            # Send confirmation back to frontend
            emit('keyboard_mode_updated', {
                'enabled': enabled,
                'mode': 'Mode 2: Arrow Movement' if enabled else 'Mode 1: WASD Movement'
            })
            
            print(f"🎮 Keyboard mode changed to: {'Mode 2 (Arrow Movement)' if enabled else 'Mode 1 (WASD Movement)'}")
                
        @self.socketio.on('emergency_land')
        def handle_emergency():
            """Handle emergency command"""
            shared_data.add_command({'type': 'emergency'})
            emit('drone_action', {'action': 'emergency', 'success': True})
        @self.socketio.on('emergency_auto')
        def handle_emergencyAuto():
            """Handle emergency command"""
            shared_data.add_command({'type': 'emergency'})
            emit('tello_status', {'connected': False, 'flying': False})
        
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
            speed = data.get('speed', 20)
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
            shared_data.add_command({'type': 'start_autonomous_mode'})
            emit('drone_action', {'action': 'autonomous_start', 'success': True})
            emit('autonomous_status', {'enabled': True})
            print("🤖 Autonomous mode start command sent")
        
        @self.socketio.on('stop_autonomous_mode')
        def handle_stop_autonomous():
            """Handle autonomous mode stop"""
            shared_data.add_command({'type': 'stop_autonomous_mode'})
            emit('drone_action', {'action': 'autonomous_stop', 'success': True})
            emit('autonomous_status', {'enabled': False})
            print("🤖 Autonomous mode stop command sent")
        @self.socketio.on('land_autonomous_mode')
        def handle_land_autonomous():
            """Handle autonomous mode land"""
            shared_data.add_command({'type': 'land_autonomous_mode'})
            emit('drone_action', {'action': 'autonomous_stop', 'success': True})
            emit('autonomous_status', {'enabled': False})
            print("🤖 Autonomous mode stop command sent")



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
                    
                    # Send autonomous status updates
                    if status.get('autonomous_mode', False):
                        self.socketio.emit('autonomous_update', {
                            'enabled': status.get('autonomous_mode', False),
                            'action': status.get('autonomous_action', 'idle'),
                            'red_detected': status.get('red_detected', False),
                            'pixel_count': status.get('pixel_count', 0)
                        })
                    
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
