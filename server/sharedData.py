import threading
import queue
import os
import numpy as np
import cv2
import base64
from config import Config

class SharedDataManager:
    """Manages communication between drone system and web server"""
    def __init__(self):
        self.status_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.status_data = {
            'connected': False,
            'flying': False,
            'battery': 0,
            'speed': '',
            'temperature': 0,
            'height': 0,
            'humans_detected': 0,
            'fps': 0,
            'recording': False,
            'ml_detection_enabled': False,
            'auto_capture_enabled': False,
            'screenshot_count': 0,
            'flight_time': 0,
            'keyboard_enabled': False,
            'autonomous_mode': False,
            'autonomous_action': 'idle',
            'red_detected': False,
            'pixel_count': 0,
            'telemetry': {
                'pitch': 0, 'roll': 0, 'yaw': 0,
                'speed_x': 0, 'speed_y': 0, 'speed_z': 0,
                'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
                'barometer': 0, 'tof': 0
            }
        }
        self.command_queue = queue.Queue()
        self.current_frame = None
        self.current_frame_base64 = None
        self._create_directories()
    def _create_directories(self):
        directories = [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    def convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    def update_status(self, status_update):
        with self.status_lock:
            converted_update = self.convert_numpy_types(status_update)
            self.status_data.update(converted_update)
    def get_status(self):
        with self.status_lock:
            return self.convert_numpy_types(self.status_data.copy())
    def add_command(self, command):
        self.command_queue.put(command)
    def get_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    def update_frame(self, frame):
        with self.frame_lock:
            if frame is not None:
                self.current_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                self.current_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    def get_frame_base64(self):
        with self.frame_lock:
            return self.current_frame_base64
shared_data = SharedDataManager()
