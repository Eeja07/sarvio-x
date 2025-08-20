import os

class Config:
    # Web server config
    WEB_HOST = '127.0.0.1'
    WEB_PORT = 5000
    WEB_DEBUG = False

    # Drone config
    FPS = 120
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    SPEED = ''
    THREAD_COUNT = 5

    # Integration config
    FRAME_SHARE_FILE = "shared_frame.jpg"
    STATUS_SHARE_FILE = "shared_status.json"
    COMMAND_SHARE_FILE = "shared_commands.json"

    # Media directories
    SCREENSHOTS_DIR = "screenshots"
    RECORDINGS_DIR = "recordings"
    MEDIA_BASE_URL = "/media"