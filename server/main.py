import sys
import signal
from config import Config
from shared_data import shared_data
from drone_system import DroneSystem, DRONE_IMPORTS_AVAILABLE
from web_server import WebServer, WEB_IMPORTS_AVAILABLE
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

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
                'speed': 20,
                "height": 0,
                "temperature": 0,
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
        if not WEB_IMPORTS_AVAILABLE or not DRONE_IMPORTS_AVAILABLE:
            print("❌ Missing dependencies for integrated mode")
            return False
        
        try:
            self.drone_system = DroneSystem()
            self.drone_system.web_integration_enabled = True
            
            if not self.drone_system.initialize_all_systems():
                print("❌ Failed to initialize drone systems")
                return False
            
            # Start drone threads
            self.drone_system.start_drone_threads()
            self.web_server = WebServer()
            
            # Run web server (this will block)
            self.web_server.run()
            
        except Exception as e:
            print(f"❌ Failed to start integrated mode: {e}")
            return False

def main():
    """Main entry point"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h']:
            return
        elif arg == '--backend-only':
            mode = 'backend-only'
        elif arg == '--drone-only':
            mode = 'drone-only'
        elif arg == '--integrated':
            mode = 'integrated'
        else:
            print(f"❌ Unknown mode: {arg}")
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


if __name__ == '__main__':
    main()
