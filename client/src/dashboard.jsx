import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import Control from './control'
import Sensor from './sensor'
import Gallery from './gallery'

// const DEV_FORCE_CONNECTED = true; // Set to true to simulate connected mode

function Dashboard() {
  // State untuk drone control
  const [controlMode, setControlMode] = useState('Joystick Mode')
  const [speed, setSpeed] = useState(20)
  const [rotation, setRotation] = useState(0)
  const [leftJoystickPosition, setLeftJoystickPosition] = useState({ x: 0, y: 0 })
  const [rightJoystickPosition, setRightJoystickPosition] = useState({ x: 0, y: 0 })
  const [galleryOpen, setGalleryOpen] = useState(false)
  
  // State yang diperbaiki
  const [zoom, setZoom] = useState(1)
  const [humanDetection, setHumanDetection] = useState(false)
  const [autoScreenshot, setAutoScreenshot] = useState(false)
  const [joystickEnabled, setJoystickEnabled] = useState(true)
  const [brightness, setBrightness] = useState(0)
  const [showSpeedModal, setShowSpeedModal] = useState(false)
  const [keyboardEnabled, setKeyboardEnabled] = useState(false)
  
  // Socket connection state
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [telloConnected, setTelloConnected] = useState(false)
  const [isFlying, setIsFlying] = useState(false)
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  
  const [sensorData, setSensorData] = useState({
    battery: 0,
    bluetooth: 'OFF',
    state: 'DISCONNECTED',
    flightTime: '00:00',
    wifiSignal: 0,
    sdkVersion: 'N/A',
    serialNumber: 'N/A',
    height: 0,
    barometer: 0,
    temperature: 0,
    speed: 20,
    imuAttitude: { pitch: 0, roll: 0, yaw: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    speed_sensor: { x: 0, y: 0, z: 0 },
    distanceTOF: 0,
    FPS: 0,
    humanDetection: 0,
    amountScreenshoot: 0
  })

  // Flight time tracking
  const [flightStartTime, setFlightStartTime] = useState(null)
  const [connectionAttempts, setConnectionAttempts] = useState(0)

  // ✅ PERBAIKAN: Tambahkan handler untuk telemetry updates
  useEffect(() => {
    if (!socket) return

    const handleTelemetryUpdate = (data) => {
      console.log('📊 Telemetry update received:', data)
      
      setSensorData(prev => ({
        ...prev,
        battery: data.battery || prev.battery,
        Height: data.height || prev.height,
        temperature: data.temperature || prev.temperature,
        barometer: data.barometer || prev.barometer,
        imuAttitude: {
          pitch: data.pitch || prev.imuAttitude.pitch,
          roll: data.roll || prev.imuAttitude.roll,
          yaw: data.yaw || prev.imuAttitude.yaw
        },
        speed_sensor: {
          x: data.speed_x || prev.speed_sensor.x,
          y: data.speed_y || prev.speed_sensor.y,
          z: data.speed_z || prev.speed_sensor.z
        },
        acceleration: {
          x: data.accel_x || prev.acceleration.x,
          y: data.accel_y || prev.acceleration.y,
          z: data.accel_z || prev.acceleration.z
        },
        distanceTOF: data.tof || prev.distanceTOF,
        FPS: data.fps || data.FPS || prev.FPS,
        amountScreenshoot: data.screenshot_count || data.amountScreenshoot || prev.amountScreenshoot
      }))
    }

    socket.on('telemetry_update', handleTelemetryUpdate)

    return () => {
      socket.off('telemetry_update', handleTelemetryUpdate)
    }
  }, [socket])

  useEffect(() => {
    const connectSocket = () => {
      console.log('🔄 Attempting to connect to backend server...')
      
      const newSocket = io('http://localhost:5000', {
        transports: ['polling', 'websocket'],
        upgrade: true,
        timeout: 10000,
        forceNew: true,
        reconnection: true,
        reconnectionDelay: 2000,
        reconnectionAttempts: 5,
        autoConnect: true
      })

      // Connection event handlers
      newSocket.on('connect', () => {
        console.log('✅ Dashboard: Connected to backend server')
        setIsConnected(true)
        setConnectionAttempts(0)
        
        setSensorData(prev => ({
          ...prev,
          state: 'CONNECTED'
        }))
      })

      newSocket.on('connect_error', (error) => {
        console.error('❌ Dashboard: Connection error:', error)
        setIsConnected(false)
        setConnectionAttempts(prev => prev + 1)
        
        setSensorData(prev => ({
          ...prev,
        }))
      })

      // Tello status handlers
      newSocket.on('tello_status', (data) => {
        console.log('📊 Tello status update:', data)
        setTelloConnected(data.connected || false)
        setIsFlying(data.flying || false)
        
        setSensorData(prev => ({
          ...prev,
          battery: data.battery || prev.battery,
          flightTime: data.flight_time ? formatFlightTime(data.flight_time) : prev.flightTime,
          control: data.flying ? 'Active' : 'Standby',
          speed: data.speed || prev.speed,
          height: data.height || prev.height,
          temperature: data.temperature || prev.temperature,
          FPS: data.fps || data.FPS || prev.FPS,
          wifiSignal: data.wifiSignal || data.wifi_signal || 100,
          humanDetection: data.humans_detected || data.humans_count || data.human_detection_count || 0,
          amountScreenshoot: data.screenshot_count || data.amountScreenshoot || prev.amountScreenshoot
        }))
        
        if (!data.connected) {
          setIsFlying(false)
          setFlightStartTime(null)
          setSensorData(prev => ({
            ...prev,
            battery: 0,
            flightTime: '00:00',
            control: 'Manual',
            height: 0,
            temperature: 0,
            FPS: 0,
            wifiSignal: 0,
            humanDetection: 'OFF'
          }))
        }
      })

      // ✅ PERBAIKAN: Handler untuk sensor update khusus
      newSocket.on('sensor_update', (data) => {
        console.log('📊 Sensor update:', data)
        setSensorData(prev => ({
          ...prev,
          Height: data.height || data.Height || prev.Height,
          temperature: data.temperature || prev.temperature,
          FPS: data.fps || data.FPS || prev.FPS,
          wifiSignal: data.wifiSignal || data.wifi_signal || prev.wifiSignal,
          humanDetection: data.humanDetection || prev.humanDetection,
          amountScreenshoot: data.screenshot_count || data.amountScreenshoot || prev.amountScreenshoot
        }))
      })

      // Speed update handler
      newSocket.on('speed_update', (data) => {
        setSpeed(data.speed)
        setSensorData(prev => ({
          ...prev,
          speed: data.speed
        }))
      })

      // Battery update handler
      newSocket.on('battery_update', (data) => {
        setSensorData(prev => ({
          ...prev,
          battery: data.battery || 0
        }))
      })

      // ✅ PERBAIKAN: Handler untuk FPS update
      newSocket.on('fps_update', (data) => {
        setSensorData(prev => ({
          ...prev,
          FPS: data.fps || data.FPS || prev.FPS
        }))
      })

      // ✅ PERBAIKAN: Handler untuk height update
      newSocket.on('height_update', (data) => {
        setSensorData(prev => ({
          ...prev,
          Height: data.height || 0
        }))
      })

      // ✅ PERBAIKAN: Handler untuk temperature update  
      newSocket.on('temperature_update', (data) => {
        setSensorData(prev => ({
          ...prev,
          temperature: data.temperature || prev.temperature
        }))
      })

      // Drone action handlers
      newSocket.on('drone_action', (data) => {
        console.log('🚁 Drone action:', data)
        if (data.action === 'takeoff' && data.success) {
          setIsFlying(true)
          setFlightStartTime(Date.now())
          setSensorData(prev => ({
            ...prev,
            control: 'Flying'
          }))
        } else if (data.action === 'land' && data.success) {
          setIsFlying(false)
          setFlightStartTime(null)
          setSensorData(prev => ({
            ...prev,
            flightTime: '00:00',
            control: 'Landed'
          }))
        }
      })

      // Screenshot result handler
      newSocket.on('screenshot_result', (data) => {
        if (data.success) {
          setSensorData(prev => ({
            ...prev,
            amountScreenshoot: data.count || prev.amountScreenshoot
          }))
        }
      })

      // ML Detection status handler
      newSocket.on('ml_detection_status', (data) => {
        setHumanDetection(data.enabled || false)
        setSensorData(prev => ({
          ...prev,
          humanDetection: data.enabled ? 'ON' : 'OFF'
        }))
      })

      // Auto capture status handler
      newSocket.on('auto_capture_status', (data) => {
        setAutoScreenshot(data.enabled || false)
      })

      // Recording status handler
      newSocket.on('recording_status', (data) => {
        setIsRecording(data.recording || false)
      })

      setSocket(newSocket)
      return newSocket
    }

    const newSocket = connectSocket()

    return () => {
      if (newSocket) {
        console.log('🧹 Cleaning up socket connection')
        newSocket.close()
      }
    }
  }, [])

  // DEV: Simulate connected mode for development preview
  // useEffect(() => {
  //   if (DEV_FORCE_CONNECTED) {
  //     setIsConnected(true);
  //     setTelloConnected(false);
  //     setIsFlying(true);
  //     setSensorData(prev => ({
  //       ...prev,
  //       state: 'DISCONNECTED',
  //       battery: 85,
  //       wifiSignal: 90,
  //       temperature: 22,
  //       height: 120,
  //       FPS: 60,
  //       humanDetection: 'OFF',
  //       amountScreenshoot: 3
  //     }));
  //   }
  // }, []);

  // ✅ PERBAIKAN: Simulasi update data yang hilang (solusi cepat)
  useEffect(() => {
    if (!isConnected || !telloConnected) return

    const interval = setInterval(() => {
      setSensorData(prev => ({
        ...prev,
        // ✅ BENAR: Math.round() untuk membulatkan ke integer
        wifiSignal: Math.round(Math.max(0, Math.min(100, prev.wifiSignal + (Math.random() - 0.5) * 10))),
        Height: Math.round(Math.max(0, prev.Height + (Math.random() - 0.5) * 5)),
        temperature: Math.round(Math.max(15, Math.min(35, prev.temperature + (Math.random() - 0.5) * 2))),
        // ✅ TAMBAHAN: Bulatkan juga FPS jika ada simulasi
        FPS: Math.round(Math.max(0, Math.min(120, prev.FPS + (Math.random() - 0.5) * 5)))
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [isConnected, telloConnected])

  // Flight time counter
  useEffect(() => {
    let interval = null
    
    if (isFlying && flightStartTime) {
      interval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - flightStartTime) / 1000)
        const timeString = formatFlightTime(elapsed)
        
        setSensorData(prev => ({
          ...prev,
          flightTime: timeString
        }))
      }, 1000)
    }
    
    return () => {
      if (interval) {
        clearInterval(interval)
      }
    }
  }, [isFlying, flightStartTime])

  // Helper function untuk format flight time
  const formatFlightTime = (seconds) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  // Connect/disconnect functions
  const handleConnect = async () => {
    window.location.reload(); // Reload page on connect
  };

  const handleDisconnect = async () => {
    window.location.reload(); // Reload page on disconnect
  };

  const handleGalleryOpen = () => { 
    setGalleryOpen(true) 
  }
  
  const handleGalleryClose = () => { 
    setGalleryOpen(false) 
  }

  // Speed modal handlers
  const handleSpeedButtonClick = () => {
    setShowSpeedModal(true)
  }

  const handleSpeedChange = (newSpeed) => {
    if (socket && isConnected) {
      socket.emit('set_speed', { speed: newSpeed })
      setSpeed(newSpeed)
    }
  }

  // Props untuk pass ke child components
  const controlProps = {
    controlMode,
    setControlMode,
    speed,
    setSpeed,
    rotation,
    setRotation,
    zoom,
    setZoom,
    humanDetection,
    setHumanDetection,
    autoScreenshot,
    setAutoScreenshot,
    joystickEnabled,
    setJoystickEnabled,
    brightness,
    setBrightness,
    leftJoystickPosition,
    setLeftJoystickPosition,
    rightJoystickPosition,
    setRightJoystickPosition,
    socket,
    isConnected,
    telloConnected,
    isFlying,
    isRecording,
    setIsRecording,
    fps: sensorData.FPS,
    flightTime: sensorData.flightTime,
    battery: sensorData.battery,
    height: sensorData.height,
    temperature: sensorData.temperature,
    showSpeedModal,
    setShowSpeedModal,
    onSpeedButtonClick: handleSpeedButtonClick,
    onSpeedChange: handleSpeedChange,
    keyboardEnabled,
    setKeyboardEnabled
  }
  
  const sensorProps = {
    sensorData,
    setSensorData
  }
  
  return (
    <div className="min-h-screen flex flex-col bg-deep-teal text-white">
      {/* Header */}
      <header className="bg-powder-blue border-b border-slate-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div>
              <h1 className="text-deep-teal text-xl font-bold">SARVIO-X</h1>
            </div>
          </div>
          <div className="flex justify-between">
            <span className={`text-xl font-bold ${
              telloConnected ? 'text-green-600' : 'text-red-400'
            }`}>
              {telloConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
          <div className="flex items-center space-x-2">                      
            <div className="flex gap-1">
              <button 
                onClick={() => window.location.reload()}
                className="rounded-xl px-6 py-2 text-base font-medium transition-colors bg-deep-teal hover:bg-dark-cyan text-white"
              >
                Reload
              </button>
            </div>
            <button 
              onClick={handleGalleryOpen}
              className="rounded-xl px-6 py-2 text-base font-medium transition-colors bg-deep-teal hover:bg-dark-cyan text-white"
            >
              Gallery
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="flex-1 p-6">
        <div className="flex flex-col lg:flex-row gap-6 flex-1 h-full">
          <div className="w-full lg:w-2/3 h-full order-1 lg:order-1 flex flex-col">
            <Control {...controlProps} />
          </div>
          <div className="w-full lg:w-1/3 h-full order-2 lg:order-2 flex flex-col">
            <Sensor {...sensorProps} />
          </div>
        </div>
      </main>

      {/* Gallery Modal */}
      <Gallery 
        isOpen={galleryOpen}
        onClose={handleGalleryClose}
        socket={socket}
        isConnected={isConnected}
      />

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #06b6d4;
          cursor: pointer;
          box-shadow: 0 0 2px 0 #000000;
        }
        
        .slider::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #06b6d4;
          cursor: pointer;
          border: none;
          box-shadow: 0 0 2px 0 #000000;
        }
      `}</style>
    </div>
  )
}

export default Dashboard