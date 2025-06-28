import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import Control from './control'
import Sensor from './sensor'
import Gallery from './gallery'

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
  const [joystickEnabled, setJoystickEnabled] = useState(true) // New state
  const [brightness, setBrightness] = useState(0)
  const [showSpeedModal, setShowSpeedModal] = useState(false) // New state
  
  // Socket connection state
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [telloConnected, setTelloConnected] = useState(false)
  const [isFlying, setIsFlying] = useState(false)
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  
  const [sensorData, setSensorData] = useState({
    battery: 0,
    bluetooth: 'Disconnected',
    state: 'OFF',
    control: 'Manual',
    flightTime: '00:00',
    wifiSignal: 0,
    sdkVersion: 'N/A',
    serialNumber: 'N/A',
    Height: 0,
    barometer: 0,
    temperature: 0,
    speed: 20, // Added speed to sensor data
    imuAttitude: { pitch: 0, roll: 0, yaw: 0 },
    acceleration: { x: 0, y: 0, z: 0 },
    speed_sensor: { x: 0, y: 0, z: 0 },
    distanceTOF: 0,
    FPS: 0,
    humanDetection: 'OFF',
    amountScreenshoot: 0
  })

  // Flight time tracking
  const [flightStartTime, setFlightStartTime] = useState(null)
  const [connectionAttempts, setConnectionAttempts] = useState(0)

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
          bluetooth: 'Connected',
          state: 'ON'
        }))
      })

      newSocket.on('connect_error', (error) => {
        console.error('❌ Dashboard: Connection error:', error)
        setIsConnected(false)
        setConnectionAttempts(prev => prev + 1)
        
        setSensorData(prev => ({
          ...prev,
          bluetooth: 'Error',
          state: 'ERROR'
        }))
      })

      newSocket.on('disconnect', (reason) => {
        console.log('❌ Dashboard: Disconnected from backend server')
        setIsConnected(false)
        setTelloConnected(false)
        setIsFlying(false)
        
        setSensorData(prev => ({
          ...prev,
          bluetooth: 'Disconnected',
          state: 'OFF',
          battery: 0,
          flightTime: '00:00'
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
          speed: data.speed || prev.speed // Update speed from backend
        }))
        
        if (!data.connected) {
          setIsFlying(false)
          setFlightStartTime(null)
          setSensorData(prev => ({
            ...prev,
            battery: 0,
            flightTime: '00:00',
            control: 'Manual'
          }))
        }
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
    if (socket && !telloConnected && isConnected) {
      console.log('🔗 Attempting to connect to Tello...')
      try {
        socket.emit('connect_tello')
        setSensorData(prev => ({
          ...prev,
          state: 'CONNECTING...'
        }))
      } catch (error) {
        console.error('❌ Failed to send connect command:', error)
      }
    } else if (!isConnected) {
      console.warn('⚠️ Backend server not connected')
    }
  }

  const handleDisconnect = async () => {
    if (socket && telloConnected) {
      console.log('🔌 Attempting to disconnect from Tello...')
      try {
        // Stop any ongoing movements first
        socket.emit('stop_movement')
        
        // Then disconnect
        socket.emit('disconnect_tello')
        setSensorData(prev => ({
          ...prev,
          state: 'DISCONNECTING...'
        }))
      } catch (error) {
        console.error('❌ Failed to send disconnect command:', error)
      }
    }
  }

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
    battery: sensorData.battery,
    showSpeedModal,
    setShowSpeedModal,
    onSpeedButtonClick: handleSpeedButtonClick,
    onSpeedChange: handleSpeedChange
  }
  
  const sensorProps = {
    sensorData,
    setSensorData
  }
  
  return (
    <div className="min-h-screen bg-deep-teal text-white">
      {/* Header */}
      <header className="bg-powder-blue border-b border-slate-700 p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div>
              <h1 className="text-deep-teal text-4xl font-bold">SARVIO-X</h1>
              <p className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                Backend: {isConnected ? 'Connected' : 'Disconnected'}
                {connectionAttempts > 0 && ` (Attempts: ${connectionAttempts})`}
              </p>
            </div>
          </div>
          <div className="flex justify-between">
            <span className={`text-4xl font-bold ${
              sensorData.state === 'ON' || sensorData.state === 'CONNECTING...' ? 'text-green-600' : 
              sensorData.state === 'ERROR' ? 'text-red-600' : 'text-red-400'
            }`}>
              {sensorData.state}
            </span>
          </div>
          
          <div className="flex items-center space-x-4">                      
            <div className="flex gap-2">
              {!telloConnected ? (
                <button 
                  onClick={handleConnect}
                  disabled={!isConnected}
                  className={`rounded-2xl px-15 py-5 text-4xl font-medium transition-colors ${
                    !isConnected
                      ? 'bg-gray-500 text-gray-300 cursor-not-allowed'
                      : 'bg-deep-teal hover:bg-dark-cyan text-white'
                  }`}
                >
                  {sensorData.state === 'CONNECTING...' ? 'Connecting...' : 'Connect'}
                </button>
              ) : (
                <button 
                  onClick={handleDisconnect}
                  className="px-15 py-5 rounded-2xl text-4xl font-medium transition-colors bg-red-600 hover:bg-red-700 text-white"
                >
                  {sensorData.state === 'DISCONNECTING...' ? 'Disconnecting...' : 'Disconnect'}
                </button>
              )}
            </div>
            <button 
              onClick={handleGalleryOpen}
              className="rounded-2xl px-15 py-5 text-4xl font-medium transition-colors bg-deep-teal hover:bg-dark-cyan text-white"
            >
              Gallery
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="p-15">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-3 w-full h-full">
            <Control {...controlProps} />
          </div>
          <div className="flex-1 w-full h-full">
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