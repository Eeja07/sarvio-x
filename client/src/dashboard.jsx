import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import Control from './control'
import Sensor from './sensor.jsx'
import Video from './video'

function Dashboard() {
  // State untuk drone control
  const [controlMode, setControlMode] = useState('Joystick Mode')
  const [speed, setSpeed] = useState(51)
  const [rotation, setRotation] = useState(90)
  const [joystickPosition, setJoystickPosition] = useState({ x: 0, y: 0 })
  
  // Socket connection state
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [telloConnected, setTelloConnected] = useState(false)
  const [isFlying, setIsFlying] = useState(false)
  
  const [sensorData, setSensorData] = useState({
    altitude: 0.2,
    flightTime: '00:00',
    battery: 0,
    wifiSignal: 75,
    temperature: 25,
    barometer: 1013.3,
    accelerometer: { x: -1.7, y: 1.9, z: 9.3 },
    gyroscope: { pitch: 1.0, roll: 3.2, yaw: -0.3 },
    gps: { latitude: 0.000000, longitude: 0.000000 }
  })

  // Flight time tracking
  const [flightStartTime, setFlightStartTime] = useState(null)

  useEffect(() => {
    // Main Socket.IO connection untuk dashboard coordination
    const newSocket = io('http://localhost:5000', {
      transports: ['polling', 'websocket'], // Coba polling dulu, lalu websocket
      upgrade: true, // Allow upgrade ke websocket
      timeout: 20000, // Timeout 20 detik
      forceNew: true, // Force new connection
      reconnection: true, // Enable reconnection
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      autoConnect: true
    })

    // Connection event handlers dengan logging detail
    newSocket.on('connect', () => {
      console.log('✅ Dashboard: Connected to backend server')
      console.log('🔗 Connection ID:', newSocket.id)
      console.log('🚀 Transport:', newSocket.io.engine.transport.name)
      setIsConnected(true)
    })

    newSocket.on('connect_error', (error) => {
      console.error('❌ Dashboard: Connection error:', error)
      console.error('📋 Error details:', error.message)
      setIsConnected(false)
    })

    newSocket.on('disconnect', (reason) => {
      console.log('❌ Dashboard: Disconnected from backend server')
      console.log('📋 Disconnect reason:', reason)
      setIsConnected(false)
      setTelloConnected(false)
      setIsFlying(false)
    })

    // Transport upgrade event
    newSocket.io.on('upgrade', () => {
      console.log('⬆️ Upgraded to transport:', newSocket.io.engine.transport.name)
    })

    // Tello status handlers
    newSocket.on('tello_status', (data) => {
      setTelloConnected(data.connected)
      setIsFlying(data.flying || false)
      
      if (data.battery !== undefined) {
        setSensorData(prev => ({
          ...prev,
          battery: data.battery
        }))
      }
    })

    newSocket.on('battery_update', (data) => {
      setSensorData(prev => ({
        ...prev,
        battery: data.battery
      }))
    })

    newSocket.on('drone_action', (data) => {
      if (data.action === 'takeoff' && data.success) {
        setIsFlying(true)
        setFlightStartTime(Date.now())
      } else if (data.action === 'land' && data.success) {
        setIsFlying(false)
        setFlightStartTime(null)
        setSensorData(prev => ({
          ...prev,
          flightTime: '00:00',
          altitude: 0.0
        }))
      }
    })

    setSocket(newSocket)

    return () => {
      if (newSocket) {
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
        const minutes = Math.floor(elapsed / 60)
        const seconds = elapsed % 60
        const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
        
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

  // Manual connect/disconnect functions
  const handleConnect = () => {
    if (socket && !telloConnected) {
      socket.emit('connect_tello')
    }
  }

  const handleDisconnect = () => {
    if (socket && telloConnected) {
      socket.emit('disconnect_tello')
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
    joystickPosition,
    setJoystickPosition,
    socket,
    isConnected,
    telloConnected,
    isFlying,
    battery: sensorData.battery
  }
  const videoProps = {
    socket,
    isConnected
  }
  const sensorProps = {
    sensorData,
    setSensorData
  }
  
  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-cyan-700 rounded flex items-center justify-center">
              <span className="text-white font-bold text-sm">SX</span>
            </div>
            <div>
              <h1 className="text-xl font-bold">SARVIO-X</h1>
              <p className="text-sm text-slate-400">DJI Tello Control System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded text-sm transition-colors">
              Media Gallery
            </button>
            
            {/* Multi-level Status */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm">Backend:</span>
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></span>
                <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className="text-sm">Tello:</span>
                <span className={`w-2 h-2 rounded-full ${telloConnected ? 'bg-green-400' : 'bg-red-400'}`}></span>
                <span className="text-sm">{telloConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
              
              {isFlying && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm">Flying:</span>
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                  <span className="text-sm text-cyan-400">Active</span>
                </div>
              )}
            </div>
            
            {/* Connection Control Buttons */}
            <div className="flex gap-2">
              {!telloConnected ? (
                <button 
                  onClick={handleConnect}
                  disabled={!isConnected}
                  className={`px-4 py-2 rounded font-medium transition-colors ${
                    !isConnected
                      ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  Connect Tello
                </button>
              ) : (
                <button 
                  onClick={handleDisconnect}
                  className="px-4 py-2 rounded font-medium transition-colors bg-red-600 hover:bg-red-700 text-white"
                >
                  Disconnect
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          <div className="order-1">
            <Control {...controlProps} />
          </div>
          <div className="order-2">
            <Video {...videoProps} />
          </div>
          <div className="order-3">
            <Sensor {...sensorProps} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-800 border-t border-slate-700 p-4">
        <div className="flex items-center justify-between text-sm text-slate-400">
          <div className="flex items-center space-x-4">
            <span>SARVIO-X v2.0</span>
            <span>•</span>
            <span>DJI Tello Compatible</span>
            <span>•</span>
            <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
              {isConnected ? 'WebSocket Connected' : 'WebSocket Disconnected'}
            </span>
          </div>
          <div className="flex items-center space-x-4">
            <span>System Status: {isConnected ? 'Online' : 'Offline'}</span>
            <span>•</span>
            <span>ML Engine: YOLOv8 (Ready)</span>
            <span>•</span>
            <span className={telloConnected ? 'text-green-400' : 'text-slate-400'}>
              {telloConnected ? 'Ready for Flight' : 'Drone Standby'}
            </span>
          </div>
        </div>
      </footer>

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