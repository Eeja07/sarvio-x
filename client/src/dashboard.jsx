import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import Control from './control'
import Sensor from './sensor'

function Dashboard() {
  // State untuk drone control
  const [controlMode, setControlMode] = useState('Joystick Mode')
  const [speed, setSpeed] = useState(51)
  const [rotation, setRotation] = useState(90)
  const [leftJoystickPosition, setLeftJoystickPosition] = useState({ x: 0, y: 0 })
  const [rightJoystickPosition, setRightJoystickPosition] = useState({ x: 0, y: 0 })
  
  // Socket connection state
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [telloConnected, setTelloConnected] = useState(false)
  const [isFlying, setIsFlying] = useState(false)
  
  const [sensorData, setSensorData] = useState({
    battery: 75,
    bluetooth: 'Connected',
    state: 'ON',
    control: 'Manual',
    flightTime: 5,
    wifiSignal: 85,
    sdkVersion: '2.0',
    serialNumber: 'TELLO123',
    Height: 150,
    barometer: '1013.25 hPa',
    temperature: 25,
    imuAttitude: { pitch: 0, roll: 0, yaw: 0 },
    acceleration: { x: 0, y: 0, z: 9.8 },
    speed: { x: 0, y: 0, z: 0 },
    distanceTOF: 100.0
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
    leftJoystickPosition,
    setLeftJoystickPosition,
    rightJoystickPosition,
    setRightJoystickPosition,
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
    <div className="min-h-screen bg-deep-teal text-white">
      {/* Header */}
      <header className="bg-powder-blue border-b border-slate-700 p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div>
              <h1 className="text-deep-teal text-4xl font-bold">SARVIO-X</h1>
            </div>
          </div>
          <div className="flex justify-between">
            <span className={`text-4xl font-bold ${sensorData.state === 'ON' ? 'text-green-600' : 'text-red-400'}`}>
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
                      ? 'bg-deep-teal text-ivory cursor-not-allowed'
                      : 'bg-deep-teal hover:bg-dark-cyan text-white'
                  }`}
                >
                  Connect
                </button>
              ) : (
                <button 
                  onClick={handleDisconnect}
                  className="px-15 py-5 rounded-2xl text-4xl rounded font-medium transition-colors bg-red-600 hover:bg-red-700 text-white"
                >
                  Disconnect
                </button>
              )}
            </div>
                <button 
                  disabled={!isConnected}
                  className={`rounded-2xl px-15 py-5 text-4xl font-medium transition-colors ${
                    !isConnected
                      ? 'bg-deep-teal text-ivory cursor-not-allowed'
                      : 'bg-deep-teal hover:bg-dark-cyan text-white'
                  }`}
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

      {/* Footer
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
      </footer> */}

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