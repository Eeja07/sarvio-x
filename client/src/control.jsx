import { useState, useEffect, useCallback, useRef } from 'react'
import VirtualJoystick from './joystick'

function Control({ 
  controlMode, 
  setControlMode,
  zoom,
  setZoom,
  speed,
  setSpeed,

  humanDetection,
  setHumanDetection,
  autoScreenshot,
  setAutoScreenshot,
  brightness,
  setBrightness,
  rotation, 
  setRotation,
  leftJoystickPosition,
  setLeftJoystickPosition,
  rightJoystickPosition,
  setRightJoystickPosition, 
  joystickPosition, 
  setJoystickPosition,
  socket,
  isConnected,
  telloConnected,
  isFlying,
  battery
}) {
  const [videoFrame, setVideoFrame] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [mlDetection, setMlDetection] = useState(false)
  const [autoCapture, setAutoCapture] = useState(false)
  // Movement control refs
  const keysPressed = useRef(new Set())
  const lastJoystickPosition = useRef({ x: 0, y: 0 })

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!telloConnected || !isFlying) return
      
      const key = event.key.toLowerCase()
      if (!keysPressed.current.has(key)) {
        keysPressed.current.add(key)
        updateMovementFromKeyboard()
      }
    }

    const handleKeyUp = (event) => {
      const key = event.key.toLowerCase()
      keysPressed.current.delete(key)
      updateMovementFromKeyboard()
    }

    if (controlMode === 'Keyboard Mode') {
      window.addEventListener('keydown', handleKeyDown)
      window.addEventListener('keyup', handleKeyUp)
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [controlMode, telloConnected, isFlying])

  // Ganti useEffect joystick yang lama dengan ini:
  useEffect(() => {
    if (controlMode === 'Joystick Mode' && isFlying && socket) {
      const leftControls = {
        left_right: (leftJoystickPosition.x / 100) * speed,
        for_back: -(leftJoystickPosition.y / 100) * speed,
        up_down: 0,
        yaw: 0
      }
      
      const rightControls = {
        left_right: 0,
        for_back: 0,
        up_down: (rightJoystickPosition.y / 100) * speed,
        yaw: (rightJoystickPosition.x / 100) * speed
      }
      
      // Combine both joystick inputs
      const combinedControls = {
        left_right: leftControls.left_right,
        for_back: leftControls.for_back,
        up_down: rightControls.up_down,
        yaw: rightControls.yaw
      }
      
      socket.emit('move_control', combinedControls)
    }
  }, [leftJoystickPosition, rightJoystickPosition, controlMode, isFlying, socket, speed])

  const updateMovementFromKeyboard = useCallback(() => {
    if (!socket || !isFlying) return

    const moveSpeed = speed
    let controls = {
      left_right: 0,
      for_back: 0,
      up_down: 0,
      yaw: 0
    }

    // Movement controls
    if (keysPressed.current.has('a') || keysPressed.current.has('arrowleft')) {
      controls.left_right = -moveSpeed
    }
    if (keysPressed.current.has('d') || keysPressed.current.has('arrowright')) {
      controls.left_right = moveSpeed
    }
    if (keysPressed.current.has('w') || keysPressed.current.has('arrowup')) {
      controls.for_back = moveSpeed
    }
    if (keysPressed.current.has('s') || keysPressed.current.has('arrowdown')) {
      controls.for_back = -moveSpeed
    }
    if (keysPressed.current.has('q')) {
      controls.up_down = moveSpeed
    }
    if (keysPressed.current.has('e')) {
      controls.up_down = -moveSpeed
    }
    if (keysPressed.current.has('z')) {
      controls.yaw = -moveSpeed
    }
    if (keysPressed.current.has('c')) {
      controls.yaw = moveSpeed
    }

    socket.emit('move_control', controls)
  }, [socket, isFlying, speed])

  const handleTakeoff = () => {
    if (socket && telloConnected && !isFlying) {
      socket.emit('takeoff')
    }
  }

  const handleLand = () => {
    if (socket && telloConnected && isFlying) {
      socket.emit('land')
    }
  }
  const handleSpeed = () => {
    if (socket && telloConnected && isFlying) {
      const newSpeed = prompt('Enter new speed (cm/s):', speed)
      if (newSpeed !== null) {
        const speedValue = parseInt(newSpeed, 10)
        if (!isNaN(speedValue) && speedValue > 0) {
          setSpeed(speedValue)
          socket.emit('set_speed', speedValue)
        } else {
          alert('Invalid speed value. Please enter a positive number.')
        }
      }
    }
  }

  const handleEmergency = () => {
    if (socket) {
      socket.emit('stop_movement')
      if (isFlying) {
        socket.emit('land')
      }
    }
  }

  const handleFlip = (direction) => {
    if (socket && isFlying) {
      // TODO: Add flip commands to backend
      console.log(`Flip ${direction} - to be implemented`)
    }
  }

  const handleRotation = (direction) => {
    if (socket && isFlying) {
      const rotSpeed = speed * 0.5 // Slower rotation
      const controls = {
        left_right: 0,
        for_back: 0,
        up_down: 0,
        yaw: direction === 'CW' ? rotSpeed : -rotSpeed
      }
      
      socket.emit('move_control', controls)
      
      // Stop rotation after short time
      setTimeout(() => {
        socket.emit('stop_movement')
      }, 500)
    }
  }
  useEffect(() => {
    // Hanya setup event listeners jika socket ada
    if (!socket) return

    // Video frame handler
    const handleVideoFrame = (data) => {
      setVideoFrame(`data:image/jpeg;base64,${data.frame}`)
    }

    // Tello status handler
    const handleTelloStatus = (data) => {
      console.log('Tello status:', data)
    }

    // Setup event listeners
    socket.on('video_frame', handleVideoFrame)
    socket.on('tello_status', handleTelloStatus)

    // Auto-start streaming saat component mount DAN socket connected
    if (isConnected) {
      socket.emit('start_stream')
      setIsStreaming(true)
    }

    // Cleanup event listeners
    return () => {
      socket.off('video_frame', handleVideoFrame)
      socket.off('tello_status', handleTelloStatus)
    }
  }, [socket, isConnected]) // Dependency pada socket dan isConnected

  // Effect terpisah untuk handle connection changes
  useEffect(() => {
    if (!isConnected) {
      setIsStreaming(false)
      setVideoFrame(null)
    }
  }, [isConnected])

  const handleCapture = () => {
    if (videoFrame) {
      const link = document.createElement('a')
      link.href = videoFrame
      link.download = `tello_capture_${new Date().getTime()}.jpg`
      link.click()
    }
  }

  const toggleRecording = () => {
    console.log('Recording toggle - to be implemented')
  }
  return (
  <div className="p-6 bg-powder-blue text-white rounded-lg shadow-lg">
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
      <div className="order-2 space-y-6">
      <div 
        className="w-full aspect-[4/3] bg-deep-teal rounded-lg flex items-center justify-center mb-6">
        {videoFrame ? (
          <img
            src={videoFrame}
            alt="Tello Live Stream"
            className="w-full h-full object-full rounded-lg"
            style={{ width: '640px', height: '480px' }}
          />
        ) : (
          <div className="text-center text-slate-400">
            <div className="w-16 h-16 mx-auto mb-4 opacity-30">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z" />
              </svg>
            </div>
            <p className="text-lg font-medium">
              {isConnected ? 'Waiting for Stream' : 'No Connection'}
            </p>
            <p className="text-sm">
              {isConnected ? 'Video stream starting...' : 'Connect drone to start streaming'}
            </p>
          </div>
        )}

        {/* Connection Status Indicator */}
          {/* Connection Status Indicator */}
          <div className="absolute top-2 right-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}>
            </div>
          </div>

          {/* Stream Status */}
          {isConnected && (
          <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
            {isStreaming ? 'LIVE' : 'PAUSED'}
          </div>
          )}
        </div>
        <div className="space-y-4">
          <div className="flex justify-center items-center gap-2">
            <div className="flex gap-1">
              <button
                onClick={() => handleFlip('left')}
                disabled={!telloConnected || !isFlying}
                className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors ${
                telloConnected && isFlying
                ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
                : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
                }`}
                >
                L
              </button>
              <button
                onClick={() => handleFlip('right')}
                disabled={!telloConnected || !isFlying}
                className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors ${
                telloConnected && isFlying
                ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
                : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
                }`}
                >
                R
              </button>
            </div>

            <button
              onClick={handleEmergency}
              disabled={!telloConnected}
              className={`w-35 h-12 rounded-xl transition-colors ${
              telloConnected
              ? 'bg-dark-cyan text-white hover:bg-red-700'
              : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
              }`}
              >
              Emergency
            </button>

            <div className="flex gap-1">
              <button
                onClick={() => handleFlip('up')}
                disabled={!telloConnected || !isFlying}
                className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors ${
                telloConnected && isFlying
                ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
                : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
                }`}
                >
                U
              </button>
              <button
                onClick={() => handleFlip('down')}
                disabled={!telloConnected || !isFlying}
                className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors ${
                telloConnected && isFlying
                ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
                : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
                }`}
                >
                D
              </button>
            </div>
          </div>

          {/* Main Action Buttons */}
          <div className="flex justify-center gap-2">
            <button
              onClick={handleTakeoff}
              disabled={!telloConnected || isFlying}
              className={`w-13 h-13 rounded-full flex items-center justify-center transition-colors ${
              telloConnected && !isFlying
              ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
              : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
              }`}
              >
              T
            </button>
            <button
              onClick={handleSpeed}
              disabled={!telloConnected || !isFlying}
              className={`w-13 h-13 rounded-full flex items-center justify-center transition-colors ${
              telloConnected && isFlying
              ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
              : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
              }`}
              >
              S
            </button>
            <button
              onClick={handleLand}
              disabled={!telloConnected || !isFlying}
              className={`w-13 h-13 rounded-full flex items-center justify-center transition-colors ${
              telloConnected && isFlying
              ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'
              : 'bg-dark-cyan text-gray-400 cursor-not-allowed'
              }`}
              >
              L
            </button>
          </div>

          {/* Dual Joystick */}
          {controlMode === 'Joystick Mode' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center px-15">
              <div className="text-center">
                <VirtualJoystick 
                joystickPosition={leftJoystickPosition}
                setJoystickPosition={setLeftJoystickPosition}
                />
                <p className="text-xs text-slate-400 mt-2">Movement</p>
              </div>
              <div className="text-center">
                <VirtualJoystick 
                joystickPosition={rightJoystickPosition}
                setJoystickPosition={setRightJoystickPosition}
                />
                <p className="text-xs text-slate-400 mt-2">Alt & Yaw</p>
              </div>
            </div>
          </div>
          )}
        </div>
      </div>  
      <div className="order-1 space-y-8">
        <div className="space-y-2">
        {['Joystick Mode', 'Button Mode', 'Keyboard Mode', 'Controller Mode', 'Autonomous Mode'].map((mode) => (
          <button
            key={mode}
            onClick={() => setControlMode(mode)}
            disabled={!telloConnected}
            className={`w-full p-2 rounded-xl text-center transition-colors ${
              controlMode === mode 
                ? 'bg-dark-cyan text-gray-500 font-medium' 
                : telloConnected 
                  ? 'bg-dark-cyan text-ivory hover:bg-deep-teal' 
                  : 'bg-dark-cyan text-ivory cursor-not-allowed'
            }`}
          >
            {mode}
          </button>
        ))}
        </div>
        <div className="bg-deep-teal p-5 rounded-2xl text-center space-y-2">
          <div className="space-y-2">
            <label className={`rounded-xl flex items-center justify-between p-3 space-x-2 transition-colors ${
              telloConnected 
              ? 'bg-dark-cyan cursor-pointer' 
              : 'bg-dark-cyan cursor-not-allowed'
              }`}>
              <span className={`${telloConnected ? 'text-ivory' : 'text-gray-400'}`}>
              Human Detection
              </span>
              <input
              type="checkbox"
              disabled={!telloConnected}  // Key: disabled prop
              className={`w-4 h-4 rounded transition-colors ${
              telloConnected
              ? 'text-light-blue bg-deep-teal border-light-blue focus:ring-light-blue cursor-pointer'
              : 'bg-light-blue border-gray-400 cursor-not-allowed'
              }`}
              checked={autoScreenshot}
              onChange={(e) => setHumanDetection(e.target.checked)}
              />
            </label>
            <label className={`rounded-xl flex items-center justify-between p-3 space-x-2 transition-colors ${
              telloConnected 
              ? 'bg-dark-cyan cursor-pointer' 
              : 'bg-dark-cyan cursor-not-allowed'
              }`}>
              <span className={`${telloConnected ? 'text-ivory' : 'text-gray-400'}`}>
              Auto Screenshot
              </span>
              <input
              type="checkbox"
              disabled={!telloConnected}  // Key: disabled prop
              className={`w-4 h-4 rounded transition-colors ${
              telloConnected
              ? 'text-light-blue bg-deep-teal border-light-blue focus:ring-light-blue cursor-pointer'
              : 'bg-light-blue border-gray-400 cursor-not-allowed'
              }`}
              checked={autoScreenshot}
              onChange={(e) => setAutoScreenshot(e.target.checked)}
              />
            </label>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => handleCapture()}  // Buat function terpisah
              disabled={!telloConnected}
              className={`flex-1 p-2 rounded-xl transition-colors ${
              telloConnected  
              ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'     
              : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
              }`}
              >
              Capture
            </button>
            <button
              onClick={() => handleRecord()}   // Buat function terpisah
              disabled={!telloConnected}
              className={`flex-1 p-2 rounded-xl transition-colors ${
              telloConnected  
              ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'     
              : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
              }`}
              >
              Record
            </button>
            </div>
          <div>
            <h3 className="text-white font-medium mb-3">Zoom</h3>
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <span>0x</span>
              <span className="flex-1 text-center">{zoom}1x</span>
              <span>2x</span>
              </div>
                <input
                type="range"
                min="0"
                max="2"
                step={0.1}
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
                className="w-full h-2 bg-dark-cyan rounded-lg appearance-none cursor-pointer slider"
                />
          </div>
          <div>
            <h3 className="text-white font-medium mb-3">Brightness</h3>
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <span>-100</span>
              <span className="flex-1 text-center">{brightness}</span>
              <span>100</span>
              </div>
                <input
                type="range"
                min="-100"
                max="100"
                value={brightness}
                onChange={(e) => setBrightness(Number(e.target.value))}
                className="w-full h-2 bg-dark-cyan rounded-lg appearance-none cursor-pointer slider"
                />
          </div>
        </div>
      </div>


      </div>
    </div>
  )
}

export default Control