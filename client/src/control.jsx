import { useState, useEffect, useCallback, useRef } from 'react'
import { 
  RotateCcw, 
  RotateCw, 
  ArrowUp, 
  ArrowDown, 
  Plane, 
  Settings, 
  PlaneLanding,
  AlertTriangle,
  Gamepad2, 
  Square, 
  Keyboard, 
  Gamepad, 
  Bot 
} from "lucide-react";

// Mapping mode ke ikon yang sesuai
const modeIcons = {
  'Joystick Mode': Gamepad2,
  'Button Mode': Square,
  'Keyboard Mode': Keyboard,
  'Controller Mode': Gamepad,
  'Autonomous Mode': Bot
};


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
  leftJoystickPosition,
  setLeftJoystickPosition,
  rightJoystickPosition,
  setRightJoystickPosition, 
  socket,
  isConnected,
  telloConnected,
  isFlying,
}) {
  const [videoFrame, setVideoFrame] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [mlDetection, setMlDetection] = useState(false)
  const [autoCapture, setAutoCapture] = useState(false)
  const leftColumnTextsKeyboard = [
  "TAKEOFF = T",
  "LANDING = L",
  "EMERGENCY = E",
  "ON/OFF DETECTION = /",
  "FLIP FORWARD = I ",
  "FLIP BACK = J",
  "FLIP LEFT = K",
  "FLIP RIGHT = L"
  ];

  const rightColumnTextsKeyboard = [
    "MOVE FORWARD = W",
    "MOVE BACKWARD = S",
    "MOVE LEFT = A",
    "MOVE RIGHT = D",
    "MOVE UP = ARROW UP",
    "MOVE DOWN = ARROW DOWN",
    "ROTATE CW = ARROW LEFT",
    "ROTATE CCW = ARROW RIGHT"
  ];
  const leftColumnTextsController = [
  "TAKEOFF = A",
  "LANDING = B",
  "EMERGENCY = E",
  "ON/OFF DETECTION = L1",
  "FLIP FORWARD = I ",
  "FLIP BACK = J",
  "FLIP LEFT = K",
  "FLIP RIGHT = L"
  ];

  const rightColumnTextsController = [
    "MOVE FORWARD = W",
    "MOVE BACKWARD = S",
    "MOVE LEFT = A",
    "MOVE RIGHT = D",
    "MOVE UP = ARROW UP",
    "MOVE DOWN = ARROW DOWN",
    "ROTATE CW = ARROW LEFT",
    "ROTATE CCW = ARROW RIGHT"
  ];
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
  const handleDirectionalMove = useCallback((direction) => {
    if (!socket || !isFlying) return
    
    const moveSpeed = speed
    let controls = {
      left_right: 0,
      for_back: 0,
      up_down: 0,
      yaw: 0
    }
    
    switch (direction) {
      case 'forward':
        controls.for_back = moveSpeed
        break
      case 'backward':
        controls.for_back = -moveSpeed  
        break
      case 'left':
        controls.left_right = -moveSpeed
        break
      case 'right':
        controls.left_right = moveSpeed 
        break
      case 'up':
        controls.up_down = moveSpeed
        break 
      case 'down':
        controls.up_down = -moveSpeed
        break
      case 'yaw_left':
        controls.yaw = -moveSpeed * 1
        break
      case 'yaw_right':
        controls.yaw = moveSpeed * 1
        break
      default:
        console.warn(`Unknown direction: ${direction}`)
        return
    }
    
    socket.emit('move_control', controls)
    
    // Stop movement setelah durasi singkat untuk button mode
    setTimeout(() => {
      if (socket) {
        socket.emit('stop_movement')
      }
    }, 300)
  }, [socket, isFlying, speed])

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
  const handleStart = () => {
    if (socket && telloConnected) {
      socket.emit('start_autonomous_mode')
      if (!isFlying) {
        socket.emit('takeoff')     
      }
    }
  }

  const handleFlip = (direction) => {
    if (!socket || !isFlying) return
    
    const moveSpeed = speed
    let controls = {
      left_right: 0,
      for_back: 0,
      up_down: 0,
      yaw: 0
    }
    
    switch (direction) {
      case 'forward':
        controls.for_back = moveSpeed
        break
      case 'backward':
        controls.for_back = -moveSpeed  
        break
      case 'left':
        controls.left_right = -moveSpeed
        break
      case 'right':
        controls.left_right = moveSpeed 
        break
      case 'up':
        controls.up_down = moveSpeed
        break 
      case 'down':
        controls.up_down = -moveSpeed
        break
      case 'yaw_left':
        controls.yaw = -moveSpeed * 1
        break
      case 'yaw_right':
        controls.yaw = moveSpeed * 1
        break
      default:
        console.warn(`Unknown direction: ${direction}`)
        return
    }
    
    socket.emit('move_control', controls)}

  const handleRotation = (direction) => {
    if (socket && isFlying) {
      const rotSpeed = speed * 1 // Slower rotation
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
  }, [socket, isConnected])

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

  const handleRecord = () => {
    console.log('Recording toggle - to be implemented')
  }

  return (
    <div className="p-6 bg-powder-blue text-white rounded-lg shadow-lg">
      <div className="w-full bg-light-blue rounded-lg p-4 mb-6">
        <h2 className="text-4xl font-bold text-deep-teal text-center">CONTROL PANEL</h2>  
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-255">
        {/* Video and Controls Section */}
        <div className="order-2 space-y-6">
          {/* Video Stream */}
          <div className="relative w-full h-140 bg-deep-teal rounded-lg flex items-center justify-center mb-5">
            {videoFrame ? (
              <img
                src={videoFrame}
                alt="Tello Live Stream"
                className="w-full h-full object-cover rounded-lg"
              />
            ) : (
              <div className="text-center text-ivory">
                <div className="w-20 h-20 mx-auto mb-2 opacity-30">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z" />
                  </svg>
                </div>
                <p className="text-xl font-medium">
                  {isConnected ? 'Waiting for Stream' : 'NO CONNECTION'}
                </p>
                <p className="text-xl p-2">
                  {isConnected ? 'Video stream starting...' : 'CONNECT DRONE TO START STREAMING'}
                </p>
              </div>
            )}

            {/* Connection Status Indicator */}
            <div className="absolute top-5 right-5">
              <div className={`w-6 h-6 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            </div>

            {/* Stream Status */}
            {isConnected && (
              <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
                {isStreaming ? 'LIVE' : 'PAUSED'}
              </div>
            )}
          </div>
          {controlMode === 'Joystick Mode' && (
              <div className="space-y-4">
                <div className="rounded-xl p-4 w-full h-43 bg-deep-teal">
                <div className="flex flex-wrap justify-center items-center gap-2">
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleFlip('left')}
                      disabled={!telloConnected || !isFlying}
                      className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                        telloConnected && isFlying
                          ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                          : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                      }`}
                    >
                      <RotateCcw className="w-8 h-8 mb-1" />
                    </button>
                    <button
                      onClick={() => handleFlip('right')}
                      disabled={!telloConnected || !isFlying}
                      className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                        telloConnected && isFlying
                          ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                          : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                      }`}
                    >
                      <RotateCw className="w-8 h-8 mb-1" />
                    </button>
                  </div>

                  <button
                    onClick={handleEmergency}
                    disabled={!telloConnected}
                    className={`w-70 h-15 rounded-xl flex items-center justify-center gap-2 transition-colors ${
                      telloConnected
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <AlertTriangle className="w-8 h-8" />
                  </button>

                  <div className="flex gap-1">
                    <button
                      onClick={() => handleFlip('up')}
                      disabled={!telloConnected || !isFlying}
                      className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                        telloConnected && isFlying
                          ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                          : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                      }`}
                    >
                      <ArrowUp className="w-8 h-8 mb-1" />
                    </button>
                    <button
                      onClick={() => handleFlip('down')}
                      disabled={!telloConnected || !isFlying}
                      className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                        telloConnected && isFlying
                          ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                          : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                      }`}
                    >
                      <ArrowDown className="w-8 h-8 mb-1" />
                    </button>
                  </div>
                </div>

                {/* Main Action Buttons */}
                <div className="flex justify-center gap-2 p-4">
                  <button
                    onClick={handleTakeoff}
                    disabled={!telloConnected || isFlying}
                    className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                      telloConnected && !isFlying
                        ? 'bg-green-600 text-ivory hover:bg-green-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <Plane className="w-8 h-8 mb-1" />
                  </button>
                  <button
                    onClick={handleSpeed}
                    disabled={!telloConnected || !isFlying}
                    className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <Settings className="w-8 h-8 mb-1" />
                  </button>
                  <button
                    onClick={handleLand}
                    disabled={!telloConnected || !isFlying}
                    className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-orange-600 text-ivory hover:bg-orange-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <PlaneLanding className="w-8 h-8 mb-1" />
                  </button>
                </div>
              </div>             
              <div className="flex flex-col md:flex-row justify-center md:justify-between items-center p-8 gap-4 px-4 md:px-15">
                <div className="text-center">
                  <VirtualJoystick 
                    joystickPosition={leftJoystickPosition}
                    setJoystickPosition={setLeftJoystickPosition}
                  />
                </div>
                <div className="text-center">
                  <VirtualJoystick 
                    joystickPosition={rightJoystickPosition}
                    setJoystickPosition={setRightJoystickPosition}
                  />
                </div>
              </div>
            </div>
          )}
          {controlMode === 'Button Mode' && (
            <div className="space-y-4">
              <div className="rounded-xl p-4 w-full h-43 bg-deep-teal">
              <div className="flex flex-wrap justify-center items-center gap-2">
                <div className="flex gap-1">
                  <button
                    onClick={() => handleFlip('left')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <RotateCcw className="w-8 h-8 mb-1" />
                  </button>
                  <button
                    onClick={() => handleFlip('right')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <RotateCw className="w-8 h-8 mb-1" />
                  </button>
                </div>

                <button
                  onClick={handleEmergency}
                  disabled={!telloConnected}
                  className={`w-70 h-15 rounded-xl flex items-center justify-center gap-2 transition-colors ${
                    telloConnected
                      ? 'bg-red-600 text-white hover:bg-red-700'
                      : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                  }`}
                >
                  <AlertTriangle className="w-8 h-8" />
                </button>

                <div className="flex gap-1">
                  <button
                    onClick={() => handleFlip('up')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <ArrowUp className="w-8 h-8 mb-1" />
                  </button>
                  <button
                    onClick={() => handleFlip('down')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-15 h-15 rounded-xl flex flex-col items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    <ArrowDown className="w-8 h-8 mb-1" />
                  </button>
                </div>
              </div>

              {/* Main Action Buttons */}
              <div className="flex justify-center gap-2 p-4">
                <button
                  onClick={handleTakeoff}
                  disabled={!telloConnected || isFlying}
                  className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                    telloConnected && !isFlying
                      ? 'bg-green-600 text-ivory hover:bg-green-700'
                      : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                  }`}
                >
                  <Plane className="w-8 h-8 mb-1" />
                </button>
                <button
                  onClick={handleSpeed}
                  disabled={!telloConnected || !isFlying}
                  className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                    telloConnected && isFlying
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                  }`}
                >
                  <Settings className="w-8 h-8 mb-1" />
                </button>
                <button
                  onClick={handleLand}
                  disabled={!telloConnected || !isFlying}
                  className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                    telloConnected && isFlying
                      ? 'bg-orange-600 text-white hover:bg-orange-700'
                      : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                  }`}
                >
                  <PlaneLanding className="w-8 h-8 mb-1" />
                </button>
              </div>
            </div>
            <div className="rounded-xl w-full h-62 bg-deep-teal">  
              {/* Directional Control Layout */}
              <div className="p-3 flex flex-col md:flex-row justify-center md:justify-between items-center gap-8 md:gap-4 px-4 md:px-8 w-full max-w-md md:max-w-full mx-auto">
                {/* Left D-Pad (Movement Controls) */}
                <div className="grid grid-cols-3 grid-rows-3 gap-1 w-fit">
                  {/* Empty top-left */}
                  <div></div>
                  {/* Forward */}
                  <button
                    onClick={() => handleDirectionalMove('forward')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ↑
                  </button>
                  {/* Empty top-right */}
                  <div></div>
                  
                  {/* Left */}
                  <button
                    onClick={() => handleDirectionalMove('left')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⟵
                  </button>
                  {/* Center (empty or logo) */}
                  <div className="w-18 h-18"></div>
                  {/* Right */}
                  <button
                    onClick={() => handleDirectionalMove('right')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⟶ 
                  </button>
                  
                  {/* Empty bottom-left */}
                  <div></div>
                  {/* Backward */}
                  <button
                    onClick={() => handleDirectionalMove('backward')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ↓
                  </button>
                  {/* Empty bottom-right */}
                  <div></div>
                </div>

                {/* Right Action Buttons */}
                <div className="grid grid-cols-3 grid-rows-3 gap-1 w-fit">
                  {/* Empty top-left */}
                  <div></div>
                  {/* Up */}
                  <button
                    onClick={() => handleDirectionalMove('up')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇈
                  </button>
                  {/* Empty top-right */}
                  <div></div>
                  
                  {/* Yaw Left */}
                  <button
                    onClick={() => handleDirectionalMove('yaw_left')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇇
                  </button>
                  {/* Center (empty) */}
                  <div className="w-18 h-18"></div>
                  {/* Yaw Right */}
                  <button
                    onClick={() => handleDirectionalMove('yaw_right')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇉
                  </button>
                  
                  {/* Empty bottom-left */}
                  <div></div>
                  {/* Down */}
                  <button
                    onClick={() => handleDirectionalMove('down')}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors ${
                      telloConnected && isFlying
                        ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇊
                  </button>
                  {/* Empty bottom-right */}
                  <div></div>
                </div>
              </div>
            </div>
          </div>
          )}
          {controlMode === 'Keyboard Mode' && (
            <div className="flex flex-col lg:flex-row w-full bg-deep-teal rounded-lg p-9 mb-2 gap-6">
              {/* Left Column */}
              <div className="flex-1 w-full space-y-2">
                <h3 className="text-xl font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">
                  Actions
                </h3>
                {leftColumnTextsKeyboard.map((text, idx) => (
                  <div key={`left-${idx}`} className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-sm">
                      {text.split(' = ')[1]}
                    </div>
                    <span className="text-lg text-ivory">
                      {text.split(' = ')[0] || text}
                    </span>
                  </div>
                ))}
              </div>

              {/* Right Column */}
              <div className="flex-1 w-full space-y-2">
                <h3 className="text-xl font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">
                  Movement
                </h3>
                {rightColumnTextsKeyboard.map((text, idx) => (
                  <div key={`right-${idx}`} className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-sm">
                      {text.includes('ARROW UP') ? '↑' : 
                        text.includes('ARROW DOWN') ? '↓' : 
                        text.includes('ARROW LEFT') ? '←' :
                        text.includes('ARROW RIGHT') ? '→' :
                        text.split(' = ')[1]}
                    </div>
                    <span className="text-lg text-ivory">
                      {text.split(' = ')[0]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {controlMode === 'Controller Mode' && (
           <div className="flex flex-col lg:flex-row w-full bg-deep-teal rounded-lg p-9 mb-2 gap-6">
              {/* Left Column */}
              <div className="flex-1 w-full space-y-2">
                <h3 className="text-xl font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">
                  Actions
                </h3>
                {leftColumnTextsController.map((text, idx) => (
                  <div key={`left-${idx}`} className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-sm">
                      {text.split(' = ')[1]}
                    </div>
                    <span className="text-lg text-ivory">
                      {text.split(' = ')[0] || text}
                    </span>
                  </div>
                ))}
              </div>

              {/* Right Column */}
              <div className="flex-1 w-full space-y-2">
                <h3 className="text-xl font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">
                  Movement
                </h3>
                {rightColumnTextsController.map((text, idx) => (
                  <div key={`right-${idx}`} className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-sm">
                      {text.includes('ARROW UP') ? '↑' : 
                        text.includes('ARROW DOWN') ? '↓' : 
                        text.includes('ARROW LEFT') ? '←' :
                        text.includes('ARROW RIGHT') ? '→' :
                        text.split(' = ')[1]}
                    </div>
                    <span className="text-lg text-ivory">
                      {text.split(' = ')[0]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {controlMode === 'Autonomous Mode' && (
            <div className="space-y-4">
              {/* Flip Controls and Emergency */}
              <div className="flex flex-wrap justify-center items-center gap-2">
                <div className="flex gap-1 p-40 space-x-5">
                <button
                  onClick={handleTakeoff}
                  disabled={!telloConnected || isFlying}
                  className={`w-32 h-25 rounded-xl flex items-center justify-center gap-2 transition-colors ${
                    telloConnected && !isFlying
                      ? 'bg-green-600 text-ivory hover:bg-green-700'
                      : 'bg-deep-teal text-dark-cyan cursor-not-allowed'
                  }`}
                >
                  <Plane className="w-12 h-12 mb-1" />
                </button>
                <button
                  onClick={handleSpeed}
                  disabled={!telloConnected || !isFlying}
                  className={`w-25 h-25 rounded-full flex flex-col items-center justify-center transition-colors ${
                    telloConnected && isFlying
                      ? 'bg-blue-600 text-ivory hover:bg-blue-700'
                      : 'bg-deep-teal text-dark-cyan cursor-not-allowed'
                  }`}
                >
                  <Settings className="w-12 h-12 mb-1" />
                </button>
                <button
                  onClick={handleEmergency}
                  disabled={!telloConnected}
                  className={`w-32 h-25 rounded-xl flex items-center justify-center gap-2 transition-colors ${
                    telloConnected
                      ? 'bg-red-600 text-ivory hover:bg-red-700'
                      : 'bg-deep-teal text-dark-cyan cursor-not-allowed'
                  }`}
                >
                  <AlertTriangle className="w-12 h-12" />
                </button>
              </div>
            </div>
            </div>
          )}</div>
        <div className="order-1 space-y-5">
          {/* Control Mode Buttons */}
          <div className="space-y-3">
            {['Joystick Mode', 'Button Mode', 'Keyboard Mode', 'Controller Mode', 'Autonomous Mode'].map((mode) => {
              const IconComponent = modeIcons[mode];
              
              return (
                <button
                  key={mode}
                  onClick={() => setControlMode(mode)}
                  className={`w-full p-7 text-2xl font-bold rounded-xl transition-colors flex items-center ${
                    controlMode === mode
                      ? 'bg-deep-teal text-gray-500'
                      : 'bg-deep-teal text-ivory hover:bg-dark-cyan'
                  }`}
                >
                  <IconComponent className="w-8 h-8 mr-4" />
                  <span className="flex-1 text-center">{mode}</span>
                </button>
              );
            })}
          </div>

          {/* Settings Panel */}
          <div className="bg-deep-teal p-7 rounded-2xl text-center space-y-4">
            {/* Detection Settings */}
            <div className="space-y-4 text-2xl">
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
                  disabled={!telloConnected}
                  className={`w-4 h-4 rounded transition-colors ${
                    telloConnected
                      ? 'text-light-blue bg-deep-teal border-light-blue focus:ring-light-blue cursor-pointer'
                      : 'bg-light-blue border-gray-400 cursor-not-allowed'
                  }`}
                  checked={humanDetection}
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
                  disabled={!telloConnected}
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

            {/* Capture and Record Buttons */}
            <div className="flex gap-2 text-2xl">
              <button
                onClick={handleCapture}
                disabled={!telloConnected}
                className={`flex-1 p-4 rounded-xl transition-colors ${
                  telloConnected  
                    ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'     
                    : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
                }`}
              >
                Capture
              </button>
              <button
                onClick={handleRecord}
                disabled={!telloConnected}
                className={`flex-1 p-4 rounded-xl transition-colors ${
                  telloConnected  
                    ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'     
                    : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
                }`}
              >
                Record
              </button>
            </div>

            {/* Zoom Control */}
            <div>
              <h3 className="text-2xl text-gray-400 font-medium mb-3">Zoom</h3>
              <div className="flex items-center gap-2 text-gray-400 text-2xl mb-2">
                <span>0x</span>
                <span className="flex-1 text-center">{zoom}1x</span>
                <span>2x</span>
              </div>
              <input
                disabled={!telloConnected}
                type="range"
                min="0"
                max="2"
                step={0.1}
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
                className="w-full h-2 bg-dark-cyan rounded-lg appearance-none cursor-pointer slider"
              />
            </div>

            {/* Brightness Control */}
            <div>
              <h3 className="text-gray-400 font-medium text-2xl mb-3">Brightness</h3>
              <div className="flex items-center gap-2 text-gray-400 text-2xl mb-2">
                <span>-100</span>
                <span className="flex-1 text-center">{brightness}0</span>
                <span>100</span>
              </div>
              <input
                disabled={!telloConnected}
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