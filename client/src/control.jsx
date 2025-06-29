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
  Bot,
  Camera,
  Video,
  X
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
  speed,
  // setSpeed,
  // rotation,
  // setRotation,
  zoom,
  // setZoom,
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
  showSpeedModal,
  setShowSpeedModal,
  onSpeedButtonClick,
  onSpeedChange,
  keyboardEnabled,
  setKeyboardEnabled
}) {
  // State declarations
  const [videoFrame, setVideoFrame] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastScreenshotTime, setLastScreenshotTime] = useState(0)
  const [tempSpeed, setTempSpeed] = useState(speed)
  
  // NEW: Hold button functionality states
  const [pressedButton, setPressedButton] = useState(null)
  const intervalRef = useRef(null)
  
  // Movement control refs
  const keysPressed = useRef(new Set())
  
  // Constants
  const leftColumnTextsKeyboard = [
    "TAKEOFF = t",
    "LANDING = q", 
    "EMERGENCY = e",
    "ON/OFF DETECTION = z",
    "FLIP FORWARD = i",
    "FLIP BACK = j",
    "FLIP LEFT = k",
    "FLIP RIGHT = l"
  ];
  
  const rightColumnTextsKeyboard = [
    "MOVE FORWARD = ↑/w",
    "MOVE BACKWARD = ↓/s",
    "MOVE LEFT = ←/a",
    "MOVE RIGHT = →/d",
    "MOVE UP = ↑/w",
    "MOVE DOWN = ↓/s",
    "ROTATE CW = ←/a",
    "ROTATE CCW = →/d"
  ];
  
  const leftColumnTextsController = [
    "TAKEOFF = A",
    "LANDING = B",
    "EMERGENCY = START",
    "ON/OFF DETECTION = /",
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
    "MOVE UP = Q",
    "MOVE DOWN = E",
    "ROTATE CW = C",
    "ROTATE CCW = Z"
  ];

  // Action handlers - defined first with stable references
  const handleTakeoff = useCallback(() => {
    if (socket && telloConnected && !isFlying && isConnected) {
      try {
        socket.emit('takeoff')
        console.log('🛫 Takeoff command sent')
      } catch (error) {
        console.error('❌ Error sending takeoff command:', error)
      }
    }
  }, [socket, telloConnected, isFlying, isConnected])

  const handleLand = useCallback(() => {
    if (socket && telloConnected && isFlying && isConnected) {
      try {
        socket.emit('land')
        console.log('🛬 Land command sent')
      } catch (error) {
        console.error('❌ Error sending land command:', error)
      }
    }
  }, [socket, telloConnected, isFlying, isConnected])

  const handleEmergency = useCallback(() => {
    if (socket && isConnected) {
      try {
        socket.emit('stop_movement')
        if (isFlying && telloConnected) {
          socket.emit('emergency_land')
        }
        console.log('🚨 Emergency command sent')
      } catch (error) {
        console.error('❌ Error sending emergency command:', error)
      }
    }
  }, [socket, isConnected, isFlying, telloConnected])

  const handleStart = useCallback(() => {
    if (socket && telloConnected && isConnected) {
      try {
        socket.emit('start_autonomous_mode')
        if (!isFlying) {
          socket.emit('takeoff')     
        }
        console.log('🤖 Autonomous mode started')
      } catch (error) {
        console.error('❌ Error starting autonomous mode:', error)
      }
    }
  }, [socket, telloConnected, isConnected, isFlying])

  // Flip function
  const handleFlip = useCallback((direction) => {
    if (!socket || !isFlying || !telloConnected) return
    
    try {
      switch (direction) {
        case 'up':
          socket.emit('flip_command', { direction: 'f' })
          break
        case 'down':
          socket.emit('flip_command', { direction: 'b' })
          break
        case 'left':
          socket.emit('flip_command', { direction: 'l' })        
          break
        case 'right':
          socket.emit('flip_command', { direction: 'r' })        
          break
        default:
          console.warn(`Unknown flip direction: ${direction}`)
          return
      }    
      console.log(`🔄 Flip ${direction} command sent`)
    } catch (error) {
      console.error('❌ Error sending flip command:', error)
    }  
  }, [socket, isFlying, telloConnected])

  // Capture handler
  const handleCapture = useCallback(() => {
    const now = Date.now()
    if (now - lastScreenshotTime < 1000) {
      console.log('⏱️ Screenshot rate limited')
      return
    }
    
    if (socket && isConnected && telloConnected) {
      try {
        socket.emit('manual_screenshot')
        setLastScreenshotTime(now)
        console.log('📸 Manual screenshot requested')
      } catch (error) {
        console.error('❌ Error taking screenshot:', error)
      }
    } else if (videoFrame) {
      const link = document.createElement('a')
      link.href = videoFrame
      link.download = `tello_capture_${new Date().getTime()}.jpg`
      link.click()
      console.log('📸 Downloaded current frame')
    }
  }, [socket, isConnected, telloConnected, lastScreenshotTime, videoFrame])

  // Record handler
  const handleRecord = useCallback(() => {
    if (socket && isConnected && telloConnected) {
      try {
        const newRecordingState = !isRecording
        socket.emit('toggle_recording', { recording: newRecordingState })
        setIsRecording(newRecordingState)
        console.log(`🎥 Recording ${newRecordingState ? 'started' : 'stopped'}`)
      } catch (error) {
        console.error('❌ Error toggling recording:', error)
      }
    } else {
      console.log('🎥 Recording feature requires Tello connection')
    }
  }, [socket, isConnected, telloConnected, isRecording, setIsRecording])

  // Movement control function - stable reference
  const updateMovementFromKeyboard = useCallback(() => {
    if (!socket || !isFlying || !telloConnected) return

    const moveSpeed = Math.min(Math.max(speed, 10), 100)
    let controls = {
      left_right: 0,
      for_back: 0,
      up_down: 0,
      yaw: 0
    }

    if (!keyboardEnabled) {
      // Mode 1: WASD untuk movement, Arrow keys untuk up/down + yaw
      if (keysPressed.current.has('a')) {
        controls.left_right = -moveSpeed
      }
      if (keysPressed.current.has('d')) {
        controls.left_right = moveSpeed
      }
      if (keysPressed.current.has('w')) {
        controls.for_back = moveSpeed
      }
      if (keysPressed.current.has('s')) {
        controls.for_back = -moveSpeed
      }
      if (keysPressed.current.has('arrowup')) {
        controls.up_down = moveSpeed
      }
      if (keysPressed.current.has('arrowdown')) {
        controls.up_down = -moveSpeed
      }
      if (keysPressed.current.has('arrowleft')) {
        controls.yaw = -moveSpeed
      }
      if (keysPressed.current.has('arrowright')) {
        controls.yaw = moveSpeed
      }
    } else {
      // Mode 2: Arrow keys untuk movement, WASD untuk up/down + yaw
      if (keysPressed.current.has('arrowleft')) {
        controls.left_right = -moveSpeed
      }
      if (keysPressed.current.has('arrowright')) {
        controls.left_right = moveSpeed
      }
      if (keysPressed.current.has('arrowup')) {
        controls.for_back = moveSpeed
      }
      if (keysPressed.current.has('arrowdown')) {
        controls.for_back = -moveSpeed
      }
      if (keysPressed.current.has('w')) {
        controls.up_down = moveSpeed
      }
      if (keysPressed.current.has('s')) {
        controls.up_down = -moveSpeed
      }
      if (keysPressed.current.has('a')) {
        controls.yaw = -moveSpeed
      }  
      if (keysPressed.current.has('d')) {
        controls.yaw = moveSpeed
      }  
    }
    
    try {
      socket.emit('move_control', controls)
      console.log(`🎮 Movement (Mode ${keyboardEnabled ? '2' : '1'}):`, controls)
    } catch (error) {
      console.error('❌ Error sending keyboard control:', error)
    }  
  }, [keyboardEnabled, socket, isFlying, telloConnected, speed])

  // NEW: Hold movement control function
  const sendContinuousMovement = useCallback((direction) => {
    if (!socket || !isFlying || !telloConnected) return
    
    const moveSpeed = Math.min(Math.max(speed, 10), 100)
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
        controls.yaw = -moveSpeed
        break
      case 'yaw_right':
        controls.yaw = moveSpeed
        break
      default:
        console.warn(`Unknown direction: ${direction}`)
        return
    }    
    
    try {
      socket.emit('move_control', controls)
      console.log(`🎮 Hold Movement ${direction}:`, controls)
    } catch (error) {
      console.error('❌ Error sending hold movement command:', error)
    }  
  }, [socket, isFlying, telloConnected, speed])

  // NEW: Handle button press (start hold)
  const handleButtonPress = useCallback((direction) => {
    if (!socket || !isFlying || !telloConnected) return
    
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
    
    setPressedButton(direction)
    
    // Send first command immediately
    sendContinuousMovement(direction)
    
    // Start interval for continuous movement
    intervalRef.current = setInterval(() => {
      sendContinuousMovement(direction)
    }, 100) // Send command every 100ms
    
    console.log(`🎮 Started holding button: ${direction}`)
  }, [socket, isFlying, telloConnected, sendContinuousMovement])

  // NEW: Handle button release (stop hold)
  const handleButtonRelease = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    
    if (pressedButton) {
      console.log(`🎮 Released button: ${pressedButton}`)
    }
    
    setPressedButton(null)
    
    // Stop movement
    if (socket && socket.connected) {
      try {
        socket.emit('stop_movement')
        console.log('🛑 Stop movement command sent')
      } catch (error) {
        console.error('❌ Error sending stop movement command:', error)
      }
    }
  }, [socket, pressedButton])

  // Directional move (for old click behavior - keeping for compatibility)
  const handleDirectionalMove = useCallback((direction) => {
    if (!socket || !isFlying || !telloConnected) return
    
    const moveSpeed = Math.min(Math.max(speed, 10), 100)
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
        controls.yaw = -moveSpeed
        break
      case 'yaw_right':
        controls.yaw = moveSpeed
        break
      default:
        console.warn(`Unknown direction: ${direction}`)
        return
    }    
    
    try {
      socket.emit('move_control', controls)
      
      setTimeout(() => {
        if (socket && socket.connected) {
          socket.emit('stop_movement')
        }  
      }, 200)  
    } catch (error) {
      console.error('❌ Error sending move command:', error)
    }  
  }, [socket, isFlying, telloConnected, speed])

  // Keyboard event handlers - STABLE REFERENCES with useCallback
  const handleKeyDown = useCallback((event) => {
    if (!telloConnected || showSpeedModal) return
    
    const key = event.key.toLowerCase()
    
    if (!keysPressed.current.has(key)) {
      keysPressed.current.add(key)
      
      switch (key) {
        case 't':
          if (!isFlying) handleTakeoff()
          return    
        case 'q':
          if (isFlying) handleLand()
          return    
        case 'o':
          handleCapture()
          return
        case 'p':
          handleRecord()
          return
        case 'z':
          setHumanDetection(prev => !prev)
          return
        case 'x':
          setAutoScreenshot(prev => !prev)
          return
        case 'v':
          setJoystickEnabled(prev => !prev)
          return
        case 'f':
          setKeyboardEnabled(prev => {
            const newMode = !prev
            console.log(`🎮 Keyboard mode switched to: ${newMode ? 'Mode 2 (Arrow Movement)' : 'Mode 1 (WASD Movement)'}`)
            
            // Sync to backend if connected
            if (socket && isConnected && telloConnected) {
              try {
                socket.emit('enable_change_keyboard', { enabled: newMode })
              } catch (error) {
                console.error('❌ Error syncing keyboard mode:', error)
              }  
            }  
            return newMode
          })  
          return
        case 'm':
          onSpeedChange(speed + 10)
          return
        case 'n':
          onSpeedChange(speed - 10) 
          return
        case 'i':
          if (isFlying) handleFlip('up')
          return    
        case 'j':
          if (isFlying) handleFlip('down')
          return    
        case 'k':
          if (isFlying) handleFlip('left')
          return    
        case 'l':
          if (isFlying) handleFlip('right')
          return    
        case 'e':
          handleEmergency()
          return
      }    
      
      // Movement keys handling - berdasarkan mode keyboard
      const movementKeys = !keyboardEnabled 
        ? ['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']
        : ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 's', 'a', 'd']
      
      if (isFlying && movementKeys.includes(key)) {
        updateMovementFromKeyboard()
      }  
    }  
  }, [
    telloConnected, 
    showSpeedModal, 
    isFlying, 
    keyboardEnabled, 
    speed,
    socket,
    isConnected,
    handleTakeoff,
    handleLand,
    handleCapture,
    handleRecord,
    handleFlip,
    handleEmergency,
    updateMovementFromKeyboard,
    setHumanDetection,
    setAutoScreenshot,
    setJoystickEnabled,
    setKeyboardEnabled,
    onSpeedChange
  ])

  const handleKeyUp = useCallback((event) => {
    const key = event.key.toLowerCase()
    keysPressed.current.delete(key)
    
    // Movement keys handling - berdasarkan mode keyboard
    const movementKeys = !keyboardEnabled 
      ? ['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']
      : ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 's', 'a', 'd']
    
    if (isFlying && movementKeys.includes(key)) {
      updateMovementFromKeyboard()
    }  
  }, [isFlying, keyboardEnabled, updateMovementFromKeyboard])

  // Speed modal handlers
  const handleSpeedModalClose = () => {
    setShowSpeedModal(false)
    setTempSpeed(speed)
  }

  const handleSpeedApply = () => {
    onSpeedChange(tempSpeed)
    setShowSpeedModal(false)
  }

  // SEPARATED EVENT LISTENER EFFECT - Minimal dependencies, stable handlers
  useEffect(() => {
    if (controlMode === 'Keyboard Mode') {
      window.addEventListener('keydown', handleKeyDown)
      window.addEventListener('keyup', handleKeyUp)
    }  

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }  
  }, [controlMode, handleKeyDown, handleKeyUp]) // MINIMAL DEPENDENCIES

  // NEW: Cleanup interval on unmount or mode change
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [])

  // NEW: Stop movement when switching away from Button Mode
  useEffect(() => {
    if (controlMode !== 'Button Mode' && intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
      setPressedButton(null)
      
      if (socket && socket.connected) {
        try {
          socket.emit('stop_movement')
        } catch (error) {
          console.error('❌ Error stopping movement on mode change:', error)
        }
      }
    }
  }, [controlMode, socket])

  // Joystick control
  useEffect(() => {
    if (controlMode === 'Joystick Mode' && joystickEnabled && isFlying && socket && telloConnected) {
      const leftControls = {
        left_right: Math.round((leftJoystickPosition.x / 100) * speed),
        for_back: Math.round(-(leftJoystickPosition.y / 100) * speed),
        up_down: 0,
        yaw: 0
      }  
      
      const rightControls = {
        left_right: 0,
        for_back: 0,
        up_down: Math.round(-(rightJoystickPosition.y / 100) * speed),
        yaw: Math.round((rightJoystickPosition.x / 100) * speed)
      }  
      
      const combinedControls = {
        left_right: leftControls.left_right,
        for_back: leftControls.for_back,
        up_down: rightControls.up_down,
        yaw: rightControls.yaw
      }  
      
      try {
        socket.emit('move_control', combinedControls)
      } catch (error) {
        console.error('❌ Error sending joystick control:', error)
      }  
    }  
  }, [leftJoystickPosition, rightJoystickPosition, controlMode, joystickEnabled, isFlying, socket, telloConnected, speed])

  // Video stream setup
  useEffect(() => {
    if (!socket) return

    const handleVideoFrame = (data) => {
      if (data && data.frame) {
        setVideoFrame(`data:image/jpeg;base64,${data.frame}`)
      }
    }

    const handleClearVideoFrame = () => {
      console.log("🧹 Clearing video frame due to disconnect")
      setVideoFrame(null)
    }

    const handleTelloStatus = (data) => {
      if (!data.connected) {
        setVideoFrame(null)
      }
    }

    const handleStreamStatus = (data) => {
      setIsStreaming(data.streaming || false)
    }

    socket.on('video_frame', handleVideoFrame)
    socket.on('clear_video_frame', handleClearVideoFrame)
    socket.on('tello_status', handleTelloStatus)
    socket.on('stream_status', handleStreamStatus)

    if (isConnected && telloConnected) {
      try {
        socket.emit('start_stream')
        setIsStreaming(true)
      } catch (error) {
        console.error('❌ Error starting stream:', error)
      }
    }

    return () => {
      socket.off('video_frame', handleVideoFrame)
      socket.off('clear_video_frame', handleClearVideoFrame)
      socket.off('tello_status', handleTelloStatus)
      socket.off('stream_status', handleStreamStatus)
    }
  }, [socket, isConnected, telloConnected])

  // Reset video when not connected
  useEffect(() => {
    if (!isConnected || !telloConnected) {
      setIsStreaming(false)
      setVideoFrame(null)
    }
  }, [isConnected, telloConnected])

  // Human detection sync
  useEffect(() => {
    if (socket && isConnected && telloConnected) {
      try {
        socket.emit('enable_ml_detection', { enabled: humanDetection })
      } catch (error) {
        console.error('❌ Error setting ML detection:', error)
      }
    }
  }, [humanDetection, socket, isConnected, telloConnected])

  // Auto screenshot sync
  useEffect(() => {
    if (socket && isConnected && telloConnected) {
      try {
        socket.emit('enable_auto_capture', { enabled: autoScreenshot })
      } catch (error) {
        console.error('❌ Error setting auto capture:', error)
      }
    }
  }, [autoScreenshot, socket, isConnected, telloConnected])

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
                style={{
                  filter: `brightness(${100 + brightness}%)`,
                  transform: `scale(${zoom})`
                }}
              />
            ) : (
              <div className="text-center text-ivory">
                <div className="w-20 h-20 mx-auto mb-2 opacity-30">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z" />
                  </svg>
                </div>
                <p className="text-xl font-medium">
                  {isConnected ? 
                    (telloConnected ? 'Waiting for Stream' : 'Tello Not Connected') : 
                    'NO CONNECTION'
                  }
                </p>
                <p className="text-xl p-2">
                  {isConnected ? 
                    (telloConnected ? 'Video stream starting...' : 'Connect Tello to start streaming') :
                    'CONNECT TO BACKEND SERVER'
                  }
                </p>
              </div>
            )}

            {/* Connection Status Indicators */}
            <div className="absolute top-5 right-5 flex space-x-2">
              <div className={`w-6 h-6 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} 
                   title={`Backend: ${isConnected ? 'Connected' : 'Disconnected'}`}></div>
              <div className={`w-6 h-6 rounded-full ${telloConnected ? 'bg-blue-400' : 'bg-gray-400'}`}
                   title={`Tello: ${telloConnected ? 'Connected' : 'Disconnected'}`}></div>
            </div>

            {/* Stream Status */}
            {isConnected && (
              <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded flex items-center space-x-1">
                <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-red-500' : 'bg-gray-500'}`}></div>
                <span>{isStreaming ? 'LIVE' : 'PAUSED'}</span>
              </div>
            )}

            {/* Recording Indicator */}
            {isRecording && (
              <div className="absolute top-2 left-2 bg-red-600 text-white text-xs px-2 py-1 rounded flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
                <span>REC</span>
              </div>
            )}

            {/* NEW: Hold Button Status Indicator */}
            {pressedButton && controlMode === 'Button Mode' && (
              <div className="absolute top-2 right-2 bg-blue-600 text-white text-xs px-2 py-1 rounded flex items-center space-x-1">
                <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
                <span>HOLD: {pressedButton.toUpperCase()}</span>
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
                    onClick={onSpeedButtonClick}
                    disabled={!telloConnected}
                    className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                      telloConnected
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
              
              {/* Joystick Area - Only show if enabled */}
              {joystickEnabled && (
                <div className="flex flex-col md:flex-row justify-center md:justify-between items-center p-8 gap-4 px-4 md:px-15">
                  <div className="text-center">
                    <VirtualJoystick 
                      joystickPosition={leftJoystickPosition}
                      setJoystickPosition={setLeftJoystickPosition}
                      telloConnected={telloConnected}
                    />
                  </div>
                  <div className="text-center">
                    <VirtualJoystick 
                      joystickPosition={rightJoystickPosition}
                      setJoystickPosition={setRightJoystickPosition}
                      telloConnected={telloConnected}
                    />
                  </div>
                </div>
              )}
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
                  onClick={onSpeedButtonClick}
                  disabled={!telloConnected}
                  className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex flex-col items-center justify-center transition-colors ${
                    telloConnected
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

            {/* MODIFIED: Button Mode Movement Controls with Hold Functionality */}
            <div className="rounded-xl w-full h-62 bg-deep-teal">  
              <div className="p-3 flex flex-col md:flex-row justify-center md:justify-between items-center gap-8 md:gap-4 px-4 md:px-8 w-full max-w-md md:max-w-full mx-auto">
                
                {/* LEFT GRID: Forward/Backward + Left/Right Movement */}
                <div className="grid grid-cols-3 grid-rows-3 gap-1 w-fit">
                  <div></div>
                  <button
                    onMouseDown={() => handleButtonPress('forward')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'forward' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ↑
                  </button>
                  <div></div>
                  
                  <button
                    onMouseDown={() => handleButtonPress('left')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'left' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⟵
                  </button>
                  <div className="w-18 h-18"></div>
                  <button
                    onMouseDown={() => handleButtonPress('right')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'right' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⟶ 
                  </button>
                  
                  <div></div>
                  <button
                    onMouseDown={() => handleButtonPress('backward')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'backward' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ↓
                  </button>
                  <div></div>
                </div>

                {/* RIGHT GRID: Up/Down + Yaw Left/Right Movement */}
                <div className="grid grid-cols-3 grid-rows-3 gap-1 w-fit">
                  <div></div>
                  <button
                    onMouseDown={() => handleButtonPress('up')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'up' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇈
                  </button>
                  <div></div>
                  
                  <button
                    onMouseDown={() => handleButtonPress('yaw_left')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'yaw_left' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇇
                  </button>
                  <div className="w-18 h-18"></div>
                  <button
                    onMouseDown={() => handleButtonPress('yaw_right')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'yaw_right' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇉
                  </button>
                  
                  <div></div>
                  <button
                    onMouseDown={() => handleButtonPress('down')}
                    onMouseUp={handleButtonRelease}
                    onMouseLeave={handleButtonRelease}
                    disabled={!telloConnected || !isFlying}
                    className={`w-18 h-18 text-4xl rounded-full flex items-center justify-center transition-colors select-none ${
                      telloConnected && isFlying
                        ? `${pressedButton === 'down' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800`
                        : 'bg-dark-cyan text-deep-teal cursor-not-allowed'
                    }`}
                  >
                    ⇊
                  </button>
                  <div></div>
                </div>
              </div>
            </div>
          </div>
          )}
          
          {controlMode === 'Keyboard Mode' && (
            <div className="flex flex-col lg:flex-row w-full bg-deep-teal rounded-lg p-9 mb-2 gap-6">
              <div className="absolute top-4 left-4 bg-blue-600 text-white px-3 py-1 rounded text-sm">
                {keyboardEnabled ? 'Mode 2: Arrow Movement' : 'Mode 1: WASD Movement'}
              </div>
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

              <div className="flex-1 w-full space-y-2">
                <h3 className="text-xl font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">
                  Movement
                </h3>
                {rightColumnTextsController.map((text, idx) => (
                  <div key={`right-${idx}`} className="flex items-center space-x-3">
                    <div className="w-12 h-8 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-xs">
                      {text.split(' = ')[1]}
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
              <div className="flex flex-wrap justify-center items-center gap-2">
                <div className="flex gap-1 p-40 space-x-5">
                <button
                  onClick={handleStart}
                  disabled={!telloConnected}
                  className={`w-32 h-25 rounded-xl flex items-center justify-center gap-2 transition-colors ${
                    telloConnected
                      ? 'bg-green-600 text-ivory hover:bg-green-700'
                      : 'bg-deep-teal text-dark-cyan cursor-not-allowed'
                  }`}
                >
                  <Bot className="w-12 h-12 mb-1" />
                </button>
                <button
                  onClick={onSpeedButtonClick}
                  disabled={!telloConnected}
                  className={`w-25 h-25 rounded-full flex flex-col items-center justify-center transition-colors ${
                    telloConnected
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
          )}
        </div>
        
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

          {/* NEW REORGANIZED Settings Panel */}
          <div className="bg-deep-teal p-7 rounded-2xl text-center space-y-6">
            {/* Detection Settings - Vertical Layout */}
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

              {/* NEW: Joystick Toggle */}
              <label className={`rounded-xl flex items-center justify-between p-3 space-x-2 transition-colors ${
                controlMode === 'Joystick Mode' 
                  ? 'bg-dark-cyan cursor-pointer' 
                  : 'bg-dark-cyan cursor-not-allowed'
              }`}>
                <span className={`${controlMode === 'Joystick Mode' ? 'text-ivory' : 'text-gray-400'}`}>
                  Joystick Control
                </span>
                <input
                  type="checkbox"
                  disabled={controlMode !== 'Joystick Mode'}
                  className={`w-4 h-4 rounded transition-colors ${
                    controlMode === 'Joystick Mode'
                      ? 'text-light-blue bg-deep-teal border-light-blue focus:ring-light-blue cursor-pointer'
                      : 'bg-light-blue border-gray-400 cursor-not-allowed'
                  }`}
                  checked={joystickEnabled}
                  onChange={(e) => setJoystickEnabled(e.target.checked)}
                />
              </label>
            </div>

            {/* Capture and Record Buttons - Vertical Layout */}
            <div className="space-y-3 text-2xl">
              <button
                onClick={handleCapture}
                disabled={!telloConnected && !videoFrame}
                className={`w-full p-4 rounded-xl transition-colors flex items-center justify-center gap-2 ${
                  (telloConnected || videoFrame)
                    ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'     
                    : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
                }`}
                title={telloConnected ? 'Take screenshot via backend' : 'Download current frame'}
              >
                <Camera className="w-5 h-5" />
                <span>Capture</span>
              </button>
              
              <button
                onClick={handleRecord}
                disabled={!telloConnected}
                className={`w-full p-4 rounded-xl transition-colors flex items-center justify-center gap-2 ${
                  telloConnected  
                    ? `${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-dark-cyan hover:bg-deep-teal'} text-ivory`
                    : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
                }`}
              >
                <Video className="w-5 h-5" />
                <span>{isRecording ? 'Stop' : 'Record'}</span>
              </button>
            </div>

            {/* Brightness Control */}
            <div>
              <h3 className="text-gray-400 font-medium text-2xl mb-3">Brightness</h3>
              <div className="flex items-center gap-2 text-gray-400 text-2xl mb-2">
                <span>-100</span>
                <span className="flex-1 text-center text-ivory">{brightness}</span>
                <span>100</span>
              </div>
              <input
                disabled={!videoFrame}
                type="range"
                min="-100"
                max="100"
                value={brightness}
                onChange={(e) => setBrightness(Number(e.target.value))}
                className={`w-full h-2 bg-dark-cyan rounded-lg appearance-none slider ${
                  videoFrame ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Speed Modal */}
      {showSpeedModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
          <div className="bg-powder-blue rounded-xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold text-deep-teal">Set Drone Speed</h3>
              <button
                onClick={handleSpeedModalClose}
                className="p-1 hover:bg-deep-teal/10 rounded-full transition-colors"
              >
                <X className="w-6 h-6 text-deep-teal" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-4xl font-bold text-deep-teal mb-2">
                  {tempSpeed} cm/s
                </div>
                <p className="text-deep-teal/70">
                  Adjust the speed of the drone (10-100 cm/s)
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-deep-teal/70">
                  <span>Slow (10)</span>
                  <span>Fast (100)</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={tempSpeed}
                  onChange={(e) => setTempSpeed(Number(e.target.value))}
                  className="w-full h-3 bg-deep-teal/20 rounded-lg appearance-none slider cursor-pointer"
                />
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={handleSpeedModalClose}
                  className="flex-1 px-4 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSpeedApply}
                  className="flex-1 px-4 py-3 bg-deep-teal text-white rounded-lg hover:bg-dark-cyan transition-colors"
                >
                  Apply
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Control