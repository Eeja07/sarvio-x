import { useState, useEffect, useCallback, useRef } from 'react'
import { 
  Gamepad2, 
  Square, 
  Keyboard, 
  Gamepad, 
  Bot,
  Camera,
  Video,
  X
} from "lucide-react";
import {
  leftColumnTextsKeyboard,
  rightColumnTextsKeyboard,
  leftColumnTextsController,
  rightColumnTextsController
} from './controlModeTexts';
import { useGamepad } from './useGamepad';
import JoystickMode from './JoystickMode';
import ButtonMode from './ButtonMode';
import KeyboardMode from './KeyboardMode';
import ControllerMode from './ControllerMode';
import AutonomousMode from './AutonomousMode';
// Mapping mode ke ikon yang sesuai
const modeIcons = {
  'Joystick Mode': Gamepad2,
  'Button Mode': Square,
  'Keyboard Mode': Keyboard,
  'Controller Mode': Gamepad,
  'Autonomous Mode': Bot
};
// const DEV_FORCE_CONNECTED = true; // Set to true for development preview
function Control({ 
  controlMode, 
  setControlMode,
  speed,
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
  setKeyboardEnabled,
  // ...other props
}) {
  const [videoFrame, setVideoFrame] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastScreenshotTime, setLastScreenshotTime] = useState(0)
  const [tempSpeed, setTempSpeed] = useState(speed)
  const [pressedButton, setPressedButton] = useState(null)
  
  // Gamepad hook
  const {
    gamepadConnected,
    detectGamepads,
    pollGamepad
  } = useGamepad({
    socket,
    telloConnected,
    isConnected,
    isFlying,
    speed,
    setHumanDetection,
    setIsRecording,
    onSpeedChange,
  });
  
  const intervalRef = useRef(null)
  const gamepadIntervalRef = useRef(null)
  const keysPressed = useRef(new Set())
  // // DEV: Simulate connected mode for development preview
  // useEffect(() => {
  //   if (DEV_FORCE_CONNECTED) {
  //     setVideoFrame(true);
  //   }
  // },);

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
  
  const handleEmergencyAuto = useCallback(() => {
    if (socket && isConnected) {
      try {
        socket.emit('stop_movement')
        socket.emit('stop_autonomous_mode')
        console.log('🚨 Emergency AUTO command sent')
      } catch (error) {
        console.error('❌ Error sending emergency command:', error)
      }
    }
  }, [socket, isConnected])
  
  const handleLandingAuto = useCallback(() => {
    if (socket && isConnected) {
      try {
        socket.emit('stop_movement')
        socket.emit('land_autonomous_mode')
        console.log('🛬 Landing AUTO command sent')
      } catch (error) {
        console.error('❌ Error sending emergency command:', error)
      }
    }
  }, [socket, isConnected])
  
  const handleStart = useCallback(() => {
    if (socket && telloConnected && isConnected) {
      try {
        socket.emit('start_autonomous_mode')
        console.log('🤖 Autonomous mode started')
      } catch (error) {
        console.error('❌ Error starting autonomous mode:', error)
      }
    }
  }, [socket, telloConnected, isConnected])
  
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
      if (keysPressed.current.has('a')) controls.left_right = -moveSpeed
      if (keysPressed.current.has('d')) controls.left_right = moveSpeed
      if (keysPressed.current.has('w')) controls.for_back = moveSpeed
      if (keysPressed.current.has('s')) controls.for_back = -moveSpeed
      if (keysPressed.current.has('arrowup')) controls.up_down = moveSpeed
      if (keysPressed.current.has('arrowdown')) controls.up_down = -moveSpeed
      if (keysPressed.current.has('arrowleft')) controls.yaw = -moveSpeed
      if (keysPressed.current.has('arrowright')) controls.yaw = moveSpeed
    } else {
      if (keysPressed.current.has('arrowleft')) controls.left_right = -moveSpeed
      if (keysPressed.current.has('arrowright')) controls.left_right = moveSpeed
      if (keysPressed.current.has('arrowup')) controls.for_back = moveSpeed
      if (keysPressed.current.has('arrowdown')) controls.for_back = -moveSpeed
      if (keysPressed.current.has('w')) controls.up_down = moveSpeed
      if (keysPressed.current.has('s')) controls.up_down = -moveSpeed
      if (keysPressed.current.has('a')) controls.yaw = -moveSpeed  
      if (keysPressed.current.has('d')) controls.yaw = moveSpeed  
    }
    
    try {
      socket.emit('move_control', controls)
      console.log(`🎮 Movement (Mode ${keyboardEnabled ? '2' : '1'}):`, controls)
    } catch (error) {
      console.error('❌ Error sending keyboard control:', error)
    }  
  }, [keyboardEnabled, socket, isFlying, telloConnected, speed])
  
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
      case 'forward': controls.for_back = moveSpeed; break
      case 'backward': controls.for_back = -moveSpeed; break
      case 'left': controls.left_right = -moveSpeed; break
      case 'right': controls.left_right = moveSpeed; break
      case 'up': controls.up_down = moveSpeed; break 
      case 'down': controls.up_down = -moveSpeed; break
      case 'yaw_left': controls.yaw = -moveSpeed; break
      case 'yaw_right': controls.yaw = moveSpeed; break
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
  
  const handleButtonPress = useCallback((direction) => {
    if (!socket || !isFlying || !telloConnected) return
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
    
    setPressedButton(direction)
    sendContinuousMovement(direction)
    
    intervalRef.current = setInterval(() => {
      sendContinuousMovement(direction)
    }, 100)
    
    console.log(`🎮 Started holding button: ${direction}`)
  }, [socket, isFlying, telloConnected, sendContinuousMovement])
  
  const handleButtonRelease = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    
    if (pressedButton) {
      console.log(`🎮 Released button: ${pressedButton}`)
    }
    
    setPressedButton(null)
    
    if (socket && socket.connected) {
      try {
        socket.emit('stop_movement')
        console.log('🛑 Stop movement command sent')
      } catch (error) {
        console.error('❌ Error sending stop movement command:', error)
      }
    }
  }, [socket, pressedButton])
  
  const handleKeyDown = useCallback((event) => {
    if (!telloConnected || showSpeedModal) return
    
    const key = event.key.toLowerCase()
    
    if (!keysPressed.current.has(key)) {
      keysPressed.current.add(key)
      
      switch (key) {
        case 't': if (!isFlying) handleTakeoff(); return    
        case 'q': if (isFlying) handleLand(); return    
        case 'o': handleCapture(); return
        case 'p': handleRecord(); return
        case 'z': setHumanDetection(prev => !prev); return
        case 'x': setAutoScreenshot(prev => !prev); return
        case 'v': setJoystickEnabled(prev => !prev); return
        case 'f':
          setKeyboardEnabled(prev => {
            const newMode = !prev
            console.log(`🎮 Keyboard mode switched to: ${newMode ? 'Mode 2 (Arrow Movement)' : 'Mode 1 (WASD Movement)'}`)
            
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
        case 'm': onSpeedChange(speed + 10); return
        case 'n': onSpeedChange(speed - 10); return
        case 'i': if (isFlying) handleFlip('up'); return    
        case 'j': if (isFlying) handleFlip('down'); return    
        case 'k': if (isFlying) handleFlip('left'); return    
        case 'l': if (isFlying) handleFlip('right'); return    
        case 'e': handleEmergency(); return
      }    
      
      const movementKeys = !keyboardEnabled 
        ? ['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']
        : ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 's', 'a', 'd']
      
      if (isFlying && movementKeys.includes(key)) {
        updateMovementFromKeyboard()
      }  
    }  
  }, [
    telloConnected, showSpeedModal, isFlying, keyboardEnabled, speed, socket, isConnected,
    handleTakeoff, handleLand, handleCapture, handleRecord, handleFlip, handleEmergency,
    updateMovementFromKeyboard, setHumanDetection, setAutoScreenshot, setJoystickEnabled,
    setKeyboardEnabled, onSpeedChange
  ])
  
  const handleKeyUp = useCallback((event) => {
    const key = event.key.toLowerCase()
    keysPressed.current.delete(key)
    
    const movementKeys = !keyboardEnabled 
      ? ['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']
      : ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 's', 'a', 'd']
    
    if (isFlying && movementKeys.includes(key)) {
      updateMovementFromKeyboard()
    }  
  }, [isFlying, keyboardEnabled, updateMovementFromKeyboard])
  
  const handleSpeedModalClose = () => {
    setShowSpeedModal(false)
    setTempSpeed(speed)
  }

  const handleSpeedApply = () => {
    onSpeedChange(tempSpeed)
    setShowSpeedModal(false)
  }

  // Gamepad event listeners
  useEffect(() => {
    const handleGamepadConnected = (event) => {
      console.log(`🎮 Gamepad connected: ${event.gamepad.id}`)
      detectGamepads()
    }

    const handleGamepadDisconnected = (event) => {
      console.log(`🎮 Gamepad disconnected: ${event.gamepad.id}`)
      detectGamepads()
    }

    window.addEventListener('gamepadconnected', handleGamepadConnected)
    window.addEventListener('gamepaddisconnected', handleGamepadDisconnected)

    // Initial detection
    detectGamepads()

    return () => {
      window.removeEventListener('gamepadconnected', handleGamepadConnected)
      window.removeEventListener('gamepaddisconnected', handleGamepadDisconnected)
    }
  }, [detectGamepads])

  // Gamepad polling effect
  useEffect(() => {
    if (controlMode === 'Controller Mode' && gamepadConnected) {
      gamepadIntervalRef.current = setInterval(pollGamepad, 16) // ~60fps
      console.log('🎮 Started gamepad polling')
    } else {
      if (gamepadIntervalRef.current) {
        clearInterval(gamepadIntervalRef.current)
        gamepadIntervalRef.current = null
        console.log('🎮 Stopped gamepad polling')
      }
    }

    return () => {
      if (gamepadIntervalRef.current) {
        clearInterval(gamepadIntervalRef.current)
        gamepadIntervalRef.current = null
      }
    }
  }, [controlMode, gamepadConnected, pollGamepad])

  // Keyboard event listeners
  useEffect(() => {
    if (controlMode === 'Keyboard Mode') {
      window.addEventListener('keydown', handleKeyDown)
      window.addEventListener('keyup', handleKeyUp)
    }  

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }  
  }, [controlMode, handleKeyDown, handleKeyUp]) 
  
  useEffect(() => {
    setTempSpeed(speed)
  }, [speed])
  
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      if (gamepadIntervalRef.current) {
        clearInterval(gamepadIntervalRef.current)
        gamepadIntervalRef.current = null
      }
    }
  }, [])
  
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
  
  useEffect(() => {
    if (!isConnected || !telloConnected) {
      setIsStreaming(false)
      setVideoFrame(null)
    }
  }, [isConnected, telloConnected])
  
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
    <div className="p-4 bg-powder-blue text-white rounded-lg shadow-lg h-full flex flex-col">
      <div className="w-full bg-light-blue rounded-lg p-2 mb-4">
        <h2 className="text-2xl font-bold text-deep-teal text-center">CONTROL PANEL</h2>  
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video and Controls Section */}
        <div className="order-2 space-y-6">
          {/* Video Stream */}
          <div className="relative w-full h-90 bg-deep-teal rounded-lg flex items-center justify-center mb-3">
            {videoFrame ? (
              <img
                src={videoFrame}
                alt="Tello Live Stream"
                className="w-full h-full object-cover rounded-lg"
                style={{
                  filter: `brightness(${100 + brightness}%)`,
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
            <JoystickMode
              telloConnected={telloConnected}
              isFlying={isFlying}
              handleFlip={handleFlip}
              handleEmergency={handleEmergency}
              handleTakeoff={handleTakeoff}
              onSpeedButtonClick={onSpeedButtonClick}
              handleLand={handleLand}
              joystickEnabled={joystickEnabled}
              leftJoystickPosition={leftJoystickPosition}
              setLeftJoystickPosition={setLeftJoystickPosition}
              rightJoystickPosition={rightJoystickPosition}
              setRightJoystickPosition={setRightJoystickPosition}
            />
          )}
          
          {controlMode === 'Button Mode' && (
            <ButtonMode
              telloConnected={telloConnected}
              isFlying={isFlying}
              handleFlip={handleFlip}
              handleEmergency={handleEmergency}
              handleTakeoff={handleTakeoff}
              onSpeedButtonClick={onSpeedButtonClick}
              handleLand={handleLand}
              handleButtonPress={handleButtonPress}
              handleButtonRelease={handleButtonRelease}
              pressedButton={pressedButton}
            />
          )}
          
          {controlMode === 'Keyboard Mode' && (
            <KeyboardMode
              leftColumnTextsKeyboard={leftColumnTextsKeyboard}
              rightColumnTextsKeyboard={rightColumnTextsKeyboard}
            />
          )}
          
          {controlMode === 'Controller Mode' && (
            <ControllerMode
              leftColumnTextsController={leftColumnTextsController}
              rightColumnTextsController={rightColumnTextsController}
            />
          )}
          
          {controlMode === 'Autonomous Mode' && (
            <AutonomousMode
              telloConnected={telloConnected}
              handleStart={handleStart}
              handleEmergencyAuto={handleEmergencyAuto}
              handleLandingAuto={handleLandingAuto}
            />
          )}
        </div>
        
        <div className="order-1 space-y-5">
          {/* Control Mode Buttons */}
          <div className="space-y-2">
            {['Joystick Mode', 'Button Mode', 'Keyboard Mode', 'Controller Mode', 'Autonomous Mode'].map((mode) => {
              const IconComponent = modeIcons[mode];
              return (
                <button
                  key={mode}
                  onClick={() => setControlMode(mode)}
                  className={`w-full p-5 text-base font-bold rounded-lg transition-colors flex items-center ${
                    controlMode === mode
                      ? 'bg-deep-teal text-gray-500'
                      : 'bg-deep-teal text-ivory hover:bg-dark-cyan'
                  }`}
                >
                  <IconComponent className="w-5 h-5 mr-2" />
                  <span className="flex-1 text-center">{mode}</span>
                </button>
              );
            })}
          </div>

          {/* NEW REORGANIZED Settings Panel */}
          <div className="bg-deep-teal p-3 rounded-xl text-center space-y-2">
            {/* Detection Settings - Vertical Layout */}
            <div className="space-y-2 text-base">
              <label className={`rounded-lg flex items-center justify-between p-2 space-x-1 transition-colors ${
                telloConnected 
                  ? 'bg-dark-cyan cursor-pointer' 
                  : 'bg-dark-cyan cursor-not-allowed'
              }`}>
                <span className={`${telloConnected ? 'text-ivory' : 'text-gray-400'}`}>Human Detection</span>
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
              <label className={`rounded-lg flex items-center justify-between p-2 space-x-1 transition-colors ${
                telloConnected 
                  ? 'bg-dark-cyan cursor-pointer' 
                  : 'bg-dark-cyan cursor-not-allowed'
              }`}>
                <span className={`${telloConnected ? 'text-ivory' : 'text-gray-400'}`}>Auto Screenshot</span>
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
              <label className={`rounded-lg flex items-center justify-between p-2 space-x-1 transition-colors ${
                controlMode === 'Joystick Mode' 
                  ? 'bg-dark-cyan cursor-pointer' 
                  : 'bg-dark-cyan cursor-not-allowed'
              }`}>
                <span className={`${controlMode === 'Joystick Mode' ? 'text-ivory' : 'text-gray-400'}`}>Joystick Control</span>
                <input
                  type="checkbox"
                  disabled={controlMode !== 'Joystick Mode'}
                  className={`w-4 h-4 rounded transition-colors ${
                    controlMode === 'Joystick Mode'
                      ? 'text-light-blue bg-deep-teal border-light-blue focus:ring-light-blue cursor-pointer'
                      : 'bg-light-blue border-gray-400 cursor-not-allowed'
                  }`}
                  checked={controlMode === 'Joystick Mode' ? joystickEnabled : false}
                  onChange={(e) => controlMode === 'Joystick Mode' && setJoystickEnabled(e.target.checked)}
                />
              </label>
            </div>
            {/* Capture and Record Buttons - Vertical Layout */}
            <div className="space-y-1 text-base">
              <button
                onClick={handleCapture}
                disabled={!telloConnected && !videoFrame}
                className={`w-full p-2 rounded-lg transition-colors flex items-center justify-center gap-1 ${
                  (telloConnected || videoFrame)
                    ? 'bg-dark-cyan text-ivory hover:bg-deep-teal'     
                    : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
                }`}
                title={telloConnected ? 'Take screenshot via backend' : 'Download current frame'}
              >
                <Camera className="w-4 h-4" />
                <span>Capture</span>
              </button>
              <button
                onClick={handleRecord}
                disabled={!telloConnected}
                className={`w-full p-2 rounded-lg transition-colors flex items-center justify-center gap-1 ${
                  telloConnected  
                    ? `${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-dark-cyan hover:bg-deep-teal'} text-ivory`
                    : 'bg-dark-cyan text-gray-400 cursor-not-allowed' 
                }`}
              >
                <Video className="w-4 h-4" />
                <span>{isRecording ? 'Stop' : 'Record'}</span>
              </button>
            </div>
            {/* Brightness Control */}
            <div>
              <h3 className="text-ivory font-medium text-base mb-1">Brightness</h3>
              <div className="flex items-center gap-1 text-ivory text-base mb-1">
                <span>-100</span>
                <span className="flex-1 text-center text-ivory">{brightness}</span>
                <span>100</span>
              </div>
              <input
                disabled={!videoFrame}
                type="range" min="-100" max="100" value={brightness} onChange={(e) => setBrightness(Number(e.target.value))}
                className={`w-full h-1 bg-dark-cyan rounded-lg appearance-none slider ${
                  videoFrame ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Speed Modal */}
      {showSpeedModal && (
        <div className="fixed inset-0 backdrop-blur-sm bg-opacity-50 z-50 flex items-center justify-center">
          <div className="bg-powder-blue rounded-xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold text-deep-teal">Set Speed</h3>
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
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-deep-teal/70">
                  <span>Slow (10)</span>
                  <span>Fast (100)</span>
                </div>
                <input
                  type="range" min="10" max="100" value={tempSpeed} onChange={(e) => setTempSpeed(Number(e.target.value))}
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