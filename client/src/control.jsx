import { useState, useEffect, useCallback, useRef } from 'react'
import VirtualJoystick from './joystick'

function Control({ 
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
  battery
}) {
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

  // Joystick movement effect
  useEffect(() => {
    if (controlMode === 'Joystick Mode' && isFlying && socket) {
      const { x, y } = joystickPosition
      
      // Only send if position changed significantly
      if (Math.abs(x - lastJoystickPosition.current.x) > 1 || 
          Math.abs(y - lastJoystickPosition.current.y) > 1) {
        
        const controls = {
          left_right: (x / 100) * speed,
          for_back: -(y / 100) * speed, // Negative because Y is inverted
          up_down: 0,
          yaw: 0
        }
        
        socket.emit('move_control', controls)
        lastJoystickPosition.current = { x, y }
      }
    }
  }, [joystickPosition, controlMode, isFlying, socket, speed])

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

  return (
    <div className="bg-slate-800 rounded-lg p-6 space-y-6">
      <h2 className="text-xl font-semibold text-cyan-400 mb-4">Control Panel</h2>
      
      {/* Connection Status */}
      <div className="bg-slate-900 rounded-lg p-3">
        <div className="text-sm space-y-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Backend:</span>
            <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Tello:</span>
            <span className={telloConnected ? 'text-green-400' : 'text-red-400'}>
              {telloConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Flying:</span>
            <span className={isFlying ? 'text-cyan-400' : 'text-slate-400'}>
              {isFlying ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Battery:</span>
            <span className="text-white">{battery}%</span>
          </div>
        </div>
      </div>
      
      {/* Control Mode */}
      <div>
        <h3 className="text-white font-medium mb-3">Control Mode</h3>
        <div className="space-y-2">
          {['Joystick Mode', 'Keyboard Mode', 'Click Control', 'External Stick', 'Autonomous'].map((mode) => (
            <button
              key={mode}
              onClick={() => setControlMode(mode)}
              disabled={!telloConnected}
              className={`w-full p-2 rounded text-left transition-colors ${
                controlMode === mode 
                  ? 'bg-cyan-500 text-white' 
                  : telloConnected 
                    ? 'bg-slate-700 text-slate-300 hover:bg-slate-600' 
                    : 'bg-slate-600 text-slate-500 cursor-not-allowed'
              }`}
            >
              {mode}
              {mode === 'Keyboard Mode' && (
                <span className="text-xs block text-slate-400">WASD/Arrows + Q/E + Z/C</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Main Commands */}
      <div>
        <h3 className="text-white font-medium mb-3">Main Commands</h3>
        <div className="flex gap-2">
          <button 
            onClick={handleTakeoff}
            disabled={!telloConnected || isFlying}
            className={`flex-1 p-2 rounded transition-colors ${
              (!telloConnected || isFlying)
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isFlying ? 'Flying' : 'Takeoff'}
          </button>
          <button 
            onClick={handleLand}
            disabled={!telloConnected || !isFlying}
            className={`flex-1 p-2 rounded transition-colors ${
              (!telloConnected || !isFlying)
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                : 'bg-orange-600 hover:bg-orange-700 text-white'
            }`}
          >
            Land
          </button>
          <button 
            onClick={handleEmergency}
            disabled={!telloConnected}
            className={`flex-1 p-2 rounded transition-colors ${
              !telloConnected
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 text-white'
            }`}
          >
            Emergency
          </button>
        </div>
      </div>

      {/* Speed Control */}
      <div>
        <h3 className="text-white font-medium mb-3">Speed Control</h3>
        <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
          <span>10%</span>
          <span className="flex-1 text-center">{speed}%</span>
          <span>100%</span>
        </div>
        <input
          type="range"
          min="10"
          max="100"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
        />
      </div>

      {/* Virtual Joystick */}
      {controlMode === 'Joystick Mode' && (
        <div>
          <h3 className="text-white font-medium mb-3">Virtual Joystick</h3>
          <div className="flex justify-center">
            <VirtualJoystick 
              joystickPosition={joystickPosition}
              setJoystickPosition={setJoystickPosition}
            />
          </div>
          {!isFlying && (
            <p className="text-xs text-slate-400 text-center mt-2">
              Takeoff first to enable joystick control
            </p>
          )}
        </div>
      )}

      {/* Rotation */}
      <div>
        <h3 className="text-white font-medium mb-3">Rotation</h3>
        <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
          <span>90°</span>
          <span className="flex-1 text-center">{rotation}°</span>
          <span>180°</span>
        </div>
        <input
          type="range"
          min="90"
          max="180"
          value={rotation}
          onChange={(e) => setRotation(e.target.value)}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
        />
        <div className="flex gap-2 mt-3">
          <button 
            onClick={() => handleRotation('CCW')}
            disabled={!isFlying}
            className={`flex-1 p-2 rounded transition-colors ${
              !isFlying
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                : 'bg-slate-700 hover:bg-slate-600 text-white'
            }`}
          >
            CCW
          </button>
          <button 
            onClick={() => handleRotation('CW')}
            disabled={!isFlying}
            className={`flex-1 p-2 rounded transition-colors ${
              !isFlying
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                : 'bg-slate-700 hover:bg-slate-600 text-white'
            }`}
          >
            CW
          </button>
        </div>
      </div>

      {/* Flip Commands */}
      <div>
        <h3 className="text-white font-medium mb-3">Flip Commands</h3>
        <div className="grid grid-cols-2 gap-2">
          {['Front', 'Back', 'Left', 'Right'].map((direction) => (
            <button 
              key={direction}
              onClick={() => handleFlip(direction)}
              disabled={!isFlying}
              className={`p-2 rounded transition-colors ${
                !isFlying
                  ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                  : 'bg-slate-700 hover:bg-slate-600 text-white'
              }`}
            >
              {direction}
            </button>
          ))}
        </div>
        {!isFlying && (
          <p className="text-xs text-slate-400 text-center mt-2">
            Available when flying
          </p>
        )}
      </div>
    </div>
  )
}

export default Control