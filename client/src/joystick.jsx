import { useCallback, useRef, useEffect } from 'react'

function VirtualJoystick({ joystickPosition, setJoystickPosition, telloConnected = false }) {
  const joystickRef = useRef(null)
  const isDragging = useRef(false)
  const animationFrame = useRef(null)

  // Improved mouse/touch handling dengan better calculation
  const handlePointerMove = useCallback((e) => {
    // Early return jika Tello belum terkoneksi atau tidak dalam dragging state
    if (!telloConnected || !isDragging.current || !joystickRef.current) {
      return
    }

    // Prevent default untuk menghindari scroll atau selection
    e.preventDefault()

    const rect = joystickRef.current.getBoundingClientRect()
    const centerX = rect.width / 2
    const centerY = rect.height / 2
    
    // Calculate position berdasarkan event type (mouse atau touch)
    let clientX, clientY
    if (e.type.includes('touch')) {
      clientX = e.touches[0]?.clientX || e.changedTouches[0]?.clientX
      clientY = e.touches[0]?.clientY || e.changedTouches[0]?.clientY
    } else {
      clientX = e.clientX
      clientY = e.clientY
    }
    
    const x = ((clientX - rect.left - centerX) / centerX * 100)
    const y = ((clientY - rect.top - centerY) / centerY * 100)
    const distance = Math.sqrt(x * x + y * y)
    
    // Limit movement ke dalam circle (max distance = 100)
    if (distance <= 100) {
      // Use requestAnimationFrame untuk smooth updates
      if (animationFrame.current) {
        cancelAnimationFrame(animationFrame.current)
      }
      
      animationFrame.current = requestAnimationFrame(() => {
        setJoystickPosition({ 
          x: parseFloat(x.toFixed(2)), 
          y: parseFloat(y.toFixed(2)) 
        })
      })
    } else {
      // Constrain ke edge of circle
      const constrainedX = (x / distance) * 100
      const constrainedY = (y / distance) * 100
      
      if (animationFrame.current) {
        cancelAnimationFrame(animationFrame.current)
      }
      
      animationFrame.current = requestAnimationFrame(() => {
        setJoystickPosition({ 
          x: parseFloat(constrainedX.toFixed(2)), 
          y: parseFloat(constrainedY.toFixed(2)) 
        })
      })
    }
  }, [telloConnected, setJoystickPosition])

  // Start dragging
  const handlePointerDown = useCallback((e) => {
    if (!telloConnected) return
    
    isDragging.current = true
    handlePointerMove(e)
    
    // Prevent context menu on long press (mobile)
    e.preventDefault()
  }, [telloConnected, handlePointerMove])

  // Stop dragging dan reset posisi
  const handlePointerUp = useCallback(() => {
    isDragging.current = false
    
    // Cancel any pending animation frame
    if (animationFrame.current) {
      cancelAnimationFrame(animationFrame.current)
      animationFrame.current = null
    }
    
    // Reset ke center dengan animation
    setJoystickPosition({ x: 0, y: 0 })
  }, [setJoystickPosition])

  // Handle pointer leave (mouse leaves area)
  const handlePointerLeave = useCallback(() => {
    if (isDragging.current) {
      handlePointerUp()
    }
  }, [handlePointerUp])

  // Global event listeners untuk mouse/touch up events
  useEffect(() => {
    const handleGlobalPointerUp = () => {
      if (isDragging.current) {
        handlePointerUp()
      }
    }

    const handleGlobalPointerMove = (e) => {
      if (isDragging.current) {
        handlePointerMove(e)
      }
    }

    // Add global listeners untuk handle movement dan release di luar element
    document.addEventListener('mouseup', handleGlobalPointerUp)
    document.addEventListener('mousemove', handleGlobalPointerMove)
    document.addEventListener('touchend', handleGlobalPointerUp)
    document.addEventListener('touchmove', handleGlobalPointerMove, { passive: false })

    return () => {
      document.removeEventListener('mouseup', handleGlobalPointerUp)
      document.removeEventListener('mousemove', handleGlobalPointerMove)
      document.removeEventListener('touchend', handleGlobalPointerUp)
      document.removeEventListener('touchmove', handleGlobalPointerMove)
      
      // Clean up animation frame
      if (animationFrame.current) {
        cancelAnimationFrame(animationFrame.current)
      }
    }
  }, [handlePointerUp, handlePointerMove])

  // Reset position ketika tello disconnected
  useEffect(() => {
    if (!telloConnected) {
      setJoystickPosition({ x: 0, y: 0 })
      isDragging.current = false
    }
  }, [telloConnected, setJoystickPosition])

  // Calculate knob position dengan proper scaling
  const knobStyle = {
    left: `${50 + (joystickPosition.x * 0.35)}%`, // Scale down untuk visual
    top: `${50 + (joystickPosition.y * 0.35)}%`,
    transform: 'translate(-50%, -50%)',
    transition: isDragging.current ? 'none' : 'all 0.2s ease-out'
  }

  return (
    <div className="relative w-48 h-48 bg-deep-teal rounded-full border-2 border-slate-600">
      <div
        ref={joystickRef}
        className={`absolute inset-3 bg-dark-cyan rounded-full transition-all duration-200 select-none ${
          telloConnected 
            ? 'cursor-pointer opacity-100' 
            : 'cursor-not-allowed opacity-50 pointer-events-none'
        }`}
        onMouseDown={handlePointerDown}
        onTouchStart={handlePointerDown}
        onMouseLeave={handlePointerLeave}
        style={{
          touchAction: 'none', // Prevent scrolling on touch devices
          userSelect: 'none'    // Prevent text selection
        }}
      >
        {/* Joystick Knob */}
        <div
          className={`absolute w-12 h-12 bg-light-blue rounded-full border-2 border-white shadow-lg ${
            telloConnected ? 'opacity-100' : 'opacity-30'
          }`}
          style={knobStyle}
        />
        
        {/* Center dot indicator */}
        <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-white rounded-full transform -translate-x-1/2 -translate-y-1/2 opacity-50" />
        
        {/* Directional indicators */}
        {telloConnected && (
          <>
            {/* Top */}
            <div className="absolute top-2 left-1/2 transform -translate-x-1/2 text-white text-xs opacity-50">
              ↑
            </div>
            {/* Bottom */}
            <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-white text-xs opacity-50">
              ↓
            </div>
            {/* Left */}
            <div className="absolute left-2 top-1/2 transform -translate-y-1/2 text-white text-xs opacity-50">
              ←
            </div>
            {/* Right */}
            <div className="absolute right-2 top-1/2 transform -translate-y-1/2 text-white text-xs opacity-50">
              →
            </div>
          </>
        )}
        
      </div>
    </div>
  )
}

export default VirtualJoystick