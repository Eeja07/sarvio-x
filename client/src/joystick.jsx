function VirtualJoystick({ joystickPosition, setJoystickPosition, telloConnected }) {
  const handleJoystickMove = (e) => {
    // Early return jika Tello belum terkoneksi
    if (!telloConnected) {
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const x = ((e.clientX - rect.left - centerX) / centerX * 100);
    const y = ((e.clientY - rect.top - centerY) / centerY * 100);
    const distance = Math.sqrt(x * x + y * y);
    
    if (distance <= 100) {
      setJoystickPosition({ x: x.toFixed(2), y: y.toFixed(2) });
    }
  };

  // Reset posisi joystick ke center ketika mouse leave atau mouse up
  const handleMouseLeave = () => {
    setJoystickPosition({ x: 0, y: 0 });
  };

  const handleMouseUp = () => {
    setJoystickPosition({ x: 0, y: 0 });
  };

  return (
    <div className="relative w-48 h-48 bg-deep-teal rounded-full border-2 border-slate-600">
      <div
        className={`absolute inset-2 bg-dark-cyan rounded-full transition-all duration-200 ${
          telloConnected 
            ? 'cursor-pointer opacity-100' 
            : 'cursor-not-allowed opacity-50 pointer-events-none'
        }`}
        onMouseMove={handleJoystickMove}
        onMouseLeave={handleMouseLeave}
        onMouseUp={handleMouseUp}
      >
        <div
          className={`absolute w-12 h-12 bg-light-blue rounded-full transform -translate-x-1/2 -translate-y-1/2 transition-all duration-200 ${
            telloConnected ? 'opacity-100' : 'opacity-30'
          }`}
          style={{
            left: `${50 + (joystickPosition.x * 0.3)}%`,
            top: `${50 + (joystickPosition.y * 0.3)}%`
          }}
        />
    
      </div>
    </div>
  );
}

export default VirtualJoystick;