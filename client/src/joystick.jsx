function VirtualJoystick({ joystickPosition, setJoystickPosition }) {
  const handleJoystickMove = (e) => {
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

  return (
    <div className="relative w-32 h-32 bg-deep-teal rounded-full border-2 border-slate-600">
      <div 
        className="absolute inset-2 bg-dark-cyan rounded-full cursor-pointer"
        onMouseMove={handleJoystickMove}
      >
        <div 
          className="absolute w-6 h-6 bg-light-blue rounded-full transform -translate-x-1/2 -translate-y-1/2"
          style={{
            left: `${50 + (joystickPosition.x * 0.3)}%`,
            top: `${50 + (joystickPosition.y * 0.3)}%`
          }}
        />
      </div>
      <div className="absolute -bottom-8 left-0 right-0 text-center text-xs text-slate-400">
        <div>X: {joystickPosition.x}</div>
        <div>Y: {joystickPosition.y}</div>
      </div>
    </div>
  );
}

export default VirtualJoystick