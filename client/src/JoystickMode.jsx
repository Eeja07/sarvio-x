import React from 'react';
import { RotateCcw, RotateCw, ArrowUp, ArrowDown, Plane, Settings, PlaneLanding, AlertTriangle } from "lucide-react";
import VirtualJoystick from './joystick';

function JoystickMode(props) {
  const {
    telloConnected,
    isFlying,
    handleFlip,
    handleEmergency,
    handleTakeoff,
    onSpeedButtonClick,
    handleLand,
    joystickEnabled,
    leftJoystickPosition,
    setLeftJoystickPosition,
    rightJoystickPosition,
    setRightJoystickPosition,
  } = props;

  return (
    <div className="space-y-4">
      <div className="rounded-xl p-4 w-full h-30 bg-deep-teal">
        <div className="flex flex-wrap justify-center items-center gap-2">
          <div className="flex gap-1">
            <button onClick={() => handleFlip('left')} disabled={!telloConnected || !isFlying} className={`w-10 h-10 rounded-xl flex flex-col items-center justify-center transition-colors ${telloConnected && isFlying ? 'bg-blue-600 text-ivory hover:bg-blue-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><RotateCcw className="w-6 h-6 mb-1" /></button>
            <button onClick={() => handleFlip('right')} disabled={!telloConnected || !isFlying} className={`w-10 h-10 rounded-xl flex flex-col items-center justify-center transition-colors ${telloConnected && isFlying ? 'bg-blue-600 text-ivory hover:bg-blue-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><RotateCw className="w-6 h-6 mb-1" /></button>
          </div>
          <button onClick={handleEmergency} disabled={!telloConnected} className={`w-50 h-10 rounded-xl flex items-center justify-center gap-2 transition-colors ${telloConnected ? 'bg-red-600 text-white hover:bg-red-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><AlertTriangle className="w-6 h-6" /></button>
          <div className="flex gap-1">
            <button onClick={() => handleFlip('up')} disabled={!telloConnected || !isFlying} className={`w-10 h-10 rounded-xl flex flex-col items-center justify-center transition-colors ${telloConnected && isFlying ? 'bg-blue-600 text-ivory hover:bg-blue-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><ArrowUp className="w-6 h-6 mb-1" /></button>
            <button onClick={() => handleFlip('down')} disabled={!telloConnected || !isFlying} className={`w-10 h-10 rounded-xl flex flex-col items-center justify-center transition-colors ${telloConnected && isFlying ? 'bg-blue-600 text-ivory hover:bg-blue-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><ArrowDown className="w-6 h-6 mb-1" /></button>
          </div>
        </div>
        <div className="flex justify-center gap-2 p-4">
          <button onClick={handleTakeoff} disabled={!telloConnected || isFlying} className={`w-10 h-10 rounded-full flex flex-col items-center justify-center transition-colors ${telloConnected && !isFlying ? 'bg-green-600 text-ivory hover:bg-green-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><Plane className="w-6 h-6 mb-1" /></button>
          <button onClick={onSpeedButtonClick} disabled={!telloConnected} className={`w-10 h-10 rounded-full flex flex-col items-center justify-center transition-colors ${telloConnected ? 'bg-blue-600 text-ivory hover:bg-blue-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><Settings className="w-6 h-6 mb-1" /></button>
          <button onClick={handleLand} disabled={!telloConnected || !isFlying} className={`w-10 h-10 rounded-full flex flex-col items-center justify-center transition-colors ${telloConnected && isFlying ? 'bg-orange-600 text-ivory hover:bg-orange-700' : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}><PlaneLanding className="w-6 h-6 mb-1" /></button>
        </div>
      </div>
      {joystickEnabled && (
        <div className="flex flex-col md:flex-row justify-center md:justify-between items-center p-2 gap-2 px-2 md:px-15">
          <div className="text-center">
            <VirtualJoystick joystickPosition={leftJoystickPosition} setJoystickPosition={setLeftJoystickPosition} telloConnected={telloConnected} />
          </div>
          <div className="text-center">
            <VirtualJoystick joystickPosition={rightJoystickPosition} setJoystickPosition={setRightJoystickPosition} telloConnected={telloConnected} />
          </div>
        </div>
      )}
    </div>
  );
}

export default JoystickMode;
