import React from 'react';
import { RotateCcw, RotateCw, ArrowUp, ArrowDown, Plane, Settings, PlaneLanding, AlertTriangle } from "lucide-react";

function ButtonMode(props) {
  const {
    telloConnected,
    isFlying,
    handleFlip,
    handleEmergency,
    handleTakeoff,
    onSpeedButtonClick,
    handleLand,
    handleButtonPress,
    handleButtonRelease,
    pressedButton,
  } = props;

  return (
    <div className="space-y-2">
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
      <div className="rounded-lg w-full h-53 bg-deep-teal">
        <div className="p-5 flex flex-col md:flex-row justify-center md:justify-between items-center gap-2 md:gap-1 px-2 md:px-4 w-full max-w-xs md:max-w-full mx-auto">
          {/* LEFT GRID: Forward/Backward + Left/Right Movement */}
          <div className="grid grid-cols-3 grid-rows-3 gap-0.5 w-fit">
            <div></div>
            <button onMouseDown={() => handleButtonPress('forward')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'forward' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>↑</button>
            <div></div>
            <button onMouseDown={() => handleButtonPress('left')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'left' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>⟵</button>
            <div className="w-14 h-14"></div>
            <button onMouseDown={() => handleButtonPress('right')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'right' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>⟶</button>
            <div></div>
            <button onMouseDown={() => handleButtonPress('backward')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'backward' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>↓</button>
            <div></div>
          </div>
          {/* RIGHT GRID: Up/Down + Yaw Left/Right Movement */}
          <div className="grid grid-cols-3 grid-rows-3 gap-0.5 w-fit">
            <div></div>
            <button onMouseDown={() => handleButtonPress('up')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'up' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>⇈</button>
            <div></div>
            <button onMouseDown={() => handleButtonPress('yaw_left')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'yaw_left' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>⇇</button>
            <div className="w-14 h-14"></div>
            <button onMouseDown={() => handleButtonPress('yaw_right')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'yaw_right' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>⇉</button>
            <div></div>
            <button onMouseDown={() => handleButtonPress('down')} onMouseUp={handleButtonRelease} onMouseLeave={handleButtonRelease} disabled={!telloConnected || !isFlying} className={`w-14 h-14 text-base rounded-full flex items-center justify-center transition-colors select-none ${telloConnected && isFlying ? `${pressedButton === 'down' ? 'bg-blue-800 scale-95' : 'bg-blue-600'} text-ivory hover:bg-blue-700 active:bg-blue-800` : 'bg-dark-cyan text-deep-teal cursor-not-allowed'}`}>⇊</button>
            <div></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ButtonMode;
