import React from 'react';
import { Bot, AlertTriangle, PlaneLanding } from "lucide-react";

function AutonomousMode(props) {
  const { telloConnected, handleStart, handleEmergencyAuto, handleLandingAuto } = props;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap justify-center items-center gap-2">
        <div className="flex gap-1 p-25 space-x-5">
          <button onClick={handleStart} disabled={!telloConnected} className={`w-32 h-25 rounded-xl flex items-center justify-center gap-2 transition-colors ${telloConnected ? 'bg-green-600 text-ivory hover:bg-green-700' : 'bg-deep-teal text-dark-cyan cursor-not-allowed'}`}><Bot className="w-12 h-12 mb-1" /></button>
          <button onClick={handleEmergencyAuto} disabled={!telloConnected} className={`w-25 h-25 rounded-full flex flex-col items-center justify-center transition-colors ${telloConnected ? 'bg-red-600 text-ivory hover:bg-blue-700' : 'bg-deep-teal text-dark-cyan cursor-not-allowed'}`}><AlertTriangle className="w-12 h-12 mb-1" /></button>
          <button onClick={handleLandingAuto} disabled={!telloConnected} className={`w-32 h-25 rounded-xl flex items-center justify-center gap-2 transition-colors ${telloConnected ? 'bg-orange-600 text-ivory hover:bg-red-700' : 'bg-deep-teal text-dark-cyan cursor-not-allowed'}`}><PlaneLanding className="w-12 h-12" /></button>
        </div>
      </div>
    </div>
  );
}

export default AutonomousMode;
