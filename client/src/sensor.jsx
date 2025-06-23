function Sensor({ sensorData }) {
  return (
    <div className="bg-slate-800 rounded-lg p-6 space-y-6">
      <h2 className="text-xl font-semibold text-cyan-400 mb-4">Sensor Data</h2>

      {/* Flight Status */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3 flex items-center">
          <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
          Flight Status
        </h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Height</span>
            <span className="text-white">{sensorData.altitude}m</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Flight Time</span>
            <span className="text-white">{sensorData.flightTime}</span>
          </div>
        </div>
      </div>

      {/* Power & Connection */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">Power & Connection</h3>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-400">Battery</span>
              <span className="text-white">{sensorData.battery}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.battery}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-400">WiFi Signal</span>
              <span className="text-white">{sensorData.wifiSignal}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.wifiSignal}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Speed */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">Speed (m/s)</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-slate-400 text-xs">X-Axis</div>
            <div className="text-cyan-400 font-mono">-0.7</div>
          </div>
          <div>
            <div className="text-slate-400 text-xs">Y-Axis</div>
            <div className="text-cyan-400 font-mono">0.1</div>
          </div>
          <div>
            <div className="text-slate-400 text-xs">Z-Axis</div>
            <div className="text-cyan-400 font-mono">0.8</div>
          </div>
        </div>
      </div>

      {/* Environmental */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">Environmental</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Temperature</span>
            <span className="text-white">{sensorData.temperature}°C</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Barometer</span>
            <span className="text-white">{sensorData.barometer} hPa</span>
          </div>
        </div>
      </div>

      {/* Accelerometer */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">Accelerometer (m/s²)</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-slate-400 text-xs">X</div>
            <div className="text-cyan-400 font-mono">{sensorData.accelerometer.x}</div>
          </div>
          <div>
            <div className="text-slate-400 text-xs">Y</div>
            <div className="text-cyan-400 font-mono">{sensorData.accelerometer.y}</div>
          </div>
          <div>
            <div className="text-slate-400 text-xs">Z</div>
            <div className="text-cyan-400 font-mono">{sensorData.accelerometer.z}</div>
          </div>
        </div>
      </div>

      {/* Gyroscope */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">Gyroscope (°)</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-slate-400 text-xs">Pitch</div>
            <div className="text-cyan-400 font-mono">{sensorData.gyroscope.pitch}</div>
          </div>
          <div>
            <div className="text-slate-400 text-xs">Roll</div>
            <div className="text-cyan-400 font-mono">{sensorData.gyroscope.roll}</div>
          </div>
          <div>
            <div className="text-slate-400 text-xs">Yaw</div>
            <div className="text-cyan-400 font-mono">{sensorData.gyroscope.yaw}</div>
          </div>
        </div>
      </div>

      {/* GPS Position */}
      <div className="bg-slate-900 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">GPS Position</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Latitude</span>
            <span className="text-white font-mono">{sensorData.gps.latitude}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Longitude</span>
            <span className="text-white font-mono">{sensorData.gps.longitude}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Sensor