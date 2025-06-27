function Sensor({ sensorData }) {
  return (
    <div className="bg-powder-blue rounded-lg p-6 space-y-6 h-290">
      <div className="w-full bg-light-blue rounded-lg p-4 mb-6">
        <h2 className="text-4xl font-bold text-deep-teal text-center">INFORMATION PANEL</h2>  
      </div>
      <div className="bg-deep-teal rounded-lg p-6">
        <h3 className="text-ivory text-3xl font-medium mb-3 flex items-center">
          <span className="w-5 h-5 bg-ivory rounded-full mr-2"></span>
          Drone
        </h3>
        <div className="space-y-2 text-xl">
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-ivory">Battery</span>
              <span className="text-ivory">{sensorData.battery}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.battery}%` }}
              />
            </div>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Bluetooth</span>
            <span className="text-ivory">{sensorData.bluetooth}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">State</span>
            <span className={`text-ivory ${sensorData.state === 'ON' ? 'text-green-400' : 'text-red-400'}`}>
              {sensorData.state}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Control</span>
            <span className={`text-ivory ${sensorData.control === 'Manual' ? 'text-yellow-400' : 'text-green-400'}`}>
              {sensorData.control}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Flight Time</span>
            <span className="text-ivory">{sensorData.flightTime} min</span>
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-ivory">WiFi Signal</span>
              <span className="text-ivory">{sensorData.wifiSignal}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.wifiSignal}%` }}
              />
            </div>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">SDK Version</span>
            <span className="text-ivory">{sensorData.sdkVersion}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Serial Number</span>
            <span className="text-ivory">{sensorData.serialNumber}</span>
          </div>
        </div>
      </div>
      <div className="bg-deep-teal rounded-lg p-6">
        <h3 className="text-ivory text-3xl font-medium mb-3 flex items-center">
          <span className="w-5 h-5 bg-ivory rounded-full mr-2"></span>
          Sensor
        </h3>
        <div className="space-y-2 text-xl">
          <div className="flex justify-between text-xl mb-1">
            <span className="text-ivory">Height</span>
            <span className="text-ivory">{sensorData.Height}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Barometer</span>
            <span className="text-ivory">{sensorData.barometer}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Temperature</span>
            <span className="text-ivory">{sensorData.temperature}°C</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">IMU Attitude(Pitch)</span>
            <span className="text-ivory">{sensorData.imuAttitude.pitch}°</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">IMU Attitude(Roll)</span>
            <span className="text-ivory">{sensorData.imuAttitude.roll}°</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">IMU Attitude(Yaw)</span>
            <span className="text-ivory">{sensorData.imuAttitude.yaw}°</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Acceleration(X)</span>
            <span className="text-ivory">{sensorData.acceleration.x} m/s²</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Acceleration(Y)</span>
            <span className="text-ivory">{sensorData.acceleration.y} m/s²</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Acceleration(Z)</span>
            <span className="text-ivory">{sensorData.acceleration.z} m/s²</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Speed(X)</span>
            <span className="text-ivory">{sensorData.speed.x} m/s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Speed(Y)</span>
            <span className="text-ivory">{sensorData.speed.y} m/s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Speed(Z)</span>
            <span className="text-ivory">{sensorData.speed.z} m/s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ivory">Distance TOF</span>
            <span className="text-ivory">{sensorData.distanceTOF}</span>
          </div>
        </div>
      </div>     
    </div>
  )
}

export default Sensor