import { 
  Battery, 
  Bluetooth, 
  Monitor, 
  Eye,
  Image, 
  Clock, 
  Wifi, 
  Code, 
  ArrowUp, 
  Gauge, 
  Thermometer, 
  Compass, 
  Move3D, 
  Zap, 
  Ruler,
  Gauge as SpeedIcon,
} from "lucide-react";

function Sensor({ sensorData }) {
  return (
    <div className="bg-powder-blue rounded-lg p-4 space-y-4 h-full flex flex-col">
      <div className="w-full bg-light-blue rounded-lg p-2 mb-4">
        <h2 className="text-2xl font-bold text-deep-teal text-center">INFORMATION PANEL</h2>  
      </div>
      {/* Drone Section */}
      <div className="bg-deep-teal rounded-lg p-4.5 text-base">
        <h3 className="text-ivory text-xl font-medium mb-2 flex items-center">
          <span className="w-4 h-4 bg-ivory rounded-full mr-2"></span>
          Drone
        </h3>
        <div className="space-y-1 text-[15px]">
          {/* Battery */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center space-x-1 text-[15px]">
                <Battery className="w-4 h-4 text-ivory" />
                <span className="text-ivory">Battery</span>
              </div>
              <span className="text-ivory">{sensorData.battery}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-1">
              <div 
                className="bg-green-500 h-1 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.battery}%` }}
              />
            </div>
          </div>
          {/* Bluetooth */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Bluetooth className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Bluetooth</span>
            </div>
            <span className="text-ivory">{sensorData.bluetooth}</span>
          </div>
          {/* Speed - NEW */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <SpeedIcon className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Speed</span>
            </div>
            <span className="text-ivory">{sensorData.speed} cm/s</span>
          </div>
          {/* FPS */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Monitor className="w-4 h-4 text-ivory" />
              <span className="text-ivory">FPS</span>
            </div>
            <span className="text-ivory">{sensorData.FPS}</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Eye className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Human Detection</span>
            </div>
            <span className={`text-ivory ${sensorData.humanCount > 0 ? 'text-ivory' : ''}`}>{sensorData.humanDetection}</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Image className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Amount Screenshot</span>
            </div>
            <span className={`text-ivory ${''}`}>{sensorData.amountScreenshoot}</span>
          </div>
          {/* Flight Time */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Clock className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Flight Time</span>
            </div>
            <span className="text-ivory">{sensorData.flightTime} min</span>
          </div>
          {/* WiFi Signal */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center space-x-1 text-[15px]">
                <Wifi className="w-4 h-4 text-ivory" />
                <span className="text-ivory">WiFi Signal</span>
              </div>
              <span className="text-ivory">{sensorData.wifiSignal}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-1">
              <div 
                className="bg-green-500 h-1 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.wifiSignal}%` }}
              />
            </div>
          </div>
        </div>
      </div>
      {/* Sensor Section */}
      <div className="bg-deep-teal rounded-lg p-3">
        <h3 className="text-ivory text-xl font-medium mb-2 flex items-center">
          <span className="w-4 h-4 bg-ivory rounded-full mr-2"></span>
          Sensor
        </h3>
        <div className="space-y-1 text-[15px]">
          {/* Height */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <ArrowUp className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Height</span>
            </div>
            <span className="text-ivory">{sensorData.height} cm</span>
          </div>
          {/* Barometer */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Gauge className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Barometer</span>
            </div>
            <span className="text-ivory">{sensorData.barometer} Hpa</span>
          </div>
          {/* Temperature */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Thermometer className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Temperature</span>
            </div>
            <span className="text-ivory">{sensorData.temperature}°C</span>
          </div>
          {/* IMU Attitude - Pitch */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Compass className="w-4 h-4 text-ivory" />
              <span className="text-ivory">IMU Attitude(Pitch)</span>
            </div>
            <span className="text-ivory">{sensorData.imuAttitude.pitch}°</span>
          </div>
          {/* IMU Attitude - Roll */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Compass className="w-4 h-4 text-ivory" />
              <span className="text-ivory">IMU Attitude(Roll)</span>
            </div>
            <span className="text-ivory">{sensorData.imuAttitude.roll}°</span>
          </div>
          {/* IMU Attitude - Yaw */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Compass className="w-4 h-4 text-ivory" />
              <span className="text-ivory">IMU Attitude(Yaw)</span>
            </div>
            <span className="text-ivory">{sensorData.imuAttitude.yaw}°</span>
          </div>
          {/* Acceleration X */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Move3D className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Acceleration(X)</span>
            </div>
            <span className="text-ivory">{sensorData.acceleration.x} m/s²</span>
          </div>
          {/* Acceleration Y */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Move3D className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Acceleration(Y)</span>
            </div>
            <span className="text-ivory">{sensorData.acceleration.y} m/s²</span>
          </div>
          {/* Acceleration Z */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Move3D className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Acceleration(Z)</span>
            </div>
            <span className="text-ivory">{sensorData.acceleration.z} m/s²</span>
          </div>
          {/* Speed X */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Zap className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Speed(X)</span>
            </div>
            <span className="text-ivory">{sensorData.speed_sensor.x} m/s</span>
          </div>
          {/* Speed Y */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Zap className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Speed(Y)</span>
            </div>
            <span className="text-ivory">{sensorData.speed_sensor.y} m/s</span>
          </div>
          {/* Speed Z */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Zap className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Speed(Z)</span>
            </div>
            <span className="text-ivory">{sensorData.speed_sensor.z} m/s</span>
          </div>
          {/* Distance TOF */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1 text-[15px]">
              <Ruler className="w-4 h-4 text-ivory" />
              <span className="text-ivory">Distance TOF</span>
            </div>
            <span className="text-ivory">{sensorData.distanceTOF} cm</span>
          </div>
        </div>
      </div>     
    </div>
  )
}

export default Sensor