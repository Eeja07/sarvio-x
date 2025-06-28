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
} from "lucide-react";

function Sensor({ sensorData }) {
  return (
    <div className="bg-powder-blue rounded-lg p-6 space-y-6 h-[72.5rem]">
      <div className="w-full bg-light-blue rounded-lg p-4 mb-6">
        <h2 className="text-4xl font-bold text-deep-teal text-center">INFORMATION PANEL</h2>  
      </div>
      
      {/* Drone Section */}
      <div className="bg-deep-teal rounded-lg p-6">
        <h3 className="text-ivory text-3xl font-medium mb-3 flex items-center">
          <span className="w-5 h-5 bg-ivory rounded-full mr-2"></span>
          Drone
        </h3>
        <div className="space-y-3 text-m">
          {/* Battery */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2 text-m">
                <Battery className="w-5 h-5 text-ivory" />
                <span className="text-ivory">Battery</span>
              </div>
              <span className="text-ivory">{sensorData.battery}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.battery}%` }}
              />
            </div>
          </div>
          
          {/* Bluetooth */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Bluetooth className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Bluetooth</span>
            </div>
            <span className="text-ivory">{sensorData.bluetooth}</span>
          </div>
          
          {/* FPS */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Monitor className="w-5 h-5 text-ivory" />
              <span className="text-ivory">FPS</span>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Eye className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Human Detection</span>
            </div>
            <span className={`text-ivory ${''}`}>
              {sensorData.humanDetection}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Image className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Amount Screenshot</span>
            </div>
            <span className={`text-ivory ${''}`}>
              {sensorData.amountScreenshoot}
            </span>
          </div>
          
          {/* Flight Time */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Clock className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Flight Time</span>
            </div>
            <span className="text-ivory">{sensorData.flightTime} min</span>
          </div>
          
          {/* WiFi Signal */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2 text-m">
                <Wifi className="w-5 h-5 text-ivory" />
                <span className="text-ivory">WiFi Signal</span>
              </div>
              <span className="text-ivory">{sensorData.wifiSignal}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${sensorData.wifiSignal}%` }}
              />
            </div>
          </div>
          
          {/* SDK Version */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Code className="w-5 h-5 text-ivory" />
              <span className="text-ivory">SDK & SN </span>
            </div>
            <span className="text-ivory">{sensorData.sdkVersion}/{sensorData.serialNumber}</span>
          </div>
        </div>
      </div>
      
      {/* Sensor Section */}
      <div className="bg-deep-teal rounded-lg p-6">
        <h3 className="text-ivory text-3xl font-medium mb-3 flex items-center">
          <span className="w-5 h-5 bg-ivory rounded-full mr-2"></span>
          Sensor
        </h3>
        <div className="space-y-3 text-m">
          {/* Height */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <ArrowUp className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Height</span>
            </div>
            <span className="text-ivory">{sensorData.Height}%</span>
          </div>
          
          {/* Barometer */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Gauge className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Barometer</span>
            </div>
            <span className="text-ivory">{sensorData.barometer}Hpa</span>
          </div>
          
          {/* Temperature */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Thermometer className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Temperature</span>
            </div>
            <span className="text-ivory">{sensorData.temperature}°C</span>
          </div>
          
          {/* IMU Attitude - Pitch */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Compass className="w-5 h-5 text-ivory" />
              <span className="text-ivory">IMU Attitude(Pitch)</span>
            </div>
            <span className="text-ivory">{sensorData.imuAttitude.pitch}°</span>
          </div>
          
          {/* IMU Attitude - Roll */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Compass className="w-5 h-5 text-ivory" />
              <span className="text-ivory">IMU Attitude(Roll)</span>
            </div>
            <span className="text-ivory">{sensorData.imuAttitude.roll}°</span>
          </div>
          
          {/* IMU Attitude - Yaw */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Compass className="w-5 h-5 text-ivory" />
              <span className="text-ivory">IMU Attitude(Yaw)</span>
            </div>
            <span className="text-ivory">{sensorData.imuAttitude.yaw}°</span>
          </div>
          
          {/* Acceleration X */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Move3D className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Acceleration(X)</span>
            </div>
            <span className="text-ivory">{sensorData.acceleration.x} m/s²</span>
          </div>
          
          {/* Acceleration Y */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Move3D className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Acceleration(Y)</span>
            </div>
            <span className="text-ivory">{sensorData.acceleration.y} m/s²</span>
          </div>
          
          {/* Acceleration Z */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Move3D className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Acceleration(Z)</span>
            </div>
            <span className="text-ivory">{sensorData.acceleration.z} m/s²</span>
          </div>
          
          {/* Speed X */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Zap className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Speed(X)</span>
            </div>
            <span className="text-ivory">{sensorData.speed.x} m/s</span>
          </div>
          
          {/* Speed Y */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Zap className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Speed(Y)</span>
            </div>
            <span className="text-ivory">{sensorData.speed.y} m/s</span>
          </div>
          
          {/* Speed Z */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Zap className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Speed(Z)</span>
            </div>
            <span className="text-ivory">{sensorData.speed.z} m/s</span>
          </div>
          
          {/* Distance TOF */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-m">
              <Ruler className="w-5 h-5 text-ivory" />
              <span className="text-ivory">Distance TOF</span>
            </div>
            <span className="text-ivory">{sensorData.distanceTOF}</span>
          </div>
        </div>
      </div>     
    </div>
  )
}

export default Sensor