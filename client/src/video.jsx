import { useState, useEffect } from 'react'

function Video({ socket, isConnected }) { // Terima socket dan isConnected sebagai props
  const [videoFrame, setVideoFrame] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [mlDetection, setMlDetection] = useState(false)
  const [autoCapture, setAutoCapture] = useState(false)

  useEffect(() => {
    // Hanya setup event listeners jika socket ada
    if (!socket) return

    // Video frame handler
    const handleVideoFrame = (data) => {
      setVideoFrame(`data:image/jpeg;base64,${data.frame}`)
    }

    // Tello status handler
    const handleTelloStatus = (data) => {
      console.log('Tello status:', data)
    }

    // Setup event listeners
    socket.on('video_frame', handleVideoFrame)
    socket.on('tello_status', handleTelloStatus)

    // Auto-start streaming saat component mount DAN socket connected
    if (isConnected) {
      socket.emit('start_stream')
      setIsStreaming(true)
    }

    // Cleanup event listeners
    return () => {
      socket.off('video_frame', handleVideoFrame)
      socket.off('tello_status', handleTelloStatus)
    }
  }, [socket, isConnected]) // Dependency pada socket dan isConnected

  // Effect terpisah untuk handle connection changes
  useEffect(() => {
    if (!isConnected) {
      setIsStreaming(false)
      setVideoFrame(null)
    }
  }, [isConnected])

  const handleCapture = () => {
    if (videoFrame) {
      const link = document.createElement('a')
      link.href = videoFrame
      link.download = `tello_capture_${new Date().getTime()}.jpg`
      link.click()
    }
  }

  const toggleRecording = () => {
    console.log('Recording toggle - to be implemented')
  }

  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold text-cyan-400 mb-4">Live Stream & ML Detection</h2>
      
      {/* Video Stream Area */}
      <div className="bg-slate-900 rounded-lg h-64 flex items-center justify-center mb-6 relative">
        {videoFrame ? (
          <img 
            src={videoFrame} 
            alt="Tello Live Stream" 
            className="max-w-full max-h-full object-contain rounded-lg"
          />
        ) : (
          <div className="text-center text-slate-400">
            <div className="w-16 h-16 mx-auto mb-4 opacity-30">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z" />
              </svg>
            </div>
            <p className="text-lg font-medium">
              {isConnected ? 'Waiting for Stream' : 'No Connection'}
            </p>
            <p className="text-sm">
              {isConnected ? 'Video stream starting...' : 'Connect drone to start streaming'}
            </p>
          </div>
        )}
        
        {/* Connection Status Indicator */}
        <div className="absolute top-2 right-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
        </div>
        
        {/* Stream Status */}
        {isConnected && (
          <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
            {isStreaming ? 'LIVE' : 'PAUSED'}
          </div>
        )}
      </div>

      {/* ML Detection */}
      <div className="space-y-4">
        <div>
          <h3 className="text-white font-medium mb-3">ML Detection</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input 
                type="checkbox" 
                className="rounded bg-slate-700 border-slate-600"
                checked={mlDetection}
                onChange={(e) => setMlDetection(e.target.checked)}
              />
              <span className="text-slate-300">YOLOv8 Detection</span>
              <span className="text-xs text-slate-500">(Coming Soon)</span>
            </label>
            <label className="flex items-center space-x-2">
              <input 
                type="checkbox" 
                className="rounded bg-slate-700 border-slate-600"
                checked={autoCapture}
                onChange={(e) => setAutoCapture(e.target.checked)}
              />
              <span className="text-slate-300">Auto Capture (3s)</span>
            </label>
          </div>
        </div>

        {/* Capture Controls */}
        <div>
          <h3 className="text-white font-medium mb-3">Capture Controls</h3>
          <div className="flex gap-2">
            <button 
              onClick={handleCapture}
              disabled={!videoFrame}
              className={`flex-1 p-2 rounded transition-colors ${
                videoFrame 
                  ? 'bg-cyan-600 hover:bg-cyan-700 text-white' 
                  : 'bg-slate-600 text-slate-400 cursor-not-allowed'
              }`}
            >
              Capture
            </button>
            <button 
              onClick={toggleRecording}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white p-2 rounded transition-colors"
            >
              Record
            </button>
          </div>
        </div>

        {/* Camera Settings */}
        <div>
          <h3 className="text-white font-medium mb-3">Camera Settings</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm text-slate-400 mb-1">
                <span>Zoom</span>
                <span>1.0x</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="5" 
                step="0.1" 
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider" 
              />
            </div>
            <div>
              <div className="flex justify-between text-sm text-slate-400 mb-1">
                <span>Brightness</span>
                <span>0</span>
              </div>
              <input 
                type="range" 
                min="-100" 
                max="100" 
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider" 
              />
            </div>
            <div>
              <div className="flex justify-between text-sm text-slate-400 mb-1">
                <span>Exposure</span>
                <span>0</span>
              </div>
              <input 
                type="range" 
                min="-100" 
                max="100" 
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider" 
              />
            </div>
            <div>
              <label className="text-sm text-slate-400">Quality</label>
              <select className="w-full mt-1 bg-slate-700 border-slate-600 text-white rounded p-2">
                <option>640x480 (Default)</option>
                <option>960x720 (HD)</option>
                <option>1280x720 (720p)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Connection Info */}
        <div className="bg-slate-900 rounded-lg p-3">
          <div className="text-xs text-slate-400 space-y-1">
            <div className="flex justify-between">
              <span>Backend:</span>
              <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Stream:</span>
              <span className={isStreaming ? 'text-green-400' : 'text-slate-400'}>
                {isStreaming ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Server:</span>
              <span>localhost:5000</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Video