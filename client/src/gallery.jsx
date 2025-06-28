import { useState, useEffect } from 'react'
import { X, Download, Eye, Calendar, Camera, Video, Trash2 } from 'lucide-react'

function Gallery({ isOpen, onClose, socket, isConnected }) {
  const [activeTab, setActiveTab] = useState('Images')
  const [mediaFiles, setMediaFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [selectedMedia, setSelectedMedia] = useState(null)
  const [previewModal, setPreviewModal] = useState(false)
  const [error, setError] = useState(null)

  // Fetch media files - WORKS WITHOUT DRONE CONNECTION
  const fetchMediaFiles = async () => {
    if (!isOpen) return
    
    setLoading(true)
    setError(null)
    
    try {
      if (socket && isConnected) {
        // Use socket if available
        console.log('📁 Fetching media via socket...')
        socket.emit('get_media_files', { type: activeTab.toLowerCase() })
      } else {
        // Fallback to direct API call
        console.log('📁 Fetching media via API fallback...')
        const response = await fetch(`http://localhost:5000/api/media/list?type=${activeTab.toLowerCase()}`)
        const data = await response.json()
        
        if (data.success) {
          setMediaFiles(data.files || [])
          console.log(`📁 Loaded ${data.files.length} files via API`)
        } else {
          throw new Error(data.error || 'Failed to fetch media files')
        }
        setLoading(false)
      }
    } catch (error) {
      console.error('Error fetching media files:', error)
      setError(`Failed to load media files: ${error.message}`)
      setMediaFiles([])
      setLoading(false)
    }
  }

  // Setup socket listeners
  useEffect(() => {
    if (!socket) return

    const handleMediaFiles = (data) => {
      console.log('📁 Received media files:', data)
      if (data.success) {
        setMediaFiles(data.files || [])
        console.log(`📁 Loaded ${data.files.length} files via socket`)
      } else {
        setError(data.error || 'Failed to load media files')
        setMediaFiles([])
      }
      setLoading(false)
    }

    const handleMediaDeleted = (data) => {
      if (data.success) {
        console.log('🗑️ File deleted, refreshing list')
        fetchMediaFiles()
      } else {
        setError(data.error || 'Failed to delete file')
      }
    }

    const handleDownloadReady = (data) => {
      if (data.success) {
        // Create download link and trigger download
        const link = document.createElement('a')
        link.href = data.url
        link.download = data.filename
        link.target = '_blank'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        console.log('📥 Download started:', data.filename)
      } else {
        setError(data.error || 'Failed to prepare download')
      }
    }

    socket.on('media_files_response', handleMediaFiles)
    socket.on('media_deleted', handleMediaDeleted)
    socket.on('download_ready', handleDownloadReady)

    return () => {
      socket.off('media_files_response', handleMediaFiles)
      socket.off('media_deleted', handleMediaDeleted)
      socket.off('download_ready', handleDownloadReady)
    }
  }, [socket])

  // Fetch files when modal opens or tab changes
  useEffect(() => {
    if (isOpen) {
      fetchMediaFiles()
    }
  }, [isOpen, activeTab])

  const handleDownload = (filename) => {
    console.log('📥 Download request:', filename)
    
    if (socket && isConnected) {
      // Use socket if available
      socket.emit('download_media', { filename })
    } else {
      // Direct download fallback
      const link = document.createElement('a')
      link.href = `http://localhost:5000/download/${filename}`
      link.download = filename
      link.target = '_blank'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const handleDelete = (filename) => {
    if (window.confirm(`Are you sure you want to delete ${filename}?`)) {
      console.log('🗑️ Delete request:', filename)
      
      if (socket && isConnected) {
        socket.emit('delete_media', { filename })
      } else {
        setError('Delete function requires backend connection')
      }
    }
  }

  const handlePreview = (file) => {
    console.log('👁️ Preview request:', file.filename)
    setSelectedMedia(file)
    setPreviewModal(true)
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (!isOpen) return null

  return (
    <>
      {/* Background Blur Overlay */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        {/* Modal Container */}
        <div 
          className="bg-powder-blue rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-light-blue p-6 flex items-center justify-between border-b border-deep-teal/20">
            <div>
              <h2 className="text-3xl font-bold text-deep-teal">Media Gallery</h2>
              {!isConnected && (
                <p className="text-sm text-orange-600 mt-1">
                  ⚠️ Backend offline - Limited functionality
                </p>
              )}
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-deep-teal/10 rounded-full transition-colors"
            >
              <X className="w-8 h-8 text-deep-teal" />
            </button>
          </div>

          {/* Tabs */}
          <div className="bg-light-blue px-6">
            <div className="flex space-x-1">
              {['Images', 'Videos'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-3 text-lg font-medium rounded-t-lg transition-colors ${
                    activeTab === tab
                      ? 'bg-powder-blue text-deep-teal border-b-2 border-deep-teal'
                      : 'text-deep-teal/70 hover:text-deep-teal hover:bg-powder-blue/50'
                  }`}
                >
                  {tab === 'Images' ? (
                    <div className="flex items-center space-x-2">
                      <Camera className="w-5 h-5" />
                      <span>{tab}</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <Video className="w-5 h-5" />
                      <span>{tab}</span>
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 mx-6 mt-4 rounded">
              <strong>Error:</strong> {error}
              <button 
                onClick={() => setError(null)}
                className="float-right text-red-700 hover:text-red-900"
              >
                ×
              </button>
            </div>
          )}

          {/* Content Area */}
          <div className="p-6 h-[60vh] overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-deep-teal mx-auto mb-4"></div>
                  <p className="text-deep-teal text-lg">Loading media files...</p>
                </div>
              </div>
            ) : mediaFiles.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-deep-teal/10 flex items-center justify-center">
                    {activeTab === 'Images' ? (
                      <Camera className="w-12 h-12 text-deep-teal/50" />
                    ) : (
                      <Video className="w-12 h-12 text-deep-teal/50" />
                    )}
                  </div>
                  <h3 className="text-xl font-semibold text-deep-teal mb-2">
                    No {activeTab.toLowerCase()} found
                  </h3>
                  <p className="text-deep-teal/70">
                    {activeTab === 'Images' 
                      ? 'Take some screenshots to see them here' 
                      : 'Record some videos to see them here'
                    }
                  </p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {mediaFiles.map((file, index) => (
                  <div
                    key={index}
                    className="bg-white rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow"
                  >
                    {/* Thumbnail */}
                    <div className="relative aspect-video bg-gray-100">
                      {activeTab === 'Images' ? (
                        <img
                          src={file.url}
                          alt={file.filename}
                          className="w-full h-full object-cover cursor-pointer"
                          onClick={() => handlePreview(file)}
                          onError={(e) => {
                            e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlPC90ZXh0Pjwvc3ZnPg=='
                          }}
                        />
                      ) : (
                        <div 
                          className="w-full h-full bg-gray-200 flex items-center justify-center cursor-pointer"
                          onClick={() => handlePreview(file)}
                        >
                          <Video className="w-12 h-12 text-gray-400" />
                        </div>
                      )}
                      
                      {/* Overlay buttons */}
                      <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center space-x-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handlePreview(file)
                          }}
                          className="p-2 bg-white rounded-full hover:bg-gray-100 transition-colors"
                          title="Preview"
                        >
                          <Eye className="w-5 h-5 text-deep-teal" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDownload(file.filename)
                          }}
                          className="p-2 bg-white rounded-full hover:bg-gray-100 transition-colors"
                          title="Download"
                        >
                          <Download className="w-5 h-5 text-deep-teal" />
                        </button>
                        {isConnected && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDelete(file.filename)
                            }}
                            className="p-2 bg-white rounded-full hover:bg-gray-100 transition-colors"
                            title="Delete"
                          >
                            <Trash2 className="w-5 h-5 text-red-600" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* File Info */}
                    <div className="p-4">
                      <h4 className="font-semibold text-deep-teal text-sm mb-2 truncate" title={file.filename}>
                        {file.filename}
                      </h4>
                      <div className="space-y-1 text-xs text-deep-teal/70">
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-3 h-3" />
                          <span>{formatDate(file.created_at || file.modified_at)}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>{formatFileSize(file.size)}</span>
                          {file.humans_detected !== undefined && file.humans_detected > 0 && (
                            <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">
                              {file.humans_detected} humans
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="bg-light-blue p-4 border-t border-deep-teal/20">
            <div className="flex items-center justify-between">
              <p className="text-deep-teal/70 text-sm">
                {mediaFiles.length} {activeTab.toLowerCase()} found
              </p>
              <button
                onClick={() => {
                  setError(null)
                  fetchMediaFiles()
                }}
                className="px-4 py-2 bg-deep-teal text-white rounded-lg hover:bg-dark-cyan transition-colors text-sm"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Preview Modal */}
      {previewModal && selectedMedia && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-90 z-60 flex items-center justify-center p-4"
          onClick={() => setPreviewModal(false)}
        >
          <div className="relative max-w-4xl max-h-full">
            <button
              onClick={() => setPreviewModal(false)}
              className="absolute -top-12 right-0 p-2 text-white hover:text-gray-300 transition-colors z-10"
            >
              <X className="w-8 h-8" />
            </button>
            
            {activeTab === 'Images' ? (
              <img
                src={selectedMedia.url}
                alt={selectedMedia.filename}
                className="max-w-full max-h-[80vh] object-contain rounded-lg"
                onClick={(e) => e.stopPropagation()}
                onError={(e) => {
                  console.error('Preview image load error:', selectedMedia.url)
                  e.target.style.display = 'none'
                  e.target.nextSibling.style.display = 'flex'
                }}
              />
            ) : (
              <video
                controls
                className="max-w-full max-h-[80vh] rounded-lg"
                onClick={(e) => e.stopPropagation()}
                onError={(e) => {
                  console.error('Preview video load error:', selectedMedia.url)
                }}
              >
                <source src={selectedMedia.url} type="video/mp4" />
                <source src={selectedMedia.url} type="video/webm" />
                <source src={selectedMedia.url} type="video/ogg" />
                Your browser does not support the video tag.
              </video>
            )}
            
            {/* Error fallback for preview */}
            <div 
              className="hidden w-96 h-64 bg-gray-800 rounded-lg items-center justify-center text-white"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="text-center">
                <div className="text-red-400 mb-2">❌</div>
                <p>Failed to load media</p>
                <button 
                  onClick={() => handleDownload(selectedMedia.filename)}
                  className="mt-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
                >
                  Download instead
                </button>
              </div>
            </div>
            
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white p-4 rounded-b-lg">
              <h3 className="font-semibold">{selectedMedia.filename}</h3>
              <p className="text-sm text-gray-300">
                {formatDate(selectedMedia.created_at || selectedMedia.modified_at)} • {formatFileSize(selectedMedia.size)}
                {selectedMedia.humans_detected > 0 && ` • ${selectedMedia.humans_detected} humans detected`}
              </p>
              <div className="flex space-x-2 mt-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDownload(selectedMedia.filename)
                  }}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
                >
                  Download
                </button>
                {isConnected && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (window.confirm(`Delete ${selectedMedia.filename}?`)) {
                        handleDelete(selectedMedia.filename)
                        setPreviewModal(false)
                      }
                    }}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default Gallery