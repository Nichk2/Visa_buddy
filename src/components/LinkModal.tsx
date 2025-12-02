
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type {LinkModalProps} from "../types/visa_buddy"


const LinkModal: React.FC<LinkModalProps> = ({ 
  isOpen, 
  onClose,
  apiUrl = '',
  connectionStatus = 'disconnected',
  onUpdateUrl,
  onClearChat
}) => {
  const [url, setUrl] = useState(apiUrl);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
      setUrl(apiUrl); // Update local state when modal opens
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose, apiUrl]);

  const handleSave = () => {
    if (onUpdateUrl) {
      onUpdateUrl(url);
    }
    onClose();
  };

  const handleClear = () => {
    setUrl('');
    if (onUpdateUrl) {
      onUpdateUrl('');
    }
    if (onClearChat) {
      onClearChat();
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'border-green-500 bg-green-500';
      case 'error':
        return 'border-red-500 bg-red-500';
      default:
        return 'border-yellow-500 bg-yellow-500';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Operating';
      case 'error':
        return 'Connection Error';
      default:
        return 'Checking Connection';
    }
  };

  const getStatusTextColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-yellow-600';
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-20">
          <motion.div
            className="absolute inset-0 bg-black/30"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
          />
          <motion.div
            className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4">
              <h3 className="text-lg font-semibold text-gray-900">API Configuration</h3>
              <button
                onClick={onClose}
                className="w-8 h-8 flex items-center cursor-pointer text-3xl justify-center text-gray-500 hover:text-gray-700 transition-colors"
              >
                ×
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Colab API URL
                </label>
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://your-tunnel-url.loca.lt"
                  className="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-red-300 focus:border-transparent"
                />
              </div>

              {/* Status Indicator */}
              <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                <div className={`w-3 h-3 rounded-full ${getStatusColor()} ${connectionStatus === 'connected' ? 'animate-pulse' : connectionStatus === 'disconnected' ? 'animate-pulse' : ''}`}>
                </div>
                <div className="flex-1">
                  <div className="text-xs text-gray-600">Connection Status</div>
                  <div className={`text-sm font-medium ${getStatusTextColor()}`}>
                    {getStatusText()}
                  </div>
                </div>
              </div>

              {/* Info Box */}
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800">
                  <strong>Note:</strong> Enter your Colab tunnel URL to connect with the Visa Buddy AI backend. 
                  
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-2">
                <button
                  onClick={handleClear}
                  className="flex-1 px-4 py-2.5 bg-gray-100 cursor-pointer hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-4xl transition-colors"
                >
                  Clear & Reset
                </button>
                <button
                  onClick={handleSave}
                  className="flex-1 px-4 py-2.5 bg-red-400 hover:bg-red-500 cursor-pointer text-white text-sm font-medium rounded-4xl transition-all ease-in"
                >
                  Save & Connect
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default LinkModal;