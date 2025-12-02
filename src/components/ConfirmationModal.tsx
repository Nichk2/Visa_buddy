// components/ConfirmationModal.tsx
import React, { useEffect } from "react";
import ReactDOM from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Trash2 } from "lucide-react";
import type { ConfirmationModalProps } from "../types/visa_buddy";

const ConfirmationModal: React.FC<ConfirmationModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title = "Confirm Deletion",
  message = "Are you sure you want to delete this chat? This action cannot be undone.",
  confirmText = "Delete",
  cancelText = "Cancel",
  isDestructive = true,
}) => {
  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  // Handle Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const modalContent = (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              {/* Icon */}
              <div className="flex justify-center mb-4">
                <div
                  className={`w-12 h-12 rounded-full ${
                    isDestructive ? "bg-red-50" : "bg-blue-50"
                  } flex items-center justify-center`}
                >
                  {isDestructive ? (
                    <Trash2 size={24} className="text-red-500" />
                  ) : (
                    <AlertTriangle size={24} className="text-blue-500" />
                  )}
                </div>
              </div>

              {/* Title */}
              <h3 className="text-xl font-semibold text-gray-900 text-center mb-2">
                {title}
              </h3>

              {/* Message */}
              <p className="text-gray-600 text-center mb-6">{message}</p>

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-2.5 border-2 border-gray-200 text-gray-700 font-medium cursor-pointer rounded-full hover:bg-gray-50 transition-colors"
                >
                  {cancelText}
                </button>
                <button
                  onClick={() => {
                    onConfirm();
                    onClose();
                  }}
                  className={`flex-1 px-4 py-2.5 font-medium rounded-full cursor-pointer transition-colors ${
                    isDestructive
                      ? "bg-red-400 text-white hover:bg-red-500"
                      : "bg-blue-500 text-white hover:bg-blue-600"
                  }`}
                >
                  {confirmText}
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );

  return ReactDOM.createPortal(modalContent, document.body);
};

export default ConfirmationModal;
