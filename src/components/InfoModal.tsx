// components/InfoModal.tsx
import React, { useEffect } from "react";
import ReactDOM from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Info, Globe, FileText, HelpCircle, MessageSquare } from "lucide-react";
import Maple from "../images/maple.svg"

interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const InfoModal: React.FC<InfoModalProps> = ({
  isOpen,
  onClose,
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
                <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center">
                  <Info size={24} className="text-blue-500" />
                </div>
              </div>

              {/* Title */}
              <h3 className="text-xl font-semibold text-gray-900 text-center mb-2">
                Welcome to Visa Buddy!
              </h3>

              {/* Welcome Message */}
              <p className="text-gray-600 text-center mb-6">
                Your AI-powered assistant for navigating visa applications to <span className="text-red-500 font-bold">CANADA</span>.
              </p>

              {/* Features List */}
              <div className="space-y-4 mb-6">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 flex items-center justify-center flex-shrink-0">
                    <img src={Maple} alt="maple" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Canada Visa Guidance</h4>
                    <p className="text-sm text-gray-600">Get detailed visa processes</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-green-50 flex items-center justify-center flex-shrink-0">
                    <FileText size={16} className="text-green-500" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Document Checklists</h4>
                    <p className="text-sm text-gray-600">Never miss a required document with our detailed lists</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-purple-50 flex items-center justify-center flex-shrink-0">
                    <HelpCircle size={16} className="text-purple-500" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Application Tips</h4>
                    <p className="text-sm text-gray-600">Learn insider tips to avoid common visa application mistakes</p>
                  </div>
                </div>
              </div>

              {/* How to Use */}
              <div className="bg-gray-50 rounded-xl p-4 mb-6">
                <div className="flex items-center gap-2 mb-3">
                  <MessageSquare size={18} className="text-gray-700" />
                  <h4 className="font-medium text-gray-900">How to Use Visa Buddy</h4>
                </div>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">1.</span>
                    Ask specific questions like "What do I need for a Canada tourist visa?"
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">2.</span>
                    Request document checklists
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">3.</span>
                    Get step-by-step application guidance
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">4.</span>
                    Save important information in separate chats
                  </li>
                </ul>
              </div>

              {/* Action Button */}
              <div className="flex justify-center">
                <button
                  onClick={onClose}
                  className="px-6 py-2.5 bg-red-400 text-white font-medium rounded-full hover:bg-red-500 transition-colors cursor-pointer"
                >
                  Let's go BUDDY!
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

export default InfoModal;