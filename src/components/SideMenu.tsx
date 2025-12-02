// components/SideMenu.tsx
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PanelLeft, MessageCirclePlus, Trash2, Pencil, Check, X } from "lucide-react";
import ConfirmationModal from "./ConfirmationModal";
import Logo from "../images/Logo.svg";
import type { ChatItem } from "../types/visa_buddy";

interface SideMenuProps {
  isOpen: boolean;
  onClose: () => void;
  chats: ChatItem[];
  activeChat: string | null;
  onChatSelect: (chatId: string) => void;
  onNewChat: () => void;
  onDeleteChat: (chatId: string) => void;
  onRenameChat: (chatId: string, newName: string) => void;
}

const SideMenu: React.FC<SideMenuProps> = ({
  isOpen,
  onClose,
  chats,
  activeChat,
  onChatSelect,
  onNewChat,
  onDeleteChat,
  onRenameChat,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = Math.floor((now.getTime() - new Date(date).getTime()) / (1000 * 60 * 60 * 24));
    
    if (diff === 0) return "Today";
    if (diff === 1) return "Yesterday";
    if (diff <= 7) return "7 Days";
    return "Older";
  };

  const groupChatsByDate = () => {
    const groups: { [key: string]: ChatItem[] } = {
      Today: [],
      Yesterday: [],
      "7 Days": [],
      Older: []
    };
    
    chats.forEach(chat => {
      groups[formatDate(chat.createdAt)].push(chat);
    });
    
    return Object.entries(groups).filter(([_, chats]) => chats.length > 0);
  };

  const startEdit = (chat: ChatItem) => {
    setEditingId(chat.id);
    setEditValue(chat.name);
  };

  const saveEdit = () => {
    if (editingId && editValue.trim()) {
      onRenameChat(editingId, editValue.trim());
    }
    setEditingId(null);
    setEditValue("");
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditValue("");
  };

  const handleDeleteClick = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setChatToDelete(chatId);
    setIsDeleteModalOpen(true);
  };

  const handleConfirmDelete = () => {
    if (chatToDelete) {
      onDeleteChat(chatToDelete);
      setChatToDelete(null);
    }
  };

  const getChatName = (chatId: string) => {
    const chat = chats.find(c => c.id === chatId);
    return chat ? chat.name : "this chat";
  };

  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="h-full bg-white border-r border-gray-200 flex flex-col shrink-0 overflow-hidden"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "tween", duration: 0.25, ease: "easeOut" }}
          >
            <div className="w-[280px] h-full flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-5 border-b cursor-pointer border-gray-300">
                <img src={Logo} alt="logo" className="h-8" onClick={onClose} />
                <button
                  onClick={onClose}
                  className="w-9 h-9 flex items-center justify-center cursor-pointer text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <PanelLeft size={20} />
                </button>
              </div>

              {/* New Chat Button */}
              <button
                onClick={onNewChat}
                className="mx-4 my-3 px-4 cursor-pointer py-2.5 bg-white border-2 border-gray-200 hover:bg-gray-50 text-gray-800 rounded-full text-sm font-medium transition-colors flex items-center gap-2 justify-center"
              >
                <MessageCirclePlus size={18} />
                New chat
              </button>

              {/* Chat List */}
              <div className="flex-1 overflow-y-auto px-3 pb-4">
                {groupChatsByDate().map(([dateGroup, groupChats]) => (
                  <div key={dateGroup} className="mb-4">
                    <div className="px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {dateGroup}
                    </div>
                    {groupChats.map((chat) => (
                      <div
                        key={chat.id}
                        className={`group relative mb-1 rounded-lg transition-colors ${
                          activeChat === chat.id ? "bg-gray-200" : "hover:bg-gray-100"
                        }`}
                      >
                        {editingId === chat.id ? (
                          <div className="p-2 flex items-center gap-1">
                            <input
                              type="text"
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") saveEdit();
                                if (e.key === "Escape") cancelEdit();
                              }}
                              className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-300"
                              autoFocus
                            />
                            <button
                              onClick={saveEdit}
                              className="w-7 h-7 flex items-center justify-center bg-red-500 text-white rounded hover:bg-red-600"
                            >
                              <Check size={14} />
                            </button>
                            <button
                              onClick={cancelEdit}
                              className="w-7 h-7 flex items-center justify-center border border-gray-300 rounded hover:bg-gray-100"
                            >
                              <X size={14} />
                            </button>
                          </div>
                        ) : (
                          <div
                            onClick={() => onChatSelect(chat.id)}
                            className="w-full px-3 py-2.5 text-left text-sm rounded-lg cursor-pointer flex items-center justify-between"
                          >
                            <div className={`truncate flex-1 ${activeChat === chat.id ? "font-medium text-gray-900" : "text-gray-700"}`}>
                              {chat.name}
                            </div>
                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  startEdit(chat);
                                }}
                                className="w-6 h-6 flex items-center justify-center cursor-pointer text-gray-500 hover:bg-gray-300 transition-all ease-in rounded"
                              >
                                <Pencil size={12} />
                              </button>
                              <button
                                onClick={(e) => handleDeleteClick(chat.id, e)}
                                className="w-6 h-6 flex items-center justify-center text-red-500 cursor-pointer hover:bg-red-50 transition-all ease-in rounded"
                              >
                                <Trash2 size={12} />
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Confirmation Modal */}
      <AnimatePresence>
      <ConfirmationModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onConfirm={handleConfirmDelete}
        title="Delete Chat"
        message={`Are you sure you want to delete "${getChatName(chatToDelete || '')}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        isDestructive={true}
      />
      </AnimatePresence>
    </>
  );
};

export default SideMenu;