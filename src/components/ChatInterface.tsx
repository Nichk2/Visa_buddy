import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import SideMenu from "../components/SideMenu";
import LinkModal from "../components/LinkModal";
import InfoModal from "../components/InfoModal"; // Add this import
import { useVisaBuddy } from "../hooks/useVisaBuddy";
import type { Message, ChatItem } from "../types/visa_buddy";
import Maple from "../images/maple.svg"
import Logo from "../images/Logo.svg";
import { PanelLeft, MessageCirclePlus, Link, Send, Info } from "lucide-react";

const STORAGE_KEY = "visabuddy-chats";

const ChatInterface: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isLinkModalOpen, setIsLinkModalOpen] = useState(false);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false); // Add this state
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    messages: visaBuddyMessages,
    loading,
    connectionStatus,
    apiUrl,
    sendMessage: sendVisaBuddyMessage,
    clearChat: clearVisaBuddyChat,
    updateApiUrl,
  } = useVisaBuddy();

  // Load chats on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved).map((chat: any) => ({
          ...chat,
          createdAt: new Date(chat.createdAt),
        }));
        setChats(parsed);
        if (parsed.length > 0) {
          setActiveChat(parsed[0].id);
        }
      } catch (error) {
        console.error("Error loading chats:", error);
      }
    }
  }, []);

  // Save chats whenever they change
  useEffect(() => {
    if (chats.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [chats]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, activeChat]);

  // Sync API messages
  useEffect(() => {
    if (!activeChat || visaBuddyMessages.length === 0) return;
    
    const lastMessage = visaBuddyMessages[visaBuddyMessages.length - 1];
    if (lastMessage.type === "assistant") {
      setChats(prev => prev.map(c => {
        if (c.id === activeChat) {
          const exists = c.messages.some(m => 
            m.type === "ai" && m.text === lastMessage.content
          );
          if (!exists) {
            return { 
              ...c, 
              messages: [...c.messages, { type: "ai", text: lastMessage.content }]
            };
          }
        }
        return c;
      }));
    }
  }, [visaBuddyMessages, activeChat]);

  const handleNewChat = () => {
    clearVisaBuddyChat();
    const newChat: ChatItem = {
      id: Date.now().toString(),
      name: "New chat",
      messages: [],
      createdAt: new Date(),
    };
    setChats(prev => [newChat, ...prev]);
    setActiveChat(newChat.id);
    setInputMessage("");
    setIsMenuOpen(true);
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    let targetChatId = activeChat;

    // Create new chat if none active
    if (!targetChatId) {
      const newChat: ChatItem = {
        id: Date.now().toString(),
        name: inputMessage.slice(0, 30) + (inputMessage.length > 30 ? "..." : ""),
        messages: [],
        createdAt: new Date(),
      };
      setChats(prev => [newChat, ...prev]);
      targetChatId = newChat.id;
      setActiveChat(targetChatId);
      clearVisaBuddyChat();
    }

    const userMessage: Message = { type: "user", text: inputMessage };

    // Add user message
    setChats(prev => prev.map(chat => {
      if (chat.id === targetChatId) {
        return {
          ...chat,
          messages: [...chat.messages, userMessage],
          name: chat.messages.length === 0 
            ? inputMessage.slice(0, 30) + (inputMessage.length > 30 ? "..." : "")
            : chat.name
        };
      }
      return chat;
    }));

    const messageText = inputMessage;
    setInputMessage("");

    // Send to API or mock
    if (connectionStatus === "connected") {
      try {
        await sendVisaBuddyMessage(messageText);
      } catch (error) {
        addMockResponse(messageText, targetChatId);
      }
    } else {
      addMockResponse(messageText, targetChatId);
    }
  };

  const addMockResponse = (message: string, chatId: string) => {
    setTimeout(() => {
      setChats(prev => prev.map(c => {
        if (c.id === chatId) {
          return {
            ...c,
            messages: [...c.messages, {
              type: "ai",
              text: `I understand you're asking about "${message}". As your visa assistant, I can help you with visa requirements, document checklists, and application processes. Could you specify which country's visa you're interested in?`
            }]
          };
        }
        return c;
      }));
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const currentChat = chats.find(c => c.id === activeChat);

  return (
    <div className="w-full h-screen flex bg-white overflow-hidden">
      <SideMenu
        isOpen={isMenuOpen}
        onClose={() => setIsMenuOpen(false)}
        chats={chats}
        activeChat={activeChat}
        onChatSelect={(id) => {
          setActiveChat(id);
          setIsMenuOpen(false);
        }}
        onNewChat={handleNewChat}
        onDeleteChat={(id) => {
          setChats(prev => prev.filter(c => c.id !== id));
          if (activeChat === id) {
            const remaining = chats.filter(c => c.id !== id);
            setActiveChat(remaining.length > 0 ? remaining[0].id : null);
          }
        }}
        onRenameChat={(id, name) => {
          setChats(prev => prev.map(c => c.id === id ? { ...c, name } : c));
        }}
      />

      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="px-5 py-6 bg-white relative z-30 shrink-0">
          <div className="flex items-center justify-between">
            {!isMenuOpen && (
              <div className="flex items-center gap-3">
                <img src={Maple} alt="maple" />
                <div className="flex items-center gap-0 border-2 border-gray-100 rounded-full px-2 py-1.5">
                  <motion.button
                    onClick={() => setIsMenuOpen(true)}
                    className="w-9 h-9 flex items-center justify-center text-gray-700 cursor-pointer hover:bg-gray-100 rounded-full transition-colors"
                    
                  >
                    <PanelLeft size={18} />
                  </motion.button>

                  <motion.button
                    onClick={handleNewChat}
                    className="w-9 h-9 flex items-center justify-center text-gray-700 hover:bg-gray-100 cursor-pointer rounded-full transition-colors"
                    
                  >
                    <MessageCirclePlus size={20} />
                  </motion.button>
                </div>

                <motion.button
                  onClick={() => setIsLinkModalOpen(true)}
                  className="w-10 h-10 flex items-center justify-center text-gray-700  hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
                  
                >
                  <Link size={22} />
                </motion.button>
              </div>
            )}

            {isMenuOpen && (
              <div className="flex items-center gap-3">
                <motion.button
                  onClick={() => setIsLinkModalOpen(true)}
                  className="w-10 h-10 flex items-center justify-center cursor-pointer text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <Link size={22} />
                </motion.button>
              </div>
            )}

            {/* Info Button - Top Right Corner */}
            <motion.button
              onClick={() => setIsInfoModalOpen(true)}
              className="w-10 h-10 flex items-center justify-center text-gray-700 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Info size={22} />
            </motion.button>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="min-h-full flex flex-col items-center justify-center px-4 py-8">
            {!currentChat || currentChat.messages.length === 0 ? (
              <div className="text-center max-w-md w-full">
                <img src={Logo} alt="logo" className="h-24 mx-auto mb-6"  />
                <h1 className="text-[18px] font-normal text-gray-900 mb-6">
                  How can we help you?
                </h1>
              </div>
            ) : (
              <div className="w-full max-w-3xl mx-auto space-y-6">
                <div className="flex items-start gap-3 mb-6">
                  <div className="text-lg font-medium text-gray-900">
                    {currentChat.name}
                  </div>
                </div>
                {currentChat.messages.map((message, index) => (
                  <div key={index} className="space-y-4">
                    <div
                      className={`${
                        message.type === "ai"
                          ? "text-gray-700"
                          : "bg-red-400 rounded-2xl px-5 py-4 text-white"
                      }`}
                    >
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {message.text}
                      </p>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-white border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 text-gray-600">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                        </div>
                        <span className="text-sm">Visa Buddy is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="px-6 mb-24 pb-20 shrink-0">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <input
                type="text"
                placeholder="Write your message"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
                className="w-full px-5 py-3.5 bg-white border-2 border-gray-200 rounded-full text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-red-300 pr-12 disabled:opacity-50 disabled:cursor-not-allowed transition-all placeholder:text-gray-400"
              />
              <button
                onClick={handleSendMessage}
                disabled={loading || !inputMessage.trim()}
                className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 rounded-full disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>

      <LinkModal
        isOpen={isLinkModalOpen}
        onClose={() => setIsLinkModalOpen(false)}
        apiUrl={apiUrl}
        connectionStatus={connectionStatus}
        onUpdateUrl={updateApiUrl}
        onClearChat={clearVisaBuddyChat}
      />

      <InfoModal
        isOpen={isInfoModalOpen}
        onClose={() => setIsInfoModalOpen(false)}
      />
    </div>
  );
};

export default ChatInterface;