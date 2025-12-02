// hooks/useChatManager.ts
import { useState, useEffect } from 'react';
import type { ChatItem, Message } from '../types/visa_buddy';
import { generateChatName } from '../utils/ChatNaming';

export const useChatManager = () => {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [activeChat, setActiveChat] = useState<string | null>(null);

  // Load chats from localStorage
  useEffect(() => {
    const savedChats = localStorage.getItem("visabuddy-chats");
    if (savedChats) {
      const parsedChats = JSON.parse(savedChats).map((chat: any) => ({
        ...chat,
        createdAt: new Date(chat.createdAt),
      }));
      setChats(parsedChats);
      if (parsedChats.length > 0) {
        setActiveChat(parsedChats[0].id);
      }
    }
  }, []);

  // Save chats to localStorage
  useEffect(() => {
    localStorage.setItem("visabuddy-chats", JSON.stringify(chats));
  }, [chats]);

  const createNewChat = (firstMessage?: string): ChatItem => {
    return {
      id: Date.now().toString(),
      name: firstMessage ? generateChatName(firstMessage) : "New chat",
      messages: [],
      createdAt: new Date(),
    };
  };

  const addNewChat = (firstMessage?: string) => {
    const newChat = createNewChat(firstMessage);
    setChats(prev => [newChat, ...prev]);
    setActiveChat(newChat.id);
    return newChat;
  };

  const selectChat = (chatId: string) => {
    setActiveChat(chatId);
  };

  const addMessageToChat = (chatId: string, message: Message) => {
    setChats(prev => prev.map(chat => {
      if (chat.id === chatId) {
        const updatedMessages = [...chat.messages, message];
        
        // Update chat name if it's the first user message
        const chatName = 
          message.type === "user" && chat.messages.length === 0
            ? generateChatName(message.text)
            : chat.name;
        
        return {
          ...chat,
          messages: updatedMessages,
          name: chatName,
        };
      }
      return chat;
    }));
  };

  const getActiveChat = () => {
    return chats.find(chat => chat.id === activeChat);
  };

  return {
    chats,
    activeChat,
    setActiveChat,
    addNewChat,
    selectChat,
    addMessageToChat,
    getActiveChat,
    createNewChat,
  };
};