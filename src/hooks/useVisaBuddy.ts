import { useState, useCallback } from 'react';
import type { ChatMessage, ConnectionStatus } from '../types/visa_buddy';
import { visaBuddyApi } from '../services/visaBuddyApi';

export const useVisaBuddy = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [apiUrl, setApiUrl] = useState('');

  const testConnection = useCallback(async (url: string): Promise<boolean> => {
    try {
      visaBuddyApi.setBaseUrl(url);
      await visaBuddyApi.checkHealth();
      setConnectionStatus('connected');
      setApiUrl(url);
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      setConnectionStatus('error');
      return false;
    }
  }, []);

  const sendMessage = useCallback(async (message: string): Promise<void> => {
    if (!message.trim() || connectionStatus !== 'connected') return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await visaBuddyApi.sendMessage(message);
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please check your connection and try again.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }, [connectionStatus]);

  const clearChat = useCallback(() => {
    setMessages([]);
  }, []);

  const updateApiUrl = useCallback(async (url: string) => {
    if (!url) {
      setConnectionStatus('disconnected');
      setApiUrl('');
      return;
    }
    
    await testConnection(url);
  }, [testConnection]);

  return {
    messages,
    loading,
    connectionStatus,
    apiUrl,
    sendMessage,
    clearChat,
    updateApiUrl,
    testConnection
  };
};