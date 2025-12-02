export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title?: string;
  message?: string;
  confirmText?: string;
  cancelText?: string;
  isDestructive?: boolean;
}

export interface ChatRequest {
  message: string;
  max_length?: number;
}

export interface Message {
  type: "user" | "ai";
  text: string;
}

export interface ChatItem {
  id: string;
  name: string;
  messages: Message[];
  createdAt: Date;
}

export interface LinkModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiUrl?: string;
  connectionStatus?: 'connected' | 'disconnected' | 'error';
  onUpdateUrl?: (url: string) => void;
  onClearChat?: () => void;
}

export interface ChatResponse {
  response: string;
}

export interface ApiHealth {
  status: string;
  model_loaded: boolean;
}

export interface Message {
  type: 'user' | 'ai';
  text: string;
}

export interface Chat {
  id: string;
  name: string;
  messages: Message[];
  createdAt: Date;
}


export interface SideMenuProps {
  isOpen: boolean;
  onClose: () => void;
  chats: ChatItem[];
  activeChat: string | null;
  onChatSelect: (chatId: string) => void;
  onNewChat: () => void;
  onDeleteChat: (chatId: string) => void;
  onRenameChat: (chatId: string, newName: string) => void;
  hoveredChatId: string | null;
  setHoveredChatId: (id: string | null) => void;
}

export type ConnectionStatus = 'disconnected' | 'connected' | 'error';