export const formatChatDate = (date: Date): string => {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const chatDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  
  const diffTime = today.getTime() - chatDate.getTime();
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays <= 7) return "7 Days";
  return "Older";
};

export const groupChatsByDate = (chats: any[]) => {
  const groups: { [key: string]: any[] } = {
    'Today': [], 'Yesterday': [], '7 Days': [], 'Older': []
  };
  
  chats.forEach(chat => {
    const dateGroup = formatChatDate(chat.createdAt);
    groups[dateGroup].push(chat);
  });
  
  const orderedGroups: { [key: string]: any[] } = {};
  ['Today', 'Yesterday', '7 Days', 'Older'].forEach(key => {
    if (groups[key].length > 0) {
      orderedGroups[key] = groups[key];
    }
  });
  
  return orderedGroups;
};