import React from 'react'
import { Plus, Trash2 } from 'lucide-react'
import { deleteConversation } from '../api'
import './Sidebar.css'

export default function Sidebar({
  conversations,
  currentConversation,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  loading
}) {
  const handleDelete = async (e, conversationId) => {
    e.stopPropagation()
    try {
      await deleteConversation(conversationId)
      onDeleteConversation(conversationId)
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button 
          className="new-chat-btn"
          onClick={onNewChat}
          disabled={loading}
        >
          <Plus size={20} />
          New chat
        </button>
      </div>

      <div className="sidebar-content">
        <div className="conversations-list">
          {conversations.length === 0 ? (
            <div className="empty-state">
              <p>No conversations yet</p>
              <p className="text-sm">Start a new chat to begin</p>
            </div>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                className={`conversation-item ${currentConversation?.id === conv.id ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv)}
              >
                <div className="conversation-info">
                  <div className="conversation-title">{conv.title}</div>
                  <div className="conversation-preview">{conv.preview}</div>
                </div>
                <button
                  className="conversation-delete"
                  onClick={(e) => handleDelete(e, conv.id)}
                  title="Delete conversation"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="footer-text">RAG Chat v1.0</div>
      </div>
    </div>
  )
}
