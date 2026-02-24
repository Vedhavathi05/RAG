import React, { useState, useEffect } from 'react'
import { getConversation } from '../api'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import './ChatWindow.css'

export default function ChatWindow({ conversation, onConversationUpdate }) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (conversation) {
      loadMessages()
    } else {
      setMessages([])
    }
  }, [conversation])

  const loadMessages = async () => {
    try {
      const response = await getConversation(conversation.id)
      setMessages(response.data.messages)
    } catch (error) {
      console.error('Failed to load messages:', error)
    }
  }

  const handleSendMessage = async (message) => {
    setLoading(true)
    try {
      // Optimistically add user message
      setMessages(prev => [...prev, {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
        citations: null
      }])

      // Send to backend
      await onConversationUpdate({ ...conversation })
      
      // Reload conversation to get updated messages
      await loadMessages()
    } catch (error) {
      console.error('Failed to send message:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-window">
      {!conversation ? (
        <div className="chat-empty">
          <div className="empty-container">
            <h1>RAG Chat</h1>
            <p>Select a conversation or create a new one to start chatting</p>
          </div>
        </div>
      ) : (
        <>
          <div className="chat-header">
            <h2>{conversation.title}</h2>
          </div>
          <MessageList messages={messages} />
          <MessageInput 
            onSendMessage={handleSendMessage}
            conversationId={conversation.id}
            loading={loading}
          />
        </>
      )}
    </div>
  )
}
