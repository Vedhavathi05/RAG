import React, { useEffect, useRef } from 'react'
import Message from './Message'
import './MessageList.css'

export default function MessageList({ messages }) {
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="no-messages">
          <p>No messages yet. Start the conversation!</p>
        </div>
      ) : (
        messages.map((msg, idx) => (
          <Message 
            key={idx}
            role={msg.role}
            content={msg.content}
            citations={msg.citations}
            timestamp={msg.timestamp}
          />
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  )
}
