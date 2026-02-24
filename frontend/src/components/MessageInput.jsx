import React, { useState } from 'react'
import { Send, Loader } from 'lucide-react'
import { sendMessage } from '../api'
import './MessageInput.css'

export default function MessageInput({ onSendMessage, conversationId, loading }) {
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)

  const handleSend = async () => {
    if (!input.trim() || isSending) return

    setIsSending(true)
    try {
      await sendMessage(conversationId, input)
      setInput('')
      await onSendMessage(input)
    } catch (error) {
      console.error('Failed to send message:', error)
      alert('Failed to send message. Please try again.')
    } finally {
      setIsSending(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="message-input">
      <div className="input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question here... (Shift+Enter for new line)"
          disabled={isSending || loading}
          rows="1"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isSending || loading}
          className="send-btn"
          title="Send message"
        >
          {isSending ? (
            <Loader size={20} className="spinner" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </div>
      <div className="input-footer">
        <p className="hint">RAG Chat • Your AI Assistant</p>
      </div>
    </div>
  )
}
