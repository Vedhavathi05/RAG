import './App.css'
import { useState, useEffect } from 'react'
import { getConversations, createConversation } from './api'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'

function App() {
  const [conversations, setConversations] = useState([])
  const [currentConversation, setCurrentConversation] = useState(null)
  const [loading, setLoading] = useState(false)

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      const response = await getConversations()
      setConversations(response.data.conversations)
    } catch (error) {
      console.error('Failed to load conversations:', error)
    }
  }

  const handleNewChat = async () => {
    setLoading(true)
    try {
      const response = await createConversation()
      const newConv = response.data
      setConversations([newConv, ...conversations])
      setCurrentConversation(newConv)
    } catch (error) {
      console.error('Failed to create conversation:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSelectConversation = (conv) => {
    setCurrentConversation(conv)
  }

  const handleDeleteConversation = (conversationId) => {
    setConversations(conversations.filter(c => c.id !== conversationId))
    if (currentConversation?.id === conversationId) {
      setCurrentConversation(null)
    }
  }

  const handleConversationUpdate = (updatedConversation) => {
    setCurrentConversation(updatedConversation)
    loadConversations()
  }

  return (
    <div className="app">
      <Sidebar 
        conversations={conversations}
        currentConversation={currentConversation}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        loading={loading}
      />
      <ChatWindow 
        conversation={currentConversation}
        onConversationUpdate={handleConversationUpdate}
      />
    </div>
  )
}

export default App
