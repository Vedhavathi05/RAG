import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  }
})

// Conversations
export const createConversation = (title = "New Conversation") =>
  api.post('/conversations/create', null, { params: { title } })

export const getConversations = () =>
  api.get('/conversations')

export const getConversation = (conversationId) =>
  api.get(`/conversations/${conversationId}`)

export const deleteConversation = (conversationId) =>
  api.delete(`/conversations/${conversationId}`)

// Chat
export const sendMessage = (conversationId, query) =>
  api.post('/chat', { conversation_id: conversationId, query })

export const healthCheck = () =>
  api.get('/health')
