import React from 'react'
import { Copy, Check } from 'lucide-react'
import './Message.css'

export default function Message({ role, content, citations, timestamp }) {
  const [copied, setCopied] = React.useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const formatTime = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className={`message message-${role}`}>
      <div className="message-content">
        <div className="message-text">
          {content}
        </div>
        
        {citations && citations.length > 0 && (
          <div className="message-citations">
            <div className="citations-header">
              <span>Sources</span>
            </div>
            <div className="citations-list">
              {citations.map((citation, idx) => (
                <div key={idx} className="citation-item">
                  <div className="citation-rank">{citation.rank}</div>
                  <div className="citation-info">
                    <div className="citation-source">{citation.source}</div>
                    {citation.text && (
                      <div className="citation-text">{citation.text}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="message-footer">
          <span className="message-time">{formatTime(timestamp)}</span>
          {role === 'assistant' && (
            <button 
              className="copy-btn"
              onClick={handleCopy}
              title="Copy message"
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
