import { useEffect, useRef, useState } from 'react'
import { connectAgUi, AgUiClient } from '../lib/agui'
import AgentMessageTodo from './AgentMessageTodo'
import AgentMessageSearch from './AgentMessageSearch'
import AgentMessageAi from './AgentMessageAi'

type Message = { id: string; role: 'user' | 'assistant' | 'system'; text: string; meta?: any }

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const clientRef = useRef<AgUiClient | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let mounted = true

    async function init() {
      try {
        // Ask the Next.js proxy for the websocket URL (this allows the backend to provide auth/signed URLs)
        const res = await fetch('/api/ag-ui-proxy', { method: 'POST' })
        let wsUrl = 'ws://localhost:8000/ag-ui/ws'
        if (res.ok) {
          const body = await res.json()
          if (body?.wsUrl) wsUrl = body.wsUrl
        }

        if (!mounted) return
        const client = connectAgUi({ url: wsUrl })
        clientRef.current = client

        client.on('open', () => console.log('ag-ui open'))
        client.on('message', (m) => {
          // Append assistant messages. Support both plain string messages and
          // structured message objects emitted by the client. Handle new AG-UI
          // schema where a message may be: { session, type: 'agent_message', payload: { text: '...', todos: [...] } }
          const id = String(Date.now())
          try {
            if (m && typeof m === 'object') {
              // determine role
              const role = (m.role === 'user' || m.role === 'assistant' || m.role === 'system') ? m.role : (m.type === 'user' ? 'user' : 'assistant')

              // If this is the newer agent_message shape, try to capture payload as structured meta
              if (m.type === 'agent_message' && m.payload) {
                const p: any = m.payload
                // preserve original payload as meta and compute a display text fallback
                let textVal: string | null = null
                if (typeof p === 'string') textVal = p
                else if (typeof p.text === 'string') textVal = p.text

                // If payload.text is a string containing JSON, try to parse it and detect structured shapes
                if (typeof p.text === 'string') {
                  try {
                    const inner = JSON.parse(p.text)
                    // inner could be search-data
                    if (inner && inner.query && Array.isArray(inner.results)) {
                      setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __search: true }), meta: { type: 'search', payload: inner } }])
                      return
                    }
                    // inner could be a todo-list
                    if (inner && Array.isArray(inner.todos)) {
                      setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __todo: inner.todos }), meta: { type: 'todo', todos: inner.todos } }])
                      return
                    }
                    // inner could be AI agent message shape
                    if (inner && inner.agent) {
                      setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __ai: true }), meta: { type: 'ai', raw: inner } }])
                      return
                    }
                  } catch (err) { /* not parseable */ }
                }

                // detect todo list directly on payload
                if (!textVal && (Array.isArray(p.todos) || Array.isArray(p.todo_list) || Array.isArray(p.items))) {
                  const todos = p.todos || p.todo_list || p.items
                  setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __todo: todos }), meta: { type: 'todo', todos } }])
                  return
                }

                // detect search-data shape where payload has query & results
                if (!textVal && typeof p === 'object' && p.query && Array.isArray(p.results)) {
                  setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __search: true }), meta: { type: 'search', payload: p } }])
                  return
                }

                // detect AI raw payloads (sometimes in payload.raw)
                if (!textVal && typeof p.raw === 'string') {
                  try {
                    const parsedRaw = JSON.parse(p.raw.replace(/'/g, '"'))
                    if (parsedRaw && parsedRaw.agent) {
                      setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __ai: true }), meta: { type: 'ai', raw: parsedRaw } }])
                      return
                    }
                  } catch (err) { /* ignore */ }
                }

                // default: use textVal or serialized payload
                if (!textVal) textVal = JSON.stringify(p)
                setMessages(prev => [...prev, { id, role, text: String(textVal), meta: { type: 'text', payload: p } }])
                return
              }

              // detect protocol messages that might contain AIMessage raw strings and render them nicely
              if (m.type === 'protocol' && m.payload && typeof m.payload.raw === 'string') {
                try {
                  const parsed = JSON.parse(m.payload.raw.replace(/'/g, '"'))
                  if (parsed && parsed.agent) {
                    setMessages(prev => [...prev, { id, role, text: JSON.stringify({ __ai: true }), meta: { type: 'ai', raw: parsed } }])
                    return
                  }
                } catch (err) { /* ignore */ }
              }

              // legacy handling for messages without agent_message type
              // extract text from a few possible locations
              let textVal: string | null = null
              if ('text' in m && typeof (m as any).text === 'string') {
                textVal = (m as any).text
              }
              if (!textVal && 'payload' in m && m.payload) {
                const p: any = (m as any).payload
                if (typeof p === 'string') {
                  textVal = p
                } else if (typeof p.text === 'string') {
                  textVal = p.text
                } else if (Array.isArray(p.todos) || Array.isArray(p.todo_list) || Array.isArray(p.items)) {
                  const todos = p.todos || p.todo_list || p.items
                  textVal = JSON.stringify({ __todo: todos })
                } else {
                  textVal = JSON.stringify(p)
                }
              }

              if (!textVal) textVal = JSON.stringify(m)
              setMessages(prev => [...prev, { id, role, text: String(textVal) }])
            } else {
              setMessages(prev => [...prev, { id, role: 'assistant', text: String(m) }])
            }
          } catch (err) {
            setMessages(prev => [...prev, { id, role: 'assistant', text: String(m) }])
          }
        })
        client.on('status', (s) => {
          if (s === 'started') setThinking(true)
          if (s === 'finished' || s === 'reset' || s === 'error') setThinking(false)
        })

        client.on('ai_message', (payload) => {
          const id = String(Date.now())
          // setMessages(prev => [...prev, { id, role: 'system', text: payload }])
           
          setMessages(prev => [...prev, { id, role: 'assistant', text: JSON.stringify({ __ai: true }), meta: { type: 'ai', raw: payload } }])
           
        })

        // listen for structured search results emitted by agui client
        client.on('search_results', (payload) => {
          const id = String(Date.now())
          const summaryText = payload.summary || `Found ${payload.results.length} results`
          // add a system message that a rich result block follows
          setMessages(prev => [...prev, { id, role: 'system', text: summaryText }])

          // prepare a concise reasoning paragraph based on top 3 results
          const top = (payload.results || []).slice(0, 3)
          let reasoning = ''
          if (top.length === 1) {
            reasoning = `I found one strong match: ${top[0].title || top[0].url}. ${top[0].content ? top[0].content.slice(0, 200) : ''}`
          } else if (top.length > 1) {
            reasoning = `Top sources include ${top.map((r: any) => r.title || r.url).join(', ')}. The snippets suggest ${top[0].content?.slice(0, 120) || ''}`
          }

          if (reasoning) {
            setMessages(prev => [...prev, { id: id + '-reason', role: 'assistant', text: reasoning }])
          }

          // inject a special message containing the structured results so we can render them
          setMessages(prev => [...prev, { id: id + '-results', role: 'assistant', text: JSON.stringify({ __search_results: payload.results }) }])
        })

      } catch (err) {
        console.error('failed to init ag-ui client', err)
      }
    }

    init()

    return () => { mounted = false; clientRef.current?.close?.() }
  }, [])

  useEffect(() => {
    // auto-scroll to bottom when messages change
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  function send() {
    const text = input.trim()
    if (!text) return
    setMessages(prev => [...prev, { id: String(Date.now()), role: 'user', text }])
    clientRef.current?.sendUserMessage(text)
    setInput('')
  }

  function resetConversation() {
    setMessages([])
    // optionally notify agent
    // clientRef.current?.send({ type: 'reset' })
  }

  return (
    <div className="agent-shell">
      <header className="agent-header">
        <div className="agent-meta">
          <div className="agent-avatar">RA</div>
          <div>
            <div className="agent-title">Research Agent</div>
            <div className="agent-sub">Agentic assistant — General research</div>
          </div>
        </div>
        <div className="agent-actions">
          <button className="btn ghost" onClick={resetConversation}>Reset</button>
        </div>
      </header>

      <section className="agent-body">
        <aside className="agent-side">
          <h4>System</h4>
          <p className="muted">You are connected to a LangGraph agent using AG-UI protocol. Use natural language to interact. Messages stream in real-time.</p>
          <div className="persona">
            <strong>Persona</strong>
            <p className="muted">Helpful research assistant with a focus on agentic deep workflow.</p>
          </div>
        </aside>

        <main className="agent-chat">
          <div ref={scrollRef} className="messages">
            {messages.length === 0 && (
              <div className="empty">No messages yet — start the conversation.</div>
            )}

            {thinking && (
              <div className="msg assistant">
                <div className="bubble thinking"><span className="dot" /> Thinking</div>
              </div>
            )}

            {messages.map(m => {
              // prefer using meta flag set when message was parsed
              const meta = (m as any).meta

              return (
                <div key={m.id} className={"msg " + (m.role === 'user' ? 'user' : m.role === 'assistant' ? 'assistant' : 'system')}>
                  <div className="bubble">
                    {meta && meta.type === 'todo' ? (
                      <AgentMessageTodo items={meta.todos} />
                    ) : meta && meta.type === 'search' ? (
                      <AgentMessageSearch payload={meta.payload} />
                    ) : meta && meta.type === 'ai' ? (
                      <AgentMessageAi raw={meta.raw} />
                    ) : (
                      // fallback: try to detect old markers like __search_results or __todo inside the text
                      (() => {
                        try {
                          const parsed = JSON.parse(String(m.text))
                          if (parsed && parsed.__search_results && Array.isArray(parsed.__search_results)) {
                            return <AgentMessageSearch payload={{ results: parsed.__search_results }} />
                          }
                          if (parsed && parsed.__todo && Array.isArray(parsed.__todo)) {
                            return <AgentMessageTodo items={parsed.__todo} />
                          }
                        } catch (e) {/* not json */ }
                        return <>{m.text}</>
                      })()
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="composer">
            <textarea aria-label="Message" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), send())} placeholder="Ask the agent something..." />
            <div className="composer-actions">
              <button className="btn primary" onClick={send}>Send</button>
            </div>
          </div>
        </main>
      </section>
    </div>
  )
}
