import { useEffect, useRef, useState } from 'react'
import { connectAgUi, AgUiClient } from '../lib/agui'

type Message = { id: string; role: 'user'|'assistant'|'system'; text: string }

export default function Chat(){
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const clientRef = useRef<AgUiClient | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(()=>{
    let mounted = true

    async function init() {
      try{
        // Ask the Next.js proxy for the websocket URL (this allows the backend to provide auth/signed URLs)
        const res = await fetch('/api/ag-ui-proxy', { method: 'POST' })
        let wsUrl = 'ws://localhost:8000/ag-ui/ws'
        if(res.ok){
          const body = await res.json()
          if(body?.wsUrl) wsUrl = body.wsUrl
        }

        if(!mounted) return
        const client = connectAgUi({ url: wsUrl })
        clientRef.current = client

        client.on('open', ()=> console.log('ag-ui open'))
        client.on('message', (m)=>{
          // Append assistant messages
          setMessages(prev=>[...prev, { id: String(Date.now()), role: 'assistant', text: String(m) }])
        })

      }catch(err){
        console.error('failed to init ag-ui client', err)
      }
    }

    init()

    return ()=>{ mounted = false; clientRef.current?.close?.() }
  }, [])

  useEffect(()=>{
    // auto-scroll to bottom when messages change
    if(scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  function send(){
    const text = input.trim()
    if(!text) return
    setMessages(prev=>[...prev, { id: String(Date.now()), role: 'user', text }])
    clientRef.current?.sendUserMessage(text)
    setInput('')
  }

  function resetConversation(){
    setMessages([])
    // optionally notify agent
    // clientRef.current?.send({ type: 'reset' })
  }

  return (
    <div className="agent-shell">
      <header className="agent-header">
        <div className="agent-meta">
          <div className="agent-avatar">DA</div>
          <div>
            <div className="agent-title">DeepAgents</div>
            <div className="agent-sub">Agentic assistant — langgraph</div>
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
            <p className="muted">Helpful research assistant with a focus on agentic workflows.</p>
          </div>
        </aside>

        <main className="agent-chat">
          <div ref={scrollRef} className="messages">
            {messages.length === 0 && (
              <div className="empty">No messages yet — start the conversation.</div>
            )}

            {messages.map(m=> (
              <div key={m.id} className={"msg " + (m.role==='user' ? 'user' : m.role==='assistant' ? 'assistant' : 'system')}>
                <div className="bubble">{m.text}</div>
              </div>
            ))}
          </div>

          <div className="composer">
            <textarea aria-label="Message" value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=> e.key==='Enter' && !e.shiftKey && (e.preventDefault(), send())} placeholder="Ask the agent something..." />
            <div className="composer-actions">
              <button className="btn primary" onClick={send}>Send</button>
            </div>
          </div>
        </main>
      </section>
    </div>
  )
}
