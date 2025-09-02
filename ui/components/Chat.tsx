import { useEffect, useRef, useState } from 'react'
import { connectAgUi, AgUiClient } from '../lib/agui'

type Message = { id: string; role: 'user'|'assistant'|'system'; text: string }

export default function Chat(){
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
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
          // Append assistant messages. Support both plain string messages and
          // structured message objects emitted by the client. Handle new AG-UI
          // schema where a message may be: { session, type: 'agent_message', payload: { text: '...', todos: [...] } }
          const id = String(Date.now())
          try{
            if(m && typeof m === 'object'){
              // determine role
              const role = (m.role === 'user' || m.role === 'assistant' || m.role === 'system') ? m.role : (m.type === 'user' ? 'user' : 'assistant')

              // extract text from a few possible locations
              let textVal: string | null = null

              // prefer top-level text if present
              if('text' in m && typeof (m as any).text === 'string'){
                textVal = (m as any).text
              }

              // handle AG-UI agent_message shape with payload
              if(!textVal && 'payload' in m && m.payload){
                const p: any = (m as any).payload
                if(typeof p === 'string'){
                  textVal = p
                }else if(typeof p.text === 'string'){
                  textVal = p.text
                }else if(Array.isArray(p.todos) || Array.isArray(p.todo_list) || Array.isArray(p.items)){
                  // inject a structured payload marker so the renderer can show a todo-list UI
                  const todos = p.todos || p.todo_list || p.items
                  textVal = JSON.stringify({ __todo: todos })
                }else{
                  // fallback to serializing the payload so it's visible in chat
                  textVal = JSON.stringify(p)
                }
              }

              // if still no text, serialize the message object
              if(!textVal){
                textVal = JSON.stringify(m)
              }

              setMessages(prev=>[...prev, { id, role, text: String(textVal) }])
            }else{
              setMessages(prev=>[...prev, { id, role: 'assistant', text: String(m) }])
            }
          }catch(err){
            // fallback to string
            setMessages(prev=>[...prev, { id, role: 'assistant', text: String(m) }])
          }
        })
        client.on('status', (s)=>{
          if(s === 'started') setThinking(true)
          if(s === 'finished' || s === 'reset' || s === 'error') setThinking(false)
        })

        // listen for structured search results emitted by agui client
        client.on('search_results', (payload)=>{
          const id = String(Date.now())
          const summaryText = payload.summary || `Found ${payload.results.length} results`
          // add a system message that a rich result block follows
          setMessages(prev=>[...prev, { id, role: 'system', text: summaryText }])

          // prepare a concise reasoning paragraph based on top 3 results
          const top = (payload.results || []).slice(0,3)
          let reasoning = ''
          if(top.length === 1){
            reasoning = `I found one strong match: ${top[0].title || top[0].url}. ${top[0].content ? top[0].content.slice(0,200) : ''}`
          }else if(top.length > 1){
            reasoning = `Top sources include ${top.map((r:any)=> r.title || r.url).join(', ')}. The snippets suggest ${top[0].content?.slice(0,120) || ''}`
          }

          if(reasoning){
            setMessages(prev=>[...prev, { id: id + '-reason', role: 'assistant', text: reasoning }])
          }

          // inject a special message containing the structured results so we can render them
          setMessages(prev=>[...prev, { id: id + '-results', role: 'assistant', text: JSON.stringify({__search_results: payload.results}) }])
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

            {thinking && (
              <div className="msg assistant">
                <div className="bubble thinking"><span className="dot"/> Thinking</div>
              </div>
            )}

            {messages.map(m=> {

              // render special search_results message objects
              let content: any = m.text
              let isResults = false
              try{
                const parsed = JSON.parse(String(m.text))
                if(parsed && parsed.__search_results){
                  isResults = true
                  content = parsed.__search_results
                }
              }catch(e){/* not json */}

              // detect todo list messages in multiple forms:
              // 1) JSON encoded payload marker {"__todo": [...]}
              // 2) human-readable "Updated todo list to [...]" strings
              let isTodo = false
              let todoItems: any[] | null = null
              try{
                // try structured marker first
                const parsed = JSON.parse(String(m.text))
                if(parsed && parsed.__todo && Array.isArray(parsed.__todo)){
                  isTodo = true
                  todoItems = parsed.__todo
                }
              }catch(e){
                // not structured JSON, fallback to legacy marker parsing
                try{
                  const marker = 'Updated todo list to '
                  if(typeof m.text === 'string' && m.text.startsWith(marker)){
                    const tail = m.text.slice(marker.length).trim()
                    const start = tail.indexOf('[')
                    const end = tail.lastIndexOf(']')
                    if(start !== -1 && end !== -1 && end > start){
                      let arrStr = tail.slice(start, end+1)
                      try{
                        todoItems = JSON.parse(arrStr)
                      }catch(err){
                        try{
                          const normalized = arrStr.replace(/\'/g, '"')
                          todoItems = JSON.parse(normalized)
                        }catch(err2){
                          todoItems = null
                        }
                      }
                      if(Array.isArray(todoItems)) isTodo = true
                    }
                  }
                }catch(e2){ /* ignore */ }
              }

              return (
                <div key={m.id} className={"msg " + (m.role==='user' ? 'user' : m.role==='assistant' ? 'assistant' : 'system')}>
                  <div className="bubble">
                    {isResults ? (
                      Array.isArray(content) ? (
                        <div className="results-grid">
                          {content.map((r:any, idx:number)=> (
                            <div key={idx} className="result-card">
                              <a href={r.url} target="_blank" rel="noreferrer" className="result-title">{r.title || r.url}</a>
                              <p className="result-snippet">{r.content || r.snippet || ''}</p>
                              <div className="result-meta"><span className="score">{(r.score||0).toFixed(2)}</span></div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <pre className="result-json">{JSON.stringify(content, null, 2)}</pre>
                      )
                    ) : isTodo ? (
                      <div className="todo-list">
                        <strong className="todo-title">Actions</strong>
                        <ul>
                          {todoItems!.map((t:any, i:number)=> (
                            <li key={i} className={t.status === 'completed' ? 'done' : (t.status === 'in_progress' ? 'in-progress' : 'pending')}>
                              <span className="todo-content">{t.content}</span>
                              <span className="todo-meta">{t.status ? ` — ${t.status}` : ''}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : (
                      <>{m.text}</>
                    )}
                  </div>
                </div>
              )
            })}
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
