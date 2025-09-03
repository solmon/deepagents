import { parse } from "path"

type EventHandler = (...args: any[]) => void

export class AgUiClient {
  ws: WebSocket
  handlers: Record<string, EventHandler[]> = {}

  constructor(ws: WebSocket) {
    this.ws = ws
    ws.onopen = () => this.emit('open')
    ws.onmessage = (ev) => this.handleRaw(ev.data as string)
    ws.onclose = () => this.emit('close')
    ws.onerror = (e) => this.emit('error', e)
  }

  on(ev: string, fn: EventHandler) {
    (this.handlers[ev] ||= []).push(fn)
  }

  emit(ev: string, ...args: any[]) {
    (this.handlers[ev] || []).forEach(fn => fn(...args))
  }

  handleRaw(raw: string) {
    try {
      const msg = JSON.parse(raw)
      // AG-UI protocol: assume messages with type 'agent_message' carry text stream
      
      if (msg.type === 'agent_message') {
        
        //Check for search_results
        const rawPayload = msg.payload?.text
        let parsed = rawPayload
        if (typeof rawPayload === 'string') {
          try { 
            parsed = JSON.parse(rawPayload) 
            console.log("may be search results or AI message");

            if (parsed && typeof parsed === 'object'){
              if (parsed.results && Array.isArray(parsed.results)) {
                  console.log("emitting search results event");
                  this.emit('search_results', { results: parsed.results, raw: parsed });
                  return;
              }
            }
          } catch (e) {
            console.error("Failed to parse search results:", e);
            this.emit('ai_message', msg.payload?.text || '');
            return    
          }         
        }
        this.emit('message', msg.payload?.text || '')
        return
      }

      // status messages (started/finished) -> emit as 'status' events
      if (msg.type === 'status') {
        this.emit('status', msg.payload?.status)
        return
      }

      // otherwise handle other protocol events
      if(msg.type == 'protocol'){
        // Try to extract structured data from protocol events (e.g., tool search results)
        try {
          const rawPayload = msg.payload?.raw
          let parsed = rawPayload
          if (typeof rawPayload === 'string') {
            try { parsed = JSON.parse(rawPayload) } catch (e) { }
          }

          // If we detect tool results produced as JSON strings in tools.messages, emit a higher-level event
          if (parsed && typeof parsed === 'object') {
            const tools = parsed.tools || parsed.tool || null
            if (tools && Array.isArray(tools.messages || tools.messages)) {
              const msgs = tools.messages || []
              // look for messages with JSON content (common pattern from agent/tool outputs)
              const results = []
              for (const m of msgs) {
                try {
                  const content = m.content
                  if (typeof content === 'string') {
                    const obj = JSON.parse(content)
                    if (obj && obj.results) results.push(...(obj.results || []))
                  }
                } catch (e) { /* ignore parse errors */ }
              }
              if (results.length) {
                this.emit('search_results', { results, raw: parsed })
              }
            }
          }
        } catch (e) { /* ignore */ }

        this.emit('protocol', msg)
      }
    } catch (e) {
      // fallback: raw text
      this.emit('message', raw)
    }
  }

  send(obj: any) {
    this.ws.send(JSON.stringify(obj))
  }

  sendUserMessage(text: string) {
    // Show an immediate initial message in the chat UI while the agent reads data
    // This helps provide feedback to the user that the query is being processed.
    this.emit('message', 'Reading Data')

    // AG-UI user input envelope
    this.send({ type: 'user_input', payload: { text } })
  }

  close() {
    this.ws.close()
  }
}

export function connectAgUi(opts: { url: string }) {
  const ws = new WebSocket(opts.url)
  return new AgUiClient(ws)
}
