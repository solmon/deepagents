import React from 'react'

export default function AgentMessageAi({ raw }: { raw: any }){
  // raw might be an object with agent/messages or AIMessage content string
  // Try to extract a human-friendly text and some metadata
  let text: string = ''
  if(!raw) raw = {}
  if(typeof raw === 'string'){
    text = raw
  }else if(typeof raw === 'object'){
    // common shapes: { agent: { messages: [ { content: '...' } ] } }
    if(raw.agent && Array.isArray(raw.agent.messages) && raw.agent.messages[0] && raw.agent.messages[0].content){
      text = raw.agent.messages[0].content
    }else if(raw.content){
      text = raw.content
    }else{
      text = JSON.stringify(raw, null, 2)
    }
  }

  return (
    <div className="ai-message">
      <div className="ai-text" dangerouslySetInnerHTML={{ __html: text.replace(/\n/g, '<br/>') }} />
    </div>
  )
}
