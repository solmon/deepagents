// Placeholder CopilotKit integration. Replace with actual CopilotKit imports per docs.

export function createCopilotSession(){
  // Example contract: returns an object with `send` and `on` for streaming
  return {
    send: (msg: string)=> console.log('copilot send', msg),
    on: (ev: string, fn: (...args:any[])=>void)=> console.log('copilot on', ev)
  }
}
