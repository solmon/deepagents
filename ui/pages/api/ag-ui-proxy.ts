// This file is a placeholder showing a recommended REST handshake for AG-UI.
// Note: Next.js API routes do not support proxying raw WebSocket connections.
// If you need a true websocket proxy, run a separate Node server (e.g., `ws` or `uWebSockets`) or configure a reverse proxy (nginx) to forward `/ag-ui/ws` to your agent.

import type { NextApiRequest, NextApiResponse } from 'next'

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST'){
    // Example: agent returns a session id or signed ws url
    res.status(200).json({ wsUrl: 'ws://localhost:8000/ag-ui/ws' })
  } else {
    res.status(405).setHeader('Allow','POST').end()
  }
}
