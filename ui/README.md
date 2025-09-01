# deepagents UI

This is a minimal Next.js UI to interact with the DeepAgents LangGraph agent using the AG-UI protocol.

Assumptions:
- The agent exposes an AG-UI compatible websocket endpoint at `ws://localhost:8000/ag-ui/ws`.
- You have Node.js installed (v18+) and will run `npm install` in the `ui` folder.

Quickstart:

```bash
cd ui
npm install
npm run dev
```

The frontend will attempt to connect to the AG-UI websocket and stream messages to/from the agent.

Next steps:
- Wire the AG-UI server url and authentication if needed.
- Replace the example websocket URL in `pages/index.tsx` with your agent's actual endpoint.
