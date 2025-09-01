UI setup notes:
- This is a minimal Next.js + copilotkit placeholder. The `copilotkit` package name is illustrative; replace with real package or SDK from CopilotKit docs.
- The frontend expects an AG-UI websocket at `ws://localhost:8000/ag-ui/ws`. Update `lib/agui.ts` if your agent exposes a different URL or auth.

To run:

```bash
cd ui
npm install
npm run dev
```

