Running the research LangGraph graph locally

This folder contains a LangGraph example (`research`) and a helper to start the LangGraph API server on a different port so it doesn't conflict with existing ClickHouse / Langfuse infra running on `localhost:8123`.

Quick start

1. Ensure you have a Python virtualenv activated and dependencies installed. The example `requirements.txt` lists `langgraph-cli[inmem]`.
2. Edit `.env` in this folder to set your `LANGFUSE_*` keys and `LANGFUSE_HOST` (default: http://localhost:3000).
3. Start the LangGraph API server on port 8124 (default) so it doesn't conflict with other local services:

```bash
./start_langgraph.sh 8124
```

Notes on observability / Langfuse

- The agent already instantiates a `langfuse.langchain.CallbackHandler` (see `research_agent.py`). Langfuse uses the `LANGFUSE_*` env vars in this folder's `.env` to send traces to your Langfuse server.
- We purposefully run LangGraph on a different port so ClickHouse (and other infra) can continue to run on `localhost:8123`.
- If you want your client code to use the LangGraph API SDK against this alternate port, set `LANGGRAPH_API_URL` or pass the `url` when creating a client (the SDK defaults to `http://localhost:8123`).

Example: invoking the research graph via the API SDK (python)

```python
from langgraph_sdk import get_client
client = get_client(url="http://localhost:8124")
resp = client.graphs.invoke_graph_sync('research', input={"messages": [{"role": "user", "content": "Write a short research summary about X"}]})
print(resp)
```

If you need help wiring Langfuse further (OTel, exporters, or server-side traces from the LangGraph runtime), I can add optional instrumentation in the runtime or show how to attach an OTel exporter that forwards spans to Langfuse.
