from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import threading
from deepagents import create_deep_agent
from typing import List, Dict, Any, Callable

import os
import uuid
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()
os.environ["CURL_CA_BUNDLE"] = "/home/solmon/github/questmind/zscaler_root.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/home/solmon/github/questmind/zscaler_root.crt"
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = "/home/solmon/github/questmind/zscaler_root.crt"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_results: int = 5
    topic: str = "general"
    include_raw_content: bool = False

def internet_search(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    include_raw_content: bool = False,
):
    """
    Run a web search using Tavily.
    """
    import os
    from tavily import TavilyClient
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    searchResults = tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return searchResults

research_instructions = """You are an expert researcher. Your job is to conduct thorough research on the internet for a given task, and then use the information gathered to recommend a shopping cart for purchase.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

@app.post("/query")
async def query_agent(request: QueryRequest):
    try:
        invoke_kwargs = {
            "messages": [{"role": "user", "content": request.query}],
            "max_results": request.max_results,
            "topic": request.topic,
            "include_raw_content": request.include_raw_content,
        }

        # For streaming, use agent's stream method
        async def generate():
            for chunk in agent.stream(invoke_kwargs):
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if hasattr(message, 'content'):
                            yield f"data: {json.dumps({'content': message.content})}\n\n"
                await asyncio.sleep(0.1)  # Small delay to simulate streaming

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _stream_worker(invoke_kwargs: Dict[str, Any], send_fn: Callable[[str], None], session_id: str = None):
    """
    Run agent.stream in a separate thread and use send_fn to push serialized messages
    back to the websocket event loop.
    """
    import inspect

    try:
        # notify started
        print(f"[_stream_worker] session={session_id} started")
        send_fn(json.dumps({"session": session_id, "type": "status", "payload": {"status": "started"}}))

        # call agent.stream; it may return a sync iterator or an async generator
        result = agent.stream(invoke_kwargs)

        processed_any = False

        async def _process_async_gen(async_gen):
            nonlocal processed_any
            async for chunk in async_gen:
                processed_any = True
                _handle_chunk(chunk, session_id, send_fn)

        def _process_sync_iter(sync_iter):
            nonlocal processed_any
            for chunk in sync_iter:
                processed_any = True
                _handle_chunk(chunk, session_id, send_fn)

        def _handle_chunk(chunk, session_id, send_fn):
            # chunk is expected to be a dict-like with possible 'messages' key
            if isinstance(chunk, dict) and "messages" in chunk:
                for message in chunk["messages"]:
                    # message may be an object with content attr or a dict
                    text = None
                    additional = None

                    # extract text/content and any additional kwargs that may contain function_call info
                    if hasattr(message, 'content'):
                        try:
                            text = getattr(message, 'content')
                        except Exception:
                            text = None
                        # LangChain/AIMessage often stores extra info here
                        if hasattr(message, 'additional_kwargs'):
                            try:
                                additional = getattr(message, 'additional_kwargs')
                            except Exception:
                                additional = None
                    elif isinstance(message, dict):
                        text = message.get('content') or message.get('text')
                        # capture any other fields for protocol events
                        additional = {k: v for k, v in message.items() if k not in ('content', 'text')}
                    else:
                        # fallback to string representation
                        text = str(message)

                    # If text is empty but there's a function call or other actionable additional info,
                    # emit a protocol event the UI can handle (function_call, metadata, etc.).
                    func_call = None
                    if isinstance(additional, dict):
                        func_call = additional.get('function_call') or additional.get('tool_call') or additional.get('function_call_args')

                    if (not text or (isinstance(text, str) and text.strip() == '')) and func_call:
                        # send structured protocol event for the function call
                        payload = {"session": session_id, "type": "protocol", "payload": {"event": "function_call", "function_call": func_call, "additional": additional}}
                        try:
                            print(f"[_stream_worker] sending function_call protocol session={session_id} func_call={repr(func_call)[:200]}")
                        except Exception:
                            print(f"[_stream_worker] sending function_call protocol session={session_id} (func_call truncated)")
                        send_fn(json.dumps(payload))
                    else:
                        # Normal agent message (may be empty string but UI will receive it)
                        payload = {"session": session_id, "type": "agent_message", "payload": {"text": text or ''}}
                        # Debug: log outgoing payload for visibility during development
                        try:
                            print(f"[_stream_worker] sending agent_message session={session_id} text={repr(text)[:200]}")
                        except Exception:
                            print(f"[_stream_worker] sending agent_message session={session_id} (text truncated)")
                        # also include additional metadata as a protocol event if present
                        if additional:
                            try:
                                send_fn(json.dumps({"session": session_id, "type": "protocol", "payload": {"event": "metadata", "metadata": additional}}))
                            except Exception:
                                pass
                        send_fn(json.dumps(payload))
            else:
                # If the chunk contains nested agent/tool messages, try to extract and forward them
                if isinstance(chunk, dict) and ("agent" in chunk or "tools" in chunk):
                    nested_msgs = []
                    try:
                        if "agent" in chunk and isinstance(chunk["agent"], dict):
                            nested_msgs.extend(chunk["agent"].get("messages", []) or [])
                    except Exception:
                        pass
                    try:
                        if "tools" in chunk and isinstance(chunk["tools"], dict):
                            nested_msgs.extend(chunk["tools"].get("messages", []) or [])
                    except Exception:
                        pass

                    for message in nested_msgs:
                        text = None
                        if hasattr(message, 'content'):
                            text = getattr(message, 'content')
                        elif isinstance(message, dict) and 'content' in message:
                            text = message['content']
                        else:
                            text = str(message)

                        payload = {"session": session_id, "type": "agent_message", "payload": {"text": text}}
                        try:
                            print(f"[_stream_worker] extracted nested agent_message session={session_id} text={repr(text)[:200]}")
                        except Exception:
                            print(f"[_stream_worker] extracted nested agent_message session={session_id} (text truncated)")
                        send_fn(json.dumps(payload))

                # send raw chunk as protocol event, but ensure it's JSON-serializable
                raw = chunk
                try:
                    json.dumps(raw)
                except Exception:
                    raw = str(chunk)
                # Debug: log raw protocol events
                try:
                    print(f"[_stream_worker] sending protocol session={session_id} raw={repr(raw)[:200]}")
                except Exception:
                    print(f"[_stream_worker] sending protocol session={session_id} (raw truncated)")
                send_fn(json.dumps({"session": session_id, "type": "protocol", "payload": {"raw": raw}}))

        if inspect.isasyncgen(result):
            # run async generator in this thread's event loop
            try:
                asyncio.run(_process_async_gen(result))
            except Exception as e:
                print("[_stream_worker] async processing error:", e)
                send_fn(json.dumps({"session": session_id, "type": "error", "payload": {"error": str(e)}}))
        elif inspect.isawaitable(result):
            # result is a coroutine that returns an async generator
            try:
                async_gen = asyncio.run(result)
                asyncio.run(_process_async_gen(async_gen))
            except Exception as e:
                print("[_stream_worker] awaitable processing error:", e)
                send_fn(json.dumps({"session": session_id, "type": "error", "payload": {"error": str(e)}}))
        else:
            # assume sync iterator
            try:
                _process_sync_iter(result)
            except Exception as e:
                print("[_stream_worker] sync processing error:", e)
                send_fn(json.dumps({"session": session_id, "type": "error", "payload": {"error": str(e)}}))

        # if nothing was produced, send a helpful fallback so the UI isn't empty
        if not processed_any:
            warn_msg = (
                "The agent produced no streaming output. This commonly happens when the model is not configured "
                "or credentials are missing. Check server logs for details."
            )
            send_fn(json.dumps({"session": session_id, "type": "protocol", "payload": {"event": "no_output"}}))
            send_fn(json.dumps({"session": session_id, "type": "agent_message", "payload": {"text": warn_msg}}))

        # notify finished
        print(f"[_stream_worker] session={session_id} finished processed_any={processed_any}")
        send_fn(json.dumps({"session": session_id, "type": "status", "payload": {"status": "finished"}}))

    except Exception as e:
        # ensure any exception is reported back to the client
        try:
            print(f"[_stream_worker] unhandled error: {e}")
            send_fn(json.dumps({"session": session_id, "type": "error", "payload": {"error": str(e)}}))
        except Exception:
            # if sending the error fails, there's not much we can do from the thread
            pass


@app.websocket("/ag-ui/ws")
async def ag_ui_ws(websocket: WebSocket):
    """
    Simple AG-UI compatible websocket endpoint.

    Protocol (minimal):
    - Client -> server: { type: 'user_input', payload: { text: '...' } }
    - Server -> client: { type: 'agent_message', payload: { text: '...' } }
      plus status/protocol/error messages.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    # active tasks may be asyncio.Tasks or background Threads
    active_tasks: List[Any] = []

    def make_send_fn():
        # capture the running loop and websocket for thread-safe sends
        def send_from_thread(data: str):
            try:
                asyncio.run_coroutine_threadsafe(websocket.send_text(data), loop)
            except Exception:
                # if sending fails, ignore — websocket may be closed
                pass
        return send_from_thread

    try:
        while True:
            msg_text = await websocket.receive_text()
            try:
                msg = json.loads(msg_text)
            except Exception:
                # ignore non-json
                continue

            if msg.get('type') == 'user_input':
                user_text = msg.get('payload', {}).get('text', '')
                print(user_text)
                if not user_text:
                    continue

                invoke_kwargs = {
                    "messages": [{"role": "user", "content": user_text}],
                    # other kwargs may be added here
                }

                # create a session id for this run so messages can be correlated
                session_id = str(uuid.uuid4())
                invoke_kwargs["session_id"] = session_id

                send_fn = make_send_fn()

                # run the blocking agent.stream synchronously inside a background Thread
                # This ensures `agent.stream` is called directly (synchronously) and can yield
                # streaming outputs that we forward to the websocket via send_fn.
                thread = threading.Thread(target=_stream_worker, args=(invoke_kwargs, send_fn, session_id), daemon=True)
                thread.start()
                active_tasks.append(thread)

                # notify client that the worker was scheduled
                try:
                    send_fn(json.dumps({"session": None, "type": "protocol", "payload": {"event": "worker_scheduled"}}))
                except Exception as e:
                    print("[ws] failed to send worker_scheduled protocol message:", e)

            elif msg.get('type') == 'reset':
                # reset conversation (client requested)
                # implementing reset is agent-specific; for now send status
                await websocket.send_text(json.dumps({"type": "status", "payload": {"status": "reset"}}))

            else:
                # unsupported messages can be ignored or echoed
                await websocket.send_text(json.dumps({"type": "protocol", "payload": {"received": msg}}))

    except WebSocketDisconnect:
        # client disconnected; cancel background tasks
        for t in active_tasks:
            try:
                # asyncio Tasks can be cancelled; threads cannot be reliably cancelled.
                if hasattr(t, 'cancel'):
                    t.cancel()
            except Exception:
                pass
        return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
