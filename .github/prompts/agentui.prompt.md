---
mode: agent
---

# create a UI to interact with the agent 
- create ui using copilotkit
- use the AG-UI protocol to stream the agent messages back to the ui.
- create the UI application separate folder under the mainworkspace as `ui`

# reference
- [CopilotKit Documentation](https://github.com/microsoft/codex)
- [AG-UI Protocol](https://github.com/ag-ui-protocol/ag-ui)

# Agent communication
- The agent is written in langgraph in the same repository.
- The agent communicates with the UI using the AG-UI protocol.
- The aim to invoke the agent via the ui interface is to provide a seamless user experience and enable efficient interaction with the agent's capabilities.