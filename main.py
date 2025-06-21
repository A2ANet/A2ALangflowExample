import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.middleware.cors import CORSMiddleware

from agent_executor import LangflowAgentExecutor

# Configuration - these must be set via environment variables
LANGFLOW_URL = os.getenv("LANGFLOW_URL")
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY")
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST", "0.0.0.0")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT", "9999"))

# Validate required environment variables
if not LANGFLOW_URL:
    raise ValueError("LANGFLOW_URL environment variable is required")

# Define the agent skill
skill = AgentSkill(
    id="langflow-chat",
    name="Langflow Chat",
    description="Process messages using Langflow AI workflows",
    tags=["chat", "langflow", "ai"],
    examples=["Hello, how are you?", "What is the weather like?", "Tell me a joke"],
)

# Define the agent card
agent_card = AgentCard(
    name="Langflow Agent",
    description="An A2A agent that processes messages using Langflow AI workflows",
    url=f"http://localhost:{A2A_SERVER_PORT}/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=AgentCapabilities(),
    skills=[skill],
)

# Create the request handler with the Langflow agent executor
request_handler: DefaultRequestHandler = DefaultRequestHandler(
    agent_executor=LangflowAgentExecutor(
        langflow_url=LANGFLOW_URL,
        api_key=LANGFLOW_API_KEY
    ), 
    task_store=InMemoryTaskStore()
)

# Create the A2A server
server: A2AStarletteApplication = A2AStarletteApplication(
    agent_card=agent_card, 
    http_handler=request_handler
)

# Build the server and wrap it with CORS middleware
app = server.build()

app = CORSMiddleware(
    app=app,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host=A2A_SERVER_HOST, port=A2A_SERVER_PORT)
    