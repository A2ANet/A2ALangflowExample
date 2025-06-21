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
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT"))

# Validate required environment variables
if not LANGFLOW_URL:
    raise ValueError("`LANGFLOW_URL` environment variable is required")
if not A2A_SERVER_HOST:
    raise ValueError("`A2A_SERVER_HOST` environment variable is required")
if not A2A_SERVER_PORT:
    raise ValueError("`A2A_SERVER_PORT` environment variable is required")

# Define the agent skills based on the Simple Agent capabilities
url_skill = AgentSkill(
    id="url-content-retrieval",
    name="URL Content Retrieval",
    description="Fetch and retrieve data from URLs. Supports plain text, raw HTML, or JSON output formats with cleaning options.",
    tags=["web", "url", "scraping", "data-retrieval"],
    examples=[
        "Get the content from https://example.com", 
        "Fetch data from this URL: https://api.example.com/data",
        "What's on this webpage: https://news.example.com"
    ],
)

calculator_skill = AgentSkill(
    id="arithmetic-calculator",
    name="Arithmetic Calculator",
    description="Perform basic arithmetic operations on mathematical expressions including addition, subtraction, multiplication, division, and exponentiation.",
    tags=["math", "calculator", "arithmetic", "computation"],
    examples=[
        "Calculate 4*4*(33/22)+12-20",
        "What is 15 + 27 * 3?",
        "Solve: (100 - 25) / 5 + 10^2"
    ],
)

general_chat_skill = AgentSkill(
    id="general-assistant",
    name="General Assistant",
    description="General conversational AI assistant that can help with questions, provide information, and perform various tasks using available tools.",
    tags=["chat", "assistant", "general", "conversation"],
    examples=[
        "Hello, how are you?",
        "Can you help me with a question?",
        "I need assistance with a task"
    ],
)

# Define the agent card
agent_card = AgentCard(
    name="Simple Agent",
    description="A simple but powerful starter agent that can fetch content from URLs, perform arithmetic calculations, and provide general assistance. The agent intelligently decides which tool to use based on your request.",
    url=f"http://localhost:{A2A_SERVER_PORT}/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=AgentCapabilities(),
    skills=[url_skill, calculator_skill, general_chat_skill],
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
    