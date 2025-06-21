import json
import requests
from typing import Dict, Any, Optional
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from loguru import logger


class LangflowAgentExecutor(AgentExecutor):
    """A2A Agent Executor that uses Langflow for processing."""
    
    def __init__(self, langflow_url: str, api_key: Optional[str] = None):
        if not langflow_url:
            raise ValueError("langflow_url is required")
        self.langflow_url = langflow_url.rstrip('/')
        self.api_key = api_key

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent using Langflow."""
        query: str = context.get_user_input()
        task: Task | None = context.current_task

        if not context.message:
            raise Exception("No message in context")

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        try:
            # Send working status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.working,
                        message=new_agent_text_message(
                            "Processing your request with Langflow...", 
                            task.contextId, 
                            task.id
                        ),
                    ),
                    final=False,
                    contextId=task.contextId,
                    taskId=task.id,
                )
            )

            # Call Langflow
            logger.info(f"Executing Langflow with query: {query}")
            response = self._call_langflow(query, session_id=task.contextId)
            
            # Extract the response text
            response_text = self._extract_message_text(response)
            
            # Send the artifact with the response
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="Langflow Response",
                        description=f"Response from Langflow for query: {query}",
                        text=response_text,
                    ),
                    contextId=task.contextId,
                    taskId=task.id,
                )
            )

            # Mark task as completed
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(state=TaskState.completed),
                    final=True,
                    contextId=task.contextId,
                    taskId=task.id,
                )
            )
            
        except Exception as e:
            logger.error(f"Error executing Langflow agent: {e}")
            
            # Send error status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            f"Error processing request: {str(e)}", 
                            task.contextId, 
                            task.id
                        ),
                    ),
                    final=True,
                    contextId=task.contextId,
                    taskId=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution."""
        raise Exception("Cancel not supported for Langflow agent")
    
    def _call_langflow(self, input_value: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a flow with the given input and return the response."""
        
        url = self.langflow_url
        
        payload = {
            "input_value": input_value,
            "output_type": "chat",
            "input_type": "chat",
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = {}
        if session_id:
            params["session_id"] = session_id
            
        try:
            logger.info(f"Calling Langflow API: {url}")
            logger.info(f"Payload: {json.dumps(payload, indent=4)}")
            
            response = requests.post(url, json=payload, headers=headers, params=params)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Langflow response received: {json.dumps(result, indent=4)}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request to Langflow: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error parsing Langflow response: {e}")
            raise
    
    def _extract_message_text(self, response: Dict[str, Any]) -> str:
        """Extract the main message text from Langflow response."""
        try:
            # Navigate the response structure to get the message text
            outputs = response.get("outputs", [])
            if outputs and len(outputs) > 0:
                first_output = outputs[0]
                results = first_output.get("outputs", [])
                if results and len(results) > 0:
                    first_result = results[0]
                    message_data = first_result.get("results", {}).get("message", {})
                    text = message_data.get("text", "")
                    if text:
                        return text
                        
            # Fallback: try to find any text in the response
            return "No message text found in response"
            
        except Exception as e:
            logger.error(f"Error extracting message text: {e}")
            return f"Error extracting response: {str(e)}"
        