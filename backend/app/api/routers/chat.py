# chat.py
from fastapi.responses import StreamingResponse
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from fastapi import APIRouter, Depends, HTTPException, Request, status
from llama_index.agent.openai import OpenAIAgent
from pydantic import BaseModel

from backend.app.utils.dependencies import get_agent
from backend.app.utils.json import json_to_model

chat_router = APIRouter()


class _Message(BaseModel):
    role: MessageRole
    content: str


class _ChatData(BaseModel):
    messages: list[_Message]


@chat_router.post("",  response_model=None)
async def chat(
    request: Request,
    data: _ChatData = Depends(json_to_model(_ChatData)),
    agent: OpenAIAgent=Depends(get_agent),  # Dependency is globally injected in main.py
):
    # Check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    last_message = data.messages.pop()
    if last_message.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )
    # Convert messages coming from the request to type ChatMessage
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]

    # Query the chat engine with streaming enabled
    response = agent.stream_chat(message=last_message.content)

    # Stream response
    async def event_generator():
        for token in response.response_gen:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            yield token

    return StreamingResponse(event_generator(), media_type="text/plain")
