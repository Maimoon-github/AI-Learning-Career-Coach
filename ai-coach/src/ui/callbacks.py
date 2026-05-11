"""Message handlers."""

# src/ui/callbacks.py

from typing import Any, Dict, List
import chainlit as cl


@cl.on_chat_start
def on_chat_start():
    """Initialize the chat session."""
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages and generate responses.
    
    Args:
        message: The user's message
    """
    # Get or initialize chat history
    history: List[Dict[str, Any]] = cl.user_session.get("history")
    if history is None:
        history = []
        cl.user_session.set("history", history)
    
    # Add user message to history
    history.append({"role": "user", "content": message.content})
    
    # Display user message
    await cl.Message(content=message.content).send()
    
    # TODO: Integrate with your AI Coach workflow
    # For now, return a placeholder response
    response_text = f"Received your message: {message.content}"
    
    # Add assistant response to history
    history.append({"role": "assistant", "content": response_text})
    
    # Display assistant response
    await cl.Message(content=response_text).send()
