"""Custom Chainlit elements."""

# src/ui/components.py

import chainlit as cl


class Avatar:
    """AI Coach avatar information."""
    
    @staticmethod
    async def display():
        """
        Display the AI Coach avatar.
        
        In a production app, this would show a profile picture.
        """
        await cl.Avatar(name="AI Coach", size="large", image="https://example.com/coach.png").send()


class StatsCard:
    """Display user statistics."""
    
    @staticmethod
    async def display(stats: dict):
        """
        Display a card with user statistics.
        
        Args:
            stats: Dictionary with statistics (e.g., {'hours_learned': 10, 'projects': 3})
        """
        content = """
        <div style="text-align: left; padding: 10px;">
            <h3>📊 Your Progress</h3>
            <ul>
        """
        
        for key, value in stats.items():
            content += f"<li><b>{key.replace('_', ' ').title()}:</b> {value}</li>"
        
        content += """
            </ul>
        </div>
        """
        
        await cl.Element(
            content=content,
            display="inline",
            name="stats-card",
            elements=[cl.Text(content=content)]
        ).send()


class QuickReplies:
    """Predefined quick reply buttons."""
    
    @staticmethod
    async def display(replies: List[str]):
        """
        Display quick reply buttons.
        
        Args:
            replies: List of reply options
        """
        elements = []
        for reply in replies:
            elements.append(cl.Action(name="reply", value=reply, label=reply))
        
        await cl.ActionSet(
            elements=elements,
            visible=True
        ).send()