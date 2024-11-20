import dspy
from typing import Optional
from datetime import datetime

import os
from models import ChatHistory, ChatMessage
from .responder import ResponderModule


class ChatterModule(dspy.Module):
    def __init__(self, examples: Optional[dict]):
        super().__init__()
        self.responder = ResponderModule()
        self.start_time = datetime.now()

    def get_time_of_day(self) -> str:
        """Determine the current time of day."""
        hour = datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        else:
            return "evening"

    def get_conversation_duration(self) -> str:
        """Calculate the duration of the conversation."""
        elapsed_time = datetime.now() - self.start_time
        if elapsed_time.total_seconds() < 300:
            return "short"
        elif elapsed_time.total_seconds() < 900:
            return "moderate"
        else:
            return "long"

    def forward(
            self,
            chat_history: Optional[ChatHistory] = None,
            question: Optional[str] = None,
    ):
        current_time = self.get_time_of_day()
        conversation_duration = self.get_conversation_duration()

        if question is not None:
            chat_history = ChatHistory(messages=[ChatMessage(from_creator=False, content=question)])

        context = {
            "chat_history": chat_history,
            "current_time": current_time,
            "conversation_duration": conversation_duration,
        }
        return self.responder(**context)
