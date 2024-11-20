import dspy

from brain.content_filter import ContentFilter
from models import ChatHistory


class Responder(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You are deciding on what your message should be.
    """

    chat_history: ChatHistory = dspy.InputField(desc="the chat history")

    reasoning: str = dspy.InputField(desc="context about timing and duration")

    reasoning_steps: str = dspy.OutputField(
        prefix="Reasoning: Let's think step by step.",
        desc="The reasoning behind the generated response.",
    )

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )
