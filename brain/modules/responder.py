import dspy

from brain.content_filter import ContentFilter
from signatures.responder import Responder
from models import ChatHistory


class ResponderModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoning_field = dspy.InputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We",
        )
        self.prog = dspy.TypedChainOfThought(
            Responder, reasoning=self.reasoning_field
        )
        self.content_filter = ContentFilter()

    def forward(
            self,
            chat_history: dict,
            current_time: str,
            conversation_duration: str,
    ):
        reasoning_context = (
            f"It's currently {current_time}, and the conversation duration is "
            f"{conversation_duration}. "
        )

        completion = self.prog(
            chat_history=ChatHistory.parse_obj(chat_history),
            reasoning=reasoning_context,
        )
        is_safe, violations = self.content_filter.check_message(completion.output)

        if not is_safe:
            filtered_message = self.content_filter.filter_message(completion.output)
            suggestions = self.content_filter.suggest_alternatives(completion.output)

            completion.reasoning_steps += "\n\nMessage was filtered to remove: "
            completion.reasoning_steps += ", ".join(violations)
            if suggestions:
                completion.reasoning_steps += "\nSuggested alternatives: "
                completion.reasoning_steps += "\n- " + "\n- ".join(suggestions)

            completion.output = filtered_message

        return completion
