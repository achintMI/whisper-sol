import dspy
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from models import ChatMessage


@dataclass
class ChatSummary:
    summary: str
    start_time: datetime
    end_time: datetime
    key_topics: List[str]
    message_count: int


class SummarizeSignature(dspy.Signature):
    """Signature for chat summarization."""
    context = dspy.InputField(desc="Recent chat messages to summarize")
    summary = dspy.OutputField(desc="A concise summary of the conversation context")
    key_topics = dspy.OutputField(desc="Key topics discussed, as a comma-separated list")


class ChatContextManager(dspy.Module):
    def __init__(self, max_messages: int = 50, summary_interval: int = 10):
        super().__init__()

        self.max_messages = max_messages
        self.summary_interval = summary_interval
        self.summaries: List[ChatSummary] = []

        self.summarize = dspy.Predict(SummarizeSignature)

    def process_messages(self, messages: List[ChatMessage]) -> Dict:
        """Process messages and maintain context through intelligent summarization."""
        current_time = datetime.now()

        # If we've exceeded max_messages, create a new summary
        if len(messages) >= self.summary_interval:
            messages_to_summarize = messages[-self.summary_interval:]

            # Convert messages to string format for summarization
            context = "\n".join([
                f"{'Creator' if msg.from_creator else 'User'}: {msg.content}"
                for msg in messages_to_summarize
            ])

            try:
                # Generate summary using DSPy
                summary_result = self.summarize(context=context)

                # Create new summary object
                new_summary = ChatSummary(
                    summary=summary_result.summary,
                    start_time=current_time - timedelta(minutes=30),
                    end_time=current_time,
                    key_topics=summary_result.key_topics.split(","),
                    message_count=len(messages_to_summarize)
                )

                self.summaries.append(new_summary)

                retain_count = min(self.max_messages // 2, len(messages))
                messages = messages[-retain_count:]
            except Exception as e:
                print(f"Warning: Failed to generate summary: {str(e)}")

        return {
            "active_messages": messages,
            "summaries": self.summaries,
            "total_context": self._generate_context(messages)
        }

    def _generate_context(self, current_messages: List[ChatMessage]) -> str:
        """Generate a complete context string combining summaries and recent messages."""
        context_parts = []

        # Add recent summaries if they exist
        if self.summaries:
            context_parts.append("Previous Context:")
            for summary in self.summaries[-2:]:  # Only use last 2 summaries
                context_parts.append(f"- {summary.summary}")
                context_parts.append(f"  Topics: {', '.join(summary.key_topics)}")
            context_parts.append("\nRecent Messages:")

        # Add current messages
        for msg in current_messages[-self.max_messages:]:
            context_parts.append(
                f"{'Creator' if msg.from_creator else 'User'}: {msg.content}"
            )

        return "\n".join(context_parts)

    def get_relevant_topics(self, query: str) -> List[str]:
        """Find relevant topics from history based on current query."""
        all_topics = []
        for summary in self.summaries:
            all_topics.extend(summary.key_topics)

        unique_topics = list(dict.fromkeys(all_topics))

        query_words = set(query.lower().split())
        relevant_topics = []

        for topic in unique_topics:
            topic_words = set(topic.lower().split())
            if query_words & topic_words:  # If there's any word overlap
                relevant_topics.append(topic)

        return relevant_topics

    def get_statistics(self) -> Dict:
        """Get conversation statistics."""
        return {
            "total_summaries": len(self.summaries),
            "total_messages_summarized": sum(s.message_count for s in self.summaries),
            "all_topics": list(set(
                topic
                for summary in self.summaries
                for topic in summary.key_topics
            )),
            "time_span": (
                self.summaries[-1].end_time - self.summaries[0].start_time
                if self.summaries else timedelta(0)
            )
        }
