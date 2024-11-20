import dspy
from dspy.teleprompt import KNNFewShot
from brain.modules.chat_context_manager import ChatContextManager
from models import ChatMessage, ChatHistory
from lms.together import Together
import json
import pickle
import os
from typing import Optional, List
from pathlib import Path


class CustomDataset:
    def __init__(self, data):
        self.data = data
        self.examples = []
        self._populate_examples()

    def _populate_examples(self):
        for item in self.data:
            chat_history = item.get("chat_history", {})
            messages = chat_history.get("messages", [])
            chat_history_list: List[ChatMessage] = [
                ChatMessage(from_creator=message.get("from_creator"), content=message.get("content"))
                for message in messages
            ]
            output = item.get("output", "")
            self.examples.append(dspy.Example(question=str(chat_history_list), answer=output).with_inputs("question"))

    def __iter__(self):
        return iter(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PersistentKNNOptimizer:
    def __init__(
            self,
            model_path: str = "./models/knn_model.json",
            k: int = 7,
            lm: Optional[Together] = None
    ):
        self.model_path = Path(model_path)
        self.k = k
        self.model = None
        self.lm = lm or Together(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.2,
            stop=["<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"],
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_or_train(self, module_class, dataset) -> dspy.Predict:
        """Load existing model if available, otherwise train new one."""
        if self.model_path.exists():
            print("Loading existing KNN model...")
            return self.load_model(module_class)

        print("Training new KNN model...")
        return self.train_model(module_class, dataset)

    def train_model(self, module_class, dataset) -> dspy.Predict:
        """Train and save a new model."""
        # Configure DSPy with the language model
        dspy.settings.configure(lm=self.lm)

        # Create and compile the KNN model
        knn_teleprompter = KNNFewShot(self.k, dataset)
        self.model = knn_teleprompter.compile(
            module_class(examples=None),
            trainset=dataset
        )

        # Save the model
        self.model.save(self.model_path)

        return self.model

    def load_model(self, module_class) -> dspy.Predict:
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No model found at {self.model_path}")

        self.model = module_class(examples=None)
        self.model.load(self.model_path)
        dspy.settings.configure(lm=self.lm)

        return self.model


def run_chat_interface(optimizer: PersistentKNNOptimizer):
    """Run the enhanced chat interface with context management."""
    chat_history = ChatHistory()
    context_manager = ChatContextManager(max_messages=5, summary_interval=1)

    print("\nEnhanced Chat Interface")
    print("Commands:")
    print("  /topics - Show relevant topics from history")
    print("  /stats - Show conversation statistics")
    print("  /context - Show current context summary")
    print("  /exit - Exit the chat")
    print("\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/exit":
                    break
                elif user_input == "/topics":
                    relevant_topics = context_manager.get_relevant_topics(
                        chat_history.messages[-1].content if chat_history.messages else ""
                    )
                    print("\nRelevant Topics:", ", ".join(relevant_topics) or "No topics found")
                    continue
                elif user_input == "/stats":
                    stats = context_manager.get_statistics()
                    print("\nConversation Statistics:")
                    print(f"Total Summaries: {stats['total_summaries']}")
                    print(f"Total Messages Summarized: {stats['total_messages_summarized']}")
                    print(f"All Topics: {', '.join(stats['all_topics'])}")
                    print(f"Time Span: {stats['time_span']}")
                    continue
                elif user_input == "/context":
                    context = context_manager._generate_context(chat_history.messages)
                    print("\nCurrent Context:")
                    print(context)
                    continue

            # Append user input to chat history
            chat_history.messages.append(
                ChatMessage(from_creator=False, content=user_input)
            )

            context_data = context_manager.process_messages(chat_history.messages)

            # print("\nContext Summary:", context_data)

            chat_history.messages = context_data["active_messages"]

            response = optimizer.model(chat_history=chat_history).output

            chat_history.messages.append(
                ChatMessage(from_creator=True, content=response)
            )

            print("\nResponse:", response, "\n")

        except Exception as e:
            print(f"Error: {str(e)}")
            continue


if __name__ == "__main__":
    # Initialize the optimizer
    optimizer = PersistentKNNOptimizer(
        model_path="./models/knn_model.pkl",
        k=7
    )

    # Load training data
    with open("./training_data/conversations.json") as f:
        data = json.load(f)

    custom_dataset = CustomDataset(data)

    # Load or train the model
    from modules.chatter import ChatterModule

    model = optimizer.load_or_train(ChatterModule, custom_dataset)

    # Run the chat interface
    run_chat_interface(optimizer)
