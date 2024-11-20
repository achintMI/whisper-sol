from models import ChatMessage, ChatHistory
from config import KNN_MODEL_PATH
from knn_optimizer import PersistentKNNOptimizer, CustomDataset, run_chat_interface
import json

# Initialize the optimizer
optimizer = PersistentKNNOptimizer(
    model_path=KNN_MODEL_PATH,
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
