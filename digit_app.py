# ── IMPORTS ──────────────────────────────────────────────
from flask import Flask, render_template, request  # Flask web framework tools
import torch                                        # PyTorch core
import torch.nn as nn                              # Neural network building blocks
from torchvision import transforms                 # Image processing tools
from PIL import Image                              # Open and manipulate image files

# ── CREATE FLASK APP ──────────────────────────────────────
app = Flask(__name__)

# ── NEURAL NETWORK BLUEPRINT ──────────────────────────────
# Defines the structure of the network — 3 layers, 784 → 128 → 64 → 10
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    # Defines how data flows through the layers when model is called
    def forward(self, x):
        x = x.view(-1, 784)                  # flatten image to 784 numbers
        x = torch.relu(self.fc1(x))          # layer 1 + clean negatives
        x = torch.relu(self.fc2(x))          # layer 2 + clean negatives
        x = self.fc3(x)                      # layer 3 → 10 scores
        return x

# ── LOAD TRAINED MODEL ────────────────────────────────────
# Create empty model, load saved weights, switch to prediction mode
model = NeuralNet()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

# ── ROUTE — HOMEPAGE ─────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        # Just show the upload form
        return render_template("digit.html")

    if request.method == "POST":
        # ── GET THE UPLOADED FILE ─────────────────────────
        file = request.files["file"]

        # ── PROCESS THE IMAGE ─────────────────────────────
        image = Image.open(file).convert("L")    # open + convert to grayscale
        image = image.resize((28, 28))           # resize to match training data

        # Convert to tensor and normalize to match training format
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

        # ── RUN THROUGH MODEL ─────────────────────────────
        with torch.no_grad():  # no training, just predicting
            output = model(image_tensor)                      # get 10 scores
            prediction = torch.argmax(output, dim=1).item()  # pick highest

        # ── SEND RESULT TO HTML ───────────────────────────
        return render_template("digit.html", prediction=prediction)

# ── START SERVER ──────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)