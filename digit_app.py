from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self,x):
        x = x.view(-1,784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "GET":
        return render_template("digit.html")
    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file).convert("L")
        image = image.resize((28, 28))
        transform = transforms.Compose([
            transforms. ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)  #calling/runing the class with the whole data and structure compressedinto image_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return render_template("digit.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
