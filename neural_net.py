import torch
import  torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle= True)
test_loader = DataLoader(test_data, batch_size=64, shuffle= False)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self. fc2 = nn.Linear(128, 64)
        self. fc3 = nn. Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
loss_fn = nn.CrossEntropyLoss() #Creating the grading tool
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001) #Settign the blueprint optimizing settings

for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)# passing the images in the neural network to run the prediction
        loss = loss_fn(outputs, labels) # Applying the grading/prediction tool
        optimizer.zero_grad()# optimizer.step() — moves every weight in that direction by a tiny amount (0.001).

        loss.backward() # Find what caused the mistake
        optimizer.step() # Actually change the weights

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "digit_model.pth")
print("Model saved!")




