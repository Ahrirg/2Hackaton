import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os

# Classes
classes = ["normal", "diabetes", "heart_disease", "alzheimerz"]

# path shit
cwd = os.getcwd()
weights_save_path = rf"{cwd}\saved_models\model.pth"
image_root_path = rf"{cwd}\all_images"

# CNN hyperparameters
training_set_fraction = 0.8
learning_rate = 0.01
epochs = 5
img_res = 16

# Look how scalable, scalable very
num_classes = len(classes)

# Define a Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.feature_size = self._calculate_feature_size(img_res)

        # Linear layers
        self.fc1 = nn.Linear(64 * self.feature_size * self.feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # 2 output classes

    def _calculate_feature_size(self, img_res):
        # Calculate the size of the feature maps after each layer
        size = img_res
        size = (size - 3 + 2 * 1) // 1 + 1  # After conv1
        size = (size - 2) // 2 + 1  # After pool1
        size = (size - 3 + 2 * 1) // 1 + 1  # After conv2
        size = (size - 2) // 2 + 1  # After pool2
        return size

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * self.feature_size * self.feature_size)  # Adjust the size based on the output of the last conv layer

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((img_res, img_res)),  # Resize images to 128x128
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
    ])
    dataset = ImageFolder(root=image_root_path, transform=transform)

    # Split into train (80%) and test (20%)
    train_size = int(training_set_fraction * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Initialize the CNN, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    print(f'TRAINING...\n{"=" * 60}')
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
    print(f'{"=" * 60}')

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'{"=" * 60}')
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    print(f'{"=" * 60}')

    # Save model state dict
    torch.save(model.state_dict(), weights_save_path)

if __name__ == "__main__":
    main()