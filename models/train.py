import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os, time

# Set device - Force CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Double-check CUDA is being used
if device.type == 'cuda':
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("WARNING: CUDA not available, using CPU instead")

# Classes
classes = ["normal", "diabetes", "heart_disease", "hyper_tensive"]

# Path configuration
cwd = os.getcwd()
weights_save_path = rf"{cwd}/models/model.pth"
image_root_path = rf"{cwd}/models/Dataset"

# CNN hyperparameters
training_set_fraction = 0.8
learning_rate = 0.005
epochs = 30
img_res = 480

# Number of classes
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
        self.fc1 = nn.Linear(64 * self.feature_size * self.feature_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, num_classes)

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
        x = x.view(-1, 64 * self.feature_size * self.feature_size)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def Eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print(f'{"=" * 60}')
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


def main():
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((img_res, img_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Verify CUDA is being used before starting
    if device.type == 'cuda':
        print("CUDA setup verified - proceeding with GPU training")
    
    dataset = ImageFolder(root=image_root_path, transform=transform)

    # Split into train and test
    train_size = int(training_set_fraction * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True)

    # Initialize the CNN, loss function, and optimizer
    model = SimpleCNN().to(device)  # Move model to CUDA
    
    # Verify model is on CUDA
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    
    criterion = nn.CrossEntropyLoss().to(device)  # Move criterion to CUDA
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    print(f'TRAINING...\n{"=" * 60}')
    # Training loop
    for epoch in range(epochs):
        time1 = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device, non_blocking=True)  # non_blocking for better performance
            labels = labels.to(device, non_blocking=True)
            
            # Verify data is on CUDA
            # if i == 0 and epoch == 0:
                # print(f"Batch data is on CUDA: {images.is_cuda}")
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Print CUDA memory usage every 10 batches
            # if i % 10 == 0 and device.type == 'cuda':
                # print(f"Batch {i}, CUDA Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Print CUDA memory after each epoch
        if device.type == 'cuda':
            print(f"CUDA Memory after epoch {epoch+1}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
        Eval(model, test_loader)
        time2 = time.time()
        print(F"Time spent on epoch: {time2-time1}s")
        print(f'{"=" * 60}')

    
    print(f'{"=" * 60}')

    # Test the model
    Eval(model, test_loader)
    torch.save(model.state_dict(), weights_save_path)
    print(f"Model saved to {weights_save_path}")

if __name__ == "__main__":
    main()
