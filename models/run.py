import torch
import train
import torchvision.transforms as transforms
from PIL import Image
import os

# Initialize the model (same architecture as before)
model = train.SimpleNN()

# Load the saved state_dict into the model
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode (important for inference)
model.eval()

# Get image
cwd = os.getcwd()
path = rf"{cwd}\all_images\normal\8_left.jpg"
img = Image.open(path)


transform = transforms.Compose([
        transforms.Resize((train.img_res, train.img_res)),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjusted for RGB
    ])
# apply transform to image
input = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model.forward(input)

_, res = torch.max(output, 1)

print(f"Predicted class:{res.item()}")