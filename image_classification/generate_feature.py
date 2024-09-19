import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model.eval()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load image and extract features
def extract_image_features(image_path):
    img = Image.open(image_path)
    # Ensure the image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img_tensor).flatten()  # Flatten the output
    return features


# Example usage
image_path = 'data/1.png'
image_features = extract_image_features(image_path)
print("Extracted Image Features:\n", image_features)