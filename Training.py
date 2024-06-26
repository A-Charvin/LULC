"""
This script trains a CNN model using PyTorch for image classification. It performs data augmentation,
trains the model, and saves the trained model and class indices to labels mapping to specified files.

Required libraries:
- torch
- torchvision
- torchsummary
- json
- pathlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import json
import pathlib
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(data_dir, model_save_path, class_indices_path, img_size=(64, 64), batch_size=32, epochs=60, learning_rate=0.001):
    data_dir = pathlib.Path(data_dir)

    # Data augmentation and normalization
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=data_dir / 'train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=data_dir / 'val', transform=val_transforms)
    
    # Debug: Print class names and corresponding directory
    print("Classes and corresponding directories in training set:")
    for class_name in train_dataset.classes:
        print(f"{class_name}: {data_dir / 'train' / class_name}")

    print("Classes and corresponding directories in validation set:")
    for class_name in val_dataset.classes:
        print(f"{class_name}: {data_dir / 'val' / class_name}")

    # Check if directories exist and contain files with supported extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    for class_name in train_dataset.classes:
        class_dir = data_dir / 'train' / class_name
        files = [f for f in os.listdir(class_dir) if any(f.endswith(ext) for ext in supported_extensions)]
        if not files:
            raise FileNotFoundError(f"No valid files found in {class_dir}. Supported extensions are: {supported_extensions}")
    
    for class_name in val_dataset.classes:
        class_dir = data_dir / 'val' / class_name
        files = [f for f in os.listdir(class_dir) if any(f.endswith(ext) for ext in supported_extensions)]
        if not files:
            raise FileNotFoundError(f"No valid files found in {class_dir}. Supported extensions are: {supported_extensions}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get class indices
    class_indices = train_dataset.class_to_idx
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved to {class_indices_path}")

    # Initialize model
    num_classes = len(class_indices)
    model = SimpleCNN(num_classes)
    model = model.cuda() if torch.cuda.is_available() else model

    # Print model summary
    summary(model, (3, img_size[0], img_size[1]))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Data paths and folders
data_dir = r"C:\Lindsay\OPS"
model_save_path = r"C:\Lindsay\OPS_model.pth"
class_indices_path = r"C:\Lindsay\OPS_class_indices.json"
train_model(data_dir, model_save_path, class_indices_path)
