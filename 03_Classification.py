"""
This script classifies a raster image using a trained PyTorch model and creates a shapefile 
with polygons representing the identified features, including class labels. The script uses 
64x64 tiles for the classification and ensures the output shapefile contains a field with 
class names for easier identification of objects.

Required libraries:
- numpy
- rasterio
- torch
- torchvision
- shapely
- geopandas
- colorama
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn
from torchvision import transforms
import geopandas as gpd
from rasterio.features import shapes
import json
from colorama import init, Fore

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

def classify_and_create_shapefile(raster_path, model_path, class_indices_path, shapefile_out_path, tile_size=(64, 64)):
    # Initialize colorama
    init(autoreset=True)

    # Load the class index to label mapping
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
        index_to_label = {v: k for k, v in class_indices.items()}
    
    num_classes = len(index_to_label)
    
    # Load the trained model
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Open the raster file
    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        crs = src.crs

        result = np.zeros((height, width), dtype=np.uint8)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(tile_size),
            transforms.ToTensor(),
        ])

        for row in range(0, height, tile_size[0]):
            for col in range(0, width, tile_size[1]):
                window = Window(col, row, tile_size[1], tile_size[0])
                tile = src.read(window=window)

                if tile.shape[0] == 4:
                    tile = tile[:3, :, :]

                tile = np.moveaxis(tile, 0, -1)

                if tile.shape[0] != tile_size[0] or tile.shape[1] != tile_size[1]:
                    continue

                tile = preprocess(tile)
                tile = tile.unsqueeze(0)

                with torch.no_grad():
                    prediction = model(tile)
                    predicted_class = prediction.argmax(dim=1).item()

                result[row:row + tile_size[0], col:col + tile_size[1]] = predicted_class

        mask = result != 0
        results = (
            {'properties': {'class_id': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(result, mask=mask, transform=src.transform)))

        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
        gdf['class_name'] = gdf['class_id'].map(index_to_label)

        gdf.to_file(shapefile_out_path)
        print(f"{Fore.CYAN}Shapefile saved successfully!")

# Example usage
raster_path = r"J:\KenReid.tif"
model_path = r"C:\Lindsay\OPS_model.pth"
class_indices_path = r"C:\Lindsay\OPS_class_indices.json"
shapefile_out_path = r"C:\Lindsay\output_KenReid.shp"
classify_and_create_shapefile(raster_path, model_path, class_indices_path, shapefile_out_path)
