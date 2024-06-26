# Pytorch based Training data creator and extractor script.
# v 2.1
# Splits the data into 256 x 256 Based tilesare put them into folders and subfolders,
# The code splits a raster dataset into 256 x 256 tiles based on geometries defined in a shapefile. 
# It also separates the tiles into training and validation sets based on a specified ratio. (80%)


import os
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
import random

def save_tile(image, meta, out_dir, base_name, tile_idx):
    """Saves an image tile to the specified directory."""
    out_tile_path = os.path.join(out_dir, f"{base_name}_{tile_idx[0]}_{tile_idx[1]}.tif")
    if os.path.exists(out_tile_path):
        print(f"Tile {tile_idx} already exists. Skipping.")
        return False

    tile_meta = meta.copy()
    tile_meta.update({
        "height": image.shape[1],
        "width": image.shape[2],
        "transform": rasterio.windows.transform(Window(tile_idx[1], tile_idx[0], 256, 256), meta['transform']), # Change the tile size as needed
        "dtype": image.dtype
    })
    with rasterio.open(out_tile_path, "w", **tile_meta) as dest:
        dest.write(image)
    print(f"Saved tile {tile_idx} to {out_tile_path}")
    return True

def split_raster(raster_path, shapefile_path, output_dir, max_tiles=3000, train_ratio=0.8): 
    # Load shapefile
    shapes = gpd.read_file(shapefile_path)

    # Create train and val directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Open the raster file
    with rasterio.open(raster_path) as src:
        for i, row in shapes.iterrows():
            geom = [row['geometry']]
            elc_code = row['ELC']
            elc_name = row['ELC_Description'] if 'ELC_Description' in row else elc_code

            print(f"Processing ELC code: {elc_code} ({elc_name})")

            # Create ELC specific directories
            elc_train_dir = os.path.join(train_dir, elc_code)
            elc_val_dir = os.path.join(val_dir, elc_code)
            os.makedirs(elc_train_dir, exist_ok=True)
            os.makedirs(elc_val_dir, exist_ok=True)

            # Check the number of existing tiles in the folder
            existing_train_tiles = len([name for name in os.listdir(elc_train_dir) if os.path.isfile(os.path.join(elc_train_dir, name))])
            existing_val_tiles = len([name for name in os.listdir(elc_val_dir) if os.path.isfile(os.path.join(elc_val_dir, name))])
            existing_tiles = existing_train_tiles + existing_val_tiles
            if existing_tiles >= max_tiles:
                print(f"Skipping {elc_name} as it already contains {existing_tiles} tiles.")
                continue

            # Mask the raster with the geometry
            out_image, out_transform = mask(src, geom, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": out_image.dtype
            })

            # Calculate the number of rows and columns
            n_rows = out_image.shape[1] // 256
            n_cols = out_image.shape[2] // 256

            # Determine the tiles to save
            tile_indices = [(row, col) for row in range(n_rows) for col in range(n_cols)]
            random.shuffle(tile_indices)

            train_count = int(len(tile_indices) * train_ratio)
            if train_count == 0:
                train_count = 1

            for idx, (row, col) in enumerate(tile_indices):
                if existing_tiles >= max_tiles:
                    print(f"Reached the maximum limit of {max_tiles} tiles for {elc_name}.")
                    break
                window = Window(col * 256, row * 256, 256, 256)
                tile = out_image[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
                if np.any(tile):  # Save the tile only if it's not empty
                    if idx < train_count:
                        if save_tile(tile, out_meta, elc_train_dir, elc_name, (row, col)):
                            existing_train_tiles += 1
                            existing_tiles += 1
                    else:
                        if save_tile(tile, out_meta, elc_val_dir, elc_name, (row, col)):
                            existing_val_tiles += 1
                            existing_tiles += 1

            print(f"Finished processing {elc_name}. Total tiles saved: {existing_tiles}")

# File Paths. Includes the Input raster and shape file as well as the output folder path.
raster_path = r"C:\IMAGERY\SCOOP\OPS.tif"
shapefile_path = r"C:\P\Shp\Pos2.shp"
output_dir = r"C:\P\Output\"
split_raster(raster_path, shapefile_path, output_dir)
