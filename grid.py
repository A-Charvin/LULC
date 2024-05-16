import os
from PIL import Image

def split_image(image_path, output_dir):
    image = Image.open(image_path)
    width, height = image.size
    tile_size = 64

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tile_filename = f"tile_{x // tile_size}_{y // tile_size}.tif"
            tile_path = os.path.join(output_dir, tile_filename)
            tile.save(tile_path)

if __name__ == "__main__":
    input_image_path = "C:/Users/AA.tif"
    output_directory = "C:/Users/Split/Aggregate/AA"
    os.makedirs(output_directory, exist_ok=True)
    split_image(input_image_path, output_directory)
    print("Image tiles saved successfully!")
