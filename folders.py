import os

# Define the land cover classes and their subfolders
land_cover_classes = {
    "Water": ["SAS", "OAO"],
    "Aggregate": ["AA", "AI"],
    "Wetlands": ["SWT", "MAS", "MAM", "BOO", "BOT", "SAF", "SAM"],
    "Agriculture": ["IAG", "NAG"],
    "Beach Bar": ["BBO"],
    "Development": ["RD", "URB"],
    "Disturbance": ["DIS"],
    "Forest": ["CUP", "CUW", "FOC", "FOD", "FOM"],
    "Manicured Open Space": ["MOS"],
    "Meadow": ["CUM", "CUS", "CUT"],
    "Rock Barren": ["RBO", "RBS", "RBT"],
    "Sand Barren": ["SBO"],
    "Treed Wetland": ["SWC", "SWD", "SWM"]
    # Add other land cover classes and their subfolders here
}

# The root directory where the folders will be created.
root_directory = "C:/Users/Folder/Split/"

def create_folders(root, classes):
    for class_name, subfolders in classes.items():
        class_directory = os.path.join(root, class_name)
        os.makedirs(class_directory, exist_ok=True)
        for subfolder in subfolders:
            subfolder_directory = os.path.join(class_directory, subfolder)
            os.makedirs(subfolder_directory, exist_ok=True)

if __name__ == "__main__":
    create_folders(root_directory, land_cover_classes)
    print("Folders and subfolders created successfully!")
