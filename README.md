# Land Cover Classification Folder Structure

This repository contains a Python script that generates a folder structure for land cover classification. It organizes different land cover classes and their subfolders, making it easier to manage and categorize data related to environmental features.

## Usage

1. Clone this repository to your local machine:

2. Navigate to the root directory:

3. Modify the `land_cover_classes` dictionary in the `folders.py` script to include additional land cover classes and their subfolders.

4. Specify the root directory where you want to create the folders by updating the `root_directory` variable in the script.

5. Run the script:

   ```bash
   python folders.py
   ```

6. Check your specified root directory to find the newly created folders and subfolders.

## Folder Structure

The generated folder structure will look like this:

```
- Water
  - SAS
  - OAO
- Aggregate
  - AA
    - AI
- Wetlands
  - SWT
  - MAS
  - MAM
  - BOO
  - BOT
  - SAF
  - SAM
- Agriculture
  - IAG
  - NAG
- ... (other land cover classes)
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.🌿🌲🌊
