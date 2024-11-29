import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class DRDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        # Store the csv file path and the transform if any
        self.csv_path = csv_path
        self.transform = transform
        # Load the CSV data (image paths and labels)
        self.image_paths, self.labels = self.load_csv_data()

    def load_csv_data(self):
        # Check if the CSV file exists
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file '{self.csv_path}' not found.")

        # Load data from CSV file into a DataFrame
        data = pd.read_csv(self.csv_path)

        # Ensure the CSV has 'image_path' and 'label' columns
        if "image_path" not in data.columns or "label" not in data.columns:
            raise ValueError("CSV file must contain 'image_path' and 'label' columns.")

        # Extract image paths and labels from the CSV
        image_paths = data["image_path"].tolist()
        labels = data["label"].tolist()

        # Verify if all image paths are valid
        invalid_image_paths = [img_path for img_path in image_paths if not os.path.isfile(img_path)]
        if invalid_image_paths:
            raise FileNotFoundError(f"Invalid image paths found: {invalid_image_paths}")

        # Convert labels to LongTensor (for use in PyTorch)
        labels = torch.LongTensor(labels)

        return image_paths, labels

    def __len__(self):
        # Return the length of the dataset (number of samples)
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path and label at the given index
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Try to load the image using torchvision's read_image function
        try:
            image = read_image(image_path)
        except Exception as e:
            raise IOError(f"Error loading image at path '{image_path}': {e}")

        # Apply transformations if any are provided
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Error applying transformations to image at path '{image_path}': {e}")

        # Return the transformed image and its label
        return image, label
