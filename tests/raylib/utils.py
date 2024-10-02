import numpy as np
import os
from PIL import Image

def create_thumbnails(root_directory, size=(250, 250)):
    root_directory = os.path.abspath(root_directory)
    thumbnail_dir = os.path.join(root_directory, 'thumbnails')
    
    # Check if thumbnail directory already exists
    if os.path.exists(thumbnail_dir):
        print("Thumbnail directory already exists. Skipping thumbnail creation.")
        return
    
    os.makedirs(thumbnail_dir, exist_ok=True)
    
    file_paths = np.load(f"{root_directory}/file_paths.npy", allow_pickle=True)
    
    for file_path in file_paths:
        try:
            with Image.open(file_path) as img:
                img.thumbnail(size)
                thumbnail_path = os.path.join(thumbnail_dir, os.path.basename(file_path))
                img.save(thumbnail_path)
        except Exception as e:
            print(f"Error processing image {file_path}: {str(e)}")
    
    print("Thumbnails created successfully.")

def load_embeddings_and_paths(root_directory):
    root_directory = os.path.abspath(root_directory)
    thumbnail_dir = os.path.join(root_directory, 'thumbnails')
    
    # Create thumbnails if they don't exist
    if not os.path.exists(thumbnail_dir):
        create_thumbnails(root_directory)
    
    embeddings = np.load(f"{root_directory}/image_embeddings.npy")
    file_paths = np.load(f"{root_directory}/file_paths.npy", allow_pickle=True)
    
    # Load label names from CSV file
    csv_path = os.path.join(root_directory, 'image_classifier_data.csv')
    label_dict = {}
    with open(csv_path, 'r') as csv_file:
        next(csv_file)  # Skip header
        for line in csv_file: #image_path,image_name,label_name,is_test,label_id,label_prediction
            parts = line.strip().split(',')
            if len(parts) >= 3:
                image_name = parts[1]
                label_name = parts[2]
                label_dict[image_name] = label_name
    
    # Replace original file paths with thumbnail paths and get corresponding labels
    thumbnail_paths = []
    labels = []
    for path in file_paths:
        image_name = os.path.basename(path)
        thumbnail_path = os.path.join(thumbnail_dir, image_name)
        thumbnail_paths.append(thumbnail_path)
        labels.append(label_dict.get(image_name, "Unknown"))
    
    return embeddings, thumbnail_paths, labels