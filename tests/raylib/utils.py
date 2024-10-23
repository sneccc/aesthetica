import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import math
from tqdm import tqdm

# Increase the maximum image size limit
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely, use with caution

# Disable DecompressionBomb warnings
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class ImageDataset(Dataset):
    def __init__(self, file_paths, thumbnail_size):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.Resize(thumbnail_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_tensor = self.transform(img)
                return img_tensor
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return torch.zeros(3, 256, 256)

def create_thumbnails_and_atlas(root_directory, thumbnail_size=(256, 256), batch_size=64, max_atlas_size=8192):
    root_directory = os.path.abspath(root_directory)
    atlas_dir = os.path.join(root_directory, 'atlas_chunks')
    positions_path = os.path.join(root_directory, 'atlas_positions.npy')
    
    # Check if atlas chunks already exist
    if os.path.exists(atlas_dir) and os.path.exists(positions_path):
        print("Atlas chunks already exist. Skipping creation.")
        return atlas_dir, positions_path
    
    os.makedirs(atlas_dir, exist_ok=True)
    
    file_paths = np.load(f"{root_directory}/file_paths.npy", allow_pickle=True)
    
    # Create dataset and dataloader
    dataset = ImageDataset(file_paths, thumbnail_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate atlas chunk size
    chunk_size = max_atlas_size - (max_atlas_size % thumbnail_size[0])
    chunk_cols = chunk_size // thumbnail_size[0]
    chunk_rows = chunk_size // thumbnail_size[1]
    thumbnails_per_chunk = chunk_cols * chunk_rows
    
    print(f"Atlas chunk size: {chunk_size}x{chunk_size}")
    print(f"Thumbnails per chunk: {thumbnails_per_chunk}")
    
    positions = []
    current_chunk = torch.zeros((3, chunk_size, chunk_size), device=device)
    chunk_index = 0
    thumbnail_count = 0
    
    print("Creating atlas chunks...")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        batch = batch.to(device)
        for j, img in enumerate(batch):
            row = (thumbnail_count % thumbnails_per_chunk) // chunk_cols
            col = (thumbnail_count % thumbnails_per_chunk) % chunk_cols
            position = (col * thumbnail_size[0], row * thumbnail_size[1], chunk_index)
            
            current_chunk[:, position[1]:position[1]+thumbnail_size[1], position[0]:position[0]+thumbnail_size[0]] = img
            positions.append(position)
            
            thumbnail_count += 1
            
            if thumbnail_count % thumbnails_per_chunk == 0 or thumbnail_count == len(file_paths):
                # Save current chunk
                chunk_np = current_chunk.cpu().numpy().transpose(1, 2, 0)
                chunk_image = Image.fromarray((chunk_np * 255).astype(np.uint8))
                chunk_path = os.path.join(atlas_dir, f'atlas_chunk_{chunk_index}.png')
                chunk_image.save(chunk_path, optimize=True, quality=85)
                
                # Reset for next chunk
                current_chunk.zero_()
                chunk_index += 1
    
    # Save positions
    np.save(positions_path, np.array(positions))
    
    print(f"Atlas chunks created successfully. Total chunks: {chunk_index}")
    return atlas_dir, positions_path

# The load_embeddings_and_paths function remains the same
def load_embeddings_and_paths(root_directory):
    root_directory = os.path.abspath(root_directory)
    
    # Create thumbnails and atlas chunks if they don't exist
    atlas_dir = os.path.join(root_directory, 'atlas_chunks')
    positions_path = os.path.join(root_directory, 'atlas_positions.npy')
    atlas_dir, positions_path = create_thumbnails_and_atlas(root_directory)
    
    embeddings = np.load(f"{root_directory}/image_embeddings.npy")
    file_paths = np.load(f"{root_directory}/file_paths.npy", allow_pickle=True)
    atlas_positions = np.load(positions_path)
    
    # Load label names from CSV file
    csv_path = os.path.join(root_directory, 'image_classifier_data.csv')
    label_dict = {}
    with open(csv_path, 'r') as csv_file:
        next(csv_file)  # Skip header
        for line in csv_file:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                image_name = parts[1]
                label_name = parts[2]
                label_dict[image_name] = label_name
    
    # Get corresponding labels
    labels = [label_dict.get(os.path.basename(path), "Unknown") for path in file_paths]
    
    return embeddings, atlas_dir, atlas_positions, labels