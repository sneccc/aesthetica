import torch
import open_clip
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import pandas as pd
import argparse
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, GaussNoise, RandomResizedCrop, ColorJitter
)
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

# Move augmentations to module level
AUGMENTATIONS = Compose([
    RandomResizedCrop(height=384, width=384, scale=(0.8, 1.0)),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    ShiftScaleRotate(p=0.5),
    RandomBrightnessContrast(p=0.5),
    GaussNoise(p=0.3),
    ColorJitter(p=0.3)
])

def custom_collate(batch):
    images = [item[0] for item in batch]  # Keep PIL images as a list
    paths = [item[1] for item in batch]   # Collect paths
    return images, paths

def prepare_embeddings(df, clip_models, num_augmentations=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    preprocessors = []
    image_embeddings = []  # Initialize as list
    file_paths = []
    batch_size = 32
    
    # Load models
    for clip_model in tqdm(clip_models, desc="Loading models"):
        if clip_model[0] == "hf-hub:timm":
            model, preprocess = open_clip.create_model_from_pretrained(clip_model[0] + "/" + clip_model[1], device=device)
        elif clip_model[0] == "nomic-ai":
            from transformers import AutoTokenizer, AutoModel, AutoImageProcessor  
            preprocess = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
            model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1], device=device)
        
        model.to(device)
        model.eval()
        models.append(model)
        preprocessors.append(preprocess)
    
    # Create dataset and dataloader with custom collate function
    dataset = ImageDataset(df)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Process batches
    all_embeddings = []  # Store all embeddings
    all_paths = []      # Store all paths
    
    pbar = tqdm(dataloader, desc="Processing images")
    for batch_images, batch_paths in pbar:
        try:
            # Preprocess batch
            batch_preprocessed_images = []
            for image in batch_images:
                if clip_model[0] == "nomic-ai":
                    inputs = preprocessors[0](image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    batch_preprocessed_images.append(inputs)
                else:
                    preprocessed = preprocessors[0](image)  # Use only first preprocessor
                    batch_preprocessed_images.append(preprocessed)
            
            # Process batch through models
            with torch.no_grad(), torch.amp.autocast('cuda'):
                if clip_model[0] == "nomic-ai":
                    features = [model(**inputs).last_hidden_state for inputs in batch_preprocessed_images]
                    features = [f.cpu().numpy() for f in features]
                else:
                    # Stack preprocessed images into a single batch tensor
                    batch_tensor = torch.stack(batch_preprocessed_images).to(device)
                    features = models[0].encode_image(batch_tensor)
                    features = features.cpu().numpy()
                
                # Store results
                all_embeddings.append(features)
                all_paths.extend(batch_paths)
            
            # Update progress
            pbar.set_postfix({'GPU_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"})
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            continue
        
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        final_embeddings = np.array([])
    
    return final_embeddings, all_paths

def load_and_preprocess_image(row):
    image = Image.open(row['image_path']).convert("RGB")
    if row['aug_version'] > 0:
        image_np = np.array(image)
        augmented = AUGMENTATIONS(image=image_np)
        image = Image.fromarray(augmented['image'])
    return image

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        if row['aug_version'] > 0:
            image_np = np.array(image)
            augmented = AUGMENTATIONS(image=image_np)
            image = Image.fromarray(augmented['image'])
        return image, row['image_path']

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for preprocessed images.")
    parser.add_argument("-i", "--input_directory", required=True, help="Path to the normalized image directory")
    args = parser.parse_args()
    num_augmentations = 5
    input_directory = os.path.abspath(args.input_directory)
    
    # Load the CSV file
    csv_path = os.path.join(input_directory, 'image_classifier_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from CSV")
    
    # Create expanded DataFrame with augmentations
    print("Creating augmented dataset structure...")
    augmented_rows = []
    for idx, row in tqdm(df.iterrows(), desc="Preparing dataset structure"):
        # Add original image
        original_row = row.copy()
        original_row['aug_version'] = 0  # 0 means original
        original_row['embedding_index'] = len(augmented_rows)
        augmented_rows.append(original_row)
        
        # Add augmented versions
        for aug_idx in range(num_augmentations):
            aug_row = row.copy()
            aug_row['aug_version'] = aug_idx + 1
            aug_row['embedding_index'] = len(augmented_rows)
            augmented_rows.append(aug_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    print(f"Created dataset with {len(augmented_df)} entries ({len(df)} original + {len(df) * num_augmentations} augmented)")
    
    # Prepare embeddings using the augmented DataFrame
    print("Preparing embeddings...")
    clip_model = [("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")]
    image_embeddings, file_paths = prepare_embeddings(augmented_df, clip_model, num_augmentations)
    
    # Verify the number of embeddings matches the DataFrame
    assert len(image_embeddings) == len(augmented_df), \
        f"Mismatch between embeddings ({len(image_embeddings)}) and DataFrame entries ({len(augmented_df)})"
    
    # Save embeddings and file paths
    np.save(f"{input_directory}/image_embeddings.npy", image_embeddings)
    np.save(f"{input_directory}/file_paths.npy", file_paths)
    
    # Save the augmented DataFrame
    augmented_df.to_csv(csv_path, index=False)
    
    print(f"\nProcessing complete:")
    print(f"Total entries in dataset: {len(augmented_df)}")
    print(f"  - Original images: {len(df)}")
    print(f"  - Augmented versions: {len(df) * num_augmentations}")
    print(f"Embedding shape: {image_embeddings.shape}")
    print(f"Files saved:")
    print(f"  - Embeddings: {input_directory}/image_embeddings.npy")
    print(f"  - File paths: {input_directory}/file_paths.npy")
    print(f"  - Updated CSV: {csv_path}")

if __name__ == "__main__":
    main()
