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
    ShiftScaleRotate, GaussNoise, RandomResizedCrop, ColorJitter,
    RandomFog, RandomShadow, RandomRain, RandomSnow, RandomSunFlare, ChannelDropout, CLAHE, Downscale, ImageCompression
)
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from embedding_processor import EmbeddingProcessor, EmbeddingConfig

# Move augmentations to module level
AUGMENTATIONS = Compose([
    RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0)),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    ShiftScaleRotate(p=0.5),
    RandomBrightnessContrast(p=0.5),
    GaussNoise(p=0.3),
    ColorJitter(p=0.3),
    RandomBrightnessContrast(p=0.3),
    RandomFog(p=0.3),
    RandomShadow(p=0.3),
    RandomRain(p=0.3),
    RandomSnow(p=0.3),
    RandomSunFlare(p=0.3),
    ChannelDropout(p=0.3),
    CLAHE(p=0.3),
    Downscale(p=0.2, scale_range=(0.1, 0.8)),
    ImageCompression(p=0.2, quality_lower=50, quality_upper=90),
])

DO_AUGMENTATIONS = False

def custom_collate(batch):
    images = [item[0] for item in batch]  # Keep PIL images as a list
    paths = [item[1] for item in batch]   # Collect paths
    return images, paths

def prepare_embeddings(df, clip_models, num_augmentations=5, pca_components=None):
    config = EmbeddingConfig(
        model_name=clip_models[0][0],
        pretrained=clip_models[0][1],
        batch_size=256,  # Increased batch size
        pca_components=pca_components,
        use_standardization=True
    )
    
    # Normalize paths in the DataFrame
    df['image_path'] = df['image_path'].apply(lambda x: os.path.normpath(x))
    
    processor = EmbeddingProcessor(config)
    
    # Create dataset and dataloader
    dataset = ImageDataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Generate embeddings
    embeddings, file_paths = processor.generate_embeddings(dataloader)
    
    # Normalize all file paths before returning
    file_paths = [os.path.normpath(path) for path in file_paths]
    
    # Apply dimensionality reduction and standardization
    embeddings = processor.fit_transform_embeddings(embeddings)
    
    return embeddings, file_paths

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
        image_path = os.path.normpath(row['image_path'])
        
        # Verify file exists before attempting to open
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        if row['aug_version'] > 0:
            image_np = np.array(image)
            if DO_AUGMENTATIONS:
                augmented = AUGMENTATIONS(image=image_np)
                image = Image.fromarray(augmented['image'])
            else:
                image = Image.fromarray(image_np)
        return image, image_path

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for preprocessed images.")
    parser.add_argument("-i", "--input_directory", required=True, help="Path to the normalized image directory")
    parser.add_argument("-a", "--augmentations", required=False, help="Whether to perform augmentations", default=1)
    parser.add_argument("--pca", type=int, help="Number of PCA components to reduce to. If not specified, no PCA reduction is performed.")
    parser.add_argument("--clip_model", default="ViT-SO400M-14-SigLIP-384", help="CLIP model to use for embeddings")
    args = parser.parse_args()
    num_augmentations = 0
    input_directory = os.path.abspath(args.input_directory)
    
    # Load the CSV file
    csv_path = os.path.join(input_directory, 'image_classifier_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from CSV")
    
    # Modified DataFrame creation logic
    print("Creating augmented dataset structure...")
    augmented_rows = []
    
    # First, add all original images
    for idx, row in tqdm(df.iterrows(), desc="Adding original images"):
        original_row = row.copy()
        original_row['aug_version'] = 0  # 0 means original
        original_row['embedding_index'] = len(augmented_rows)
        augmented_rows.append(original_row)
    
    # Then, add all augmented versions
    for idx, row in tqdm(df.iterrows(), desc="Adding augmented versions"):
        for aug_idx in range(num_augmentations):
            aug_row = row.copy()
            aug_row['aug_version'] = aug_idx + 1
            aug_row['embedding_index'] = len(augmented_rows)
            augmented_rows.append(aug_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    print(f"Created dataset with {len(augmented_df)} entries ({len(df)} original + {len(df) * num_augmentations} augmented)")
    
    # Prepare embeddings using the augmented DataFrame
    print("Preparing embeddings...")
    # Use the clip_model from command line arguments - fix model selection logic
    if args.clip_model in ["ViT-SO400M-14-SigLIP-384", "ViT-B-16-SigLIP-512", "ViT-L-14-SigLIP-384"]:
        # These are timm models
        clip_model = [("hf-hub:timm", args.clip_model)]
    else:
        # Standard OpenAI CLIP models (ViT-B/32, ViT-L/14, RN50, etc.)
        clip_model = [(args.clip_model, "openai")]
    
    print(f"Using CLIP model: {clip_model}")
    image_embeddings, file_paths = prepare_embeddings(augmented_df, clip_model, num_augmentations, args.pca)
    
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
    print(f"PCA components: {args.pca if args.pca else 'None (no dimensionality reduction)'}")
    print(f"CLIP model used: {clip_model}")
    print(f"Files saved:")
    print(f"  - Embeddings: {input_directory}/image_embeddings.npy")
    print(f"  - File paths: {input_directory}/file_paths.npy")
    print(f"  - Updated CSV: {csv_path}")

if __name__ == "__main__":
    main()
