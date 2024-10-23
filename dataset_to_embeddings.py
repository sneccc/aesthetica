import torch
import open_clip
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import pandas as pd
import argparse

def prepare_embeddings(df, clip_models):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models = []
    preprocessors = []
    
    for clip_model in clip_models:
        if clip_model[0] == "hf-hub:timm":
            model, preprocess = open_clip.create_model_from_pretrained(clip_model[0] + "/" + clip_model[1], device=device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1], device=device)
        
        model.to(device)
        model.eval()
        models.append(model)
        preprocessors.append(preprocess)
    
    image_embeddings = []
    file_paths = []
    batch_size = 256
    
    # Use tqdm to show progress over the DataFrame
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        
        batch_images = []
        batch_paths = []
        
        for _, row in batch_df.iterrows():
            try:
                image_path = row['image_path']
                # Images are already preprocessed to 224x224
                image = Image.open(image_path).convert("RGB")
                preprocessed_images = [preprocessor(image).unsqueeze(0).to(device) for preprocessor in preprocessors]
                batch_images.append(torch.cat(preprocessed_images, dim=0))
                batch_paths.append(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue
        
        if not batch_images:
            continue
        
        batch_images_tensor = torch.cat(batch_images, dim=0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_embeddings = []
            for model in models:
                features = model.encode_image(batch_images_tensor)
                batch_embeddings.append(features.detach().cpu().numpy())
            
            np_batch_embeddings = np.concatenate(batch_embeddings, axis=1)
            
            image_embeddings.extend(np_batch_embeddings)
            file_paths.extend(batch_paths)
    
    return np.array(image_embeddings), file_paths

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for preprocessed images.")
    parser.add_argument("-i", "--input_directory", required=True, help="Path to the normalized image directory")
    args = parser.parse_args()

    input_directory = os.path.abspath(args.input_directory)
    
    # Load the CSV file
    csv_path = os.path.join(input_directory, 'image_classifier_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from CSV")
    
    # Prepare embeddings
    print("Preparing embeddings...")
    clip_model = [("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")]
    image_embeddings, file_paths = prepare_embeddings(df, clip_model)
    
    # Save embeddings and file paths
    np.save(f"{input_directory}/image_embeddings.npy", image_embeddings)
    np.save(f"{input_directory}/file_paths.npy", file_paths)
    
    # Add embeddings to DataFrame
    embedding_df = pd.DataFrame({
        'image_path': file_paths,
        'embedding_index': range(len(file_paths))
    })
    
    # Merge with original DataFrame to maintain all information
    final_df = df.merge(embedding_df, on='image_path', how='left')
    final_df.to_csv(csv_path, index=False)
    
    print(f"\nProcessing complete:")
    print(f"Total images processed: {len(file_paths)}")
    print(f"Embedding shape: {image_embeddings.shape}")
    print(f"Embeddings saved to: {input_directory}/image_embeddings.npy")
    print(f"File paths saved to: {input_directory}/file_paths.npy")
    print(f"Updated CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
