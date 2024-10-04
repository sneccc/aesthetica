import torch
import open_clip
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(old_path, new_path):
    if os.path.exists(new_path):
        return new_path, os.path.basename(new_path)

    try:
        with Image.open(old_path) as img:
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            img.save(new_path, 'PNG', optimize=True)
        return new_path, os.path.basename(new_path)
    except Exception as e:
        print(f"Error processing {old_path}: {str(e)}")
        return None

def normalize_data(image_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_data = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for file in os.listdir(image_directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                old_path = os.path.join(image_directory, file)
                new_name = f"image_{len(futures):04d}.png"
                new_path = os.path.join(output_directory, new_name)
                futures.append(executor.submit(process_image, old_path, new_path))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing images"):
            result = future.result()
            if result:
                image_data.append(result)

    return zip(*image_data)

def prepare_embeddings(image_directory, clip_models):
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
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for start_idx in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[start_idx:start_idx + batch_size]
        
        batch_images = []
        batch_paths = []
        
        for image_file in batch_files:
            try:
                image_path = os.path.join(image_directory, image_file)
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
        
        batch_embeddings = []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for model in models:
                features = model.encode_image(batch_images_tensor)
                batch_embeddings.append(features.detach().cpu().numpy())
        
        np_batch_embeddings = np.concatenate(batch_embeddings, axis=1)
        
        image_embeddings.extend(np_batch_embeddings)
        file_paths.extend(batch_paths)
    
    return image_embeddings, file_paths

def main():
    parser = argparse.ArgumentParser(description="Process image dataset, create embeddings, and save CSV file.")
    parser.add_argument("-i", "--input_directory", required=True, help="Path to the input image directory")
    args = parser.parse_args()

    input_directory = os.path.abspath(args.input_directory)
    output_directory = os.path.join(os.path.dirname(input_directory), f"normalized_{os.path.basename(input_directory)}")
    
    # Normalize data
    print("Normalizing images...")
    image_paths, image_names = normalize_data(input_directory, output_directory)
    
    # Prepare embeddings
    print("Preparing embeddings...")
    clip_model = [("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")]
    image_embeddings, file_paths = prepare_embeddings(output_directory, clip_model)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': file_paths,
        'image_name': [os.path.basename(path) for path in file_paths],
        'label_name': 'no_label',
        'is_test': False,
        'label_id': -2,
        'label_prediction': pd.NA
    })
    
    # Save embeddings and file paths
    np.save(f"{output_directory}/image_embeddings.npy", image_embeddings)
    np.save(f"{output_directory}/file_paths.npy", file_paths)
    
    # Save CSV file
    csv_path = os.path.join(output_directory, 'image_classifier_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Total images processed: {len(df)}")
    print(f"Embeddings saved to: {output_directory}/image_embeddings.npy")
    print(f"File paths saved to: {output_directory}/file_paths.npy")
    print(f"CSV file saved to: {csv_path}")

if __name__ == "__main__":
    main()