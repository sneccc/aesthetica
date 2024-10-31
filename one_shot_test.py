import torch
import pathlib
import open_clip
from train_model import MultiLayerPerceptron
from PIL import Image
import os
from tqdm import tqdm
import albumentations as A
import cv2
import numpy as np
import argparse
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
def load_model_and_config(model_path):
    """Load model and its configuration"""
    # Convert string path to Path object
    model_path = pathlib.Path(model_path)
    
    # Load config
    config_path = model_path.parent / "model_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model with config parameters
    model = MultiLayerPerceptron(
        input_size=config['input_size'],
        num_classes=config['num_classes'],
        hidden_units=tuple(config['hidden_units'])
    )
    
    # Load weights
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config

def preprocess_image(image_path, output_path):
    """Normalize image to 224x224 using Lanczos resampling"""
    try:
        with Image.open(image_path) as pil_img:
            # Convert CMYK/P to RGB if needed
            if pil_img.mode in ['CMYK', 'P']:
                pil_img = pil_img.convert('RGB')
            
            # Convert to numpy for albumentations
            img = np.array(pil_img)
            
            transform = A.Compose([
                A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4)
            ])
            
            transformed = transform(image=img)['image']
            output_img = Image.fromarray(transformed)
            
            # Strip metadata and save
            output_img.info.clear()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path, 'PNG', optimize=True, icc_profile=None, pnginfo=None)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def prepare_clip_models(clip_models):
    """Initialize CLIP models and preprocessors"""
    models = []
    preprocessors = []
    total_dim = 0
    
    for clip_model in clip_models:
        if clip_model[0] == "hf-hub:timm":
            config = open_clip.get_model_config(clip_model[1])
            model, preprocess = open_clip.create_model_from_pretrained(
                clip_model[0] + "/" + clip_model[1])
            model.to(device)
        else:
            config = open_clip.get_model_config(clip_model[0])
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model[0], pretrained=clip_model[1], device=device)
            
        if config is not None and 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")
        
        models.append(model)
        preprocessors.append(preprocess)
    
    return models, preprocessors, total_dim

def process_folder(input_folder, model_path, output_folder, batch_size=64):
    """Process a folder of images through preprocessing, embedding, and classification"""
    input_folder = pathlib.Path(input_folder)
    output_folder = pathlib.Path(output_folder)
    
    # Create processing folder for normalized images
    process_folder = output_folder / "processed"
    process_folder.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocess images
    print("Preprocessing images...")
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        image_files.extend(input_folder.glob(ext))
    
    processed_paths = []
    for img_path in tqdm(image_files, desc="Normalizing images"):
        output_path = process_folder / img_path.name
        if preprocess_image(img_path, output_path):
            processed_paths.append(output_path)
    
    if not processed_paths:
        print("No valid images found to process!")
        return
    
    # Load model and config
    mlp_model, config = load_model_and_config(model_path)
    class_labels = config['class_labels']  # Get class labels from config
    
    # Initialize CLIP models from config
    clip_models = config['clip_models']
    models, preprocessors, total_dim = prepare_clip_models(clip_models)
    
    # Verify dimensions match
    assert total_dim == config['input_size'], f"CLIP model dimension {total_dim} doesn't match config {config['input_size']}"

    # Step 3: Process images in batches
    print("\nGenerating predictions...")
    predictions = []
    
    for start_idx in tqdm(range(0, len(processed_paths), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(processed_paths))
        batch_paths = processed_paths[start_idx:end_idx]
        
        batch_images = []
        current_paths = []
        
        for img_path in batch_paths:
            try:
                with Image.open(img_path) as pil_image:
                    images = [preprocessor(pil_image).unsqueeze(0).to(device) 
                             for preprocessor in preprocessors]
                    batch_images.append(torch.cat(images, dim=0))
                    current_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        if not batch_images:
            continue
        
        # Generate embeddings and predictions
        batch_tensor = torch.cat(batch_images, dim=0)
        with torch.no_grad(), torch.amp.autocast(device):
            image_features = []
            for model in models:
                features = model.encode_image(batch_tensor)
                image_features.append(features)
            
            concatenated_features = torch.cat(image_features, dim=1)
            im_emb_arr = concatenated_features.cpu().detach().numpy()
            prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device))
            predicted_labels = torch.argmax(prediction, dim=1).cpu().numpy()
            
            for path, label in zip(current_paths, predicted_labels):
                predictions.append((path, label))
    
    # Step 4: Organize results into folders
    print("\nOrganizing results...")
    for img_path, label in predictions:
        # Create label folder using both numerical label and name
        label_name = class_labels[label]
        label_folder = output_folder / f"class_{label}_{label_name}"
        label_folder.mkdir(exist_ok=True)
        
        # Copy original image to label folder
        original_path = input_folder / img_path.name
        if original_path.exists():
            import shutil
            shutil.copy2(original_path, label_folder / img_path.name)
    
    print(f"\nProcessing complete! Results saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Process a folder of images through classification model")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to input image folder")
    parser.add_argument("-m", "--model_path", required=True, help="Path to trained model.pth")
    parser.add_argument("-o", "--output_folder", required=True, help="Path to output folder")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for processing")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.model_path, args.output_folder, args.batch_size)

if __name__ == "__main__":
    main()