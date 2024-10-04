import torch
import open_clip
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import os

def prepare_embeddings(root_directory, clip_models):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # transform path into absolute path
    root_directory = os.path.abspath(root_directory)
    path = os.path.join(root_directory, "image_classifier_data.csv")
    database = pd.read_csv(path)
    
    df = database[database.label_id != -1].reset_index(drop=True)  # remove test data
    
    for clip_model in clip_models:
        print("Clip Model -> ", clip_model[0], clip_model[1], "full is ", clip_model, " type is ->", type(clip_model))
    
    models = []
    preprocessors = []
    
    for clip_model in clip_models:
        if clip_model[0] == "hf-hub:timm":
            model, preprocess = open_clip.create_model_from_pretrained(clip_model[0] + "/" + clip_model[1], device=device)
            model.to(device)
            model.eval()
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1], device=device)
        
        models.append(model)
        preprocessors.append(preprocess)
    
    # input features, labels, and file paths
    image_embeddings = []
    class_labels = []
    file_paths = []
    batch_size = 256
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]
        
        # prepare batch data
        batch_images = []
        batch_labels = []
        batch_paths = []
        
        for _, row in batch_df.iterrows():
            try:
                image = Image.open(row.image_path).convert("RGB")
                preprocessed_images = [preprocessor(image).unsqueeze(0).to(device) for preprocessor in preprocessors]
                batch_images.append(torch.cat(preprocessed_images, dim=0))
                batch_labels.append(float(row.label_id))
                batch_paths.append(row.image_path)
            except Exception as e:
                print(f"Error processing image {row.image_path}: {str(e)}")
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
        
        np_labels = np.array(batch_labels).astype(int)
        class_labels.extend(np_labels)
        file_paths.extend(batch_paths)
        
    # save embeddings, labels, and file paths in the root directory
    np.save(f"{root_directory}/image_embeddings.npy", image_embeddings)
    np.save(f"{root_directory}/class_labels.npy", class_labels)
    np.save(f"{root_directory}/file_paths.npy", file_paths)

    # Return the data as well
    return image_embeddings, class_labels, file_paths



if __name__ == "__main__":
    clip_model=[("hf-hub:timm","ViT-SO400M-14-SigLIP-384")]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.join(current_dir, "../data/normalized_art_categories")
    
    image_embeddings, class_labels, file_paths = prepare_embeddings(root_directory, clip_model)
