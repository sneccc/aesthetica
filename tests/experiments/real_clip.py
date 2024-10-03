import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_real_clip():
    """Load the CLIP model"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def encode_images(model, preprocess, image_paths):
    """Encode images using CLIP"""
    encoded_images = []
    for img_path in tqdm(image_paths, desc="Encoding images"):
        print("processing image: ", img_path)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        encoded_images.append(image_features)
    return torch.cat(encoded_images)

def encode_text(model, text_prompts):
    """Encode text prompts using CLIP"""
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

def compute_cosine_similarities(image_features, text_features):
    """Compute cosine similarities between image and text features"""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return (100.0 * image_features @ text_features.T).softmax(dim=-1)

def autoclip_aggregation(cosine_similarities, beta=0.85):
    # ... (keep the existing autoclip_aggregation function, but ensure it uses CUDA tensors)
    pass  # Placeholder for brevity

def run_real_clip_experiment(image_paths, class_names, prompt_templates, beta=0.85):
    model, preprocess = load_real_clip()
    
    # Encode images
    image_features = encode_images(model, preprocess, image_paths)
    
    # Generate text prompts
    text_prompts = [template.format(c) for c in class_names for template in prompt_templates]
    
    # Encode text prompts
    text_features = encode_text(model, text_prompts)
    
    # Compute cosine similarities
    cosine_similarities = compute_cosine_similarities(image_features, text_features)
    
    # Reshape cosine similarities to match the expected shape (icgd)
    n_instances = len(image_paths)
    n_classes = len(class_names)
    n_descriptors = len(prompt_templates)
    cosine_similarities = cosine_similarities.view(n_instances, n_classes, n_descriptors)
    
    # Perform AutoCLIP aggregation
    logits = autoclip_aggregation(cosine_similarities, beta)
    
    # Compute classifications
    classifications = logits.argmax(dim=1)
    
    return classifications

# Example usage
if __name__ == "__main__":
    import os
    image_paths = ["data/normalized_3d_test/thumbnails/bad_0000.png", "data/normalized_3d_test/thumbnails/bad_0001.png"]
    #transform in full path using current working directory
    image_paths = [os.path.join(os.getcwd(), img_path) for img_path in image_paths]
    
    class_names = ["dog", "cat", "bird", ...]
    prompt_templates = ["a photo of a {}", "an image of a {}"]
    
    classifications = run_real_clip_experiment(image_paths, class_names, prompt_templates)
    print("Classifications:", classifications)