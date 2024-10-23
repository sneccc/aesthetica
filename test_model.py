import torch
import pathlib
import open_clip
from train_model import MultiLayerPerceptron
import pandas as pd
from tqdm import tqdm
from PIL import Image
import shutil

#root_folder here is the folder where the model.pth is located
#the predict will predict the score for all images in the test_data folder
#and will create subfolders based on the predicted labels
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_score(root_folder, clip_models):
    print(f"🐍root_folder: {root_folder}\n🐍clip_models: {clip_models}")
    #paths
    path = pathlib.Path(root_folder)
    csv_path = path / "image_classifier_data.csv"
    model_path = path / "model.pth"

    #load clip models
    models, preprocessors, total_dim = prepare_clip_models(clip_models)
    
    #load csv
    df = pd.read_csv(csv_path)
    #load label_id and label_name as dict, unique values but ignore label_name "test"
    label_id_to_name = dict(zip(df[df['label_name'] != 'test']['label_id'].unique(), df[df['label_name'] != 'test']['label_name'].unique()))
    print(f"🐍label_id_to_name: {label_id_to_name}")
    
    #load model
    mlp_model = MultiLayerPerceptron(total_dim,num_classes=len(label_id_to_name))  
    state_dict = torch.load(model_path)
    mlp_model.load_state_dict(state_dict)
    mlp_model.to(device)
    mlp_model.eval()
    
    #load arbitrary test image folder
    test_folder = path / "test_data"
    test_images = list(test_folder.glob("*.png"))
    print(f"🐍 found {len(test_images)} test images")
    
    
    batch_size = 256
    total_predictions = []

    # Process test images in batches
    indices_to_drop = []
    total_predictions = []
    
    for start_idx in tqdm(range(0, len(test_images), batch_size), desc="Predicting scores"):
        end_idx = min(start_idx + batch_size, len(test_images))
        batch_paths = test_images[start_idx:end_idx]

        batch_images = []
        current_paths = []  # Track current batch paths for successful processing

        for img_path in batch_paths:
            try:
                with Image.open(img_path) as pil_image:
                    images = [preprocessor(pil_image).unsqueeze(0).to(device) for preprocessor in preprocessors]
                    batch_images.append(torch.cat(images, dim=0))
                    current_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}, skipping this image")
                indices_to_drop.append(img_path)

        if not batch_images:
            continue

        # Process batch
        batch_images_tensor = torch.cat(batch_images, dim=0)
        image_features_list = []
        with torch.no_grad():
            for model in models:
                features = model.encode_image(batch_images_tensor)
                image_features_list.append(features)

        concatenated_features = torch.cat(image_features_list, dim=1)
        im_emb_arr = concatenated_features.cpu().detach().numpy()

        with torch.no_grad(), torch.cuda.amp.autocast():
            prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device))
            predicted_labels = torch.argmax(prediction, dim=1).cpu().numpy()

        # Create results for successful predictions
        for path, label in zip(current_paths, predicted_labels):
            predicted_name = label_id_to_name[label]
            total_predictions.append((path, predicted_name))

    # Create directories and move files
    prediction_folder = test_folder.parent / "prediction"
    prediction_folder.mkdir(exist_ok=True)
    
    for img_path, label_name in total_predictions:
        target_dir = prediction_folder / label_name
        target_dir.mkdir(exist_ok=True)
        
        # Copy the file to the predicted label directory instead of moving
        img_path = pathlib.Path(img_path)
        import shutil
        shutil.copy2(img_path, target_dir / img_path.name)


def prepare_clip_models(clip_models):
    models = []
    preprocessors = []
    total_dim = 0
    for clip_model in clip_models:
        if clip_model[0] == "hf-hub:timm":
            config = open_clip.get_model_config(clip_model[1])  # ["embed_dim"]
            print(f"SigLip model with {config['embed_dim']} dimension")
            model, preprocess = open_clip.create_model_from_pretrained(
                clip_model[0] + "/" + clip_model[1])  # for hf-hub:timm/ViT-SO400M-14-SigLIP-384 format
            model.to(device)
        else:
            config = open_clip.get_model_config(clip_model[0])
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1],
                                                                         device=device)
        if config is not None and 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")
        
        models.append(model)
        preprocessors.append(preprocess)
    
    return models, preprocessors, total_dim
