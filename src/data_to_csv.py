import pandas as pd
import os
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from torchvision.transforms import functional as F
from PIL import ImageFile
import cv2
import albumentations as A
from joblib import Parallel, delayed
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images
"""
-> Looks at the path, looks at the subfolders, and creates a CSV file with the dataset information like : image_path, image_name, label_name, is_test
"""

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def process_image(old_path, new_path, label_name, is_test, force_overwrite=True):
    try:
        # Store absolute path of original image
        original_path = os.path.abspath(old_path)
        
        # Always process the image, no early returns
        # Read with PIL first to handle CMYK conversion
        with Image.open(old_path) as pil_img:
            # Convert CMYK to RGB if needed
            if pil_img.mode in ['CMYK', 'P']:
                pil_img = pil_img.convert('RGB')
            
            # Convert to numpy array for albumentations
            img = np.array(pil_img)
            
            # Apply transformation
            transform = A.Compose([
                A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4)
            ])
            
            transformed = transform(image=img)['image']
            
            # Convert back to PIL and save with minimal metadata
            output_img = Image.fromarray(transformed)
            
            # Strip all metadata and profiles
            output_img.info.clear()
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Save with minimal options, overwriting any existing file
            output_img.save(
                new_path, 
                'PNG',
                optimize=True,
                icc_profile=None,
                pnginfo=None
            )
        
        return new_path, os.path.basename(new_path), label_name, is_test, original_path
    except Exception as e:
        print(f"Error processing {old_path}: {str(e)}")
        return None

def normalize_data(root_directory, output_directory, images_per_class=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    tasks = []
    moved_to_test = {}  # Track statistics
    
    # Process label folders
    label_folders = [d for d in os.listdir(root_directory) 
                    if os.path.isdir(os.path.join(root_directory, d)) 
                    and d != "test_data"]
    
    # Ensure test_data directory exists
    new_test_path = os.path.join(output_directory, "test_data")
    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)

    for label_name in label_folders:
        label_path = os.path.join(root_directory, label_name)
        new_label_path = os.path.join(output_directory, label_name)
        if not os.path.exists(new_label_path):
            os.makedirs(new_label_path)
        
        # Get all image files for this label
        image_files = [f for f in os.listdir(label_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        moved_to_test[label_name] = 0
        
        for idx, file in enumerate(image_files):
            old_path = os.path.join(label_path, file)
            
            # Decide if this should go to test based on the limit
            if images_per_class and idx >= images_per_class:
                new_name = f"test_{label_name}_{idx:04d}.png"
                new_path = os.path.join(new_test_path, new_name)
                tasks.append((old_path, new_path, "test", True))
                moved_to_test[label_name] += 1
            else:
                new_name = f"{label_name}_{idx:04d}.png"
                new_path = os.path.join(new_label_path, new_name)
                tasks.append((old_path, new_path, label_name, False))

    # Process images in root directory as test images
    root_images = [f for f in os.listdir(root_directory) 
                   if os.path.isfile(os.path.join(root_directory, f))
                   and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for idx, file in enumerate(root_images):
        old_path = os.path.join(root_directory, file)
        new_name = f"test_root_{idx:04d}.png"
        new_path = os.path.join(new_test_path, new_name)
        tasks.append((old_path, new_path, "test", True))

    # Process existing test_data folder
    test_data_path = os.path.join(root_directory, "test_data")
    if os.path.exists(test_data_path):
        for idx, file in enumerate(os.listdir(test_data_path)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                old_path = os.path.join(test_data_path, file)
                new_name = f"test_orig_{idx:04d}.png"
                new_path = os.path.join(new_test_path, new_name)
                tasks.append((old_path, new_path, "test", True))

    # Process all images in parallel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_image)(*task) for task in tqdm(tasks)
    )
    
    # Print statistics
    print("\nImages moved to test set:")
    for label, count in moved_to_test.items():
        if count > 0:
            print(f"{label}: {count} images")
    
    # Filter out None results from failed processing
    results = [r for r in results if r is not None]
    return zip(*results)

def create_image_dataframe(root_directory, output_directory, images_per_class=None):
    image_paths, image_names, label_names, is_test, original_paths = normalize_data(root_directory, output_directory, images_per_class)
    
    # Create DataFrame with original_path added
    df = pd.DataFrame({
        'image_path': image_paths,
        'image_name': image_names,
        'label_name': label_names,
        'is_test': is_test,
        'original_path': original_paths
    })
    
    # Create label_id using simple incremental approach
    unique_labels = df[df['label_name'] != "test"]['label_name'].unique()
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    label_to_id["test"] = -1  # Ensure test data keeps -1 as label_id
    
    df['label_id'] = df['label_name'].map(label_to_id)
    
    # Initialize label_prediction column with NaN
    df['label_prediction'] = pd.NA
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Process image dataset and create a DataFrame.")
    parser.add_argument("-i", "--input_directory", required=True, help="Path to the input image directory")
    parser.add_argument("-o", "--output_directory", help="Path to the output directory (optional)")
    parser.add_argument("--images-per-class", type=int, help="Maximum number of images per class (excess will be moved to test)")
    args = parser.parse_args()
    print(f"Input directory: {args.input_directory}")
    root_directory = os.path.abspath(args.input_directory)
    
    # Determine output directory
    if args.output_directory:
        output_directory = os.path.abspath(args.output_directory)
    elif os.path.basename(root_directory).startswith("normalized_"):
        output_directory = root_directory
    else:
        # Find the project root and create normalized directory in datasets/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        datasets_dir = os.path.join(project_root, 'datasets', 'normalized')
        dataset_name = os.path.basename(root_directory)
        output_directory = os.path.join(datasets_dir, dataset_name)
    
    print(f"Output directory: {output_directory}")
    
    image_df = create_image_dataframe(root_directory, output_directory, args.images_per_class)

    # Save the DataFrame to a CSV file in the normalized output directory
    csv_path = os.path.join(output_directory, 'image_classifier_data.csv')
    image_df.to_csv(csv_path, index=False)

    # Print some information about the dataset
    print(f"Total images: {len(image_df)}")
    print(f"Training images: {len(image_df[~image_df['is_test']])}")
    print(f"Test images: {len(image_df[image_df['is_test']])}")
    print(f"Unique labels: {image_df['label_name'].nunique() - 1}")  # -1 to exclude test label

    # Verify image sizes
    print("\nVerifying image sizes...")
    incorrect_sizes = []
    for idx, row in image_df.iterrows():
        with Image.open(row['image_path']) as img:
            if img.size != (224, 224):
                incorrect_sizes.append((row['image_path'], img.size))
    
    if incorrect_sizes:
        print("Found images with incorrect sizes:")
        for path, size in incorrect_sizes:
            print(f"{path}: {size}")
    else:
        print("All images are correctly sized to 224x224")

if __name__ == "__main__":
    main()
