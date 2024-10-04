import pandas as pd
import os
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# Get the full path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def process_image(old_path, new_path, label_name, is_test):
    if os.path.exists(new_path):
        return new_path, os.path.basename(new_path), label_name, is_test

    try:
        with Image.open(old_path) as img:
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            img.save(new_path, 'PNG', optimize=True)
        return new_path, os.path.basename(new_path), label_name, is_test
    except Exception as e:
        print(f"Error processing {old_path}: {str(e)}")
        return None

def normalize_data(root_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_data = []
    
    # Process label folders (excluding test_data)
    label_folders = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d)) and d != "test_data"]
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for label_name in label_folders:
            label_path = os.path.join(root_directory, label_name)
            new_label_path = os.path.join(output_directory, label_name)
            if not os.path.exists(new_label_path):
                os.makedirs(new_label_path)
            
            for file in os.listdir(label_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    old_path = os.path.join(label_path, file)
                    new_name = f"{label_name}_{len(futures):04d}.png"
                    new_path = os.path.join(new_label_path, new_name)
                    futures.append(executor.submit(process_image, old_path, new_path, label_name, False))

        # Process test_data folder
        test_data_path = os.path.join(root_directory, "test_data")
        new_test_data_path = os.path.join(output_directory, "test_data")
        if os.path.exists(test_data_path):
            if not os.path.exists(new_test_data_path):
                os.makedirs(new_test_data_path)
            
            for file in os.listdir(test_data_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    old_path = os.path.join(test_data_path, file)
                    new_name = f"test_{len(futures):04d}.png"
                    new_path = os.path.join(new_test_data_path, new_name)
                    futures.append(executor.submit(process_image, old_path, new_path, "test", True))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result:
                image_data.append(result)

    return zip(*image_data)

def create_image_dataframe(root_directory, output_directory):
    image_paths, image_names, label_names, is_test = normalize_data(root_directory, output_directory)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'image_name': image_names,
        'label_name': label_names,
        'is_test': is_test
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
    args = parser.parse_args()
    print(f"Input directory: {args.input_directory}")
    root_directory = os.path.abspath(args.input_directory)
    output_directory = os.path.join(os.path.dirname(root_directory), f"normalized_{os.path.basename(root_directory)}")

    image_df = create_image_dataframe(root_directory, output_directory)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(project_root, 'image_classifier_data.csv')
    image_df.to_csv(csv_path, index=False)

    # Print some information about the dataset
    print(f"Total images: {len(image_df)}")
    print(f"Training images: {len(image_df[~image_df['is_test']])}")
    print(f"Test images: {len(image_df[image_df['is_test']])}")
    print(f"Unique labels: {image_df['label_name'].nunique() - 1}")  # -1 to exclude test label

if __name__ == "__main__":
    main()