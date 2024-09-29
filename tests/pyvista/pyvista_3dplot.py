import os
import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.manifold import TSNE
from PIL import Image
import base64
import io
from tqdm import tqdm

def load_data(root_directory):
    """Load embeddings, labels, and file paths."""
    try:
        embeddings = np.load(os.path.join(root_directory, 'image_embeddings.npy'))
        label_ids = np.load(os.path.join(root_directory, 'class_labels.npy'))
        file_paths = np.load(os.path.join(root_directory, 'file_paths.npy'))
        label_df = pd.read_csv(os.path.join(root_directory, 'image_classifier_data.csv'))
        label_id_to_name = dict(zip(label_df['label_id'], label_df['label_name']))
        label_names = [label_id_to_name.get(id, f"Unknown ({id})") for id in label_ids]
        return embeddings, label_ids, file_paths, label_names
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def perform_dimensionality_reduction(embeddings):
    """Perform TSNE dimensionality reduction."""
    try:
        reducer = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        return None

def process_thumbnails(file_paths, thumbnail_size=(256, 256), cache_dir='thumbnails'):
    """Process and cache thumbnails for the given file paths."""
    os.makedirs(cache_dir, exist_ok=True)
    thumbnails = []

    for file_path in tqdm(file_paths, desc="Processing thumbnails"):
        cache_path = os.path.join(cache_dir, os.path.basename(file_path) + '.thumbnail')
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                thumbnail = Image.open(io.BytesIO(f.read()))
        else:
            try:
                img = Image.open(file_path)
                img.thumbnail(thumbnail_size)
                thumbnail = img.convert('RGB')
                
                with open(cache_path, 'wb') as f:
                    thumbnail.save(f, format='PNG')
            except Exception as e:
                print(f"Error processing thumbnail for {file_path}: {e}")
                thumbnail = Image.new('RGB', thumbnail_size, color='gray')
        
        thumbnails.append(thumbnail)
    
    return thumbnails

def create_pyvista_plot(reduced_embeddings, label_names, thumbnails):
    """Create a PyVista 3D plot of the embeddings."""
    # Create a PyVista point cloud
    cloud = pv.PolyData(reduced_embeddings)
    
    # Add labels as a scalar array
    cloud['labels'] = label_names
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Add the point cloud to the plotter
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=10, scalars='labels', cmap='Set1')
    
    # Add image planes for all images
    for i, thumbnail in enumerate(thumbnails):
        # Convert image to numpy array
        img_array = np.array(thumbnail)
        
        # Create a PyVista image plane
        plane = pv.Plane(center=reduced_embeddings[i], direction=(0, 0, 1), i_size=2, j_size=2)
        
        # Add the image texture to the plane
        tex = pv.numpy_to_texture(img_array)
        plotter.add_mesh(plane, texture=tex)
    
    # Set up camera and other plot properties
    plotter.set_background('white')
    plotter.add_axes()
    plotter.add_scalar_bar('Labels', vertical=True)
    
    return plotter

def main():
    # Configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    root_directory = os.path.join(project_root, 'data', 'normalized_3d_test')
    thumbnail_cache_dir = os.path.join(project_root, 'data', 'thumbnail_cache')

    # Load data
    embeddings, label_ids, file_paths, label_names = load_data(root_directory)
    if embeddings is None:
        print("Failed to load data. Exiting.")
        return

    # Process thumbnails
    thumbnails = process_thumbnails(file_paths, cache_dir=thumbnail_cache_dir)

    # Dimensionality reduction
    reduced_embeddings = perform_dimensionality_reduction(embeddings)
    if reduced_embeddings is None:
        print("Dimensionality reduction failed. Exiting.")
        return

    # Create and show the PyVista plot
    plotter = create_pyvista_plot(reduced_embeddings, label_names, thumbnails)
    plotter.show()

if __name__ == '__main__':
    main()
