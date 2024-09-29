import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.manifold import TSNE
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import base64
from PIL import Image

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

def encode_image(image_path):
    """Encode an image to Base64."""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def create_dash_app(reduced_embeddings, label_names, file_paths, limit=10):
    """
    Create and configure the Dash app.

    Parameters:
    - reduced_embeddings: NumPy array of reduced embeddings.
    - label_names: List of label names corresponding to embeddings.
    - file_paths: List of image file paths.
    - limit: Maximum number of images to embed in the plot.
    """
    app = Dash(__name__)

    # Pre-encode all images
    encoded_images = [encode_image(path) for path in file_paths]

    # Select indices for embedding images (e.g., first 'limit' valid images)
    embedded_indices = []
    for idx, img in enumerate(encoded_images):
        if img:
            embedded_indices.append(idx)
        if len(embedded_indices) == limit:
            break

    # Define the layout
    app.layout = html.Div([
        html.H1("3D Visualization of Image Embeddings with Images"),
        dcc.Graph(id='scatter-plot', style={
            'width': '100%',
            'height': '800px'
        }),
        html.Div([
            html.Img(id='selected-image', style={
                'max-width': '100%',
                'max-height': '100%',
                'object-fit': 'contain',
            }),
            html.P(id='image-label', style={'text-align': 'center'})
        ], style={
            'width': '30%',
            'height': '800px',
            'display': 'inline-block',
            'vertical-align': 'top',
            'border': '1px solid #ddd',
            'padding': '10px',
            'box-sizing': 'border-box',
            'position': 'absolute',
            'right': '10px',
            'top': '100px'
        })
    ], style={'position': 'relative', 'width': '100%', 'height': '800px'})

    # Create the scatter plot with limited image annotations
    def generate_figure():
        traces = []
        unique_labels = sorted(set(label_names))
        color_scale = px.colors.qualitative.Plotly[:len(unique_labels)]
        color_map = {label: color for label, color in zip(unique_labels, color_scale)}

        for label in unique_labels:
            mask = np.array(label_names) == label
            traces.append(go.Scatter3d(
                x=reduced_embeddings[mask, 0],
                y=reduced_embeddings[mask, 1],
                z=reduced_embeddings[mask, 2],
                mode='markers',
                name=label,
                marker=dict(
                    size=5,
                    color=color_map[label],
                    opacity=0.8
                ),
                text=[f"Label: {name}" for name in np.array(label_names)[mask]],
                hoverinfo='text',
                customdata=np.arange(len(label_names))[mask]
            ))

        fig = go.Figure(data=traces)

        # Add images as annotations for limited points
        for idx in embedded_indices:
            x, y, z = reduced_embeddings[idx]
            img_src = encoded_images[idx]
            if img_src:
                fig.add_layout_image(
                    dict(
                        source=img_src,
                        xref="x",
                        yref="y",
                        zref="z",
                        x=x,
                        y=y,
                        z=z,
                        sizex=2,  # Adjust size as needed
                        sizey=2,
                        sizex_units="x",
                        sizey_units="y",
                        sizing="contain",
                        opacity=0.8,
                        layer="above"
                    )
                )

        fig.update_layout(
            title='3D Visualization of Image Embeddings with Images',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            legend_title='Labels',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            uirevision='constant',
            height=800,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    # Initial figure
    app.layout.children[1].figure = generate_figure()

    # Update the selected image and label
    @app.callback(
        [Output('selected-image', 'src'),
         Output('selected-image', 'style'),
         Output('image-label', 'children')],
        Input('scatter-plot', 'clickData')
    )
    def display_click_image(clickData):
        if clickData and clickData['points']:
            point_index = clickData['points'][0]['customdata']
            image_path = file_paths[point_index]
            label = label_names[point_index]
            try:
                encoded_image = encode_image(image_path)
                with Image.open(image_path) as img:
                    width, height = img.size

                aspect_ratio = width / height

                if aspect_ratio > 1:
                    style = {
                        'max-width': '100%',
                        'height': 'auto',
                        'object-fit': 'contain',
                    }
                else:
                    style = {
                        'max-height': '100%',
                        'width': 'auto',
                        'object-fit': 'contain',
                    }

                return encoded_image, style, f"Label: {label}"
            except Exception as e:
                print(f"Error loading image: {e}")
                return "", {}, "Error loading image"
        return "", {}, ""

    return app

def main():
    # Configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    root_directory = os.path.join(project_root, 'data', 'normalized_3d_test')

    # Load data
    embeddings, label_ids, file_paths, label_names = load_data(root_directory)
    if embeddings is None:
        print("Failed to load data. Exiting.")
        return

    # Dimensionality reduction
    reduced_embeddings = perform_dimensionality_reduction(embeddings)
    if reduced_embeddings is None:
        print("Dimensionality reduction failed. Exiting.")
        return

    # Create and run the Dash app with a limit of 10 images
    app = create_dash_app(reduced_embeddings, label_names, file_paths, limit=10)
    print("Starting the Dash app. Please open a web browser and go to http://127.0.0.1:8050/")
    app.run_server(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()