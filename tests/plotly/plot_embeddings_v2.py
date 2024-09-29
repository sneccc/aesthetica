import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import umap
import os
from PIL import Image
import base64
from io import BytesIO
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Configuration
root_directory = f'{project_root}/data/normalized_3d_test'
embeddings_file = f'{root_directory}/image_embeddings.npy'
labels_file = f'{root_directory}/class_labels.npy'
paths_file = f'{root_directory}/file_paths.npy'

# Load embeddings, labels, and file paths
embeddings = np.load(os.path.join(root_directory, embeddings_file))
label_ids = np.load(os.path.join(root_directory, labels_file))
file_paths = np.load(os.path.join(root_directory, paths_file))

# Load or create label names
# Assuming we have a CSV file with label_id and label_name columns
label_df = pd.read_csv(f'{root_directory}/image_classifier_data.csv')
label_id_to_name = dict(zip(label_df['label_id'], label_df['label_name']))

# Map label_ids to label_names
label_names = [label_id_to_name.get(id, f"Unknown ({id})") for id in label_ids]

# Perform dimensionality reduction
reducer = TSNE(n_components=3, perplexity=30, n_iter=1000)
reduced_embeddings = reducer.fit_transform(embeddings)

# Create the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("3D Visualization of Image Embeddings"),
    html.Div([
        dcc.Graph(id='scatter-plot', style={
            'width': '70%',
            'height': '800px',
            'display': 'inline-block'
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
        })
    ], style={'width': '100%', 'height': '800px'})
])

# Create the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-plot', 'clickData')
)
def update_graph(clickData):
    # Create a list of unique labels and their corresponding colors
    unique_labels = sorted(set(label_names))
    color_scale = px.colors.qualitative.Plotly[:len(unique_labels)]
    color_map = {label: color for label, color in zip(unique_labels, color_scale)}

    # Create traces for each label
    traces = []
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
            customdata=np.arange(len(label_ids))[mask]
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title='3D Visualization of Image Embeddings',
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

# Update the selected image and label
@app.callback(
    Output('selected-image', 'src'),
    Output('selected-image', 'style'),
    Output('image-label', 'children'),
    Input('scatter-plot', 'clickData')
)
def display_click_image(clickData):
    if clickData and clickData['points']:
        point_index = clickData['points'][0]['customdata']
        image_path = file_paths[point_index]
        label = label_names[point_index]
        try:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            
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
            
            return f"data:image/png;base64,{encoded_image}", style, f"Label: {label}"
        except Exception as e:
            print(f"Error loading image: {e}")
            return "", {}, "Error loading image"
    return "", {}, ""

# Run the app
if __name__ == '__main__':
    print("Starting the Dash app. Please open a web browser and go to http://127.0.0.1:8050/")
    app.run_server(debug=True, use_reloader=False)