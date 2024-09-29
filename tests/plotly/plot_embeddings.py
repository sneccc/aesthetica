import torch
import numpy as np
import plotly.graph_objs as go
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
import os
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import clip
from PIL import Image
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import umap
 
# Configuration
positive_class_idx = 3  # 'cat' class in CIFAR-10
cavtemp_dir = 'cavtemp_cifar10'
os.makedirs(cavtemp_dir, exist_ok=True)
 
# Dataset transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match CLIP's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
# Load CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
# Combine train and test sets for more data
cifar10_dataset = cifar10_train + cifar10_test
 
# Filter dataset to create positive and negative examples
positive_indices = [i for i, (img, label) in enumerate(cifar10_dataset) if label == positive_class_idx]
negative_indices = [i for i, (img, label) in enumerate(cifar10_dataset) if label != positive_class_idx]
 
positive_dataset = torch.utils.data.Subset(cifar10_dataset, positive_indices)
negative_dataset = torch.utils.data.Subset(cifar10_dataset, negative_indices)
 
# Define dataloaders
positive_loader = DataLoader(positive_dataset, batch_size=32, shuffle=True)
negative_loader = DataLoader(negative_dataset, batch_size=32, shuffle=True)
 
# Function to generate embeddings
def generate_embeddings(model, data_loader):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda()
            labels = labels.cuda()
            image_features = model.encode_image(images)
            all_embeddings.append(image_features.cpu())
            all_labels.append(labels.cpu())
 
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    return all_embeddings, all_labels
 
# Load the CLIP model
model, preprocess = clip.load("ViT-L/14", device="cuda:0")
 
# Generate or load embeddings
positive_embeddings_path = os.path.join(cavtemp_dir, f'cifar10_cat_embeddings.pt')
negative_embeddings_path = os.path.join(cavtemp_dir, 'cifar10_negative_embeddings.pt')
 
try:
    positive_embeddings = torch.load(positive_embeddings_path)
    negative_embeddings = torch.load(negative_embeddings_path)
except:
    print("No saved embeddings found, generating...")
    positive_embeddings, _ = generate_embeddings(model, positive_loader)
    negative_embeddings, _ = generate_embeddings(model, negative_loader)
    torch.save(positive_embeddings, positive_embeddings_path)
    torch.save(negative_embeddings, negative_embeddings_path)
 
# Create labels for the embeddings
positive_labels = torch.ones(positive_embeddings.size(0))
negative_labels = torch.zeros(negative_embeddings.size(0))
 
# Combine embeddings and labels
all_embeddings = torch.cat([positive_embeddings, negative_embeddings])
all_labels = torch.cat([positive_labels, negative_labels])
 
# Flatten embeddings for dimensionality reduction
flat_embeddings = all_embeddings.view(all_embeddings.size(0), -1).numpy()
 
# Perform dimensionality reduction (e.g., t-SNE or UMAP)
reducer = TSNE(n_components=3, perplexity=30, n_iter=1000)
# Or use UMAP:
# reducer = umap.UMAP(n_components=3)
reduced_embeddings = reducer.fit_transform(flat_embeddings)
 
# Create 3D scatter plot with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    z=reduced_embeddings[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=all_labels,  # Use labels for coloring
        colorscale='bluered',  # Choose a color scale
        opacity=0.8
    )
)])
 
fig.update_layout(
    title='3D Visualization of CIFAR-10 Embeddings (Cat vs Others)',
    scene=dict(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        zaxis_title='Component 3'
    )
)
 
fig.show()