import os
from utils import load_embeddings_and_paths
from pyray import *
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load embeddings and paths
data_dir = os.path.join(current_dir, "../../data/normalized_3d_test")
embeddings, thumbnail_paths, labels = load_embeddings_and_paths(data_dir)
init_window(1280, 720, "Texture Loading Test")
textures = []
print("Total paths", len(thumbnail_paths))
for i, path in enumerate(thumbnail_paths):
    image = load_image(path)
    texture = load_texture_from_image(image)
    textures.append(texture)

print(f"{len(textures)}/{len(thumbnail_paths)} textures loaded")

for texture in textures:
    unload_texture(texture)
