import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import open_clip
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import joblib
import pathlib
import json


@dataclass
class EmbeddingConfig:
    model_name: str
    pretrained: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 4
    pca_components: Optional[int] = 256
    use_standardization: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class TorchPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        
        # Center the data
        self.mean_ = torch.mean(X_tensor, dim=0)
        X_centered = X_tensor - self.mean_
        
        # Compute SVD
        U, S, V = torch.svd(X_centered)
        
        # Get components and explained variance
        self.components_ = V[:, :self.n_components].T
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        
        self.explained_variance_ = explained_variance[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components]
        
        # Transform the data
        transformed = torch.matmul(X_centered, self.components_.T)
        
        return transformed.cpu().numpy()

    def transform(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        X_centered = X_tensor - self.mean_
        transformed = torch.matmul(X_centered, self.components_.T)
        return transformed.cpu().numpy()

class EmbeddingProcessor:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.preprocess = None
        self.pca = None
        self.scaler = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize the model based on configuration."""
        if self.config.model_name.startswith("hf-hub:timm"):
            self.model, self.preprocess = open_clip.create_model_from_pretrained(
                self.config.model_name + "/" + self.config.pretrained,
                device=self.config.device
            )
        elif self.config.model_name == "nomic-ai":
            from transformers import AutoModel, AutoImageProcessor
            self.preprocess = AutoImageProcessor.from_pretrained(self.config.pretrained)
            self.model = AutoModel.from_pretrained(self.config.pretrained, trust_remote_code=True)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained,
                device=self.config.device
            )
        
        self.model.to(self.config.device)
        self.model.eval()

    def process_batch(self, batch_images: List) -> np.ndarray:
        """Process a batch of images and return embeddings."""
        try:
            batch_preprocessed = []
            for image in batch_images:
                if self.config.model_name == "nomic-ai":
                    inputs = self.preprocess(image, return_tensors="pt")
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    batch_preprocessed.append(inputs)
                else:
                    preprocessed = self.preprocess(image)
                    batch_preprocessed.append(preprocessed)

            with torch.no_grad(), torch.amp.autocast(self.config.device):
                if self.config.model_name == "nomic-ai":
                    features = [self.model(**inputs).last_hidden_state for inputs in batch_preprocessed]
                    features = [f.cpu().numpy() for f in features]
                else:
                    batch_tensor = torch.stack(batch_preprocessed).to(self.config.device)
                    features = self.model.encode_image(batch_tensor)
                    features = features.cpu().numpy()

            return features

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return None

    def generate_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for all images in the dataloader."""
        all_embeddings = []
        all_paths = []
        
        pbar = tqdm(dataloader, desc="Generating embeddings")
        for batch_images, batch_paths in pbar:
            features = self.process_batch(batch_images)
            if features is not None:
                all_embeddings.append(features)
                all_paths.extend(batch_paths)
            
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            
            pbar.set_postfix({'GPU_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"})
        
        final_embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])
        return final_embeddings, all_paths

    def fit_transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction and standardization to embeddings."""
        if self.config.use_standardization:
            self.scaler = StandardScaler()
            embeddings = self.scaler.fit_transform(embeddings)
        
        if self.config.pca_components:
            self.pca = TorchPCA(n_components=self.config.pca_components)
            embeddings = self.pca.fit_transform(embeddings)
            explained_variance = sum(self.pca.explained_variance_ratio_)
            print(f"PCA explained variance ratio: {explained_variance:.4f}")
        
        return embeddings

    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted PCA and scaler."""
        if self.scaler is not None:
            embeddings = self.scaler.transform(embeddings)
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
        return embeddings

    @property
    def output_dimension(self) -> int:
        """Get the dimension of the processed embeddings."""
        if self.pca is not None:
            return self.pca.n_components
        if self.model is not None:
            # Get the original embedding dimension from the model
            if self.config.model_name.startswith("hf-hub:timm"):
                # Extract just the model name part after the prefix
                model_name = self.config.pretrained
                config = open_clip.get_model_config(model_name)
                if config is not None and "embed_dim" in config:
                    return config["embed_dim"]
                # Fallback to model's embed_dim if config lookup fails
                return self.model.embed_dim
            return self.model.embed_dim
        raise ValueError("Model not initialized")

    def save_processors(self, save_dir: str | pathlib.Path):
        """Save PCA, StandardScaler states, and config."""
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.pca is not None:
            joblib.dump(self.pca, save_dir / 'pca.joblib')
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / 'scaler.joblib')
        
        # Save configurations
        processor_config = {
            'embedding_config': self.config.to_dict(),
            'output_dimension': self.output_dimension,
            'use_pca': self.pca is not None,
            'use_standardization': self.scaler is not None
        }
        
        with open(save_dir / 'processor_config.json', 'w') as f:
            json.dump(processor_config, f, indent=2)

    def load_processors(self, save_dir: str | pathlib.Path):
        """Load PCA and StandardScaler states."""
        save_dir = pathlib.Path(save_dir)
        
        pca_path = save_dir / 'pca.joblib'
        scaler_path = save_dir / 'scaler.joblib'
        
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

    @classmethod
    def from_config_file(cls, config_path: str | pathlib.Path):
        """Create EmbeddingProcessor instance from saved config."""
        config_path = pathlib.Path(config_path)
        with open(config_path / 'processor_config.json', 'r') as f:
            processor_config = json.load(f)
        
        # Create instance with embedding config
        embedding_config = EmbeddingConfig.from_dict(processor_config['embedding_config'])
        processor = cls(embedding_config)
        
        # Load PCA and StandardScaler if they exist
        processor.load_processors(config_path)
        
        return processor

class EmbeddingClassifier:
    """Placeholder for the classifier model."""
    def __init__(self):
        # Add classifier initialization here
        pass
