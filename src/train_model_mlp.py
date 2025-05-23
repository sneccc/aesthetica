import pytorch_lightning as pl
import torch,pathlib,open_clip
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from pytorch_lamb import Lamb

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_units=(2048,512,64), class_weights=None):
        super().__init__()
        
        # Train Metrics
        self.train_acc = Accuracy(num_classes=num_classes, average='macro',task="multiclass")
        self.train_f1_score = F1Score(num_classes=num_classes, average='macro',task="multiclass")
        self.train_precision = Precision(num_classes=num_classes, average='macro',task="multiclass")
        self.train_recall = Recall(num_classes=num_classes, average='macro',task="multiclass")

        # Validation Metrics
        self.val_acc = Accuracy(num_classes=num_classes, average='micro',task="multiclass")
        self.val_f1_score = F1Score(num_classes=num_classes, average='macro',task="multiclass")
        self.val_precision = Precision(num_classes=num_classes, average='macro',task="multiclass")
        self.val_recall = Recall(num_classes=num_classes, average='macro',task="multiclass")

        all_layers = [nn.Flatten()]
        for index, hidden_unit in enumerate(hidden_units):
            all_layers.append(nn.Linear(input_size, hidden_unit))
            all_layers.append(nn.BatchNorm1d(hidden_unit))
            all_layers.append(nn.LeakyReLU(negative_slope=0.02))
            # all_layers.append(nn.ReLU())
            #all_layers.append(nn.GELU())
            input_size = hidden_unit
            if index < len(hidden_units) - 1:
                all_layers.append(nn.Dropout(0.7))
            else:
                all_layers.append(nn.Dropout(0.5))

        all_layers.append(nn.Linear(hidden_units[-1], num_classes))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, y)

        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        self.val_f1_score.update(logits, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(logits, y)
        self.val_acc.update(preds, y)

        return loss
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        # Add training augmentations here
        #x = self.training_augmentations(x)
        
        logits = self(x)
        
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        self.train_f1_score.update(logits, y)
        self.train_precision.update(preds, y)
        self.train_recall.update(logits, y)
        self.train_acc.update(preds, y)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.log('train_f1_score', self.train_f1_score.compute(), prog_bar=False)
        self.log('train_precision', self.train_precision.compute())
        self.log('train_recall', self.train_recall.compute(), prog_bar=False)

        # Reset metrics at the end of each epoch
        self.train_acc.reset()
        self.train_f1_score.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute())
        self.log('val_f1_score', self.val_f1_score.compute(), prog_bar=False)
        self.log('val_precision', self.val_precision.compute(), prog_bar=False)
        self.log('val_recall', self.val_recall.compute(), prog_bar=False)

        # Reset metrics at the end of each epoch
        self.val_acc.reset()
        self.val_f1_score.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
            optimizer = "lion"
            if optimizer == "prodigy":
                from prodigyopt import Prodigy
                optimizer = Prodigy(self.parameters(), lr=1, weight_decay=0.0,d_coef=1)
                
                # Warmup configuration
                warmup_epochs = 250
                base_lr = 1
                max_lr = 1
                
                def lr_lambda(current_epoch):
                    if current_epoch < warmup_epochs:
                        # Linear warm-up from base_lr to max_lr
                        return (max_lr / base_lr) * (current_epoch / warmup_epochs)
                    else:
                        # Post warm-up: cosine decay
                        return 0.5 * (1 + np.cos(np.pi * (current_epoch - warmup_epochs) / (self.trainer.max_epochs - warmup_epochs)))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return [optimizer], [scheduler]
            if optimizer == "lion":   
                from bitsandbytes.optim import Lion
                
                params = list(self.parameters())
                optimizer = Lion(
                    params,
                    lr=5e-5,                    # Lion's default learning rate
                    betas=(0.95, 0.98),         # Recommended momentum parameters
                    weight_decay=0.0,           # Recommended weight decay
                    optim_bits=32,
                    min_8bit_size=4096,
                    percentile_clipping=100,
                    block_wise=True,
                    is_paged=False
                )
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_epochs,
                    eta_min=1e-6
                )
                
                return [optimizer], [scheduler]
            if optimizer == "lamb":
                params = list(self.parameters())
                optimizer = Lamb(adam=True,betas=(0.9, 0.999),eps=1e-8,weight_decay=0.05,params=params,lr=1e-5)
                return optimizer
            
            elif optimizer == "Adam":
                optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=0.01)

                # Warmup configuration
                warmup_epochs = 2500
                base_lr = 1e-6
                max_lr = 2e-4
                
                def lr_lambda(current_epoch):
                    if current_epoch < warmup_epochs:
                        # Linear warm-up from base_lr to max_lr
                        return (max_lr / base_lr) * (current_epoch / warmup_epochs)
                    else:
                        # Post warm-up: cosine decay
                        return 0.5 * (1 + np.cos(np.pi * (current_epoch - warmup_epochs) / (self.trainer.max_epochs - warmup_epochs)))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return [optimizer], [scheduler]
                    
            elif optimizer == "warmup":
                warmup_epochs = 2000
                base_lr = 0.0001
                max_lr = 0.003
                optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.05)
                def lr_lambda(current_epoch):
                    if current_epoch < warmup_epochs:
                        # Linear warm-up from base_lr to max_lr
                        return (max_lr / base_lr) * (current_epoch / warmup_epochs)
                    else:
                        # Post warm-up: you can define decay or constant rate here
                        return 1.0
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                
                return [optimizer], [scheduler]
            elif optimizer == "cosine":
                optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
                # Use cosine annealing with warm restarts
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=50,  # Reset every 10 epochs
                    T_mult=2,  # Double the reset period after each restart
                    eta_min=1e-6  # Minimum learning rate
                )                
            else:
                stepping_batches = self.trainer.estimated_stepping_batches
                print(f"🐍 Total number of steps {stepping_batches}")
                max_lr = 3e-4
                default_lr = 1e-4
                optimizer = torch.optim.SGD(self.parameters(), lr=default_lr, momentum=0.9, weight_decay=0.0)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.01, patience=100, verbose=True)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                    pct_start=0.1,
                    div_factor=25.0,
                    final_div_factor=1e4
                )

                return [optimizer], [scheduler]
            
    def training_augmentations(self, x):
        # Add random noise
        noise = torch.randn_like(x) * 0.1
        x = x + noise
        
        # Random feature dropout (randomly zero out some features)
        mask = torch.bernoulli(torch.ones_like(x) * 0.9)  # Keep 90% of features
        x = x * mask
        
        return x

@dataclass
class ModelConfig:
    input_size: int
    num_classes: int
    hidden_units: Tuple[int, ...]
    clip_models: List[Tuple[str, str]]
    class_labels: List[str]  # Map from class index to label name

def start_training(root_folder, database_file, train_from, clip_models, val_percentage=0.25, epochs=5000, batch_size=1000):
    train_dataloader, val_dataloader, num_classes = setup_dataset(root_folder=root_folder, database_file=database_file,train_from=train_from)
    input_size = get_total_dim(clip_models)
    print(f"input size: {input_size}\nNumber of classes: {num_classes}")

    # Get class labels mapping
    df = pd.read_csv(pathlib.Path(root_folder) / 'image_classifier_data.csv')
    class_labels = []
    for i in range(num_classes):
        label = df[df['label_id'] == i]['label_name'].iloc[0]
        class_labels.append(label)

    # Create config
    config = ModelConfig(
        input_size=input_size,
        num_classes=num_classes,
        hidden_units=(4096, 1024, 512,128),  # Your default architecture
        clip_models=clip_models,
        class_labels=class_labels
    )

    net = MultiLayerPerceptron(
        input_size=config.input_size, 
        num_classes=config.num_classes,
        hidden_units=config.hidden_units
    )
    
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode='max', monitor="val_recall"),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_recall',
            min_delta=0.0000001,
            patience=25,
            verbose=True,
            mode='max'
        )
    ]  # save top 1 model
    logger = TensorBoardLogger('tb_logs', name="my_logger", log_graph=True)
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        logger=logger,  # This is your TensorBoardLogger
        max_epochs=epochs,
        devices="auto",
        accelerator="cuda",
        callbacks=callbacks,
        check_val_every_n_epoch=10  # Check validation every 10 epochs
    )

    trainer.fit(net, train_dataloader, val_dataloader)

    # Save both model and config
    root_path = pathlib.Path(root_folder)
    save_path = root_path / "model.pth"
    config_path = root_path / "model_config.json"
    
    print("-> saving model to:", save_path)
    torch.save(net.state_dict(), save_path)
    
    print("-> saving config to:", config_path)
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)


def setup_dataset(root_folder, database_file, train_from, val_percentage=0.25):
    out_path = pathlib.Path(root_folder)
    x_embeddings_path = out_path / "image_embeddings.npy"
    
    # Load the CSV with augmentation information
    df = pd.read_csv(out_path / 'image_classifier_data.csv')
    print(f"Loaded CSV with {len(df)} entries")
    
    # Load embeddings
    x_embeddings = np.load(x_embeddings_path)
    print(f"Loaded embeddings with shape: {x_embeddings.shape}")
    
    # Get labels from DataFrame
    y_features = df['label_id'].values
    
    # Verify lengths match using embedding_index
    assert len(x_embeddings) == len(df), \
        f"Mismatch between embeddings ({len(x_embeddings)}) and DataFrame entries ({len(df)})"
    
    # Filter out test data (negative labels) before splitting into train/val
    valid_mask = y_features >= 0
    x_features = x_embeddings[valid_mask]
    y_features = y_features[valid_mask]
    filtered_df = df[valid_mask]
    
    # Now split the valid data into train/val
    # Use stratification based on labels, but handle classes with only 1 sample
    try:
        # Try stratification by class labels first
        x_train, x_val, y_train, y_val = train_test_split(
            x_features, 
            y_features,
            test_size=0.25,
            random_state=42,
            stratify=y_features  # Stratify by actual class labels
        )
    except ValueError as e:
        if "least populated class" in str(e):
            # Some classes have only 1 sample, use random split without stratification
            print("Warning: Some classes have very few samples. Using random split without stratification.")
            x_train, x_val, y_train, y_val = train_test_split(
                x_features, 
                y_features,
                test_size=0.25,
                random_state=42,
                stratify=None  # No stratification
            )
        else:
            raise e
    
    # Convert to tensors
    train_tensor_x = torch.Tensor(x_train)
    train_tensor_y = torch.Tensor(y_train).long()
    val_tensor_x = torch.Tensor(x_val)
    val_tensor_y = torch.Tensor(y_val).long()
    
    # Debug prints
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_tensor_x)}")
    print(f"Validation samples: {len(val_tensor_x)}")
    print(f"Label range: {torch.min(train_tensor_y).item()} to {torch.max(train_tensor_y).item()}")
    print(f"Data range: {torch.min(train_tensor_x).item():.2f} to {torch.max(train_tensor_x).item():.2f}")
    
    # Calculate class distribution
    class_counts = np.bincount(y_train)
    print("\nClass Distribution:")
    for class_idx, count in enumerate(class_counts):
        print(f"Class {class_idx}: {count} samples")
    num_classes = len(class_counts)
    
    # Create dataloaders
    batch_size = 256
    
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, num_classes

def custom_collate_fn(batch):
    x_ = default_collate(batch)
    return tuple(item.to(device) for item in x_)

def get_total_dim(clip_models):
    total_dim = 0
    for clip_model in clip_models:
        # get_model_config("ViT-B-16-SigLIP-512")["embed_dim"]
        if clip_model[0] == "hf-hub:timm":
            config = open_clip.get_model_config(clip_model[1])  # ["embed_dim"]
            print(f"SigLip model with {config['embed_dim']} dimension")
        else:
            config = open_clip.get_model_config(clip_model[0])

        if config is not None and 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")

    # Use the total dimension for the MLP model
    print("total_dim: ", total_dim)
    return total_dim

# --- Add command-line execution support ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the MultiLayer Perceptron model.")
    parser.add_argument("-i", "--input_directory", required=True,
                        help="Path to the normalized dataset directory containing image_embeddings.npy and image_classifier_data.csv")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val_percentage", type=float, default=0.25, help="Validation split percentage")

    args = parser.parse_args()

    # Define parameters from command line
    root_folder = args.input_directory
    database_file = 'image_classifier_data.csv' # Assumed to be inside root_folder
    train_from = 'embeddings' # Assumed based on previous context
    # Using the default CLIP model from the original start_training function
    clip_models = [("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")]

    print(f"Starting training from command line for folder: {root_folder}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Validation percentage: {args.val_percentage}")
    print(f"  - CLIP model: {clip_models[0]}")
    
    start_training(
        root_folder=root_folder,
        database_file=database_file,
        train_from=train_from,
        clip_models=clip_models,
        val_percentage=args.val_percentage,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    print("Training finished successfully!")
