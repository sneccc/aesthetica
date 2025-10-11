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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from pytorch_lamb import Lamb
from PIL import Image
import wandb
from datetime import datetime
import time
import threading

# System monitoring imports
try:
    import psutil
    import GPUtil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False

# Import wandb utilities
try:
    from .wandb_utils import setup_wandb_key, get_wandb_enabled
except ImportError:
    # Handle case when running as script
    import sys
    sys.path.append('.')
    from src.wandb_utils import setup_wandb_key, get_wandb_enabled

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
        
        # Store validation data for visualization
        self.val_sample_data = None
        self.val_sample_info = None
        self.class_labels = None
        self.root_folder = None
        self.validation_viz_count = 0
        self.validation_viz_epochs = []

    def forward(self, x):
        x = self.model(x)
        return x
    
    def setup_validation_visualization_simple(self, val_dataloader, val_df_info, class_labels, root_folder, num_samples=10):
        """Setup validation visualization with direct DataFrame mapping."""
        self.class_labels = class_labels
        self.root_folder = pathlib.Path(root_folder)
        
        # Get a sample of validation data
        val_dataset = val_dataloader.dataset
        dataset_size = len(val_dataset)
        num_samples = min(num_samples, dataset_size, len(val_df_info))
        
        # Sample random indices
        indices = torch.randperm(dataset_size)[:num_samples]
        
        sample_embeddings = []
        sample_labels = []
        sample_info = []
        
        for idx in indices:
            idx = int(idx.item())
            if idx < len(val_dataset) and idx < len(val_df_info):
                try:
                    embedding, label = val_dataset[idx]
                    row = val_df_info.iloc[idx]
                    
                    sample_embeddings.append(embedding)
                    sample_labels.append(label)
                    sample_info.append({
                        'image_path': row['image_path'],
                        'image_name': row['image_name'],
                        'label_name': row['label_name'],
                        'label_id': row['label_id']
                    })
                except Exception as e:
                    print(f"Warning: Could not process validation sample {idx}: {e}")
                    continue
        
        if sample_embeddings:
            self.val_sample_data = torch.stack(sample_embeddings)
            self.val_sample_labels = torch.stack(sample_labels)
            self.val_sample_info = sample_info
        else:
            print("Warning: No validation samples could be prepared for visualization")

    def setup_validation_visualization(self, val_dataloader, df, class_labels, root_folder, num_samples=10):
        """Setup validation visualization by sampling some validation data."""
        self.class_labels = class_labels
        self.root_folder = pathlib.Path(root_folder)
        
        # Get a sample of validation data
        val_dataset = val_dataloader.dataset
        dataset_size = len(val_dataset)
        indices = torch.randperm(dataset_size)[:min(num_samples, dataset_size)]
        
        sample_embeddings = []
        sample_labels = []
        sample_info = []
        
        # Get filtered DataFrame (same filtering as in setup_dataset)
        val_mask = df['label_id'] >= 0
        filtered_df = df[val_mask].reset_index(drop=True)
        
        # Since we're using the same random seed (42), the split should be identical
        # We can get validation indices by recreating the split
        from sklearn.model_selection import train_test_split
        y_features = filtered_df['label_id'].values
        
        try:
            train_indices, val_indices = train_test_split(
                range(len(filtered_df)), 
                test_size=0.25,
                random_state=42,
                stratify=y_features
            )
        except ValueError:
            train_indices, val_indices = train_test_split(
                range(len(filtered_df)), 
                test_size=0.25,
                random_state=42,
                stratify=None
            )
        
        # Convert to DataFrame with validation subset
        val_df_subset = filtered_df.iloc[val_indices].reset_index(drop=True)
        
        for idx_pos, dataset_idx in enumerate(indices):
            dataset_idx = int(dataset_idx.item())  # Convert tensor to int
            if dataset_idx < len(val_dataset) and dataset_idx < len(val_df_subset):
                try:
                    row = val_df_subset.iloc[dataset_idx]
                    
                    embedding, label = val_dataset[dataset_idx]
                    sample_embeddings.append(embedding)
                    sample_labels.append(label)
                    sample_info.append({
                        'image_path': row['image_path'],
                        'image_name': row['image_name'],
                        'label_name': row['label_name'],
                        'label_id': row['label_id']
                    })
                except Exception as e:
                    print(f"Warning: Could not process validation sample {dataset_idx}: {e}")
                    continue
        
        if sample_embeddings:
            self.val_sample_data = torch.stack(sample_embeddings)
            self.val_sample_labels = torch.stack(sample_labels)
            self.val_sample_info = sample_info
        else:
            print("Warning: No validation samples could be prepared for visualization")

    def log_validation_predictions(self):
        """Log validation predictions to wandb with images."""
        
        if self.val_sample_data is None:
            return False
        
        # Find wandb logger from the list of loggers
        wandb_logger = None
        if hasattr(self.trainer, 'loggers'):
            for logger in self.trainer.loggers:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                    # Check if it's a wandb logger
                    if hasattr(logger.experiment, 'project'):
                        wandb_logger = logger
                        break
        
        if wandb_logger is None:
            return False
            
        try:
            # Get predictions
            self.eval()
            with torch.no_grad():
                logits = self(self.val_sample_data.to(self.device))
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
            
            # Create wandb table data
            table_data = []
            successful_rows = 0
            
            for i, info in enumerate(self.val_sample_info):
                
                # Load image
                try:
                    image_path = pathlib.Path(info['image_path'])
                    if not image_path.is_absolute():
                        image_path = self.root_folder / image_path
                    
                    if image_path.exists():
                        pil_img = Image.open(image_path).convert('RGB')
                        # Resize for display
                        pil_img.thumbnail((224, 224), Image.Resampling.LANCZOS)
                        wandb_image = wandb.Image(pil_img)
                    else:
                        print(f"Warning: Image not found: {image_path}")
                        wandb_image = None
                except Exception as e:
                    print(f"Warning: Could not load image {info['image_path']}: {e}")
                    wandb_image = None
                
                pred_id = predictions[i].item()
                true_id = self.val_sample_labels[i].item()
                pred_label = self.class_labels[pred_id] if pred_id < len(self.class_labels) else f"Unknown_{pred_id}"
                true_label = self.class_labels[true_id] if true_id < len(self.class_labels) else f"Unknown_{true_id}"
                confidence = probabilities[i][pred_id].item()
                
                table_data.append([
                    wandb_image,
                    info['image_name'],
                    true_label,
                    pred_label,
                    f"{confidence:.3f}",
                    "CORRECT" if pred_id == true_id else "WRONG"
                ])
                successful_rows += 1
            
            if successful_rows == 0:
                return False
            
            # Create and log table
            table = wandb.Table(
                columns=["Image", "Filename", "True Label", "Predicted Label", "Confidence", "Correct"],
                data=table_data
            )
            
            wandb_logger.experiment.log({
                f"validation_predictions_epoch_{self.current_epoch}": table
            })
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to log validation predictions: {e}")
            import traceback
            traceback.print_exc()
            return False

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
        # Write validation info to a debug file
        debug_file = self.root_folder / "validation_debug.txt" if self.root_folder else pathlib.Path("validation_debug.txt")
        with open(debug_file, "a") as f:
            f.write(f"VALIDATION_RUN: epoch={self.current_epoch}, time={datetime.now()}\n")
        
        self.log("val_acc", self.val_acc.compute())
        self.log('val_f1_score', self.val_f1_score.compute(), prog_bar=False)
        self.log('val_precision', self.val_precision.compute(), prog_bar=False)
        self.log('val_recall', self.val_recall.compute(), prog_bar=False)

        # Log validation predictions more frequently - every 50 epochs OR every 5 validation runs
        # Since validation runs every 5 epochs, this means visualization at epochs: 0, 25, 50, 75, 100, etc.
        validation_run_count = (self.current_epoch // 5) + 1  # Approximate validation run number
        should_visualize = (
            self.current_epoch == 0 or  # Always visualize at epoch 0
            self.current_epoch % 25 == 0 or  # Every 25 epochs (more frequent than before)
            validation_run_count % 5 == 0  # Every 5 validation runs
        )
        
        if should_visualize:
            with open(debug_file, "a") as f:
                f.write(f"VISUALIZATION_ATTEMPT: epoch={self.current_epoch}, validation_run={validation_run_count}, time={datetime.now()}\n")
            
            success = self.log_validation_predictions()
            if success:
                self.validation_viz_count += 1
                self.validation_viz_epochs.append(self.current_epoch)
                with open(debug_file, "a") as f:
                    f.write(f"VISUALIZATION_SUCCESS: epoch={self.current_epoch}, count={self.validation_viz_count}\n")
            else:
                with open(debug_file, "a") as f:
                    f.write(f"VISUALIZATION_FAILED: epoch={self.current_epoch}\n")
        else:
            print(f">> Skipping validation visualization at epoch {self.current_epoch} (next at epoch {((self.current_epoch // 25) + 1) * 25})")

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

    def test_validation_visualization_setup(self, trainer=None):
        """Test if validation visualization is properly set up."""
        print(">> Testing validation visualization setup:")
        print(f">>   val_sample_data: {'OK' if self.val_sample_data is not None else 'MISSING'}")
        print(f">>   val_sample_info: {'OK' if self.val_sample_info is not None else 'MISSING'}")
        print(f">>   class_labels: {'OK' if self.class_labels is not None else 'MISSING'}")
        print(f">>   root_folder: {'OK' if self.root_folder is not None else 'MISSING'}")
        
        if self.val_sample_data is not None:
            print(f">>   Sample data shape: {self.val_sample_data.shape}")
            print(f">>   Number of samples: {len(self.val_sample_info) if self.val_sample_info else 0}")
        
        # Check for trainer and loggers
        current_trainer = trainer if trainer is not None else getattr(self, 'trainer', None)
        if current_trainer and hasattr(current_trainer, 'loggers'):
            wandb_logger_found = False
            for logger in current_trainer.loggers:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'project'):
                    wandb_logger_found = True
                    break
            print(f">>   wandb logger: {'OK' if wandb_logger_found else 'MISSING'}")
        else:
            print(f">>   wandb logger: PENDING (trainer not attached yet)")
        
        return (self.val_sample_data is not None and 
                self.val_sample_info is not None and 
                self.class_labels is not None)

@dataclass
class ModelConfig:
    input_size: int
    num_classes: int
    hidden_units: Tuple[int, ...]
    clip_models: List[Tuple[str, str]]
    class_labels: List[str]  # Map from class index to label name

def start_training(root_folder, database_file, train_from, clip_models, val_percentage=0.25, epochs=5000, batch_size=1000, enable_wandb=True, enable_dashboard=True):
    # Setup wandb if enabled
    wandb_enabled = False
    if enable_wandb:
        wandb_key = setup_wandb_key(root_folder, ask_user=True)
        wandb_enabled = wandb_key is not None

    train_dataloader, val_dataloader, num_classes, val_df_info = setup_dataset(root_folder=root_folder, database_file=database_file,train_from=train_from)
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
            patience=50,  # Increased patience since validation is more frequent
            verbose=True,
            mode='max'
        )
    ]
    
    # Add simple dashboard if enabled
    if enable_dashboard:
        callbacks.append(SimpleDashboard(total_epochs=epochs))
    else:
        print(">> Training without dashboard")
    
    # Setup loggers
    loggers = []
    
    # Always add TensorBoard logger
    tb_logger = TensorBoardLogger('tb_logs', name="my_logger", log_graph=True)
    loggers.append(tb_logger)
    
    # Add wandb logger if enabled
    wandb_logger = None
    if wandb_enabled:
        # Create wandb config with training hyperparameters
        wandb_config = {
            "architecture": "MLP",
            "input_size": config.input_size,
            "num_classes": config.num_classes,
            "hidden_units": config.hidden_units,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_percentage": val_percentage,
            "clip_models": [f"{model[0]}:{model[1]}" for model in clip_models],
            "class_labels": config.class_labels,
            "optimizer": "lion",  # Based on current default in configure_optimizers
            "monitor_metric": "val_recall",
            "early_stopping_patience": 50
        }
        
        wandb_logger = WandbLogger(
            project="aesthetica-training",
            name=f"mlp-{len(config.class_labels)}classes-{input_size}dim",
            config=wandb_config,
            log_model="all",  # Log model checkpoints
            save_dir="wandb_logs"
        )
        loggers.append(wandb_logger)
        
        # Setup validation visualization for wandb with the validation DataFrame info
        net.setup_validation_visualization_simple(
            val_dataloader=val_dataloader,
            val_df_info=val_df_info,
            class_labels=config.class_labels,
            root_folder=root_folder,
            num_samples=10
        )
    else:
        print(">> Wandb logging disabled")

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        logger=loggers,  # Use list of loggers
        max_epochs=epochs,
        devices="auto",
        accelerator="cuda",
        callbacks=callbacks,
        check_val_every_n_epoch=5  # Check validation every 5 epochs for more frequent visualization
    )

    # Final test of validation visualization setup
    if wandb_enabled:
        if not net.test_validation_visualization_setup(trainer):
            print("WARNING: Validation visualization setup failed!")

    trainer.fit(net, train_dataloader, val_dataloader)

    # Write final training info to debug file
    debug_file = pathlib.Path(root_folder) / "validation_debug.txt"
    with open(debug_file, "a") as f:
        f.write(f"TRAINING_COMPLETED: final_epoch={trainer.current_epoch}, max_epochs={epochs}, viz_count={net.validation_viz_count}\n")
    
    print(f">> Training completed at epoch {trainer.current_epoch} (max was {epochs})")

    # Save both model and config
    root_path = pathlib.Path(root_folder)
    save_path = root_path / "model.pth"
    config_path = root_path / "model_config.json"
    
    print("-> saving model to:", save_path)
    torch.save(net.state_dict(), save_path)
    
    print("-> saving config to:", config_path)
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Log final model artifact to wandb if enabled
    if wandb_enabled and wandb_logger:
        try:
            # Log the final model as an artifact
            artifact = wandb.Artifact(
                name="final_model",
                type="model",
                description=f"Final trained MLP model with {config.num_classes} classes"
            )
            artifact.add_file(str(save_path))
            artifact.add_file(str(config_path))
            wandb_logger.experiment.log_artifact(artifact)
            print(">> Saved model artifacts to wandb")
            
            # Save wandb run info for testing to resume the same run
            try:
                wandb_run_info = {
                    "run_id": wandb_logger.experiment.id,
                    "project": wandb_logger.experiment.project,
                    "name": wandb_logger.experiment.name,
                    "tags": list(wandb_logger.experiment.tags) if wandb_logger.experiment.tags else []
                }
                
                wandb_info_path = pathlib.Path(root_folder) / "wandb_run_info.json"
                with open(wandb_info_path, 'w') as f:
                    json.dump(wandb_run_info, f, indent=2)
                print(f">> Saved wandb run info to {wandb_info_path} for testing continuation")
                
            except Exception as e:
                print(f"WARNING: Failed to save wandb run info: {e}")
                
        except Exception as e:
            print(f"WARNING: Failed to log artifacts to wandb: {e}")
    
    # Return trainer, net, and validation visualization stats
    viz_stats = {
        'count': net.validation_viz_count,
        'epochs': net.validation_viz_epochs
    }
    
    # Print validation visualization stats for CLI parsing
    print(f"VALIDATION_VIZ_STATS: count={net.validation_viz_count}, epochs={net.validation_viz_epochs}")
    
    return trainer, net, viz_stats


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
        x_train, x_val, y_train, y_val, train_indices, val_indices = train_test_split(
            x_features, 
            y_features,
            range(len(filtered_df)),
            test_size=0.25,
            random_state=42,
            stratify=y_features  # Stratify by actual class labels
        )
    except ValueError as e:
        if "least populated class" in str(e):
            # Some classes have only 1 sample, use random split without stratification
            print("Warning: Some classes have very few samples. Using random split without stratification.")
            x_train, x_val, y_train, y_val, train_indices, val_indices = train_test_split(
                x_features, 
                y_features,
                range(len(filtered_df)),
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
    
    # Create validation DataFrame info for visualization
    val_df_info = filtered_df.iloc[val_indices].reset_index(drop=True)
    
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
    
    return train_dataloader, val_dataloader, num_classes, val_df_info

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

def print_training_dashboard(epoch, total_epochs, metrics, start_time, epoch_start_time=None):
    """Print a simple training dashboard every epoch"""
    
    # Safety check - don't run if start_time is None (during sanity check)
    if start_time is None:
        return
    
    # Calculate timing
    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s"
    
    if epoch > 0:
        avg_epoch_time = elapsed / epoch
        remaining_epochs = total_epochs - epoch
        eta = avg_epoch_time * remaining_epochs
        eta_str = f"{eta//3600:.0f}h {(eta%3600)//60:.0f}m {eta%60:.0f}s"
    else:
        eta_str = "Calculating..."
    
    epoch_time = ""
    if epoch_start_time:
        epoch_duration = time.time() - epoch_start_time
        epoch_time = f" (Last epoch: {epoch_duration:.1f}s)"
    
    progress_percent = (epoch / total_epochs) * 100
    progress_bar = "#" * int(progress_percent / 5) + "-" * (20 - int(progress_percent / 5))
    
    # Get system stats if available
    gpu_info = "N/A"
    cpu_info = "N/A"
    try:
        if SYSTEM_MONITORING_AVAILABLE:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_info = f"{cpu_percent:.1f}%"
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = f"{gpu.load*100:.1f}% / {gpu.memoryUtil*100:.1f}%"
            except:
                pass
    except:
        pass
    
    # Format metrics
    train_loss = metrics.get('train_loss', 0.0)
    val_loss = metrics.get('val_loss', 0.0)
    train_acc = metrics.get('train_acc', 0.0)
    val_acc = metrics.get('val_acc', 0.0)
    train_f1 = metrics.get('train_f1_score', 0.0)
    val_f1 = metrics.get('val_f1_score', 0.0)
    lr = metrics.get('lr', 0.0)
    
    # Wandb status
    wandb_status = "OFFLINE"
    if hasattr(wandb, 'run') and wandb.run is not None:
        wandb_status = "ONLINE"
    
    print("\n" + "="*80)
    print("                    AESTHETICA TRAINING DASHBOARD")
    print("="*80)
    print(f"Epoch: {epoch:4d}/{total_epochs}  [{progress_bar}] {progress_percent:5.1f}%{epoch_time}")
    print(f"Time:  Elapsed: {elapsed_str:>12} | ETA: {eta_str:>12}")
    print("-"*80)
    print("METRICS               TRAIN        VALIDATION     STATUS")
    print("-"*80)
    print(f"Loss                  {train_loss:8.4f}     {val_loss:8.4f}       {'DOWN' if val_loss > 0 and train_loss > val_loss else 'UP' if val_loss > 0 else 'WAIT'}")
    print(f"Accuracy              {train_acc:8.1%}     {val_acc:8.1%}       {'GOOD' if val_acc > 0.8 else 'OK' if val_acc > 0.5 else 'TRAIN'}")
    print(f"F1 Score              {train_f1:8.1%}     {val_f1:8.1%}       {'GREAT' if val_f1 > 0.8 else 'GOOD' if val_f1 > 0.5 else 'TRAIN'}")
    print(f"Learning Rate         {lr:8.2e}     {'—':>8}       ACTIVE")
    print("-"*80)
    print(f"System: GPU {gpu_info:>12} | CPU {cpu_info:>6} | Wandb: {wandb_status}")
    print("="*80)

class SimpleDashboard(pl.Callback):
    """Simple dashboard that prints training status every epoch"""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_start_time = None
        self.current_metrics = {}
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print("\n>>> Training started with simple dashboard")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip dashboard during sanity check
        if trainer.sanity_checking:
            return
            
        # Update metrics from trainer logs
        if hasattr(trainer, 'logged_metrics'):
            logs = trainer.logged_metrics
            self.current_metrics.update(logs)
            
        # Get learning rate
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            self.current_metrics['lr'] = optimizer.param_groups[0]['lr']
        
        # Print dashboard every validation epoch
        print_training_dashboard(
            epoch=trainer.current_epoch,
            total_epochs=self.total_epochs,
            metrics=self.current_metrics,
            start_time=self.start_time,
            epoch_start_time=self.epoch_start_time
        )
        
    def on_train_end(self, trainer, pl_module):
        print("\n>>> Training completed!")
        print_training_dashboard(
            epoch=trainer.current_epoch,
            total_epochs=self.total_epochs,
            metrics=self.current_metrics,
            start_time=self.start_time,
            epoch_start_time=self.epoch_start_time
        )

# --- Add command-line execution support ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the MultiLayer Perceptron model.")
    parser.add_argument("-i", "--input_directory", required=True,
                        help="Path to the normalized dataset directory containing image_embeddings.npy and image_classifier_data.csv")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val_percentage", type=float, default=0.25, help="Validation split percentage")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable rich dashboard")

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
    print(f"  - Wandb enabled: {not args.no_wandb}")
    print(f"  - Dashboard enabled: {not args.no_dashboard}")
    
    start_training(
        root_folder=root_folder,
        database_file=database_file,
        train_from=train_from,
        clip_models=clip_models,
        val_percentage=args.val_percentage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        enable_wandb=not args.no_wandb,
        enable_dashboard=not args.no_dashboard
    )
    print("Training finished successfully!")
