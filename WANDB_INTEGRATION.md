# Weights & Biases (wandb) Integration

This project now includes full integration with [Weights & Biases](https://wandb.ai/) for experiment tracking, metrics visualization, and model management.

## Features

- 🔐 **Automatic API Key Management**: Checks for `keys.txt`, prompts for key if needed
- 📊 **Rich Metrics Logging**: Automatically logs training/validation losses, accuracies, F1 scores, precision, and recall
- 🎯 **Hyperparameter Tracking**: Logs model architecture, optimizer settings, and training configuration
- 📦 **Model Artifacts**: Saves trained models and configs as wandb artifacts
- 🎨 **Beautiful Dashboards**: Real-time visualization of training progress
- 🚀 **Easy Setup**: One-command integration with interactive prompts

## Quick Start

### 1. Install Dependencies

```bash
pip install -r pipeline_requirements.txt
```

### 2. Test Integration (Optional)

```bash
python src/test_wandb_integration.py
```

### 3. Set Up wandb (Interactive)

```bash
python example_wandb_setup.py
```

This will:
- Guide you through getting a wandb API key
- Save it securely to `keys.txt`
- Test the integration
- Show example usage

### 4. Train with wandb

```bash
# With wandb enabled (default)
python src/train_model_mlp.py -i /path/to/your/dataset

# Disable wandb if needed
python src/train_model_mlp.py -i /path/to/your/dataset --no-wandb
```

## How It Works

### Key Management

1. **Automatic Detection**: Checks for `keys.txt` in the root folder
2. **Format**: Stores as `WANDB_KEY=your_api_key_here`
3. **Security**: Automatically added to `.gitignore`
4. **User Prompt**: Asks for key interactively if not found

### What Gets Logged

#### Metrics (per epoch):
- `train_loss` / `val_loss`
- `train_acc` / `val_acc` 
- `train_f1_score` / `val_f1_score`
- `train_precision` / `val_precision`
- `train_recall` / `val_recall`

#### Hyperparameters:
- Model architecture details
- Optimizer configuration
- Training parameters (epochs, batch size, etc.)
- CLIP model information
- Class labels

#### Artifacts:
- Final trained model (`model.pth`)
- Model configuration (`model_config.json`)
- Automatically versioned and tagged

### Integration Details

The wandb integration uses PyTorch Lightning's `WandbLogger` which provides:
- Seamless metric logging via `self.log()`
- Automatic hyperparameter detection
- Model checkpointing as artifacts
- Integration with Lightning callbacks

## Manual API Key Setup

If you prefer to set up the API key manually:

1. Get your key from https://wandb.ai/authorize
2. Create `keys.txt` in the project root:
   ```
   WANDB_KEY=your_api_key_here
   ```
3. The training script will automatically detect and use it

## Environment Variable Alternative

You can also set the key as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
python src/train_model_mlp.py -i /path/to/dataset
```

## Project Configuration

wandb runs are organized as follows:
- **Project Name**: `aesthetica-training`
- **Run Names**: Automatically generated based on model configuration
- **Tags**: Based on model architecture and dataset size

## Viewing Results

After starting training:
1. Check the console for the wandb URL
2. Visit your wandb dashboard
3. View real-time metrics, system stats, and logs
4. Compare different runs and experiments
5. Download model artifacts

## Advanced Usage

### Programmatic Integration

```python
from src.wandb_utils import setup_wandb_key, initialize_wandb
from src.train_model_mlp import start_training

# Setup wandb
setup_wandb_key(".", ask_user=False)  # Silent mode

# Train with custom config
start_training(
    root_folder="./my_dataset",
    database_file="image_classifier_data.csv", 
    train_from="embeddings",
    clip_models=[("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")],
    epochs=100,
    batch_size=64,
    enable_wandb=True
)
```

### Custom Metrics

The PyTorch Lightning model automatically logs metrics using `self.log()`. To add custom metrics:

```python
# In your LightningModule
def training_step(self, batch, batch_idx):
    # ... existing code ...
    
    # Custom metric
    custom_metric = compute_my_metric(outputs, targets)
    self.log("custom_metric", custom_metric)
    
    return loss
```

## Testing

### Run Integration Tests

```bash
# Test all wandb functionality
python src/test_wandb_integration.py

# Or run from project root
python -m src.test_wandb_integration
```

The test script will verify:
- All imports work correctly
- Key detection functionality
- Training script integration

## Troubleshooting

### Common Issues

1. **"No wandb API key found"**
   - Run `python example_wandb_setup.py` to set up your key
   - Or manually create `keys.txt` with `WANDB_KEY=your_key`

2. **"Failed to initialize wandb"**
   - Check your internet connection
   - Verify your API key is valid
   - Try logging in manually: `wandb login`

3. **"Permission denied"**
   - Ensure you have write permissions in the project directory
   - Check that `.gitignore` includes `keys.txt`

### Offline Mode

If you need to run without internet:

```python
# In your training script
os.environ['WANDB_MODE'] = 'offline'
```

Logs will be saved locally and can be synced later with `wandb sync`.

## Best Practices

1. **Security**: Never commit `keys.txt` to version control
2. **Organization**: Use descriptive run names for easy identification
3. **Tagging**: Add tags to group related experiments
4. **Notes**: Add run descriptions for important experiments
5. **Artifacts**: Use artifacts for model versioning and deployment

## API Reference

### `wandb_utils.py`

- `setup_wandb_key(root_folder, ask_user)`: Set up wandb API key
- `initialize_wandb(project_name, run_name, config, enabled)`: Initialize wandb run
- `get_wandb_enabled()`: Check if wandb is configured

### Training Parameters

- `enable_wandb` (bool): Enable/disable wandb logging
- `--no-wandb` (CLI flag): Disable wandb via command line

## Links

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Lightning wandb Integration](https://docs.wandb.ai/guides/integrations/lightning/)
- [wandb Python API](https://docs.wandb.ai/ref/python/) 