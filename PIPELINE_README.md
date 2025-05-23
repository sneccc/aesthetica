# 🎨 Aesthetica ML Training Pipeline CLI

A beautiful, comprehensive command-line interface for the complete machine learning workflow using the Rich library for stunning terminal output.

## 🚀 Features

- **Complete ML Pipeline**: From raw images to trained models with a single command
- **Beautiful CLI**: Rich terminal interface with progress bars, tables, and colored output
- **Interactive Mode**: Guided workflow with prompts and validation
- **Modular Execution**: Run individual steps or the complete pipeline
- **Real-time Progress**: Live updates and detailed logging
- **Comprehensive Testing**: Built-in model evaluation and metrics

## 📋 Pipeline Steps

1. **CSV Generation** 📊 - Convert raw image folders to structured CSV datasets
2. **Embedding Generation** 🔮 - Generate CLIP embeddings from images  
3. **Model Training** 🤖 - Train MLP classifier on embeddings
4. **Model Testing** 📈 - Evaluate model performance and generate predictions

## 🛠️ Installation

### Install Dependencies

```bash
# Install pipeline-specific requirements
pip install -r pipeline_requirements.txt

# Or install from main requirements
pip install -r requirements.txt
```

### Required Dependencies

- `rich>=13.0.0` - Beautiful terminal interface
- `torch>=1.9.0` - PyTorch for deep learning
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - ML utilities
- `open-clip-torch` - CLIP models
- `timm` - Vision transformers
- `numpy>=1.21.0` - Numerical computing

## 📁 Project Structure

Your project should be organized as follows:

```
aesthetica/
├── data/
│   ├── datasets/           # Raw datasets go here
│   │   ├── tattoo/        # Example dataset
│   │   └── artwork/       # Another dataset
│   └── normalized_*/      # Processed datasets (auto-generated)
├── scripts/               # Processing scripts
│   ├── data_to_csv.py
│   ├── dataset_to_embeddings.py
│   ├── train_model_mlp.py
│   └── test_model.py
├── pipeline_cli.py        # Main pipeline CLI
└── requirements.txt
```

## 🎯 Usage

### Interactive Mode (Recommended)

Run the pipeline in interactive mode for a guided experience:

```bash
python pipeline_cli.py
```

This will:
- Display a beautiful banner and status overview
- Show available datasets and their current status
- Guide you through configuration options
- Execute the complete pipeline with real-time progress

### Command Line Mode

#### List Available Datasets

```bash
python pipeline_cli.py --list-datasets
```

#### Run Complete Pipeline

```bash
python pipeline_cli.py --dataset tattoo --epochs 100 --batch-size 32
```

#### Run Individual Steps

```bash
# Generate CSV only
python pipeline_cli.py --dataset tattoo --csv-only

# Generate embeddings only
python pipeline_cli.py --dataset tattoo --embeddings-only

# Train model only
python pipeline_cli.py --dataset tattoo --train-only

# Test model only
python pipeline_cli.py --dataset tattoo --test-only
```

### Advanced Options

```bash
python pipeline_cli.py \
    --dataset tattoo \
    --clip-model "ViT-SO400M-14-SigLIP-384" \
    --epochs 200 \
    --batch-size 64 \
    --project-root /path/to/project
```

## 📊 Command Reference

### Main Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset name to process | - |
| `--list-datasets` | `-l` | List available datasets | - |
| `--auto` | - | Run full pipeline without prompts | False |

### Pipeline Control

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv-only` | Only generate CSV | False |
| `--embeddings-only` | Only generate embeddings | False |
| `--train-only` | Only train model | False |
| `--test-only` | Only test model | False |

### Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--clip-model` | CLIP model to use | "ViT-SO400M-14-SigLIP-384" |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--project-root` | Project root directory | Auto-detect |

## 🎨 Example Workflows

### 1. First Time Setup

```bash
# 1. Add your raw dataset to data/datasets/your_dataset/
# 2. Run the complete pipeline
python pipeline_cli.py --dataset your_dataset
```

### 2. Experiment with Different Models

```bash
# Try different CLIP models
python pipeline_cli.py --dataset tattoo --clip-model "ViT-L/14"
python pipeline_cli.py --dataset tattoo --clip-model "ViT-SO400M-14-SigLIP-384"
```

### 3. Hyperparameter Tuning

```bash
# Experiment with training parameters
python pipeline_cli.py --dataset tattoo --epochs 200 --batch-size 64
python pipeline_cli.py --dataset tattoo --epochs 500 --batch-size 128
```

### 4. Incremental Processing

```bash
# Process step by step
python pipeline_cli.py --dataset tattoo --csv-only
python pipeline_cli.py --dataset tattoo --embeddings-only
python pipeline_cli.py --dataset tattoo --train-only
python pipeline_cli.py --dataset tattoo --test-only
```

## 📈 Output and Results

### Status Overview

The pipeline provides a beautiful status table showing:
- ✅ Completed steps
- ❌ Pending steps  
- 🔴 Needs CSV
- 🟡 Needs Embeddings
- 🔵 Ready to Train
- 🟢 Trained

### Generated Files

For each dataset, the pipeline creates:

```
data/normalized_dataset_name/
├── image_classifier_data.csv    # Dataset metadata
├── image_embeddings.npy         # CLIP embeddings
├── file_paths.npy              # File path mappings
├── model.pth                   # Trained model weights
├── model_config.json           # Model configuration
└── test_results.csv            # Test predictions
```

### Model Testing Output

The testing phase provides:
- Validation accuracy and metrics
- Classification report
- Confusion matrix
- Confidence statistics
- Sample predictions
- Detailed CSV results

## 🔧 Customization

### Adding New CLIP Models

Edit the `dataset_to_embeddings.py` script to support new models:

```python
if args.clip_model == "your-custom-model":
    clip_model = [("custom-hub", "your-model-name")]
```

### Modifying Training Parameters

The training script supports various parameters:
- Architecture changes in `train_model_mlp.py`
- Learning rate schedules
- Augmentation strategies
- Validation splits

### Custom Data Processing

Modify `data_to_csv.py` for custom:
- Image preprocessing
- Label extraction
- Data filtering
- Train/test splits

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   ❌ Dataset 'name' not found
   ```
   - Ensure your dataset folder exists in `data/datasets/`

2. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named 'rich'
   ```
   - Install requirements: `pip install -r pipeline_requirements.txt`

3. **CUDA Memory Issues**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce batch size: `--batch-size 16`
   - Use CPU: Set `CUDA_VISIBLE_DEVICES=""`

4. **File Not Found Errors**
   - Check file paths in error messages
   - Ensure previous pipeline steps completed successfully
   - Verify dataset structure

### Debug Mode

For detailed debugging, check the logs:
- Flask app logs: `flask_app/logs/app.log`
- Script output: Displayed in real-time during execution

## 🚀 Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Optimize Batch Size**: Balance memory usage and training speed
3. **Use SSD**: Store datasets on SSD for faster I/O
4. **Monitor Resources**: Watch CPU/GPU utilization during processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test with the pipeline CLI
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎉 Example Output

When you run the pipeline, you'll see beautiful output like this:

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🎨 AESTHETICA ML TRAINING PIPELINE 🤖                ║
║                                                           ║
║     Complete Machine Learning Workflow                    ║
║     From Raw Data → Trained Model → Predictions          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

📊 Dataset Status Overview
┌─────────────┬─────┬────────────┬───────┬─────────────────┐
│ Dataset     │ CSV │ Embeddings │ Model │ Status          │
│ Name        │     │            │       │                 │
├─────────────┼─────┼────────────┼───────┼─────────────────┤
│ tattoo      │ ✅  │ ✅         │ ✅    │ 🟢 Trained     │
│ artwork     │ ✅  │ ❌         │ ❌    │ 🟡 Needs       │
│             │     │            │       │ Embeddings     │
└─────────────┴─────┴────────────┴───────┴─────────────────┘
```

Happy training! 🎨🤖 