# Aesthetica ML Pipeline - Main Scripts

This repository contains all the essential scripts for the Aesthetica ML training pipeline with a clean, organized structure.

## Directory Structure

```
aesthetica/
├── src/                     # Source code
│   ├── train_model_mlp.py   # MLP model training
│   ├── test_model.py        # Model testing with Rich formatting
│   ├── dataset_to_embeddings.py  # Generate embeddings
│   ├── data_to_csv.py       # Convert raw datasets to CSV
│   └── setup.py             # Setup and verification script
├── datasets/                # All dataset files
│   ├── raw/                 # Raw datasets go here
│   │   ├── tattoo/          # Example raw dataset
│   │   └── dataset2/
│   └── normalized/          # Processed datasets (auto-created)
│       ├── tattoo/          # Example normalized dataset
│       └── dataset2/
├── pipeline_cli.py          # Main CLI interface
├── pipeline_requirements.txt
├── PIPELINE_README.md       # Detailed documentation
└── README.md               # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r pipeline_requirements.txt
   ```

2. **Run the interactive pipeline:**
   ```bash
   python pipeline_cli.py
   ```

3. **Or run specific steps:**
   ```bash
   # Generate CSV for a specific dataset
   python pipeline_cli.py --dataset your_dataset_name --csv-only
   
   # Generate embeddings
   python pipeline_cli.py --dataset your_dataset_name --embeddings-only
   
   # Train model
   python pipeline_cli.py --dataset your_dataset_name --train-only
   
   # Test model
   python pipeline_cli.py --dataset your_dataset_name --test-only
   
   # Run complete pipeline
   python pipeline_cli.py --dataset your_dataset_name
   ```

## Setup from src/ folder

You can also run the setup script from the src folder:
```bash
cd src
python setup.py
```

## Key Features

✅ **Rich Terminal UI** - Beautiful progress bars, tables, and colored output  
✅ **Interactive Mode** - Guided workflow with menu selection  
✅ **Modular Execution** - Run individual steps or complete pipeline  
✅ **Multiple Datasets** - Support for multiple datasets with status tracking  
✅ **Windows Compatible** - Fixed Unicode encoding issues for Windows terminals  
✅ **Comprehensive Testing** - Detailed model evaluation with visual statistics  
✅ **Clean Organization** - Source code separated from configuration and data

## Usage Examples

### Interactive Mode
```bash
python pipeline_cli.py
```

### Command Line Mode
```bash
# List available datasets
python pipeline_cli.py --list-datasets

# Run full pipeline on specific dataset
python pipeline_cli.py --dataset tattoo

# Train only (assumes CSV and embeddings exist)
python pipeline_cli.py --dataset tattoo --train-only --epochs 50 --batch-size 64

# Test existing model
python pipeline_cli.py --dataset tattoo --test-only
```

## File Descriptions

- **`pipeline_cli.py`** - Main CLI interface with Rich formatting and interactive mode
- **`src/train_model_mlp.py`** - MLP model training script 
- **`src/test_model.py`** - Model testing script with beautiful Rich tables and statistics
- **`src/dataset_to_embeddings.py`** - Generate embeddings from images using CLIP models
- **`src/data_to_csv.py`** - Convert raw image datasets to normalized CSV format
- **`src/setup.py`** - Setup and verification script
- **`pipeline_requirements.txt`** - Required Python packages
- **`PIPELINE_README.md`** - Detailed documentation

The pipeline will automatically create the normalized dataset directories in `datasets/normalized/` as needed. 