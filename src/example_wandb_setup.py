#!/usr/bin/env python3
"""
Example script demonstrating wandb integration with the aesthetica training pipeline.

This script shows how to:
1. Set up wandb API keys
2. Enable wandb logging during training
3. View training metrics in wandb dashboard

Usage:
    python example_wandb_setup.py --input_directory /path/to/your/dataset
"""

import sys
import pathlib

# Add src to path for imports
sys.path.append(str(pathlib.Path(__file__).parent / "src"))

from wandb_utils import setup_wandb_key, initialize_wandb, get_wandb_enabled
from rich.console import Console

console = Console()

def main():
    """Main function demonstrating wandb setup and usage."""
    
    console.print("🎯 [bold]Wandb Integration Example[/bold]", style="blue")
    console.print("This example shows how to set up and use wandb logging with the aesthetica training pipeline.\n")
    
    # Step 1: Setup wandb API key
    console.print("Step 1: Setting up wandb API key...", style="yellow")
    wandb_key = setup_wandb_key(root_folder=".", ask_user=True)
    
    if wandb_key:
        console.print("✅ Wandb key configured successfully!", style="green")
        
        # Step 2: Test wandb initialization
        console.print("\nStep 2: Testing wandb initialization...", style="yellow")
        
        # Example config that would be used during training
        example_config = {
            "architecture": "MLP",
            "input_size": 1024,
            "num_classes": 10,
            "hidden_units": [4096, 1024, 512, 128],
            "epochs": 100,
            "batch_size": 32,
            "optimizer": "lion",
        }
        
        run = initialize_wandb(
            project_name="aesthetica-example",
            run_name="test-setup",
            config=example_config,
            enabled=True
        )
        
        if run:
            console.print("✅ Wandb run initialized successfully!", style="green")
            console.print(f"📊 View your run at: {run.url}", style="blue")
            
            # Example of logging metrics (this would happen during training)
            console.print("\nStep 3: Example metric logging...", style="yellow")
            import wandb
            
            # Simulate some training metrics
            for epoch in range(3):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": 0.5 - epoch * 0.1,
                    "val_loss": 0.6 - epoch * 0.09,
                    "train_acc": 0.7 + epoch * 0.1,
                    "val_acc": 0.65 + epoch * 0.09
                })
            
            console.print("✅ Example metrics logged!", style="green")
            
            # Finish the run
            wandb.finish()
            console.print("🏁 Wandb run finished", style="blue")
            
        else:
            console.print("❌ Failed to initialize wandb run", style="red")
            
    else:
        console.print("⏭️  Wandb setup skipped", style="yellow")
    
    # Step 4: Show how to use with training
    console.print("\n" + "="*50, style="blue")
    console.print("🚀 [bold]How to use with training:[/bold]", style="blue")
    console.print("\n1. With wandb enabled (default):")
    console.print("   python src/train_model_mlp.py -i /path/to/dataset", style="cyan")
    console.print("\n2. With wandb disabled:")
    console.print("   python src/train_model_mlp.py -i /path/to/dataset --no-wandb", style="cyan")
    console.print("\n3. Training will automatically:")
    console.print("   • Check for keys.txt file")
    console.print("   • Ask for wandb key if not found")
    console.print("   • Log metrics, hyperparameters, and model artifacts")
    console.print("   • Create beautiful dashboards for experiment tracking")
    
    console.print("\n📁 Your wandb key has been saved to keys.txt", style="green")
    console.print("🔒 This file is automatically added to .gitignore for security", style="green")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example wandb setup for aesthetica training")
    parser.add_argument("--input_directory", help="Optional: path to dataset for full training example")
    
    args = parser.parse_args()
    
    try:
        main()
        
        if args.input_directory:
            console.print(f"\n🎯 [bold]Running full training example with {args.input_directory}[/bold]", style="blue")
            # Import and run training
            from train_model_mlp import start_training
            
            clip_models = [("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")]
            start_training(
                root_folder=args.input_directory,
                database_file='image_classifier_data.csv',
                train_from='embeddings',
                clip_models=clip_models,
                epochs=5,  # Short example
                batch_size=32,
                enable_wandb=True
            )
            
    except KeyboardInterrupt:
        console.print("\n⏹️  Setup interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        raise 