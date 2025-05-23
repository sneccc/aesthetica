#!/usr/bin/env python3
"""
Aesthetica ML Training Pipeline CLI
A comprehensive command-line interface for the complete machine learning pipeline.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Rich imports for beautiful CLI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown

# Initialize Rich console
console = Console()

class AestheticaPipeline:
    """Main pipeline orchestrator for the Aesthetica ML workflow."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.datasets_dir = self.project_root / "datasets"  # Base datasets directory
        self.scripts_dir = self.project_root / "src"  # Scripts are in src/ folder
        self.raw_datasets_dir = self.datasets_dir / "raw"  # Raw datasets in datasets/raw/
        self.normalized_datasets_dir = self.datasets_dir / "normalized"  # Normalized in datasets/normalized/
        
        # Ensure directories exist
        self.datasets_dir.mkdir(exist_ok=True)
        self.raw_datasets_dir.mkdir(exist_ok=True)
        self.normalized_datasets_dir.mkdir(exist_ok=True)
        
        # Pipeline status
        self.pipeline_status = {
            "csv_generation": {"completed": False, "time": None, "output_path": None},
            "embedding_generation": {"completed": False, "time": None, "output_path": None},
            "model_training": {"completed": False, "time": None, "output_path": None},
            "model_testing": {"completed": False, "time": None, "results": None}
        }
    
    def display_banner(self):
        """Display the application banner."""
        banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🎨 AESTHETICA ML TRAINING PIPELINE 🤖                 ║
    ║                                                           ║
    ║     Complete Machine Learning Workflow                    ║
    ║     From Raw Data → Trained Model → Predictions           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold magenta", box=box.DOUBLE))
    
    def scan_datasets(self) -> List[Dict]:
        """Scan for available datasets and their status."""
        datasets = []
        
        if not self.raw_datasets_dir.exists():
            return datasets
        
        for item in self.raw_datasets_dir.iterdir():
            if item.is_dir():
                dataset_name = item.name
                normalized_path = self.normalized_datasets_dir / dataset_name
                
                # Check status of various pipeline stages
                csv_path = normalized_path / "image_classifier_data.csv"
                embeddings_path = normalized_path / "image_embeddings.npy"
                model_path = normalized_path / "model.pth"
                config_path = normalized_path / "model_config.json"
                
                status = {
                    "name": dataset_name,
                    "raw_path": str(item),
                    "normalized_path": str(normalized_path),
                    "has_csv": csv_path.exists(),
                    "has_embeddings": embeddings_path.exists(),
                    "has_model": model_path.exists() and config_path.exists(),
                    "csv_path": str(csv_path),
                    "embeddings_path": str(embeddings_path),
                    "model_path": str(model_path)
                }
                
                # Determine overall status
                if not status["has_csv"]:
                    status["stage"] = "needs_csv"
                elif not status["has_embeddings"]:
                    status["stage"] = "needs_embeddings"
                elif not status["has_model"]:
                    status["stage"] = "ready_to_train"
                else:
                    status["stage"] = "trained"
                
                datasets.append(status)
        
        return datasets
    
    def display_datasets_status(self, datasets: List[Dict]):
        """Display a beautiful table of dataset statuses."""
        table = Table(title="📊 Dataset Status Overview", box=box.ROUNDED)
        
        table.add_column("Dataset Name", style="bold cyan", no_wrap=True)
        table.add_column("CSV", justify="center")
        table.add_column("Embeddings", justify="center")
        table.add_column("Model", justify="center")
        table.add_column("Status", style="bold")
        
        status_colors = {
            "needs_csv": "red",
            "needs_embeddings": "yellow",
            "ready_to_train": "blue",
            "trained": "green"
        }
        
        status_messages = {
            "needs_csv": "🔴 Needs CSV",
            "needs_embeddings": "🟡 Needs Embeddings",
            "ready_to_train": "🔵 Ready to Train",
            "trained": "🟢 Trained"
        }
        
        for dataset in datasets:
            csv_status = "✅" if dataset["has_csv"] else "❌"
            embeddings_status = "✅" if dataset["has_embeddings"] else "❌"
            model_status = "✅" if dataset["has_model"] else "❌"
            
            table.add_row(
                dataset["name"],
                csv_status,
                embeddings_status,
                model_status,
                Text(status_messages[dataset["stage"]], style=status_colors[dataset["stage"]])
            )
        
        console.print(table)
    
    def run_command_with_progress(self, command: List[str], description: str, cwd: str = None) -> Tuple[bool, str]:
        """Run a command with a progress spinner and capture output."""
        output_lines = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(description, total=None)
            
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=cwd or self.project_root,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Read output line by line
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.strip())
                    # Update description with latest output (keep it short)
                    if line.strip():
                        progress.update(task, description=f"{description} - {line.strip()[:50]}...")
                
                process.wait()
                success = process.returncode == 0
                
            except Exception as e:
                success = False
                output_lines.append(f"Error: {str(e)}")
        
        return success, "\n".join(output_lines)
    
    def generate_csv(self, dataset_name: str, raw_path: str) -> bool:
        """Generate CSV from raw image dataset."""
        console.print(f"\n🔄 [bold blue]Step 1: Generating CSV for dataset '{dataset_name}'[/bold blue]")
        
        # Prepare output path
        normalized_path = self.normalized_datasets_dir / dataset_name
        
        # Display input/output paths with rich formatting
        paths_table = Table(title="📁 CSV Generation Paths", box=box.ROUNDED)
        paths_table.add_column("Type", style="bold cyan")
        paths_table.add_column("Path", style="white")
        paths_table.add_row("Raw Dataset", str(raw_path))
        paths_table.add_row("Normalized Output", str(normalized_path))
        paths_table.add_row("CSV Output", str(normalized_path / "image_classifier_data.csv"))
        console.print(paths_table)
        
        command = [
            sys.executable,
            str(self.scripts_dir / "data_to_csv.py"),
            "-i", raw_path,
            "-o", str(normalized_path)  # Pass the output directory explicitly
        ]
        
        success, output = self.run_command_with_progress(
            command,
            f"Processing images and generating CSV..."
        )
        
        if success:
            csv_path = normalized_path / "image_classifier_data.csv"
            
            if csv_path.exists():
                # Rich success panel with details
                success_panel = Panel(
                    f"✅ [bold green]CSV Generated Successfully![/bold green]\n\n"
                    f"📄 CSV File: [cyan]{csv_path}[/cyan]\n"
                    f"📁 Normalized Images: [cyan]{normalized_path}[/cyan]\n"
                    f"🔗 Dataset: [yellow]{dataset_name}[/yellow]",
                    title="Step 1 Complete",
                    style="green",
                    box=box.DOUBLE
                )
                console.print(success_panel)
                
                self.pipeline_status["csv_generation"] = {
                    "completed": True,
                    "time": datetime.now(),
                    "output_path": str(csv_path)
                }
                return True
            else:
                console.print(f"❌ [bold red]CSV file not found after generation[/bold red]")
                return False
        else:
            console.print(f"❌ [bold red]CSV generation failed[/bold red]")
            console.print(Panel(output, title="Error Output", style="red"))
            return False
    
    def generate_embeddings(self, normalized_path: str, clip_model: str = "ViT-B/32") -> bool:
        """Generate embeddings from the CSV dataset."""
        console.print(f"\n🔄 [bold blue]Step 2: Generating embeddings[/bold blue]")
        
        # Display input/output paths with rich formatting
        paths_table = Table(title="🧠 Embeddings Generation Paths", box=box.ROUNDED)
        paths_table.add_column("Type", style="bold cyan")
        paths_table.add_column("Path/Value", style="white")
        paths_table.add_row("Input CSV", str(Path(normalized_path) / "image_classifier_data.csv"))
        paths_table.add_row("Normalized Images", str(normalized_path))
        paths_table.add_row("CLIP Model", f"[yellow]{clip_model}[/yellow]")
        paths_table.add_row("Embeddings Output", str(Path(normalized_path) / "image_embeddings.npy"))
        paths_table.add_row("Labels Output", str(Path(normalized_path) / "labels.npy"))
        console.print(paths_table)
        
        command = [
            sys.executable,
            str(self.scripts_dir / "dataset_to_embeddings.py"),
            "-i", normalized_path,
            "--clip_model", clip_model
        ]
        
        success, output = self.run_command_with_progress(
            command,
            f"Generating embeddings with {clip_model}..."
        )
        
        if success:
            embeddings_path = Path(normalized_path) / "image_embeddings.npy"
            labels_path = Path(normalized_path) / "labels.npy"
            
            if embeddings_path.exists():
                # Rich success panel with details
                success_panel = Panel(
                    f"✅ [bold green]Embeddings Generated Successfully![/bold green]\n\n"
                    f"🧠 Embeddings File: [cyan]{embeddings_path}[/cyan]\n"
                    f"🏷️  Labels File: [cyan]{labels_path}[/cyan]\n"
                    f"🤖 CLIP Model: [yellow]{clip_model}[/yellow]\n"
                    f"📊 Dataset: [magenta]{Path(normalized_path).name}[/magenta]",
                    title="Step 2 Complete",
                    style="green",
                    box=box.DOUBLE
                )
                console.print(success_panel)
                
                self.pipeline_status["embedding_generation"] = {
                    "completed": True,
                    "time": datetime.now(),
                    "output_path": str(embeddings_path)
                }
                return True
            else:
                console.print(f"❌ [bold red]Embeddings file not found after generation[/bold red]")
                return False
        else:
            console.print(f"❌ [bold red]Embeddings generation failed[/bold red]")
            console.print(Panel(output, title="Error Output", style="red"))
            return False
    
    def train_model(self, normalized_path: str, epochs: int = 100, batch_size: int = 32) -> bool:
        """Train the MLP model."""
        console.print(f"\n🔄 [bold blue]Step 3: Training model[/bold blue]")
        
        # Display training parameters and paths with rich formatting
        training_table = Table(title="🚀 Model Training Configuration", box=box.ROUNDED)
        training_table.add_column("Parameter", style="bold cyan")
        training_table.add_column("Value", style="white")
        training_table.add_row("Input Embeddings", str(Path(normalized_path) / "image_embeddings.npy"))
        training_table.add_row("Input Labels", str(Path(normalized_path) / "labels.npy"))
        training_table.add_row("Training Epochs", f"[yellow]{epochs}[/yellow]")
        training_table.add_row("Batch Size", f"[yellow]{batch_size}[/yellow]")
        training_table.add_row("Model Output", str(Path(normalized_path) / "model.pth"))
        training_table.add_row("Config Output", str(Path(normalized_path) / "model_config.json"))
        console.print(training_table)
        
        command = [
            sys.executable,
            str(self.scripts_dir / "train_model_mlp.py"),
            "-i", normalized_path,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size)
        ]
        
        success, output = self.run_command_with_progress(
            command,
            f"Training MLP model ({epochs} epochs, batch size {batch_size})..."
        )
        
        if success:
            model_path = Path(normalized_path) / "model.pth"
            config_path = Path(normalized_path) / "model_config.json"
            
            if model_path.exists() and config_path.exists():
                # Rich success panel with details
                success_panel = Panel(
                    f"✅ [bold green]Model Trained Successfully![/bold green]\n\n"
                    f"🤖 Model File: [cyan]{model_path}[/cyan]\n"
                    f"⚙️  Config File: [cyan]{config_path}[/cyan]\n"
                    f"📊 Epochs: [yellow]{epochs}[/yellow] | Batch Size: [yellow]{batch_size}[/yellow]\n"
                    f"📁 Dataset: [magenta]{Path(normalized_path).name}[/magenta]",
                    title="Step 3 Complete",
                    style="green",
                    box=box.DOUBLE
                )
                console.print(success_panel)
                
                self.pipeline_status["model_training"] = {
                    "completed": True,
                    "time": datetime.now(),
                    "output_path": str(model_path)
                }
                return True
            else:
                console.print(f"❌ [bold red]Model files not found after training[/bold red]")
                return False
        else:
            console.print(f"❌ [bold red]Model training failed[/bold red]")
            console.print(Panel(output, title="Error Output", style="red"))
            return False
    
    def test_model(self, normalized_path: str) -> bool:
        """Test the trained model."""
        console.print(f"\n🔄 [bold blue]Step 4: Testing model[/bold blue]")
        
        # Display test configuration with rich formatting
        test_table = Table(title="🧪 Model Testing Configuration", box=box.ROUNDED)
        test_table.add_column("Resource", style="bold cyan")
        test_table.add_column("Path", style="white")
        test_table.add_row("Model File", str(Path(normalized_path) / "model.pth"))
        test_table.add_row("Config File", str(Path(normalized_path) / "model_config.json"))
        test_table.add_row("Test Data", str(Path(normalized_path) / "image_classifier_data.csv"))
        test_table.add_row("Dataset", f"[magenta]{Path(normalized_path).name}[/magenta]")
        console.print(test_table)
        
        command = [
            sys.executable,
            str(self.scripts_dir / "test_model.py"),
            "-i", normalized_path
        ]
        
        success, output = self.run_command_with_progress(
            command,
            "Running model evaluation..."
        )
        
        if success:
            # Rich success panel with details
            success_panel = Panel(
                f"✅ [bold green]Model Testing Completed![/bold green]\n\n"
                f"📊 Test Results Available Below\n"
                f"🎯 Dataset: [magenta]{Path(normalized_path).name}[/magenta]\n"
                f"📁 Model: [cyan]{Path(normalized_path) / 'model.pth'}[/cyan]",
                title="Step 4 Complete",
                style="green",
                box=box.DOUBLE
            )
            console.print(success_panel)
            
            # Display the test results in a panel
            console.print(Panel(output, title="📊 Model Test Results", style="blue", box=box.ROUNDED))
            
            self.pipeline_status["model_testing"] = {
                "completed": True,
                "time": datetime.now(),
                "results": output
            }
            return True
        else:
            console.print(f"❌ [bold red]Model testing failed[/bold red]")
            console.print(Panel(output, title="Error Output", style="red"))
            return False
    
    def display_pipeline_summary(self):
        """Display a summary of the completed pipeline."""
        console.print(f"\n🎉 [bold green]Pipeline Summary[/bold green]")
        
        table = Table(title="Pipeline Execution Summary", box=box.ROUNDED)
        table.add_column("Stage", style="bold cyan")
        table.add_column("Status", justify="center")
        table.add_column("Completion Time")
        table.add_column("Output/Notes")
        
        stages = [
            ("CSV Generation", "csv_generation"),
            ("Embedding Generation", "embedding_generation"),
            ("Model Training", "model_training"),
            ("Model Testing", "model_testing")
        ]
        
        for stage_name, stage_key in stages:
            status_info = self.pipeline_status[stage_key]
            
            if status_info["completed"]:
                status = "✅ Complete"
                time_str = status_info["time"].strftime("%H:%M:%S") if status_info["time"] else "N/A"
                output = status_info.get("output_path", status_info.get("results", "N/A"))
                if len(str(output)) > 50:
                    output = str(output)[:47] + "..."
            else:
                status = "❌ Not Run"
                time_str = "N/A"
                output = "N/A"
            
            table.add_row(stage_name, status, time_str, str(output))
        
        console.print(table)
    
    def run_full_pipeline(self, dataset_name: str, raw_path: str, 
                         clip_model: str = "ViT-B/32", epochs: int = 100, 
                         batch_size: int = 32) -> bool:
        """Run the complete pipeline for a dataset."""
        console.print(f"\n🚀 [bold magenta]Starting Full Pipeline for '{dataset_name}'[/bold magenta]")
        
        normalized_path = self.normalized_datasets_dir / dataset_name
        
        # Step 1: Generate CSV
        if not self.generate_csv(dataset_name, raw_path):
            return False
        
        # Step 2: Generate Embeddings
        if not self.generate_embeddings(str(normalized_path), clip_model):
            return False
        
        # Step 3: Train Model
        if not self.train_model(str(normalized_path), epochs, batch_size):
            return False
        
        # Step 4: Test Model
        if not self.test_model(str(normalized_path)):
            return False
        
        # Display summary
        self.display_pipeline_summary()
        
        console.print(f"\n🎉 [bold green]Pipeline completed successfully for '{dataset_name}'![/bold green]")
        return True
    
    def interactive_mode(self):
        """Run the pipeline in interactive mode."""
        self.display_banner()
        
        # Scan for datasets
        datasets = self.scan_datasets()
        
        if not datasets:
            console.print("❌ [bold red]No datasets found in the raw datasets directory[/bold red]")
            console.print(f"📁 Please add datasets to: {self.raw_datasets_dir}")
            return
        
        # Display current status
        self.display_datasets_status(datasets)
        
        # Ask user to select a dataset
        console.print(f"\n🤔 [bold yellow]Select a dataset to process:[/bold yellow]")
        dataset_choices = [f"{i+1}. {ds['name']}" for i, ds in enumerate(datasets)]
        
        for choice in dataset_choices:
            console.print(f"   {choice}")
        
        try:
            selection = int(Prompt.ask("Enter your choice", default="1")) - 1
            if selection < 0 or selection >= len(datasets):
                console.print("❌ [bold red]Invalid selection[/bold red]")
                return
            
            selected_dataset = datasets[selection]
            console.print(f"✅ Selected dataset: [bold cyan]{selected_dataset['name']}[/bold cyan]")
            
            # Get pipeline parameters
            console.print(f"\n⚙️  [bold yellow]Configure Pipeline Parameters:[/bold yellow]")
            
            clip_model = Prompt.ask("CLIP Model", default="ViT-SO400M-14-SigLIP-384")
            epochs = int(Prompt.ask("Training Epochs", default="100"))
            batch_size = int(Prompt.ask("Batch Size", default="32"))
            
            # Confirm and run
            console.print(f"\n📋 [bold yellow]Pipeline Configuration:[/bold yellow]")
            console.print(f"   Dataset: {selected_dataset['name']}")
            console.print(f"   CLIP Model: {clip_model}")
            console.print(f"   Epochs: {epochs}")
            console.print(f"   Batch Size: {batch_size}")
            
            if Confirm.ask("Start pipeline?", default=True):
                self.run_full_pipeline(
                    selected_dataset['name'], 
                    selected_dataset['raw_path'],
                    clip_model, 
                    epochs, 
                    batch_size
                )
            else:
                console.print("👋 Pipeline cancelled")
                
        except (ValueError, KeyboardInterrupt):
            console.print("\n👋 Pipeline cancelled")
            return


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aesthetica ML Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --dataset tattoo --auto          # Run full pipeline for 'tattoo' dataset
  %(prog)s --dataset tattoo --csv-only      # Only generate CSV
  %(prog)s --list-datasets                   # List available datasets
        """
    )
    
    parser.add_argument("--dataset", "-d", help="Dataset name to process")
    parser.add_argument("--auto", action="store_true", help="Run full pipeline without prompts")
    parser.add_argument("--csv-only", action="store_true", help="Only generate CSV")
    parser.add_argument("--embeddings-only", action="store_true", help="Only generate embeddings")
    parser.add_argument("--train-only", action="store_true", help="Only train model")
    parser.add_argument("--test-only", action="store_true", help="Only test model")
    parser.add_argument("--list-datasets", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--clip-model", default="ViT-SO400M-14-SigLIP-384", help="CLIP model to use")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AestheticaPipeline(args.project_root)
    
    if args.list_datasets:
        pipeline.display_banner()
        datasets = pipeline.scan_datasets()
        pipeline.display_datasets_status(datasets)
        return
    
    if args.dataset:
        datasets = pipeline.scan_datasets()
        dataset = next((ds for ds in datasets if ds['name'] == args.dataset), None)
        
        if not dataset:
            console.print(f"❌ [bold red]Dataset '{args.dataset}' not found[/bold red]")
            return
        
        pipeline.display_banner()
        console.print(f"🎯 Processing dataset: [bold cyan]{args.dataset}[/bold cyan]")
        
        normalized_path = pipeline.normalized_datasets_dir / args.dataset
        
        if args.csv_only:
            pipeline.generate_csv(args.dataset, dataset['raw_path'])
        elif args.embeddings_only:
            pipeline.generate_embeddings(str(normalized_path), args.clip_model)
        elif args.train_only:
            pipeline.train_model(str(normalized_path), args.epochs, args.batch_size)
        elif args.test_only:
            pipeline.test_model(str(normalized_path))
        else:
            # Full pipeline
            pipeline.run_full_pipeline(
                args.dataset, 
                dataset['raw_path'],
                args.clip_model, 
                args.epochs, 
                args.batch_size
            )
    else:
        # Interactive mode
        pipeline.interactive_mode()


if __name__ == "__main__":
    main() 