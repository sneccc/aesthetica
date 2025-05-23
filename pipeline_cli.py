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

# Import wandb utilities
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from src.wandb_utils import setup_wandb_key, get_wandb_enabled, check_keys_file
except ImportError:
    # Fallback import
    try:
        from wandb_utils import setup_wandb_key, get_wandb_enabled, check_keys_file
    except ImportError:
        console.print("⚠️  wandb_utils not found. Wandb integration will be disabled.", style="yellow")
        def setup_wandb_key(*args, **kwargs): return None
        def get_wandb_enabled(): return False
        def check_keys_file(*args, **kwargs): return None

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
        
        # Wandb configuration
        self.wandb_enabled = False
        self.wandb_configured = False
    
    def setup_wandb(self, ask_user: bool = True) -> bool:
        """Setup wandb integration for the pipeline."""
        if self.wandb_configured:
            return self.wandb_enabled
            
        console.print("\n🔧 [bold blue]Wandb Integration Setup[/bold blue]")
        
        # Check if key already exists
        existing_key = check_keys_file(str(self.project_root))
        if existing_key:
            self.wandb_enabled = True
            self.wandb_configured = True
            console.print("✅ [green]Wandb key found and configured[/green]")
            return True
        
        if not ask_user:
            self.wandb_configured = True
            return False
            
        # Setup wandb key interactively
        wandb_key = setup_wandb_key(str(self.project_root), ask_user=True)
        
        self.wandb_enabled = wandb_key is not None
        self.wandb_configured = True
        
        if self.wandb_enabled:
            console.print("✅ [green]Wandb integration enabled[/green]")
        else:
            console.print("⏭️  [yellow]Wandb integration disabled[/yellow]")
            
        return self.wandb_enabled
    
    def display_wandb_status(self) -> None:
        """Display current wandb configuration status."""
        if not self.wandb_configured:
            status = "🔧 Not Configured"
            style = "yellow"
        elif self.wandb_enabled:
            status = "✅ Enabled"
            style = "green"
        else:
            status = "⏭️  Disabled"
            style = "yellow"
            
        console.print(f"Wandb Logging: [{style}]{status}[/{style}]")
    
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
        last_update_line = ""
        
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
                    line_stripped = line.strip()
                    output_lines.append(line_stripped)
                    
                    # Only update progress for meaningful lines and avoid repetition
                    if line_stripped and line_stripped != last_update_line:
                        # Preserve important validation debug messages
                        important_validation_patterns = [
                            ">> validation epoch end:",
                            ">> triggering validation", 
                            ">> successfully completed validation",
                            ">> failed validation",
                            "validation_viz_stats:",
                            ">> setup validation visualization"
                        ]
                        
                        is_important_validation = any(pattern in line_stripped.lower() for pattern in important_validation_patterns)
                        
                        # Filter out common repetitive patterns but preserve important validation messages
                        if is_important_validation or not any(pattern in line_stripped.lower() for pattern in [
                            "epoch", "batch", "loss:", "accuracy:", "learning rate", 
                            "val_loss", "val_acc", "step", "/", "%"
                        ]):
                            # Only show important progress messages, truncated
                            update_text = line_stripped[:60] + "..." if len(line_stripped) > 60 else line_stripped
                            progress.update(task, description=f"{description} - {update_text}")
                            last_update_line = line_stripped
                
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
    
    def train_model(self, normalized_path: str, epochs: int = 100, batch_size: int = 32, enable_wandb: bool = None) -> bool:
        """Train the MLP model."""
        console.print(f"\n🔄 [bold blue]Step 3: Training model[/bold blue]")
        
        # Setup wandb if not configured yet
        if enable_wandb is None:
            enable_wandb = self.setup_wandb(ask_user=True)
        elif enable_wandb and not self.wandb_configured:
            enable_wandb = self.setup_wandb(ask_user=False)
        
        # Display training parameters and paths with rich formatting
        training_table = Table(title="🚀 Model Training Configuration", box=box.ROUNDED)
        training_table.add_column("Parameter", style="bold cyan")
        training_table.add_column("Value", style="white")
        training_table.add_row("Input Embeddings", str(Path(normalized_path) / "image_embeddings.npy"))
        training_table.add_row("Input Labels", str(Path(normalized_path) / "labels.npy"))
        training_table.add_row("Training Epochs", f"[yellow]{epochs}[/yellow]")
        training_table.add_row("Batch Size", f"[yellow]{batch_size}[/yellow]")
        training_table.add_row("Wandb Logging", f"[{'green' if enable_wandb else 'yellow'}]{'Enabled' if enable_wandb else 'Disabled'}[/{'green' if enable_wandb else 'yellow'}]")
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
        
        # Add wandb flag if disabled
        if not enable_wandb:
            command.append("--no-wandb")
        
        success, output = self.run_command_with_progress(
            command,
            f"Training MLP model ({epochs} epochs, batch size {batch_size}, wandb {'enabled' if enable_wandb else 'disabled'})..."
        )
        
        if success:
            model_path = Path(normalized_path) / "model.pth"
            config_path = Path(normalized_path) / "model_config.json"
            
            if model_path.exists() and config_path.exists():
                # Parse validation visualization stats from output
                viz_count = 0
                viz_epochs = []
                for line in output.split('\n'):
                    if line.startswith('VALIDATION_VIZ_STATS:'):
                        try:
                            # Parse: "VALIDATION_VIZ_STATS: count=2, epochs=[0, 100]"
                            parts = line.split(': ')[1]
                            count_part = parts.split(', epochs=')[0]
                            epochs_part = parts.split(', epochs=')[1]
                            
                            viz_count = int(count_part.split('=')[1])
                            viz_epochs = eval(epochs_part)  # Safe since we control the input
                        except Exception as e:
                            console.print(f"Warning: Could not parse validation visualization stats: {e}")
                
                # Rich success panel with details
                wandb_info = ""
                if enable_wandb:
                    wandb_info = "\n🎯 Training metrics logged to wandb dashboard"
                    if viz_count > 0:
                        epoch_list = ", ".join(map(str, viz_epochs))
                        wandb_info += f"\n📸 Validation visualizations: {viz_count} (epochs: {epoch_list})"
                
                success_panel = Panel(
                    f"✅ [bold green]Model Trained Successfully![/bold green]\n\n"
                    f"🤖 Model File: [cyan]{model_path}[/cyan]\n"
                    f"⚙️  Config File: [cyan]{config_path}[/cyan]\n"
                    f"📊 Epochs: [yellow]{epochs}[/yellow] | Batch Size: [yellow]{batch_size}[/yellow]\n"
                    f"📁 Dataset: [magenta]{Path(normalized_path).name}[/magenta]"
                    f"{wandb_info}",
                    title="Step 3 Complete",
                    style="green",
                    box=box.DOUBLE
                )
                console.print(success_panel)
                
                self.pipeline_status["model_training"] = {
                    "completed": True,
                    "time": datetime.now(),
                    "output_path": str(model_path),
                    "validation_viz_count": viz_count,
                    "validation_viz_epochs": viz_epochs
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
                         batch_size: int = 32, enable_wandb: bool = None) -> bool:
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
        if not self.train_model(str(normalized_path), epochs, batch_size, enable_wandb):
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
        
        # Setup wandb first
        console.print(f"\n🔧 [bold blue]Setup Configuration[/bold blue]")
        enable_wandb = self.setup_wandb(ask_user=True)
        
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
            
            # Allow user to override wandb setting
            if not self.wandb_enabled:
                enable_wandb_override = Confirm.ask("Enable wandb logging for this run?", default=False)
                if enable_wandb_override:
                    enable_wandb = self.setup_wandb(ask_user=True)
            else:
                enable_wandb_override = not Confirm.ask("Disable wandb logging for this run?", default=False)
                enable_wandb = enable_wandb_override
            
            # Confirm and run
            console.print(f"\n📋 [bold yellow]Pipeline Configuration:[/bold yellow]")
            console.print(f"   Dataset: {selected_dataset['name']}")
            console.print(f"   CLIP Model: {clip_model}")
            console.print(f"   Epochs: {epochs}")
            console.print(f"   Batch Size: {batch_size}")
            self.display_wandb_status() if not enable_wandb else console.print(f"   Wandb Logging: [green]✅ Enabled[/green]")
            
            if Confirm.ask("Start pipeline?", default=True):
                self.run_full_pipeline(
                    selected_dataset['name'], 
                    selected_dataset['raw_path'],
                    clip_model, 
                    epochs, 
                    batch_size,
                    enable_wandb
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
  %(prog)s --dataset tattoo --no-wandb      # Run without wandb logging
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
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--setup-wandb", action="store_true", help="Setup wandb API key and exit")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AestheticaPipeline(args.project_root)
    
    # Handle wandb setup command
    if args.setup_wandb:
        pipeline.display_banner()
        console.print(f"\n🔧 [bold blue]Wandb Setup Mode[/bold blue]")
        pipeline.setup_wandb(ask_user=True)
        return
    
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
        
        # Determine wandb setting
        if args.no_wandb:
            enable_wandb = False
        else:
            enable_wandb = None  # Let pipeline decide based on configuration
        
        normalized_path = pipeline.normalized_datasets_dir / args.dataset
        
        if args.csv_only:
            pipeline.generate_csv(args.dataset, dataset['raw_path'])
        elif args.embeddings_only:
            pipeline.generate_embeddings(str(normalized_path), args.clip_model)
        elif args.train_only:
            pipeline.train_model(str(normalized_path), args.epochs, args.batch_size, enable_wandb)
        elif args.test_only:
            pipeline.test_model(str(normalized_path))
        else:
            # Full pipeline
            pipeline.run_full_pipeline(
                args.dataset, 
                dataset['raw_path'],
                args.clip_model, 
                args.epochs, 
                args.batch_size,
                enable_wandb
            )
    else:
        # Interactive mode
        pipeline.interactive_mode()


if __name__ == "__main__":
    main() 