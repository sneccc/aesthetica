import torch
import pathlib
import pandas as pd
import numpy as np
from train_model_mlp import MultiLayerPerceptron
import json
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout
from rich.align import Align
from rich import box
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Wandb imports (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

console = Console(force_terminal=True, width=120)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_classification_table(y_true, y_pred, class_labels):
    """Create a Rich table for classification report"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Create table
    table = Table(title="[bold]Classification Report[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Precision", justify="center", style="green")
    table.add_column("Recall", justify="center", style="yellow") 
    table.add_column("F1-Score", justify="center", style="blue")
    table.add_column("Support", justify="center", style="white")
    table.add_column("Performance", justify="center")
    
    # Add rows for each class
    for i, (class_name, prec, rec, f1_score, supp) in enumerate(zip(class_labels, precision, recall, f1, support)):
        # Performance indicator
        if f1_score >= 0.8:
            perf = "[green]Excellent[/green]"
            f1_style = "bold green"
        elif f1_score >= 0.6:
            perf = "[yellow]Good[/yellow]"
            f1_style = "bold yellow"
        elif f1_score >= 0.3:
            perf = "[orange3]Fair[/orange3]"
            f1_style = "bold orange3"
        else:
            perf = "[red]Poor[/red]"
            f1_style = "bold red"
            
        table.add_row(
            class_name,
            f"{prec:.3f}",
            f"{rec:.3f}",
            Text(f"{f1_score:.3f}", style=f1_style),
            str(int(supp)),
            perf
        )
    
    # Add summary rows
    table.add_row("", "", "", "", "", "")  # Separator
    table.add_row(
        "[bold]Overall Accuracy[/bold]",
        "", "", 
        f"[bold cyan]{accuracy:.3f}[/bold cyan]",
        str(len(y_true)),
        "[cyan]Total[/cyan]"
    )
    
    return table

def create_stats_panel(stats_dict, title, color="blue"):
    """Create a Rich panel for statistics"""
    stats_text = Text()
    for key, value in stats_dict.items():
        if isinstance(value, float):
            stats_text.append(f"{key}: ", style="bold")
            stats_text.append(f"{value:.4f}\n", style=color)
        else:
            stats_text.append(f"{key}: ", style="bold")
            stats_text.append(f"{value}\n", style=color)
    
    return Panel(
        Align.center(stats_text),
        title=f"[bold]{title}[/bold]",
        border_style=color,
        expand=False
    )

def create_predictions_table(test_data, class_labels, max_samples=8):
    """Create a Rich table for sample predictions"""
    table = Table(title="[bold]Sample Test Predictions[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Image Path", style="cyan", max_width=40)
    table.add_column("Predicted Class", style="green", justify="center")
    table.add_column("Confidence", style="yellow", justify="center")
    table.add_column("Confidence Level", justify="center")
    
    # Sample some predictions
    if len(test_data) > max_samples:
        sample_indices = np.random.choice(len(test_data), max_samples, replace=False)
        sample_data = test_data.iloc[sample_indices]
    else:
        sample_data = test_data
    
    for i, (_, row) in enumerate(sample_data.iterrows(), 1):
        pred_class = row['predicted_class']
        confidence = row['confidence_score']
        class_name = class_labels[pred_class]
        image_path = pathlib.Path(row['image_path']).name  # Just filename
        
        # Confidence level indicator
        if confidence >= 0.7:
            conf_level = "[green]High[/green]"
            conf_style = "bold green"
        elif confidence >= 0.4:
            conf_level = "[yellow]Medium[/yellow]"
            conf_style = "bold yellow"
        else:
            conf_level = "[red]Low[/red]"
            conf_style = "bold red"
        
        table.add_row(
            str(i),
            image_path,
            class_name,
            Text(f"{confidence:.3f}", style=conf_style),
            conf_level
        )
    
    return table

def create_class_distribution_table(predictions, class_labels, total_samples):
    """Create a Rich table for class distribution"""
    table = Table(title="[bold]Test Set Class Distribution[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("Class ID", style="dim", justify="center")
    table.add_column("Class Name", style="cyan")
    table.add_column("Predictions", style="green", justify="center")
    table.add_column("Percentage", style="yellow", justify="center")
    table.add_column("Bar", style="blue", justify="left", max_width=20)
    
    unique_classes, counts = np.unique(predictions, return_counts=True)
    
    # Sort by count descending
    sorted_indices = np.argsort(counts)[::-1]
    
    for idx in sorted_indices:
        class_idx = unique_classes[idx]
        count = counts[idx]
        class_name = class_labels[class_idx]
        percentage = (count / total_samples) * 100
        
        # Create a simple bar visualization using safe characters
        bar_length = int((percentage / 100) * 15)
        bar = "#" * bar_length + "-" * (15 - bar_length)
        
        table.add_row(
            str(class_idx),
            class_name,
            str(count),
            f"{percentage:.1f}%",
            bar
        )
    
    return table

def create_confusion_matrix_display(y_true, y_pred, class_labels, title="Confusion Matrix"):
    """Create a Rich display for confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_labels)
    
    # Create table with better formatting
    table = Table(title=f"[bold]{title}[/bold]", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    
    # Add header row - first column is "True/Pred", then predicted class names
    table.add_column("True \\ Pred", style="bold cyan", justify="center", min_width=12)
    for class_name in class_labels:
        # Truncate long class names for display
        display_name = class_name[:8] + "..." if len(class_name) > 8 else class_name
        table.add_column(display_name, justify="center", min_width=8)
    
    # Add data rows
    for i, class_name in enumerate(class_labels):
        row_data = [class_name[:10] + "..." if len(class_name) > 10 else class_name]
        
        for j in range(n_classes):
            value = cm[i, j]
            if i == j:  # Diagonal (correct predictions)
                if value > 0:
                    row_data.append(Text(str(value), style="bold green"))
                else:
                    row_data.append(Text("0", style="dim"))
            else:  # Off-diagonal (incorrect predictions)
                if value > 0:
                    row_data.append(Text(str(value), style="bold red"))
                else:
                    row_data.append(Text("0", style="dim"))
        
        table.add_row(*row_data)
    
    return table, cm

def create_confusion_matrix_stats(cm, class_labels):
    """Create statistics panel for confusion matrix"""
    n_classes = len(class_labels)
    total_samples = cm.sum()
    
    # Calculate per-class accuracy (recall)
    class_accuracies = []
    for i in range(n_classes):
        if cm[i].sum() > 0:
            accuracy = cm[i, i] / cm[i].sum()
            class_accuracies.append((class_labels[i], accuracy))
        else:
            class_accuracies.append((class_labels[i], 0.0))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    # Overall accuracy
    overall_accuracy = cm.diagonal().sum() / total_samples
    
    # Create stats table
    stats_table = Table(title="[bold]Per-Class Performance[/bold]", show_header=True, header_style="bold magenta")
    stats_table.add_column("Class", style="cyan")
    stats_table.add_column("Recall", justify="center", style="yellow")
    stats_table.add_column("Samples", justify="center", style="white")
    stats_table.add_column("Performance", justify="center")
    
    for class_name, recall in class_accuracies:
        class_idx = class_labels.index(class_name)
        sample_count = cm[class_idx].sum()
        
        # Performance indicator
        if recall >= 0.9:
            perf = "[green]Excellent[/green]"
            recall_style = "bold green"
        elif recall >= 0.7:
            perf = "[yellow]Good[/yellow]"
            recall_style = "bold yellow"
        elif recall >= 0.5:
            perf = "[orange3]Fair[/orange3]"
            recall_style = "bold orange3"
        else:
            perf = "[red]Poor[/red]"
            recall_style = "bold red"
        
        display_name = class_name[:15] + "..." if len(class_name) > 15 else class_name
        stats_table.add_row(
            display_name,
            Text(f"{recall:.3f}", style=recall_style),
            str(int(sample_count)),
            perf
        )
    
    # Create summary panel
    summary_stats = {
        "Overall Accuracy": f"{overall_accuracy:.4f}",
        "Total Samples": int(total_samples),
        "Classes": n_classes,
        "Best Class": f"{class_accuracies[0][0]} ({class_accuracies[0][1]:.3f})",
        "Worst Class": f"{class_accuracies[-1][0]} ({class_accuracies[-1][1]:.3f})"
    }
    
    summary_panel = create_stats_panel(summary_stats, "Confusion Matrix Summary", "magenta")
    
    return stats_table, summary_panel

def log_confusion_matrix_to_wandb(cm, class_labels, dataset_name, phase="validation"):
    """Log publication-quality confusion matrix to wandb if available and active"""
    if not WANDB_AVAILABLE or wandb is None:
        return False
        
    # Check if wandb run is active
    if not hasattr(wandb, 'run') or wandb.run is None:
        console.print("[yellow]WARNING: No active wandb run found for confusion matrix logging[/yellow]")
        return False
    
    try:
        # Create publication-quality matrices
        console.print("[cyan]Creating publication-quality confusion matrices...[/cyan]")
        
        # Raw counts matrix
        fig_counts, _ = create_publication_confusion_matrix(
            cm, class_labels, dataset_name, normalize=False
        )
        
        # Normalized matrix
        fig_normalized, _ = create_publication_confusion_matrix(
            cm, class_labels, dataset_name, normalize=True
        )
        
        # Log to wandb
        log_data = {}
        
        if fig_counts:
            log_data[f"confusion_matrix_{phase}_counts"] = wandb.Image(fig_counts)
            plt.close(fig_counts)
            
        if fig_normalized:
            log_data[f"confusion_matrix_{phase}_normalized"] = wandb.Image(fig_normalized)
            plt.close(fig_normalized)
        
        # Also log as interactive table
        log_data[f"confusion_matrix_{phase}_table"] = wandb.Table(
            columns=["True"] + class_labels,
            data=[[class_labels[i]] + cm[i].tolist() for i in range(len(class_labels))]
        )
        
        # Log all data
        wandb.log(log_data)
        
        console.print(f"[green]SUCCESS: Publication-quality confusion matrices logged to wandb ({phase})[/green]")
        return True
            
    except Exception as e:
        console.print(f"[red]ERROR: Failed to log confusion matrix to wandb: {e}[/red]")
        return False

def load_model_and_config(root_folder: str):
    """Load the trained model and its configuration"""
    root_path = pathlib.Path(root_folder)
    model_path = root_path / "model.pth"
    config_path = root_path / "model_config.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model using MLP
    model = MultiLayerPerceptron(
        input_size=config['input_size'],
        num_classes=config['num_classes'],
        hidden_units=tuple(config['hidden_units'])
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config

def predict_score(root_folder: str, clip_models: List[Tuple[str, str]] = None, enable_publication_plots: bool = True):
    """Test the model on embeddings and return detailed results"""
    root_path = pathlib.Path(root_folder)
    
    # Simple loading messages without Unicode
    console.print("[cyan]Loading model and configuration...[/cyan]")
    
    # Load model and config
    model, config = load_model_and_config(root_folder)
    console.print("[cyan]Loading embeddings and CSV data...[/cyan]")
    
    # Load embeddings and CSV data
    embeddings_path = root_path / "image_embeddings.npy"
    csv_path = root_path / "image_classifier_data.csv"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load data
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(csv_path)
    
    console.print("[cyan]Running model predictions...[/cyan]")
    
    # Convert embeddings to tensor
    embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(embeddings_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
        confidence_scores = torch.max(probabilities, dim=1)[0]
    
    # Move results to CPU for analysis
    predicted_classes = predicted_classes.cpu().numpy()
    confidence_scores = confidence_scores.cpu().numpy()
    
    console.print("[cyan]Analyzing results...[/cyan]")
    
    # Add predictions to dataframe
    df['predicted_class'] = predicted_classes
    df['confidence_score'] = confidence_scores
    
    # Separate validation data from test data
    val_data = df[df['label_id'] >= 0]  # Training/validation data
    test_data = df[df['label_id'] == -1]  # Test data
    
    # Display dataset info
    console.print()
    dataset_info = Panel(
        f"[bold cyan]Dataset:[/bold cyan] {root_path.name}\n"
        f"[bold green]Total Samples:[/bold green] {len(df):,}\n"
        f"[bold yellow]Validation Samples:[/bold yellow] {len(val_data):,}\n"
        f"[bold blue]Test Samples:[/bold blue] {len(test_data):,}\n"
        f"[bold magenta]Model Classes:[/bold magenta] {config['num_classes']}\n"
        f"[bold white]Input Dimensions:[/bold white] {config['input_size']}",
        title="[bold]Dataset Information[/bold]",
        border_style="cyan"
    )
    console.print(dataset_info)
    
    # Evaluate on validation data (if any)
    if len(val_data) > 0:
        console.print("\n" + "="*80)
        console.print(Align.center(Text("VALIDATION SET EVALUATION", style="bold magenta")))
        console.print("="*80)
        
        val_true = val_data['label_id'].values
        val_pred = val_data['predicted_class'].values
        val_confidence = val_data['confidence_score'].values
        
        # Create and display classification table
        classification_table = create_classification_table(val_true, val_pred, config['class_labels'])
        console.print(classification_table)
        
        # Create and display confusion matrix
        console.print("\n" + "-"*80)
        console.print(Align.center(Text("CONFUSION MATRIX ANALYSIS", style="bold cyan")))
        console.print("-"*80)
        
        confusion_table, cm = create_confusion_matrix_display(val_true, val_pred, config['class_labels'])
        stats_table, summary_panel = create_confusion_matrix_stats(cm, config['class_labels'])
        
        # Display confusion matrix
        console.print(confusion_table)
        console.print()
        
        # Display stats side by side
        console.print(Columns([summary_panel, stats_table], equal=True))
        
        # Save confusion matrix data
        cm_path = root_path / "confusion_matrix.csv"
        cm_df = pd.DataFrame(cm, index=config['class_labels'], columns=config['class_labels'])
        cm_df.to_csv(cm_path)
        
        # Create publication-quality confusion matrices for papers/presentations
        if enable_publication_plots:
            matrices_dir = root_path / "confusion_matrices"
            matrices = create_multiple_confusion_matrices(
                cm, config['class_labels'], root_path.name, save_dir=matrices_dir
            )
            
            cm_save_panel = Panel(
                f"[bold green]Confusion matrix files saved:[/bold green]\n"
                f"[cyan]DATA (CSV): {cm_path}[/cyan]\n"
                f"[cyan]PUBLICATION PLOTS: {matrices_dir}/[/cyan]\n"
                f"[dim]  • confusion_matrix_counts.png (raw counts)[/dim]\n"
                f"[dim]  • confusion_matrix_normalized.png (percentages)[/dim]",
                title="[bold]Matrix Export[/bold]",
                border_style="green"
            )
        else:
            cm_save_panel = Panel(
                f"[bold green]Confusion matrix data saved:[/bold green]\n[cyan]{cm_path}[/cyan]",
                title="[bold]Matrix Export[/bold]",
                border_style="green"
            )
        console.print(cm_save_panel)
        
        # Log confusion matrix to wandb
        log_confusion_matrix_to_wandb(cm, config['class_labels'], root_path.name, "validation")
        
        # Validation confidence statistics
        val_stats = {
            "Mean Confidence": val_confidence.mean(),
            "Min Confidence": val_confidence.min(),
            "Max Confidence": val_confidence.max(),
            "Std Confidence": val_confidence.std(),
            "Samples": len(val_data)
        }
        val_stats_panel = create_stats_panel(val_stats, "Validation Confidence Statistics", "green")
        console.print(val_stats_panel)
    
    # Analyze test data predictions
    if len(test_data) > 0:
        console.print("\n" + "="*80)
        console.print(Align.center(Text("TEST SET PREDICTIONS", style="bold cyan")))
        console.print("="*80)
        
        test_pred = test_data['predicted_class'].values
        test_confidence = test_data['confidence_score'].values
        
        # Class distribution table
        distribution_table = create_class_distribution_table(test_pred, config['class_labels'], len(test_data))
        console.print(distribution_table)
        
        # Test confidence statistics
        test_stats = {
            "Mean Confidence": test_confidence.mean(),
            "Min Confidence": test_confidence.min(),
            "Max Confidence": test_confidence.max(),
            "Std Confidence": test_confidence.std(),
            "Samples": len(test_data)
        }
        test_stats_panel = create_stats_panel(test_stats, "Test Confidence Statistics", "blue")
        
        # Sample predictions table
        predictions_table = create_predictions_table(test_data, config['class_labels'])
        
        # Display side by side
        console.print(Columns([test_stats_panel, predictions_table], equal=True))
    
    # Overall statistics
    console.print("\n" + "="*80)
    console.print(Align.center(Text("OVERALL STATISTICS", style="bold yellow")))
    console.print("="*80)
    
    overall_stats = {
        "Total Samples": len(df),
        "Model Input Size": config['input_size'],
        "Number of Classes": config['num_classes'],
        "Mean Confidence": confidence_scores.mean(),
        "Min Confidence": confidence_scores.min(),
        "Max Confidence": confidence_scores.max(),
        "Std Confidence": confidence_scores.std()
    }
    
    overall_panel = create_stats_panel(overall_stats, "Overall Model Statistics", "yellow")
    
    # Create class labels table
    labels_table = Table(title="[bold]Class Labels[/bold]", show_header=True, header_style="bold magenta")
    labels_table.add_column("ID", style="dim", justify="center", width=4)
    labels_table.add_column("Label", style="cyan")
    
    for i, label in enumerate(config['class_labels']):
        labels_table.add_row(str(i), label)
    
    # Display side by side
    console.print(Columns([overall_panel, labels_table], equal=True))
    
    # Save results
    results_path = root_path / "test_results.csv"
    df.to_csv(results_path, index=False)
    
    save_panel = Panel(
        f"[bold green]Results saved to:[/bold green]\n[cyan]{results_path}[/cyan]",
        title="[bold]Export Complete[/bold]",
        border_style="green"
    )
    console.print(save_panel)
    
    return predicted_classes, confidence_scores

def setup_wandb_for_testing(dataset_name, enable_wandb=False, dataset_path=None):
    """Setup wandb for test logging if requested, preferring existing runs"""
    if not enable_wandb or not WANDB_AVAILABLE:
        return False
        
    # Check if wandb run is already active (from training)
    if hasattr(wandb, 'run') and wandb.run is not None:
        console.print("[green]SUCCESS: Using existing wandb run for confusion matrix logging[/green]")
        return True
    
    # Check for saved wandb run info from training
    try:
        # Look in the dataset directory first, then current directory
        search_paths = []
        if dataset_path:
            search_paths.append(pathlib.Path(dataset_path) / "wandb_run_info.json")
        search_paths.append(pathlib.Path(".") / "wandb_run_info.json")
        
        wandb_info_path = None
        for path in search_paths:
            if path.exists():
                wandb_info_path = path
                break
                
        if wandb_info_path:
            import json
            with open(wandb_info_path, 'r') as f:
                run_info = json.load(f)
            
            # Resume the existing run
            console.print(f"[cyan]Found existing wandb run info, resuming run: {run_info['run_id']}[/cyan]")
            wandb.init(
                project=run_info['project'],
                id=run_info['run_id'],
                resume="allow",
                name=run_info.get('name', f"test-{dataset_name}"),
                tags=run_info.get('tags', []) + ["testing", "evaluation", "confusion_matrix"]
            )
            console.print("[green]SUCCESS: Resumed existing wandb run for confusion matrix logging[/green]")
            
            # Clean up the run info file after successful resume
            try:
                wandb_info_path.unlink()
                console.print(f"[dim]Cleaned up wandb run info file: {wandb_info_path}[/dim]")
            except Exception:
                pass  # Ignore cleanup errors
                
            return True
            
    except Exception as e:
        console.print(f"[yellow]WARNING: Could not resume existing wandb run: {e}[/yellow]")
        
    try:
        # Try to import wandb utilities for key management
        try:
            from src.wandb_utils import get_wandb_enabled, check_keys_file
        except ImportError:
            console.print("[yellow]WARNING: Wandb utilities not found, using basic wandb setup[/yellow]")
            
        # Initialize wandb for test logging (same project as training)
        wandb.init(
            project="aesthetica-training", 
            name=f"test-{dataset_name}",
            tags=["testing", "evaluation", "confusion_matrix"],
            reinit=True
        )
        console.print("[green]SUCCESS: Wandb initialized for confusion matrix logging[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]ERROR: Failed to setup wandb for testing: {e}[/red]")
        return False

def auto_setup_wandb_if_available(dataset_name, dataset_path=None):
    """Automatically setup wandb if an existing run is active or wandb is configured"""
    if not WANDB_AVAILABLE:
        return False
        
    # If wandb run is already active (from training), use it
    if hasattr(wandb, 'run') and wandb.run is not None:
        console.print("[green]Auto-detected active wandb run - enabling confusion matrix logging[/green]")
        return True
        
    # Check if wandb is configured (has API key) and auto-enable
    try:
        from src.wandb_utils import check_keys_file
        if check_keys_file("."):
            return setup_wandb_for_testing(dataset_name, enable_wandb=True, dataset_path=dataset_path)
    except ImportError:
        pass
        
    return False

def create_publication_confusion_matrix(cm, class_labels, dataset_name, save_path=None, normalize=False):
    """Create a publication-quality confusion matrix visualization for ML papers"""
    try:
        # Set up the matplotlib style for publication
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Normalize if requested (common in papers)
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title_suffix = " (Normalized)"
            cmap = 'Blues'
            vmax = 1.0
        else:
            cm_display = cm
            fmt = 'd'
            title_suffix = ""
            cmap = 'Blues'
            vmax = None
        
        # Create figure with appropriate size
        n_classes = len(class_labels)
        fig_size = max(8, min(16, n_classes * 0.8))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Truncate long class names for better display
        display_labels = [label[:12] + "..." if len(label) > 12 else label for label in class_labels]
        
        # Create the heatmap
        sns.heatmap(cm_display, 
                   annot=True, 
                   fmt=fmt,
                   cmap=cmap,
                   xticklabels=display_labels,
                   yticklabels=display_labels,
                   cbar_kws={'label': 'Count' if not normalize else 'Percentage'},
                   square=True,
                   linewidths=0.5,
                   ax=ax,
                   vmax=vmax)
        
        # Set labels and title
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
        
        # Calculate overall accuracy for title
        accuracy = np.trace(cm) / np.sum(cm)
        ax.set_title(f'Confusion Matrix - {dataset_name}{title_suffix}\n'
                    f'Overall Accuracy: {accuracy:.1%} | Classes: {n_classes} | Samples: {np.sum(cm):,}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            console.print(f"[green]SUCCESS: Publication-quality confusion matrix saved to: {save_path}[/green]")
        
        return fig, ax
        
    except ImportError as e:
        console.print(f"[red]ERROR: Cannot create publication matrix: {e}[/red]")
        console.print("[yellow]Install matplotlib and seaborn for publication-quality plots[/yellow]")
        return None, None
    except Exception as e:
        console.print(f"[red]ERROR: Error creating publication matrix: {e}[/red]")
        return None, None

def create_multiple_confusion_matrices(cm, class_labels, dataset_name, save_dir=None):
    """Create multiple confusion matrix variants (raw counts, normalized, etc.)"""
    matrices = {}
    
    if save_dir:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # 1. Raw counts matrix
    fig1, ax1 = create_publication_confusion_matrix(
        cm, class_labels, dataset_name, 
        save_path=save_dir / "confusion_matrix_counts.png" if save_dir else None,
        normalize=False
    )
    if fig1:
        matrices['counts'] = fig1
    
    # 2. Normalized matrix (percentages)
    fig2, ax2 = create_publication_confusion_matrix(
        cm, class_labels, dataset_name,
        save_path=save_dir / "confusion_matrix_normalized.png" if save_dir else None, 
        normalize=True
    )
    if fig2:
        matrices['normalized'] = fig2
    
    return matrices

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained model on embeddings")
    parser.add_argument("-i", "--input_directory", required=True, 
                        help="Path to normalized dataset directory containing model, embeddings, and CSV")
    parser.add_argument("--clip_model", default="ViT-SO400M-14-SigLIP-384", 
                        help="CLIP model name (for compatibility)")
    parser.add_argument("--wandb", action="store_true", 
                        help="Enable wandb logging for confusion matrix and metrics")
    parser.add_argument("--publication-plots", action="store_true", default=True,
                        help="Generate publication-quality confusion matrix plots (default: True)")
    parser.add_argument("--no-publication-plots", dest="publication_plots", action="store_false",
                        help="Disable publication-quality confusion matrix plots")
    
    args = parser.parse_args()
    
    dataset_name = pathlib.Path(args.input_directory).name
    console.print(f"\n[bold cyan]Testing model in directory:[/bold cyan] [yellow]{args.input_directory}[/yellow]")
    
    # Setup wandb automatically if available, or force if requested
    wandb_enabled = auto_setup_wandb_if_available(dataset_name, args.input_directory)
    if not wandb_enabled and args.wandb:
        wandb_enabled = setup_wandb_for_testing(dataset_name, enable_wandb=True, dataset_path=args.input_directory)
    
    try:
        predicted_classes, confidence_scores = predict_score(args.input_directory, enable_publication_plots=args.publication_plots)
        
        success_panel = Panel(
            "[bold green]Model testing completed successfully![/bold green]\n"
            f"[cyan]Processed {len(predicted_classes):,} samples[/cyan]",
            title="[bold green]Success[/bold green]",
            border_style="green"
        )
        console.print(success_panel)
        
    except Exception as e:
        error_panel = Panel(
            f"[bold red]Error during model testing:[/bold red]\n[yellow]{str(e)}[/yellow]",
            title="[bold red]Error[/bold red]",
            border_style="red"
        )
        console.print(error_panel)
        raise
