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
import time

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

def predict_score(root_folder: str, clip_models: List[Tuple[str, str]] = None):
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained model on embeddings")
    parser.add_argument("-i", "--input_directory", required=True, 
                        help="Path to normalized dataset directory containing model, embeddings, and CSV")
    parser.add_argument("--clip_model", default="ViT-SO400M-14-SigLIP-384", 
                        help="CLIP model name (for compatibility)")
    
    args = parser.parse_args()
    
    console.print(f"\n[bold cyan]Testing model in directory:[/bold cyan] [yellow]{args.input_directory}[/yellow]")
    
    try:
        predicted_classes, confidence_scores = predict_score(args.input_directory)
        
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
