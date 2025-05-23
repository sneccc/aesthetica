import os
import pathlib
from rich.console import Console
from rich.prompt import Prompt, Confirm
import wandb

console = Console()

def check_keys_file(root_folder="."):
    """Check if keys.txt exists in the root folder and return the wandb key if found."""
    keys_path = pathlib.Path(root_folder) / "keys.txt"
    
    if keys_path.exists():
        try:
            with open(keys_path, 'r') as f:
                content = f.read().strip()
                
            # Look for WANDB_KEY= line
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('WANDB_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key:
                        console.print(f">> Found wandb key in {keys_path}", style="green")
                        return key
                        
            console.print(f"WARNING: Found {keys_path} but no WANDB_KEY= entry", style="yellow")
            return None
            
        except Exception as e:
            console.print(f"ERROR: Error reading {keys_path}: {e}", style="red")
            return None
    
    console.print(f"INFO: No {keys_path} found", style="blue")
    return None

def setup_wandb_key(root_folder=".", ask_user=True):
    """Setup wandb key either from keys.txt or by asking the user."""
    
    # First, check if key exists in keys.txt
    existing_key = check_keys_file(root_folder)
    if existing_key:
        os.environ['WANDB_API_KEY'] = existing_key
        return existing_key
    
    # Check if key is already in environment
    env_key = os.environ.get('WANDB_API_KEY')
    if env_key:
        console.print(">> Found WANDB_API_KEY in environment variables", style="green")
        return env_key
    
    # If we're not asking the user, return None
    if not ask_user:
        return None
    
    # Ask user if they want to set up wandb logging
    console.print("\n>> [bold]Weights & Biases (wandb) Integration[/bold]")
    console.print("Wandb provides experiment tracking, metrics visualization, and model management.")
    
    use_wandb = Confirm.ask("Would you like to enable wandb logging?", default=True)
    
    if not use_wandb:
        console.print(">> Skipping wandb setup", style="yellow")
        return None
    
    # Ask for wandb key
    console.print("\n>> To use wandb, you need an API key:")
    console.print("   1. Go to https://wandb.ai/authorize")
    console.print("   2. Copy your API key")
    console.print("   3. Paste it below")
    
    wandb_key = Prompt.ask("\nEnter your wandb API key", password=True)
    
    if not wandb_key or not wandb_key.strip():
        console.print("ERROR: No key provided, skipping wandb setup", style="red")
        return None
    
    wandb_key = wandb_key.strip()
    
    # Save key to keys.txt
    try:
        keys_path = pathlib.Path(root_folder) / "keys.txt"
        
        # Read existing content if file exists
        existing_content = ""
        if keys_path.exists():
            with open(keys_path, 'r') as f:
                existing_content = f.read()
        
        # Check if WANDB_KEY already exists in file
        lines = existing_content.split('\n')
        wandb_key_exists = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('WANDB_KEY='):
                lines[i] = f'WANDB_KEY={wandb_key}'
                wandb_key_exists = True
                break
        
        # If WANDB_KEY doesn't exist, add it
        if not wandb_key_exists:
            if existing_content and not existing_content.endswith('\n'):
                existing_content += '\n'
            lines.append(f'WANDB_KEY={wandb_key}')
        
        # Write the file
        with open(keys_path, 'w') as f:
            f.write('\n'.join(lines))
        
        console.print(f">> Saved wandb key to {keys_path}", style="green")
        
    except Exception as e:
        console.print(f"WARNING: Failed to save key to file: {e}", style="yellow")
        console.print("Key will be used for this session only", style="yellow")
    
    # Set environment variable
    os.environ['WANDB_API_KEY'] = wandb_key
    return wandb_key

def initialize_wandb(project_name="aesthetica-training", run_name=None, config=None, enabled=True):
    """Initialize wandb run with given parameters."""
    
    if not enabled:
        console.print(">> Wandb logging disabled", style="yellow")
        return None
    
    # Check if wandb key is available
    if not os.environ.get('WANDB_API_KEY'):
        console.print("ERROR: No wandb API key found. Run setup_wandb_key() first.", style="red")
        return None
    
    try:
        # Try to login to wandb
        wandb.login()
        
        # Initialize the run
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True
        )
        
        console.print(f">> Initialized wandb run: {run.name}", style="green")
        console.print(f">> View at: {run.url}", style="blue")
        
        return run
        
    except Exception as e:
        console.print(f"ERROR: Failed to initialize wandb: {e}", style="red")
        return None

def get_wandb_enabled():
    """Check if wandb should be enabled based on available API key."""
    return bool(os.environ.get('WANDB_API_KEY')) 