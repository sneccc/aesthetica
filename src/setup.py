#!/usr/bin/env python3
"""
Setup script for Aesthetica ML Pipeline
Installs dependencies and verifies the installation.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages from requirements file."""
    print("🔧 Installing dependencies...")
    
    requirements_file = Path(__file__).parent.parent / "pipeline_requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: pipeline_requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify that all required packages are installed."""
    print("\n🔍 Verifying installation...")
    
    # Package name mappings: (pip_name, import_name)
    required_packages = [
        ("rich", "rich"),
        ("torch", "torch"), 
        ("torchvision", "torchvision"),
        ("pandas", "pandas"), 
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("opencv-python", "cv2"),
        ("albumentations", "albumentations")
    ]
    
    missing_packages = []
    
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name} - Missing!")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All packages installed correctly!")
        return True

def test_pipeline():
    """Test the pipeline CLI."""
    print("\n🧪 Testing pipeline CLI...")
    
    pipeline_script = Path(__file__).parent.parent / "pipeline_cli.py"
    
    if not pipeline_script.exists():
        print("❌ Error: pipeline_cli.py not found!")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(pipeline_script), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Pipeline CLI is working!")
            return True
        else:
            print(f"❌ Pipeline CLI error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Pipeline CLI test timed out!")
        return False
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        return False

def main():
    """Main setup function."""
    print("🎨 Aesthetica ML Pipeline Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Install dependencies
    if not install_requirements():
        return False
    
    # Verify installation
    if not verify_installation():
        return False
    
    # Test pipeline
    if not test_pipeline():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your raw datasets in: ../datasets/raw/")
    print("2. Run: python ../pipeline_cli.py")
    print("3. Follow the interactive prompts!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 