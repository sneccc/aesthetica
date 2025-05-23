#!/usr/bin/env python3
"""
Simple test script to verify wandb integration functionality.
"""

import sys
import os
import pathlib

def test_imports():
    """Test that all wandb-related imports work."""
    print("Testing imports...")
    
    try:
        from wandb_utils import check_keys_file, setup_wandb_key, get_wandb_enabled
        print("✅ wandb_utils imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import wandb_utils: {e}")
        return False
        
    try:
        import pytorch_lightning as pl
        print("✅ PyTorch Lightning imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import pytorch_lightning: {e}")
        return False
        
    try:
        from pytorch_lightning.loggers import WandbLogger
        print("✅ WandbLogger imports successfully")  
    except ImportError as e:
        print(f"❌ Failed to import WandbLogger: {e}")
        return False
        
    return True

def test_key_detection():
    """Test the key detection functionality."""
    print("\nTesting key detection...")
    
    from wandb_utils import check_keys_file
    
    # Test with parent directory (project root)
    result = check_keys_file("..")
    if result is None:
        print("✅ Correctly detected no keys.txt file")
    else:
        print(f"✅ Found existing wandb key: {result[:10]}...")
        
    return True

def test_training_script_import():
    """Test that the training script imports correctly."""
    print("\nTesting training script import...")
    
    try:
        from train_model_mlp import start_training, MultiLayerPerceptron
        print("✅ Training script imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import training script: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Wandb Integration")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_key_detection, 
        test_training_script_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Wandb integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_wandb_setup.py")
        print("2. Follow the prompts to set up your wandb API key")
        print("3. Start training with wandb enabled!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    main() 