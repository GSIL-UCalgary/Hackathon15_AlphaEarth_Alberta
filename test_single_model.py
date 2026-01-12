"""
Test a single trained model on test set
"""

import torch
import json
from pathlib import Path
from train import create_model, Trainer
import argparse

def test_model(model_path, data_root, sensor_name, device='cuda'):
    """Test a trained model"""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Update paths
    config['data_root'] = data_root
    config['sensor_name'] = sensor_name
    config['use_wandb'] = False
    
    # Create trainer (but don't initialize wandb)
    trainer = Trainer(config)
    
    # Load model state
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataloaders
    _, _, test_loader = trainer.create_dataloaders()
    
    # Test
    print(f"Testing model: {config['model_name']}")
    print(f"Sensor: {sensor_name}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    test_metrics = trainer.validate(test_loader, 0, mode='test')
    
    # Print results
    print("\nTest Results:")
    for key, value in test_metrics.items():
        if not key.startswith('class_'):
            print(f"  {key}: {value:.4f}")
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str,
                       default=r'D:\Hackathon15_AlphaEarth\train_val_test_patches\patches',
                       help='Root directory of dataset')
    parser.add_argument('--sensor_name', type=str, required=True,
                       choices=['landsat8', 'sentinel2', 'alphaearth'],
                       help='Sensor to test on')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    test_metrics = test_model(
        args.model_path,
        args.data_root,
        args.sensor_name,
        args.device
    )
    
    # Save results
    output_path = Path(args.model_path).parent / 'test_results.json'
    with open(output_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()