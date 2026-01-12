"""
Run hyperparameter sweeps with Weights & Biases
"""

import wandb
import yaml
from train import Trainer
import argparse

def sweep_train():
    """Train function for wandb sweep"""
    
    # Initialize wandb
    wandb.init()
    
    # Get config from wandb
    config = wandb.config
    
    # Update config with wandb parameters
    trainer_config = {
        'model_name': config.model_name,
        'sensor_name': config.sensor_name,
        'data_root': r'D:\Hackathon15_AlphaEarth\train_val_test_patches\patches',
        'dataset_config': r'D:\Hackathon15_AlphaEarth\train_val_test_patches\multisensor_dataset_config.json',
        'output_dir': './experiments',
        'epochs': 100,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'loss_fn': config.loss_fn,
        'optimizer': config.optimizer,
        'scheduler': 'reduce_on_plateau',
        'ignore_index': -99,
        'use_amp': True,
        'num_workers': 4,
        'save_every': 10,
        'use_wandb': True,
        'wandb_project': 'multisensor-segmentation-sweeps',
    }
    
    # Train
    trainer = Trainer(trainer_config)
    test_metrics = trainer.train()
    
    # Log test metrics
    wandb.log({
        'test/miou': test_metrics['mean_iou'],
        'test/mean_f1': test_metrics['mean_f1'],
        'test/accuracy': test_metrics['overall_accuracy']
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_config', type=str, default='config.yaml',
                       help='Path to sweep configuration file')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of sweep runs')
    args = parser.parse_args()
    
    # Load sweep config
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project='multisensor-segmentation-sweeps')
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train, count=args.count)

if __name__ == '__main__':
    main()