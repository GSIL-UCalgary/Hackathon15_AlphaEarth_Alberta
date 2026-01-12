# test_wandb_setup.py
import os
import sys

# Add the current directory to path
sys.path.append('.')

from train import setup_wandb_api_key

print("Testing WandB setup...")
print(f"Current directory: {os.getcwd()}")

# Test the setup
success = setup_wandb_api_key()

if success:
    if 'WANDB_API_KEY' in os.environ:
        key = os.environ['WANDB_API_KEY']
        print(f" Success! API Key loaded (first 10 chars): {key[:10]}...")
    else:
        print(" Success! (Already logged in)")
else:
    print(" Failed to setup WandB")