"""
Plot Loss Curves Only - Up to Epoch 45
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
CHECKPOINT_DIR = Path('./checkpoints_old')
STOP_EPOCH = 45

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_DIR / 'latest_checkpoint.pth', map_location='cpu')

# Get loss metrics (up to epoch 45)
train_loss = checkpoint['train_metrics']['loss'][:STOP_EPOCH]
val_loss = checkpoint['val_metrics']['loss'][:STOP_EPOCH]

epochs = range(1, STOP_EPOCH + 1)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training and validation loss
ax.plot(epochs, train_loss, linewidth=2.5, 
        label='Training Loss', color='#e74c3c', alpha=0.8)
ax.plot(epochs, val_loss, linewidth=2.5, 
        label='Validation Loss', color='#3498db', alpha=0.8)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Training and Validation Loss (Epochs 1-45)', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([1, STOP_EPOCH])

# Set y-axis to start from 0 for better visualization
ymin = min(min(train_loss), min(val_loss))
ymax = max(max(train_loss), max(val_loss))
ax.set_ylim([ymin * 0.9, ymax * 1.1])

# Add text box with final values
final_text = "Final Loss Values (Epoch 45):\n"
final_text += f"  Training:   {train_loss[-1]:.4f}\n"
final_text += f"  Validation: {val_loss[-1]:.4f}"

ax.text(0.02, 0.98, final_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
output_file = CHECKPOINT_DIR / 'loss_curves_epoch45.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved loss plot to: {output_file}")

# Also show final values
print(f"\n📊 Final Loss Values (Epoch {STOP_EPOCH}):")
print("=" * 40)
print(f"  Training Loss:   {train_loss[-1]:.6f}")
print(f"  Validation Loss: {val_loss[-1]:.6f}")
print("=" * 40)

# Optional: Show statistics
print(f"\n📈 Loss Statistics (Epochs 1-{STOP_EPOCH}):")
print("=" * 40)
print(f"Training Loss:")
print(f"  Min:    {min(train_loss):.6f}")
print(f"  Max:    {max(train_loss):.6f}")
print(f"  Avg:    {np.mean(train_loss):.6f}")
print(f"  Std:    {np.std(train_loss):.6f}")
print()
print(f"Validation Loss:")
print(f"  Min:    {min(val_loss):.6f}")
print(f"  Max:    {max(val_loss):.6f}")
print(f"  Avg:    {np.mean(val_loss):.6f}")
print(f"  Std:    {np.std(val_loss):.6f}")
print("=" * 40)
