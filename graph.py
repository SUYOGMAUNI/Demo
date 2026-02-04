"""
Plot Accuracy Curve Only - Up to Epoch 45
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
CHECKPOINT_DIR = Path('./checkpoints')
STOP_EPOCH = 45

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_DIR / 'latest_checkpoint.pth', map_location='cpu')

# Get accuracy metrics (up to epoch 45)
val_metrics = checkpoint['val_metrics']

# Check what accuracy metrics are available
acc_keys = [k for k in val_metrics.keys() if 'acc@' in k]
print(f"Available accuracy metrics: {acc_keys}")

if not acc_keys:
    print("❌ No accuracy metrics found in checkpoint!")
    print("Run training with the updated validate() method first.")
    exit()

epochs = range(1, STOP_EPOCH + 1)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for different thresholds
colors = {
    'acc@2.0deg': '#e74c3c',   # Red
    'acc@3.0deg': '#f39c12',   # Orange
    'acc@5.0deg': '#2ecc71',   # Green
    'acc@10.0deg': '#3498db'   # Blue
}

labels = {
    'acc@2.0deg': '< 2°',
    'acc@3.0deg': '< 3°',
    'acc@5.0deg': '< 5°',
    'acc@10.0deg': '< 10°'
}

# Plot each accuracy threshold
for key in acc_keys:
    if key in val_metrics:
        acc_values = val_metrics[key][:STOP_EPOCH]
        color = colors.get(key, 'gray')
        label = labels.get(key, key)
        
        # Removed marker='o' and markersize=4
        ax.plot(epochs, acc_values, linewidth=2.5, 
                label=label, color=color, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Gaze Estimation Accuracy (Epochs 1-45)', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.9, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([1, STOP_EPOCH])
ax.set_ylim([0, 105])

# Add text box with final values
if acc_keys:
    final_text = "Final Accuracy (Epoch 45):\n"
    for key in sorted(acc_keys):
        if key in val_metrics:
            final_val = val_metrics[key][STOP_EPOCH - 1]
            label = labels.get(key, key)
            final_text += f"  {label:6s}: {final_val:5.1f}%\n"
    
    ax.text(0.02, 0.98, final_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
output_file = CHECKPOINT_DIR / 'accuracy_curve_epoch45.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved accuracy plot to: {output_file}")

# Also show final values
print(f"\n📊 Final Accuracy Values (Epoch {STOP_EPOCH}):")
print("=" * 40)
for key in sorted(acc_keys):
    if key in val_metrics:
        final_val = val_metrics[key][STOP_EPOCH - 1]
        label = labels.get(key, key)
        print(f"  {label:10s}: {final_val:6.2f}%")
print("=" * 40)
