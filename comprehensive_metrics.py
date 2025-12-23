"""
Metrics Included:
- Angular error (mean, median, std)
- Euclidean distance error
- Per-axis accuracy (X and Y)
- Percentile analysis (90th, 95th, 99th)
- Per-person performance
- Error distribution histograms
- Scatter plots (predicted vs actual)
- Heatmaps of gaze distribution
- Learning curves with confidence intervals

python comprehensive_metrics.py --checkpoint ./max_regularized_checkpoints/best_max_regularized.pth --test_data ./diagnostic_gaze_data/test_metadata.json --checkpoint_dir ./max_regularized_checkpoints --output_dir ./evaluation_results --batch_size 64
python comprehensive_metrics.py --checkpoint ./max_regularized_checkpoints/best_max_regularized.pth --test_data ./diagnostic_gaze_data/test_metadata.json
Usage:
    python comprehensive_evaluation.py \
        --checkpoint ./max_regularized_checkpoints/best_max_regularized.pth \
        --test_data ./diagnostic_gaze_data/test_metadata.json \
        --output_dir ./evaluation_results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class EvaluationDataset(Dataset):
    """Simple dataset for evaluation"""
    def __init__(self, metadata_path):
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.samples = self.metadata['samples']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        left = cv2.cvtColor(cv2.imread(s['left_eye_path']), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(s['right_eye_path']), cv2.COLOR_BGR2RGB)
        face = cv2.cvtColor(cv2.imread(s['face_path']), cv2.COLOR_BGR2RGB)
        
        left = cv2.resize(left, (224, 224))
        right = cv2.resize(right, (224, 224))
        face = cv2.resize(face, (224, 224))
        
        def to_tensor(img):
            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (img - mean) / std
        
        return {
            'left_eye': to_tensor(left),
            'right_eye': to_tensor(right),
            'face': to_tensor(face),
            'head_pose': torch.FloatTensor(s['head_pose']),
            'gaze': torch.FloatTensor(s['gaze']),
            'person_id': s['person_id'],
            'sample_id': s['sample_id']
        }


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, model, device='cuda', output_dir='./evaluation_results'):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions = []
        self.targets = []
        self.person_ids = []
        self.sample_ids = []
        
    def evaluate(self, dataloader):
        """Run evaluation and collect predictions"""
        print("\n" + "="*80)
        print("🔍 RUNNING COMPREHENSIVE EVALUATION")
        print("="*80)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                left = batch['left_eye'].to(self.device)
                right = batch['right_eye'].to(self.device)
                face = batch['face'].to(self.device)
                pose = batch['head_pose'].to(self.device)
                gaze = batch['gaze'].to(self.device)
                
                pred = self.model(left, right, face, pose)
                pred = torch.clamp(pred, 0, 1)
                
                self.predictions.append(pred.cpu().numpy())
                self.targets.append(gaze.cpu().numpy())
                self.person_ids.extend(batch['person_id'])
                self.sample_ids.extend(batch['sample_id'])
        
        self.predictions = np.vstack(self.predictions)
        self.targets = np.vstack(self.targets)
        
        print(f"✅ Evaluated {len(self.predictions)} samples")
        
    def compute_all_metrics(self):
        """Compute comprehensive metrics"""
        print("\n" + "="*80)
        print("📊 COMPUTING METRICS")
        print("="*80)
        
        metrics = {}
        
        # Angular error (degrees)
        angular_errors = self._compute_angular_errors()
        metrics['angular_error_mean'] = np.mean(angular_errors)
        metrics['angular_error_median'] = np.median(angular_errors)
        metrics['angular_error_std'] = np.std(angular_errors)
        metrics['angular_error_min'] = np.min(angular_errors)
        metrics['angular_error_max'] = np.max(angular_errors)
        
        # Percentiles
        metrics['angular_error_90th'] = np.percentile(angular_errors, 90)
        metrics['angular_error_95th'] = np.percentile(angular_errors, 95)
        metrics['angular_error_99th'] = np.percentile(angular_errors, 99)
        
        # Euclidean distance error (normalized 0-1 space)
        euclidean_errors = np.sqrt(((self.predictions - self.targets) ** 2).sum(axis=1))
        metrics['euclidean_error_mean'] = np.mean(euclidean_errors)
        metrics['euclidean_error_median'] = np.median(euclidean_errors)
        metrics['euclidean_error_std'] = np.std(euclidean_errors)
        
        # Per-axis errors
        x_errors = np.abs(self.predictions[:, 0] - self.targets[:, 0])
        y_errors = np.abs(self.predictions[:, 1] - self.targets[:, 1])
        
        metrics['x_error_mean'] = np.mean(x_errors)
        metrics['x_error_median'] = np.median(x_errors)
        metrics['y_error_mean'] = np.mean(y_errors)
        metrics['y_error_median'] = np.median(y_errors)
        
        # MSE and RMSE
        mse = np.mean((self.predictions - self.targets) ** 2)
        metrics['mse'] = mse
        metrics['rmse'] = np.sqrt(mse)
        
        # R² score per axis
        metrics['r2_x'] = self._compute_r2(self.predictions[:, 0], self.targets[:, 0])
        metrics['r2_y'] = self._compute_r2(self.predictions[:, 1], self.targets[:, 1])
        metrics['r2_overall'] = (metrics['r2_x'] + metrics['r2_y']) / 2
        
        # Accuracy at thresholds (angular error < threshold)
        for threshold in [1, 2, 3, 5, 10]:
            acc = np.mean(angular_errors < threshold) * 100
            metrics[f'accuracy_within_{threshold}deg'] = acc
        
        # Per-person statistics
        person_metrics = self._compute_per_person_metrics(angular_errors)
        metrics['per_person'] = person_metrics
        
        self.metrics = metrics
        self._print_metrics()
        
        return metrics
    
    def _compute_angular_errors(self):
        """Compute angular error in degrees"""
        # Convert normalized coords back to angles
        pred_angles = self._coords_to_angles(self.predictions)
        target_angles = self._coords_to_angles(self.targets)
        
        # Compute angular distance
        diff = np.sqrt(((pred_angles - target_angles) ** 2).sum(axis=1))
        return diff * (180 / np.pi)  # Convert to degrees
    
    def _coords_to_angles(self, coords):
        """Convert normalized screen coords to angles"""
        fov = 1.2
        x = (coords[:, 0] * 2 - 1) * np.sin(fov)
        y = (coords[:, 1] * 2 - 1) * np.sin(fov)
        theta = np.arcsin(np.clip(x, -1, 1))
        phi = np.arcsin(np.clip(y, -1, 1))
        return np.stack([theta, phi], axis=1)
    
    def _compute_r2(self, pred, target):
        """Compute R² score"""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _compute_per_person_metrics(self, angular_errors):
        """Compute metrics per person"""
        person_errors = defaultdict(list)
        
        for error, person_id in zip(angular_errors, self.person_ids):
            person_errors[person_id].append(error)
        
        person_metrics = {}
        for person_id, errors in person_errors.items():
            person_metrics[person_id] = {
                'mean': np.mean(errors),
                'median': np.median(errors),
                'std': np.std(errors),
                'count': len(errors)
            }
        
        return person_metrics
    
    def _print_metrics(self):
        """Print metrics in nice format"""
        m = self.metrics
        
        print("\n" + "="*80)
        print("📈 ACCURACY METRICS")
        print("="*80)
        
        print("\n🎯 Angular Error (degrees):")
        print(f"  Mean:     {m['angular_error_mean']:.2f}°")
        print(f"  Median:   {m['angular_error_median']:.2f}°")
        print(f"  Std Dev:  {m['angular_error_std']:.2f}°")
        print(f"  Min:      {m['angular_error_min']:.2f}°")
        print(f"  Max:      {m['angular_error_max']:.2f}°")
        
        print("\n📊 Percentiles:")
        print(f"  90th:     {m['angular_error_90th']:.2f}°")
        print(f"  95th:     {m['angular_error_95th']:.2f}°")
        print(f"  99th:     {m['angular_error_99th']:.2f}°")
        
        print("\n📏 Distance Metrics:")
        print(f"  Euclidean (mean):   {m['euclidean_error_mean']:.4f}")
        print(f"  MSE:                {m['mse']:.6f}")
        print(f"  RMSE:               {m['rmse']:.4f}")
        
        print("\n📐 Per-Axis Error:")
        print(f"  X-axis (mean):      {m['x_error_mean']:.4f}")
        print(f"  Y-axis (mean):      {m['y_error_mean']:.4f}")
        
        print("\n🎓 R² Scores:")
        print(f"  X-axis:             {m['r2_x']:.4f}")
        print(f"  Y-axis:             {m['r2_y']:.4f}")
        print(f"  Overall:            {m['r2_overall']:.4f}")
        
        print("\n✅ Accuracy Within Threshold:")
        for threshold in [1, 2, 3, 5, 10]:
            key = f'accuracy_within_{threshold}deg'
            print(f"  < {threshold:2d}°:              {m[key]:.2f}%")
        
        print("\n👥 Per-Person Statistics:")
        person_means = [p['mean'] for p in m['per_person'].values()]
        print(f"  Best person:        {min(person_means):.2f}°")
        print(f"  Worst person:       {max(person_means):.2f}°")
        print(f"  Mean across people: {np.mean(person_means):.2f}°")
        print(f"  Std across people:  {np.std(person_means):.2f}°")
        
        print("="*80)
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "="*80)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*80)
        
        angular_errors = self._compute_angular_errors()
        
        # 1. Error distribution histogram
        self._plot_error_histogram(angular_errors)
        
        # 2. Scatter plot (predicted vs actual)
        self._plot_scatter()
        
        # 3. Per-axis comparison
        self._plot_per_axis_comparison()
        
        # 4. Error heatmap
        self._plot_error_heatmap(angular_errors)
        
        # 5. Gaze distribution
        self._plot_gaze_distribution()
        
        # 6. Per-person performance
        self._plot_per_person_performance()
        
        # 7. Cumulative error distribution
        self._plot_cumulative_error(angular_errors)
        
        # 8. Box plots
        self._plot_box_plots()
        
        print(f"\n✅ All plots saved to: {self.output_dir}")
    
    def _plot_error_histogram(self, angular_errors):
        """Plot error distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(angular_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(angular_errors), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(angular_errors):.2f}°')
        ax.axvline(np.median(angular_errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(angular_errors):.2f}°')
        
        ax.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Angular Errors', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Error histogram")
    
    def _plot_scatter(self):
        """Plot predicted vs actual (scatter)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (ax, axis_name) in enumerate(zip(axes, ['X-axis', 'Y-axis'])):
            pred = self.predictions[:, idx]
            target = self.targets[:, idx]
            
            ax.scatter(target, pred, alpha=0.3, s=10, c='steelblue')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
            
            r2 = self._compute_r2(pred, target)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'Actual {axis_name}', fontsize=12)
            ax.set_ylabel(f'Predicted {axis_name}', fontsize=12)
            ax.set_title(f'{axis_name} Predictions', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Scatter plots")
    
    def _plot_per_axis_comparison(self):
        """Plot per-axis error comparison"""
        x_errors = np.abs(self.predictions[:, 0] - self.targets[:, 0])
        y_errors = np.abs(self.predictions[:, 1] - self.targets[:, 1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        positions = [1, 2]
        bp = ax.boxplot([x_errors, y_errors], positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
            patch.set_facecolor(color)
        
        ax.set_xticklabels(['X-axis Error', 'Y-axis Error'])
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Per-Axis Error Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_axis_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Per-axis comparison")
    
    def _plot_error_heatmap(self, angular_errors):
        """Plot 2D heatmap of errors"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create 2D bins
        x_bins = np.linspace(0, 1, 20)
        y_bins = np.linspace(0, 1, 20)
        
        # Compute average error in each bin
        heatmap = np.zeros((len(y_bins)-1, len(x_bins)-1))
        counts = np.zeros((len(y_bins)-1, len(x_bins)-1))
        
        for i in range(len(angular_errors)):
            x_idx = np.digitize(self.targets[i, 0], x_bins) - 1
            y_idx = np.digitize(self.targets[i, 1], y_bins) - 1
            
            if 0 <= x_idx < len(x_bins)-1 and 0 <= y_idx < len(y_bins)-1:
                heatmap[y_idx, x_idx] += angular_errors[i]
                counts[y_idx, x_idx] += 1
        
        # Average
        mask = counts > 0
        heatmap[mask] = heatmap[mask] / counts[mask]
        heatmap[~mask] = np.nan
        
        im = ax.imshow(heatmap, origin='lower', aspect='auto', cmap='YlOrRd',
                      extent=[0, 1, 0, 1])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Angular Error (degrees)', fontsize=11)
        
        ax.set_xlabel('X Coordinate (Screen)', fontsize=12)
        ax.set_ylabel('Y Coordinate (Screen)', fontsize=12)
        ax.set_title('Spatial Distribution of Errors', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Error heatmap")
    
    def _plot_gaze_distribution(self):
        """Plot distribution of gaze points"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Target distribution
        ax = axes[0]
        ax.scatter(self.targets[:, 0], self.targets[:, 1], 
                  alpha=0.3, s=5, c='blue')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Ground Truth Gaze Distribution', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Prediction distribution
        ax = axes[1]
        ax.scatter(self.predictions[:, 0], self.predictions[:, 1], 
                  alpha=0.3, s=5, c='red')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Predicted Gaze Distribution', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gaze_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Gaze distribution")
    
    def _plot_per_person_performance(self):
        """Plot per-person performance"""
        person_metrics = self.metrics['per_person']
        
        persons = sorted(person_metrics.keys())
        means = [person_metrics[p]['mean'] for p in persons]
        stds = [person_metrics[p]['std'] for p in persons]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(persons))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.axhline(np.mean(means), color='red', linestyle='--', 
                   linewidth=2, label=f'Overall Mean: {np.mean(means):.2f}°')
        
        ax.set_xlabel('Person ID', fontsize=12)
        ax.set_ylabel('Mean Angular Error (degrees)', fontsize=12)
        ax.set_title('Per-Person Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(persons, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_person_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Per-person performance")
    
    def _plot_cumulative_error(self, angular_errors):
        """Plot cumulative error distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_errors = np.sort(angular_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        
        ax.plot(sorted_errors, cumulative, linewidth=2, color='steelblue')
        
        # Add reference lines
        for threshold in [1, 2, 3, 5]:
            pct = np.mean(angular_errors < threshold) * 100
            ax.axvline(threshold, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(pct, color='gray', linestyle='--', alpha=0.5)
            ax.text(threshold, pct + 2, f'{pct:.1f}%', fontsize=9)
        
        ax.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(20, np.percentile(angular_errors, 99.5)))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cumulative_error.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Cumulative error distribution")
    
    def _plot_box_plots(self):
        """Create comprehensive box plots"""
        angular_errors = self._compute_angular_errors()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall error distribution
        ax = axes[0, 0]
        bp = ax.boxplot([angular_errors], widths=0.5, patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Angular Error (degrees)', fontsize=11)
        ax.set_title('Overall Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['All Samples'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Per-axis errors
        ax = axes[0, 1]
        x_err = np.abs(self.predictions[:, 0] - self.targets[:, 0])
        y_err = np.abs(self.predictions[:, 1] - self.targets[:, 1])
        bp = ax.boxplot([x_err, y_err], widths=0.5, patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        ax.set_ylabel('Absolute Error', fontsize=11)
        ax.set_title('Per-Axis Errors', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['X-axis', 'Y-axis'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Error by quadrant
        ax = axes[1, 0]
        quadrants = []
        labels = []
        for i, (x_range, y_range, label) in enumerate([
            ((0, 0.5), (0, 0.5), 'Bottom-Left'),
            ((0.5, 1), (0, 0.5), 'Bottom-Right'),
            ((0, 0.5), (0.5, 1), 'Top-Left'),
            ((0.5, 1), (0.5, 1), 'Top-Right')
        ]):
            mask = ((self.targets[:, 0] >= x_range[0]) & (self.targets[:, 0] < x_range[1]) &
                   (self.targets[:, 1] >= y_range[0]) & (self.targets[:, 1] < y_range[1]))
            if mask.sum() > 0:
                quadrants.append(angular_errors[mask])
                labels.append(label)
        
        bp = ax.boxplot(quadrants, widths=0.5, patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.set_ylabel('Angular Error (degrees)', fontsize=11)
        ax.set_title('Error by Screen Quadrant', fontsize=12, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
        Summary Statistics
        
        Mean Error:      {np.mean(angular_errors):.2f}°
        Median Error:    {np.median(angular_errors):.2f}°
        Std Dev:         {np.std(angular_errors):.2f}°
        
        90th Percentile: {np.percentile(angular_errors, 90):.2f}°
        95th Percentile: {np.percentile(angular_errors, 95):.2f}°
        99th Percentile: {np.percentile(angular_errors, 99):.2f}°
        
        Accuracy < 1°:   {np.mean(angular_errors < 1)*100:.2f}%
        Accuracy < 2°:   {np.mean(angular_errors < 2)*100:.2f}%
        Accuracy < 5°:   {np.mean(angular_errors < 5)*100:.2f}%
        
        R² (overall):    {self.metrics['r2_overall']:.4f}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Comprehensive box plots")

    def save_metrics_report(self):
        """Save detailed metrics to JSON and text files"""
        # JSON report
        json_path = self.output_dir / 'metrics_report.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n✅ Metrics saved to: {json_path}")

        # Text report
        txt_path = self.output_dir / 'metrics_report.txt'
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("ANGULAR ERROR METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean:              {self.metrics['angular_error_mean']:.2f}°\n")
            f.write(f"Median:            {self.metrics['angular_error_median']:.2f}°\n")
            f.write(f"Std Dev:           {self.metrics['angular_error_std']:.2f}°\n")
            f.write(f"Min:               {self.metrics['angular_error_min']:.2f}°\n")
            f.write(f"Max:               {self.metrics['angular_error_max']:.2f}°\n\n")

            f.write("PERCENTILES\n")
            f.write("-" * 80 + "\n")
            f.write(f"90th percentile:   {self.metrics['angular_error_90th']:.2f}°\n")
            f.write(f"95th percentile:   {self.metrics['angular_error_95th']:.2f}°\n")
            f.write(f"99th percentile:   {self.metrics['angular_error_99th']:.2f}°\n\n")

            f.write("DISTANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Euclidean (mean):  {self.metrics['euclidean_error_mean']:.4f}\n")
            f.write(f"MSE:               {self.metrics['mse']:.6f}\n")
            f.write(f"RMSE:              {self.metrics['rmse']:.4f}\n\n")

            f.write("PER-AXIS METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"X-axis mean error: {self.metrics['x_error_mean']:.4f}\n")
            f.write(f"X-axis median:     {self.metrics['x_error_median']:.4f}\n")
            f.write(f"Y-axis mean error: {self.metrics['y_error_mean']:.4f}\n")
            f.write(f"Y-axis median:     {self.metrics['y_error_median']:.4f}\n\n")

            f.write("R² SCORES\n")
            f.write("-" * 80 + "\n")
            f.write(f"X-axis R²:         {self.metrics['r2_x']:.4f}\n")
            f.write(f"Y-axis R²:         {self.metrics['r2_y']:.4f}\n")
            f.write(f"Overall R²:        {self.metrics['r2_overall']:.4f}\n\n")

            f.write("ACCURACY WITHIN THRESHOLDS\n")
            f.write("-" * 80 + "\n")
            for threshold in [1, 2, 3, 5, 10]:
                key = f'accuracy_within_{threshold}deg'
                f.write(f"< {threshold:2d}°:              {self.metrics[key]:.2f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("PER-PERSON STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            for person_id in sorted(self.metrics['per_person'].keys()):
                pm = self.metrics['per_person'][person_id]
                f.write(f"{person_id}:\n")
                f.write(f"  Mean:   {pm['mean']:.2f}°\n")
                f.write(f"  Median: {pm['median']:.2f}°\n")
                f.write(f"  Std:    {pm['std']:.2f}°\n")
                f.write(f"  Count:  {pm['count']}\n\n")

            f.write("=" * 80 + "\n")

        print(f"✅ Text report saved to: {txt_path}")

        # CSV report for per-person metrics
        csv_path = self.output_dir / 'per_person_metrics.csv'
        person_data = []
        for person_id, metrics in self.metrics['per_person'].items():
            person_data.append({
                'person_id': person_id,
                'mean_error': metrics['mean'],
                'median_error': metrics['median'],
                'std_error': metrics['std'],
                'sample_count': metrics['count']
            })

        df = pd.DataFrame(person_data)
        df = df.sort_values('mean_error')
        df.to_csv(csv_path, index=False)
        print(f"✅ Per-person CSV saved to: {csv_path}")

    def plot_learning_curves(self, checkpoint_dir):
        """Plot training and validation curves from checkpoints"""
        print("\n📈 Generating learning curves...")

        checkpoint_path = Path(checkpoint_dir) / 'checkpoint_best.pth'

        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])

            if not train_losses or not val_losses:
                print("⚠️ No training history found in checkpoint")
                return

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            epochs = range(1, len(train_losses) + 1)

            # Loss curves
            ax = axes[0]
            ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
            ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (MSE)', fontsize=12)
            ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Loss difference (overfitting indicator)
            ax = axes[1]
            diff = np.array(val_losses) - np.array(train_losses)
            ax.plot(epochs, diff, 'g-', linewidth=2)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
            ax.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("  ✓ Learning curves")

            # Save training summary
            summary_path = self.output_dir / 'training_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("TRAINING SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total epochs:        {len(train_losses)}\n")
                f.write(f"Best epoch:          {np.argmin(val_losses) + 1}\n")
                f.write(f"Best val loss:       {min(val_losses):.6f}\n")
                f.write(f"Final train loss:    {train_losses[-1]:.6f}\n")
                f.write(f"Final val loss:      {val_losses[-1]:.6f}\n")
                f.write(f"Final overfitting:   {val_losses[-1] - train_losses[-1]:.6f}\n")

            print(f"✅ Training summary saved to: {summary_path}")

        except Exception as e:
            print(f"❌ Error loading training history: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--checkpoint_dir', default='./max_regularized_checkpoints',help='Directory containing training checkpoints for learning curves')
    parser.add_argument('--output_dir', default='./evaluation_results')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (you'll need to import your model class)
    from anti_overfit_itracker import MaxRegularizediTrackerModel
    model = MaxRegularizediTrackerModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, device, args.output_dir)

    # Load test data
    test_dataset = EvaluationDataset(args.test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=4)

    # Run evaluation
    evaluator.evaluate(test_loader)
    evaluator.compute_all_metrics()
    evaluator.generate_all_plots()

    # Plot learning curves
    evaluator.plot_learning_curves(args.checkpoint_dir)

    evaluator.save_metrics_report()

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()