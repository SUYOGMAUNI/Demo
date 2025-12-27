"""
Professional MPIIGaze Training System
======================================
Optimized gaze estimation specifically for MPIIGaze dataset with proper
angular metrics, realistic augmentations, and production-ready features.

Key Improvements:
1. Proper handling of spherical coordinates (pitch, yaw in radians)
2. Correct angular error computation for gaze vectors
3. MPIIGaze-appropriate augmentation intensity
4. Combined MSE + Angular loss
5. Comprehensive evaluation metrics
6. Easy configuration and extensibility

Dataset Format:
- Gaze: [pitch, yaw] in radians
- Head pose: [pitch, yaw] in radians
- Images: RGB, will be resized to 224×224

Expected Performance:
- Mean Angular Error: 4.5-5.5° (state-of-the-art: ~4.3°)
- Training time: 2-3 hours on single GPU

Usage:
    python mpiigaze_training.py --train_data ./MPIIGaze/train_metadata.json \
                                --val_data ./MPIIGaze/val_metadata.json \
                                --checkpoint_dir ./mpiigaze_checkpoints \
                                --epochs 100 --batch_size 64
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from tqdm import tqdm
import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MPIIGaze')


@dataclass
class MPIIGazeAugmentationConfig:
    """Augmentation configuration optimized for MPIIGaze (lab conditions)"""

    # Color augmentations (gentle for controlled lab environment)
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.95, 1.05)
    saturation_range: Tuple[float, float] = (0.95, 1.05)
    hue_shift_range: float = 0.02  # Small hue variation

    # Geometric augmentations (small for head-mounted camera data)
    rotation_range: Tuple[float, float] = (-3, 3)  # ±3 degrees
    translation_range: Tuple[float, float] = (-0.03, 0.03)  # ±3%
    scale_range: Tuple[float, float] = (0.95, 1.05)  # ±5%

    # Image quality (mild - lab cameras are decent quality)
    blur_prob: float = 0.2
    blur_kernel_range: Tuple[int, int] = (1, 3)
    noise_prob: float = 0.2
    noise_std_range: Tuple[float, float] = (0.5, 1.5)

    # Regularization augmentations
    cutout_prob: float = 0.2
    cutout_size_range: Tuple[float, float] = (0.1, 0.2)  # 10-20% of image

    # Advanced augmentations
    mixup_alpha: float = 0.2  # Gentle mixup
    cutmix_alpha: float = 0.5  # Gentle cutmix
    horizontal_flip_prob: float = 0.5

    # MPIIGaze specific
    temporal_jitter_prob: float = 0.0  # Not used for static images

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'rotation_range': self.rotation_range,
            'blur_prob': self.blur_prob,
            'noise_prob': self.noise_prob,
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha,
        }


class MPIIGazeAugmentor:
    """Augmentation pipeline for MPIIGaze dataset"""

    def __init__(self, config: MPIIGazeAugmentationConfig):
        self.config = config

    def apply_color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color augmentations"""
        # Brightness
        if random.random() < 0.5:
            alpha = random.uniform(*self.config.brightness_range)
            image = np.clip(image * alpha, 0, 255)

        # Contrast
        if random.random() < 0.5:
            alpha = random.uniform(*self.config.contrast_range)
            mean = image.mean()
            image = np.clip((image - mean) * alpha + mean, 0, 255)

        # Saturation
        if random.random() < 0.3:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            alpha = random.uniform(*self.config.saturation_range)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        # Hue shift
        if random.random() < 0.2:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            shift = random.uniform(-self.config.hue_shift_range, self.config.hue_shift_range)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift * 180) % 180
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        return image

    def apply_geometric(self, image: np.ndarray) -> np.ndarray:
        """Apply geometric transformations (gaze-preserving for small transforms)"""
        h, w = image.shape[:2]

        # Small rotation
        if random.random() < 0.3:
            angle = random.uniform(*self.config.rotation_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

        # Small translation
        if random.random() < 0.2:
            tx = random.uniform(*self.config.translation_range) * w
            ty = random.uniform(*self.config.translation_range) * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Small scaling
        if random.random() < 0.2:
            scale = random.uniform(*self.config.scale_range)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))

            # Crop or pad to original size
            if scale > 1:
                # Center crop
                start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
                image = image[start_y:start_y + h, start_x:start_x + w]
            else:
                # Pad
                pad_y, pad_x = (h - new_h) // 2, (w - new_w) // 2
                image = cv2.copyMakeBorder(
                    image, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                    cv2.BORDER_REFLECT
                )

        return image

    def apply_quality_degradation(self, image: np.ndarray) -> np.ndarray:
        """Apply mild quality degradations"""
        # Gaussian blur
        if random.random() < self.config.blur_prob:
            k = random.choice(self.config.blur_kernel_range)
            if k > 0:
                image = cv2.GaussianBlur(image, (k * 2 + 1, k * 2 + 1), 0)

        # Gaussian noise
        if random.random() < self.config.noise_prob:
            std = random.uniform(*self.config.noise_std_range)
            noise = np.random.normal(0, std, image.shape)
            image = np.clip(image + noise, 0, 255)

        return image

    def apply_cutout(self, image: np.ndarray) -> np.ndarray:
        """Apply cutout augmentation"""
        if random.random() < self.config.cutout_prob:
            h, w = image.shape[:2]

            # 1-2 cutouts
            n_cutouts = random.randint(1, 2)
            for _ in range(n_cutouts):
                size = int(random.uniform(*self.config.cutout_size_range) * min(h, w))
                x = random.randint(0, max(1, w - size))
                y = random.randint(0, max(1, h - size))

                # Fill with mean color
                mean_color = np.mean(image, axis=(0, 1))
                image[y:y + size, x:x + size] = mean_color

        return image

    def apply_horizontal_flip(self, left: np.ndarray, right: np.ndarray,
                              face: np.ndarray, gaze: np.ndarray,
                              head_pose: np.ndarray) -> Tuple:
        """
        Apply horizontal flip with CORRECT gaze transformation for radians.

        MPIIGaze gaze format: [pitch, yaw] in radians
        - pitch: vertical angle (up/down) - stays same on horizontal flip
        - yaw: horizontal angle (left/right) - negated on horizontal flip
        """
        # Swap eyes and flip face
        left_new = np.fliplr(right).copy()
        right_new = np.fliplr(left).copy()
        face_new = np.fliplr(face).copy()

        # Transform gaze: negate yaw (horizontal component)
        gaze_new = gaze.copy()
        gaze_new[1] = -gaze_new[1]  # Negate yaw
        # pitch (gaze_new[0]) stays the same

        # Transform head pose: negate yaw
        head_pose_new = head_pose.copy()
        head_pose_new[1] = -head_pose_new[1]  # Negate head yaw

        return left_new, right_new, face_new, gaze_new, head_pose_new

    def __call__(self, left: np.ndarray, right: np.ndarray,
                 face: np.ndarray, gaze: np.ndarray,
                 head_pose: np.ndarray) -> Tuple:
        """Apply all augmentations"""
        # Copy to avoid modifying originals
        left = left.copy().astype(np.float32)
        right = right.copy().astype(np.float32)
        face = face.copy().astype(np.float32)
        gaze = gaze.copy()
        head_pose = head_pose.copy()

        # Horizontal flip (with correct gaze/pose transformation)
        if random.random() < self.config.horizontal_flip_prob:
            left, right, face, gaze, head_pose = self.apply_horizontal_flip(
                left, right, face, gaze, head_pose
            )

        # Apply augmentations to each image independently
        left = self.apply_color_jitter(left)
        left = self.apply_geometric(left)
        left = self.apply_quality_degradation(left)
        left = self.apply_cutout(left)

        right = self.apply_color_jitter(right)
        right = self.apply_geometric(right)
        right = self.apply_quality_degradation(right)
        right = self.apply_cutout(right)

        face = self.apply_color_jitter(face)
        face = self.apply_geometric(face)
        face = self.apply_quality_degradation(face)
        face = self.apply_cutout(face)

        # Clamp to valid range
        left = np.clip(left, 0, 255).astype(np.uint8)
        right = np.clip(right, 0, 255).astype(np.uint8)
        face = np.clip(face, 0, 255).astype(np.uint8)

        return left, right, face, gaze, head_pose


class MPIIGazeDataset(Dataset):
    """Dataset for MPIIGaze with proper augmentation"""

    def __init__(self, metadata_path: str, augment: bool = False,
                 config: Optional[MPIIGazeAugmentationConfig] = None,
                 image_size: int = 224):
        super().__init__()

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.samples = metadata['samples']
        self.augment = augment
        self.image_size = image_size

        # Setup augmenter
        if config is None:
            config = MPIIGazeAugmentationConfig()
        self.augmenter = MPIIGazeAugmentor(config)

        # Normalization (ImageNet stats)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        logger.info(f"Loaded {len(self.samples)} samples, augment={augment}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load and convert image to RGB"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor"""
        # Convert to [0, 1] range and change to CHW format
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        return self.normalize(img_tensor)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load images
        left = self._load_image(sample['left_eye_path'])
        right = self._load_image(sample['right_eye_path'])
        face = self._load_image(sample['face_path'])

        # Load labels (in radians)
        gaze = np.array(sample['gaze'], dtype=np.float32)  # [pitch, yaw]
        head_pose = np.array(sample['head_pose'], dtype=np.float32)  # [pitch, yaw]

        # Resize to standard size
        left = cv2.resize(left, (self.image_size, self.image_size))
        right = cv2.resize(right, (self.image_size, self.image_size))
        face = cv2.resize(face, (self.image_size, self.image_size))

        # Apply augmentations
        if self.augment:
            left, right, face, gaze, head_pose = self.augmenter(
                left, right, face, gaze, head_pose
            )

        # Convert to tensors
        return {
            'left_eye': self._to_tensor(left),
            'right_eye': self._to_tensor(right),
            'face': self._to_tensor(face),
            'head_pose': torch.FloatTensor(head_pose),
            'gaze': torch.FloatTensor(gaze),
            'sample_id': sample.get('id', str(idx))
        }


class StochasticDepth(nn.Module):
    """Stochastic depth (DropPath) for regularization"""

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)

        if keep_prob > 0:
            mask.div_(keep_prob)

        return x * mask


class MPIIGazeModel(nn.Module):
    """
    Enhanced iTracker-style model for MPIIGaze.

    Key improvements:
    - Proper weight initialization
    - Batch normalization for stability
    - Moderate dropout (0.4) for MPIIGaze scale
    - NO output activation (radians are unbounded)
    - Stochastic depth for regularization
    """

    def __init__(self, dropout_rate: float = 0.4, use_stochastic_depth: bool = True):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.use_stochastic_depth = use_stochastic_depth

        # Eye feature extractor (shared for both eyes)
        self.eye_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3)
        )

        # Face feature extractor
        self.face_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3)
        )

        # Head pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Eye feature projection
        eye_features = 256 * 4 * 4
        self.eye_projection = nn.Sequential(
            nn.Linear(eye_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Face feature projection
        face_features = 512 * 4 * 4
        self.face_projection = nn.Sequential(
            nn.Linear(face_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Gaze prediction head
        total_features = 128 * 2 + 256 + 64  # left + right + face + pose
        self.gaze_head = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 2)  # [pitch, yaw] in radians - NO activation!
        )

        # Stochastic depth layers
        if use_stochastic_depth:
            self.eye_stochastic = StochasticDepth(drop_prob=0.1)
            self.face_stochastic = StochasticDepth(drop_prob=0.1)
        else:
            self.eye_stochastic = nn.Identity()
            self.face_stochastic = nn.Identity()

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Model: dropout={dropout_rate}, stoch_depth={use_stochastic_depth}")

    def _initialize_weights(self):
        """Kaiming initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor,
                face: torch.Tensor, head_pose: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            gaze: [batch_size, 2] tensor with [pitch, yaw] in radians
        """
        # Extract eye features (shared weights)
        left_feat = self.eye_extractor(left_eye)
        right_feat = self.eye_extractor(right_eye)

        # Apply stochastic depth
        left_feat = self.eye_stochastic(left_feat)
        right_feat = self.eye_stochastic(right_feat)

        # Project eye features
        left_proj = self.eye_projection(left_feat.view(left_feat.size(0), -1))
        right_proj = self.eye_projection(right_feat.view(right_feat.size(0), -1))

        # Extract and project face features
        face_feat = self.face_extractor(face)
        face_feat = self.face_stochastic(face_feat)
        face_proj = self.face_projection(face_feat.view(face_feat.size(0), -1))

        # Encode head pose
        pose_proj = self.pose_encoder(head_pose)

        # Concatenate all features
        combined = torch.cat([left_proj, right_proj, face_proj, pose_proj], dim=1)

        # Predict gaze (in radians)
        gaze = self.gaze_head(combined)

        return gaze


class MixupCutmix:
    """Mixup and CutMix augmentations"""

    @staticmethod
    def mixup(images: torch.Tensor, labels: torch.Tensor,
              alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation"""
        if alpha <= 0:
            return images, labels, 1.0

        lam = np.random.beta(alpha, alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels, lam

    @staticmethod
    def cutmix(images: torch.Tensor, labels: torch.Tensor,
               alpha: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply cutmix augmentation"""
        if alpha <= 0:
            return images, labels, 1.0

        lam = np.random.beta(alpha, alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        # Random bounding box
        h, w = images.size(2), images.size(3)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels, lam


class MPIIGazeTrainer:
    """Professional trainer for MPIIGaze dataset"""

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model.to(self.device)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.angular_weight = config.get('angular_weight', 0.5)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 5e-4),
            betas=config.get('betas', (0.9, 0.999))
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_angular = float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Augmentation settings
        self.use_mixup = config.get('use_mixup', True)
        self.use_cutmix = config.get('use_cutmix', True)
        self.mixup_alpha = config.get('mixup_alpha', 0.2)
        self.cutmix_alpha = config.get('cutmix_alpha', 0.5)

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}")
        logger.info(f"Angular loss weight: {self.angular_weight}")

    def _create_scheduler(self):
        """Create cosine annealing scheduler with warmup"""
        total_epochs = self.config.get('epochs', 100)
        warmup_epochs = self.config.get('warmup_epochs', 5)
        min_lr_ratio = self.config.get('min_lr_ratio', 0.001)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine = 0.5 * (1 + np.cos(np.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_angular_error_degrees(self, pred: torch.Tensor,
                                      target: torch.Tensor) -> torch.Tensor:
        def spherical_to_cartesian(angles):
            """Convert (pitch, yaw) to 3D unit vector"""
            pitch = angles[:, 0]
            yaw = angles[:, 1]

            x = -torch.cos(pitch) * torch.sin(yaw)
            y = -torch.sin(pitch)
            z = -torch.cos(pitch) * torch.cos(yaw)

            return torch.stack([x, y, z], dim=1)

        pred_vec = spherical_to_cartesian(pred)
        target_vec = spherical_to_cartesian(target)

        # Cosine similarity
        cos_sim = torch.sum(pred_vec * target_vec, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Angular error in degrees
        angular_error_degrees = torch.acos(cos_sim) * 180.0 / np.pi

        return angular_error_degrees

    def compute_angular_loss(self, pred: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """
        Compute angular loss in RADIANS (for backpropagation).

        This is the CORRECT way - compute directly in radians without
        degree conversion confusion.

        Args:
            pred: [batch_size, 2] predicted [pitch, yaw] in radians
            target: [batch_size, 2] ground truth [pitch, yaw] in radians

        Returns:
            Mean angular error in RADIANS (scalar for loss)
        """

        def spherical_to_cartesian(angles):
            pitch = angles[:, 0]
            yaw = angles[:, 1]

            x = -torch.cos(pitch) * torch.sin(yaw)
            y = -torch.sin(pitch)
            z = -torch.cos(pitch) * torch.cos(yaw)

            return torch.stack([x, y, z], dim=1)

        pred_vec = spherical_to_cartesian(pred)
        target_vec = spherical_to_cartesian(target)

        # Cosine similarity
        cos_sim = torch.sum(pred_vec * target_vec, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Angular error in RADIANS (for loss computation)
        angular_error_radians = torch.acos(cos_sim)

        return angular_error_radians.mean()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation support"""
        self.model.train()
        metrics = defaultdict(float)

        # Gradient accumulation
        accum_steps = self.config.get('accumulation_steps', 1)

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            left_eye = batch['left_eye'].to(self.device)
            right_eye = batch['right_eye'].to(self.device)
            face = batch['face'].to(self.device)
            head_pose = batch['head_pose'].to(self.device)
            gaze_target = batch['gaze'].to(self.device)

            # Apply mixup or cutmix CONSISTENTLY across all images
            # FIX: Use same lambda for all images!
            if self.use_mixup and random.random() < 0.5:
                # Get lambda once
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                batch_size = left_eye.size(0)
                index = torch.randperm(batch_size, device=self.device)

                # Apply same mixing to all images and labels
                left_eye = lam * left_eye + (1 - lam) * left_eye[index]
                right_eye = lam * right_eye + (1 - lam) * right_eye[index]
                face = lam * face + (1 - lam) * face[index]
                gaze_target = lam * gaze_target + (1 - lam) * gaze_target[index]
                head_pose = lam * head_pose + (1 - lam) * head_pose[index]

            elif self.use_cutmix and random.random() < 0.5:
                # Get lambda and box once
                lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                batch_size = left_eye.size(0)
                index = torch.randperm(batch_size, device=self.device)

                # Get consistent bounding box
                h, w = left_eye.size(2), left_eye.size(3)
                cut_ratio = np.sqrt(1.0 - lam)
                cut_h = int(h * cut_ratio)
                cut_w = int(w * cut_ratio)
                cx = np.random.randint(w)
                cy = np.random.randint(h)
                x1 = np.clip(cx - cut_w // 2, 0, w)
                x2 = np.clip(cx + cut_w // 2, 0, w)
                y1 = np.clip(cy - cut_h // 2, 0, h)
                y2 = np.clip(cy + cut_h // 2, 0, h)

                # Apply same cutmix to all images
                left_eye = left_eye.clone()
                right_eye = right_eye.clone()
                face = face.clone()
                left_eye[:, :, y1:y2, x1:x2] = left_eye[index, :, y1:y2, x1:x2]
                right_eye[:, :, y1:y2, x1:x2] = right_eye[index, :, y1:y2, x1:x2]
                face[:, :, y1:y2, x1:x2] = face[index, :, y1:y2, x1:x2]

                # Adjust lambda and mix labels
                lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
                gaze_target = lam * gaze_target + (1 - lam) * gaze_target[index]
                head_pose = lam * head_pose + (1 - lam) * head_pose[index]

            # Forward pass
            gaze_pred = self.model(left_eye, right_eye, face, head_pose)

            # Compute losses
            mse_loss = self.mse_loss(gaze_pred, gaze_target)

            # FIX: Compute angular loss directly in radians (no conversion!)
            angular_loss = self.compute_angular_loss(gaze_pred, gaze_target)

            # Combined loss
            total_loss = mse_loss + self.angular_weight * angular_loss

            # Scale loss for gradient accumulation
            total_loss = total_loss / accum_steps

            # Backward pass
            total_loss.backward()

            # Update weights only after accumulation steps
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track metrics (unscaled)
            with torch.no_grad():
                # For reporting, compute angular error in degrees
                angular_error_deg = self.compute_angular_error_degrees(
                    gaze_pred, gaze_target
                ).mean().item()

                metrics['loss'] += (total_loss.item() * accum_steps)
                metrics['mse_loss'] += mse_loss.item()
                metrics['angular_error'] += angular_error_deg
                metrics['batch_count'] += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item() * accum_steps:.4f}",
                'angle': f"{angular_error_deg:.2f}°"
            })

        # Clear any remaining gradients
        self.optimizer.zero_grad()

        # Average metrics
        for key in ['loss', 'mse_loss', 'angular_error']:
            metrics[key] /= metrics['batch_count']

        return dict(metrics)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        FIX: Only compute metrics we need, don't compute loss for training
        (we just need predictions for angular error).
        """
        self.model.eval()
        metrics = defaultdict(float)
        all_angular_errors = []
        all_mse_errors = []

        progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in progress_bar:
            # Move to device
            left_eye = batch['left_eye'].to(self.device)
            right_eye = batch['right_eye'].to(self.device)
            face = batch['face'].to(self.device)
            head_pose = batch['head_pose'].to(self.device)
            gaze_target = batch['gaze'].to(self.device)

            # Forward pass
            gaze_pred = self.model(left_eye, right_eye, face, head_pose)

            # Compute angular error in degrees (for reporting)
            angular_errors = self.compute_angular_error_degrees(gaze_pred, gaze_target)

            # Compute MSE (for monitoring)
            mse = torch.mean((gaze_pred - gaze_target) ** 2, dim=1)

            # Store for statistics
            all_angular_errors.extend(angular_errors.cpu().numpy().tolist())
            all_mse_errors.extend(mse.cpu().numpy().tolist())

            # Track batch metrics
            metrics['angular_error'] += angular_errors.mean().item()
            metrics['mse_loss'] += mse.mean().item()
            metrics['batch_count'] += 1

            # Update progress bar
            progress_bar.set_postfix({
                'angle': f"{angular_errors.mean().item():.2f}°"
            })

        # Average batch metrics
        metrics['angular_error'] /= metrics['batch_count']
        metrics['mse_loss'] /= metrics['batch_count']

        # Compute combined loss for tracking (not for training)
        # This is just for logging purposes
        metrics['loss'] = metrics['mse_loss'] + self.angular_weight * (
                metrics['angular_error'] * np.pi / 180.0
        )

        # Additional statistics
        all_angular_errors = np.array(all_angular_errors)
        metrics['angular_median'] = np.median(all_angular_errors)
        metrics['angular_p95'] = np.percentile(all_angular_errors, 95)
        metrics['angular_std'] = np.std(all_angular_errors)
        metrics['angular_min'] = np.min(all_angular_errors)
        metrics['angular_max'] = np.max(all_angular_errors)

        return dict(metrics)

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_angular': self.best_val_angular,
            'patience_counter': self.patience_counter,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'config': self.config
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

            # Also save just model weights
            weights_path = self.checkpoint_dir / 'best_model_weights.pth'
            torch.save(self.model.state_dict(), weights_path)

            logger.info(f"✅ Saved best model: {self.best_val_angular:.3f}°")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load training checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_angular = checkpoint['best_val_angular']
            self.patience_counter = checkpoint['patience_counter']
            self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
            self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))

            logger.info(f"✅ Resumed from epoch {checkpoint['epoch']}")
            logger.info(f"   Best angular error: {self.best_val_angular:.3f}°")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def plot_training_curves(self):
        """Plot and save training curves (with robust LR tracking)"""
        if len(self.train_metrics['loss']) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.train_metrics['loss'], label='Train', color='blue', linewidth=2)
        ax.plot(self.val_metrics['loss'], label='Val', color='red', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Total Loss', fontsize=11)
        ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Angular error curves
        ax = axes[0, 1]
        ax.plot(self.train_metrics['angular_error'], label='Train',
                color='blue', linewidth=2)
        ax.plot(self.val_metrics['angular_error'], label='Val',
                color='red', linewidth=2)
        ax.axhline(y=self.best_val_angular, color='green',
                   linestyle='--', label=f'Best: {self.best_val_angular:.2f}°')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Angular Error (degrees)', fontsize=11)
        ax.set_title('Angular Error Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Overfitting indicator
        ax = axes[1, 0]
        if len(self.train_metrics['loss']) == len(self.val_metrics['loss']):
            diff = [v - t for t, v in zip(self.train_metrics['loss'],
                                          self.val_metrics['loss'])]
            ax.plot(diff, color='purple', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.fill_between(range(len(diff)), 0, diff,
                            where=[d > 0 for d in diff],
                            color='red', alpha=0.3, label='Overfitting')
            ax.fill_between(range(len(diff)), 0, diff,
                            where=[d <= 0 for d in diff],
                            color='green', alpha=0.3, label='Good')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Val Loss - Train Loss', fontsize=11)
            ax.set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Learning rate schedule (FIXED: robust method)
        ax = axes[1, 1]
        try:
            # Method 1: Try to get from optimizer history (if available)
            if hasattr(self, '_lr_history'):
                lrs = self._lr_history
            else:
                # Method 2: Reconstruct from scheduler lambda
                if hasattr(self.scheduler, 'lr_lambdas'):
                    base_lr = self.config.get('learning_rate', 1e-3)
                    lrs = [base_lr * self.scheduler.lr_lambdas[0](i)
                           for i in range(len(self.train_metrics['loss']))]
                else:
                    # Method 3: Just show current LR as constant (fallback)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    lrs = [current_lr] * len(self.train_metrics['loss'])

            ax.plot(lrs, color='orange', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Learning Rate', fontsize=11)
            ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.warning(f"Could not plot LR schedule: {e}")
            ax.text(0.5, 0.5, 'LR plot unavailable',
                    ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved training curves to {plot_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 20)
        early_stop = self.config.get('early_stop', True)

        # Initialize LR history for plotting
        self._lr_history = []

        logger.info(f"\n{'=' * 70}")
        logger.info("MPIIGaze Training Started")
        logger.info(f"{'=' * 70}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {epochs}, Patience: {patience}")
        logger.info(f"Learning rate: {self.config.get('learning_rate')}")
        logger.info(f"Weight decay: {self.config.get('weight_decay')}")
        logger.info(f"Gradient accumulation: {self.config.get('accumulation_steps', 1)}")
        logger.info(f"{'=' * 70}\n")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Store current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            self._lr_history.append(current_lr)

            # Train
            train_metrics = self.train_epoch(train_loader)
            for key, value in train_metrics.items():
                if key != 'batch_count':
                    self.train_metrics[key].append(value)

            # Validate
            val_metrics = self.validate(val_loader)
            for key, value in val_metrics.items():
                if key != 'batch_count':
                    self.val_metrics[key].append(value)

            # Update learning rate
            self.scheduler.step()

            # Check for improvement
            current_angular = val_metrics['angular_error']
            is_best = current_angular < self.best_val_angular

            if is_best:
                self.best_val_angular = current_angular
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Log epoch summary
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'=' * 70}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.6f}, "
                        f"Angular: {train_metrics['angular_error']:.3f}°")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.6f}, "
                        f"Angular: {val_metrics['angular_error']:.3f}°")
            logger.info(f"        Median: {val_metrics['angular_median']:.3f}°, "
                        f"P95: {val_metrics['angular_p95']:.3f}°, "
                        f"Std: {val_metrics['angular_std']:.3f}°")
            logger.info(f"        Range: [{val_metrics['angular_min']:.2f}°, "
                        f"{val_metrics['angular_max']:.2f}°]")
            logger.info(f"Best  - Angular: {self.best_val_angular:.3f}° "
                        f"{'🎉 NEW!' if is_best else ''}")
            logger.info(f"LR: {current_lr:.6f}, Patience: {self.patience_counter}/{patience}")
            logger.info(f"{'=' * 70}")

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Plot curves periodically
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()

            # Early stopping
            if early_stop and self.patience_counter >= patience:
                logger.info(f"\n⚠️  Early stopping at epoch {epoch + 1}")
                break

        # Final plot
        self.plot_training_curves()

        logger.info(f"\n{'=' * 70}")
        logger.info("✅ Training Completed!")
        logger.info(f"{'=' * 70}")
        logger.info(f"Best validation angular error: {self.best_val_angular:.3f}°")
        logger.info(f"Best model saved to: {self.checkpoint_dir / 'best_model.pth'}")
        logger.info(f"{'=' * 70}\n")


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    # Augmentation config
    aug_config = MPIIGazeAugmentationConfig()

    # Create datasets
    train_dataset = MPIIGazeDataset(
        metadata_path=config['train_data'],
        augment=True,
        config=aug_config,
        image_size=config.get('image_size', 224)
    )

    val_dataset = MPIIGazeDataset(
        metadata_path=config['val_data'],
        augment=False,
        config=aug_config,
        image_size=config.get('image_size', 224)
    )

    # Create dataloaders
    batch_size = config.get('batch_size', 64)
    num_workers = min(8, torch.multiprocessing.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")

    return train_loader, val_loader


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='MPIIGaze Training (CORRECTED)')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training metadata JSON')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation metadata JSON')

    # Training arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./mpiigaze_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--no_stochastic_depth', action='store_true',
                        help='Disable stochastic depth')

    # Augmentation arguments
    parser.add_argument('--no_mixup', action='store_true',
                        help='Disable mixup')
    parser.add_argument('--no_cutmix', action='store_true',
                        help='Disable cutmix')

    # Other
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Learning rate warmup epochs')
    parser.add_argument('--angular_weight', type=float, default=0.5,
                        help='Weight for angular loss')

    args = parser.parse_args()

    # Build config
    config = {
        'train_data': args.train_data,
        'val_data': args.val_data,
        'checkpoint_dir': args.checkpoint_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'accumulation_steps': args.accumulation_steps,
        'dropout_rate': args.dropout,
        'use_stochastic_depth': not args.no_stochastic_depth,
        'use_mixup': not args.no_mixup,
        'use_cutmix': not args.no_cutmix,
        'patience': args.patience,
        'warmup_epochs': args.warmup_epochs,
        'angular_weight': args.angular_weight,
        'image_size': 224,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    model = MPIIGazeModel(
        dropout_rate=config['dropout_rate'],
        use_stochastic_depth=config['use_stochastic_depth']
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Create trainer
    trainer = MPIIGazeTrainer(model, config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint()

    # Train
    trainer.train(train_loader, val_loader)

    logger.info("🎯 Training completed successfully!")


if __name__ == '__main__':
    main()