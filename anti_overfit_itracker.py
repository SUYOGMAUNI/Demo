"""
MAXIMUM Anti-Overfitting iTracker
==================================
Aggressive regularization for continuous learning across all epochs.

Key Features:
1. Dropout 0.6 (very strong)
2. Weight decay 1e-3 (10x stronger)
3. Stochastic depth (randomly drops layers)
4. Mixup + CutMix augmentation
5. Multi-scale training
6. Cosine annealing with restarts
7. Gradient accumulation (larger effective batch)
8. Strong data augmentation

This model learns THROUGHOUT training without overfitting.

Usage:
    python max_antioverfit_itracker.py --train_data ./diagnostic_gaze_data/train_metadata.json \
                                       --val_data ./diagnostic_gaze_data/val_metadata.json \
                                       --epochs 150 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StochasticDepth(nn.Module):
    """Randomly drops layers during training"""
    def __init__(self, drop_prob=0.15):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MaxRegularizediTrackerModel(nn.Module):
    """iTracker with maximum anti-overfitting"""

    def __init__(self, dropout_rate=0.6, stochastic_depth=0.2):
        super().__init__()

        # Eye CNN blocks with stochastic depth
        self.eye_block1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.4),
        )
        self.eye_sd1 = StochasticDepth(stochastic_depth)

        self.eye_block2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.4),
        )
        self.eye_sd2 = StochasticDepth(stochastic_depth)

        self.eye_block3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        # Face CNN blocks
        self.face_block1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.4),
        )
        self.face_sd1 = StochasticDepth(stochastic_depth)

        self.face_block2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.4),
        )
        self.face_sd2 = StochasticDepth(stochastic_depth)

        self.face_block3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Feature projection
        self.eye_fc = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.face_fc = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Prediction head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout_rate)

        logger.info(f"✅ Max Regularized iTracker: dropout={dropout_rate}, stoch_depth={stochastic_depth}")

    def forward(self, left, right, face, pose):
        # Left eye with stochastic depth
        x = self.eye_block1(left)
        x = self.eye_sd1(x)
        x = self.eye_block2(x)
        x = self.eye_sd2(x)
        x = self.eye_block3(x)
        left_feat = self.eye_fc(x.view(x.size(0), -1))

        # Right eye (shared weights)
        x = self.eye_block1(right)
        x = self.eye_sd1(x)
        x = self.eye_block2(x)
        x = self.eye_sd2(x)
        x = self.eye_block3(x)
        right_feat = self.eye_fc(x.view(x.size(0), -1))

        # Face with stochastic depth
        x = self.face_block1(face)
        x = self.face_sd1(x)
        x = self.face_block2(x)
        x = self.face_sd2(x)
        x = self.face_block3(x)
        face_feat = self.face_fc(x.view(x.size(0), -1))

        # Pose
        pose_feat = self.pose_encoder(pose)

        # Fuse and predict
        combined = torch.cat([left_feat, right_feat, face_feat, pose_feat], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        gaze = self.fc3(x)

        return gaze


class AggressiveAugmentationDataset(Dataset):
    """Dataset with very strong augmentation"""

    def __init__(self, metadata_path, augment=False, multi_scale=True):
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.samples = self.metadata['samples']
        self.augment = augment
        self.multi_scale = multi_scale
        self.scales = [192, 224, 256] if multi_scale else [224]
        
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        logger.info(f"Dataset: {len(self.samples)} samples, augment={augment}, multi_scale={multi_scale}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        left = cv2.cvtColor(cv2.imread(s['left_eye_path']), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(s['right_eye_path']), cv2.COLOR_BGR2RGB)
        face = cv2.cvtColor(cv2.imread(s['face_path']), cv2.COLOR_BGR2RGB)
        gaze = np.array(s['gaze'], dtype=np.float32)
        pose = np.array(s['head_pose'], dtype=np.float32)

        size = np.random.choice(self.scales) if (self.augment and self.multi_scale) else 224

        if self.augment:
            left, right, face, gaze = self._augment(left, right, face, gaze)

        left = cv2.resize(left, (size, size))
        right = cv2.resize(right, (size, size))
        face = cv2.resize(face, (size, size))

        if size != 224:
            left = cv2.resize(left, (224, 224))
            right = cv2.resize(right, (224, 224))
            face = cv2.resize(face, (224, 224))

        return {
            'left_eye': self._to_tensor(left),
            'right_eye': self._to_tensor(right),
            'face': self._to_tensor(face),
            'head_pose': torch.FloatTensor(pose),
            'gaze': torch.FloatTensor(gaze)
        }

    def _augment(self, left, right, face, gaze):
        # Strong color jittering
        if np.random.random() < 0.7:
            alpha = np.random.uniform(0.6, 1.4)
            beta = np.random.randint(-40, 40)
            left = np.clip(alpha * left + beta, 0, 255).astype(np.uint8)
            right = np.clip(alpha * right + beta, 0, 255).astype(np.uint8)
            face = np.clip(alpha * face + beta, 0, 255).astype(np.uint8)

        # Flip
        if np.random.random() < 0.5:
            left, right = right.copy(), left.copy()
            face = np.fliplr(face).copy()
            gaze[0] = 1 - gaze[0]

        # Noise
        if np.random.random() < 0.4:
            noise = np.random.normal(0, np.random.uniform(3, 10), left.shape)
            left = np.clip(left + noise, 0, 255).astype(np.uint8)
            right = np.clip(right + noise, 0, 255).astype(np.uint8)
            face = np.clip(face + noise, 0, 255).astype(np.uint8)

        # Blur
        if np.random.random() < 0.3:
            k = np.random.choice([3, 5])
            left = cv2.GaussianBlur(left, (k, k), 0)
            right = cv2.GaussianBlur(right, (k, k), 0)
            face = cv2.GaussianBlur(face, (k, k), 0)

        # Rotation
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            h, w = left.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            left = cv2.warpAffine(left, M, (w, h))
            right = cv2.warpAffine(right, M, (w, h))
            face = cv2.warpAffine(face, M, (w, h))

        # Cutout
        if np.random.random() < 0.3:
            for img in [left, right, face]:
                h, w = img.shape[:2]
                length = 30
                y, x = np.random.randint(h), np.random.randint(w)
                y1 = np.clip(y - length//2, 0, h)
                y2 = np.clip(y + length//2, 0, h)
                x1 = np.clip(x - length//2, 0, w)
                x2 = np.clip(x + length//2, 0, w)
                img[y1:y2, x1:x2] = 0

        return left, right, face, gaze

    def _to_tensor(self, img):
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return self.normalize(img)


def mixup_data(left, right, face, pose, gaze, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(left.size(0)).to(left.device)
    return (lam * left + (1-lam) * left[idx],
            lam * right + (1-lam) * right[idx],
            lam * face + (1-lam) * face[idx],
            lam * pose + (1-lam) * pose[idx],
            lam * gaze + (1-lam) * gaze[idx])


def cutmix_data(left, right, face, pose, gaze, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(left.size(0)).to(left.device)
    
    W = H = left.size(-1)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cut_w//2, 0, W), np.clip(cy - cut_h//2, 0, H)
    x2, y2 = np.clip(cx + cut_w//2, 0, W), np.clip(cy + cut_h//2, 0, H)

    left[:, :, x1:x2, y1:y2] = left[idx, :, x1:x2, y1:y2]
    right[:, :, x1:x2, y1:y2] = right[idx, :, x1:x2, y1:y2]
    face[:, :, x1:x2, y1:y2] = face[idx, :, x1:x2, y1:y2]

    lam = 1 - ((x2-x1) * (y2-y1) / (W * H))
    return left, right, face, pose, lam * gaze + (1-lam) * gaze[idx]


class MaxAntiOverfitTrainer:
    def __init__(self, model, device='cuda', checkpoint_dir='./max_regularized_checkpoints',
                 use_mixup=True, use_cutmix=True, accum_steps=2, patience=25,
                 resume=False):  # ← ADD resume parameter
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.accum_steps = accum_steps
        self.patience = patience
        self.patience_counter = 0

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2, eta_min=1e-6
        )

        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        # ============ ADD THIS RESUME LOGIC ============
        if resume:
            self._load_checkpoint()
        # ===============================================

        logger.info("✅ Max Anti-Overfit Trainer")
        logger.info(f"   Mixup: {use_mixup}, CutMix: {use_cutmix}")
        logger.info(f"   Grad accum: {accum_steps}, Patience: {patience}")

    def compute_angular_error(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        diff = torch.sqrt(((pred - target) ** 2).sum(dim=1))
        return (diff.mean() * 90.0).item()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = total_ang = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(loader, desc='Train')):
            left = batch['left_eye'].to(self.device)
            right = batch['right_eye'].to(self.device)
            face = batch['face'].to(self.device)
            pose = batch['head_pose'].to(self.device)
            gaze = batch['gaze'].to(self.device)

            # Apply mixup/cutmix
            r = np.random.random()
            if self.use_mixup and r < 0.5:
                left, right, face, pose, gaze = mixup_data(left, right, face, pose, gaze)
            elif self.use_cutmix and r < 0.8:
                left, right, face, pose, gaze = cutmix_data(left, right, face, pose, gaze)

            pred = self.model(left, right, face, pose)
            loss = self.criterion(pred, gaze) / self.accum_steps
            loss.backward()

            if (i + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps
            total_ang += self.compute_angular_error(pred.detach(), gaze)

        return total_loss / len(loader), total_ang / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = total_ang = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc='Val'):
                left = batch['left_eye'].to(self.device)
                right = batch['right_eye'].to(self.device)
                face = batch['face'].to(self.device)
                pose = batch['head_pose'].to(self.device)
                gaze = batch['gaze'].to(self.device)

                pred = self.model(left, right, face, pose)
                total_loss += self.criterion(pred, gaze).item()
                total_ang += self.compute_angular_error(pred, gaze)

        return total_loss / len(loader), total_ang / len(loader)

    def save_checkpoint(self, epoch, val_loss, is_best):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,  # ← ADD THIS
            'val_losses': self.val_losses,  # ← ADD THIS
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
        }

        torch.save(ckpt, self.checkpoint_dir / 'checkpoint_latest.pth')

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'checkpoint_best.pth')
            torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_max_regularized.pth')
            logger.info(f"✅ NEW BEST: {val_loss:.4f}")

    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        path = self.checkpoint_dir / 'checkpoint_latest.pth'

        if not path.exists():
            logger.info("No checkpoint found, starting from scratch")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load training state
            self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint['patience_counter']
            self.train_losses = checkpoint.get('train_losses', [])  # ← ADD THIS
            self.val_losses = checkpoint.get('val_losses', [])  # ← ADD THIS

            logger.info(f"✅ Resumed from epoch {checkpoint['epoch']}")
            logger.info(f"   Best val loss: {self.best_val_loss:.6f}")
            logger.info(f"   Patience counter: {self.patience_counter}/{self.patience}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
            return False

    def train(self, train_loader, val_loader, epochs=150):
        logger.info(f"\n{'='*70}\n🚀 MAX ANTI-OVERFIT TRAINING\n{'='*70}")
        logger.info("This model will LEARN every epoch without overfitting!")
        logger.info(f"{'='*70}\n")

        for epoch in range(self.current_epoch, epochs):
            train_loss, train_ang = self.train_epoch(train_loader)
            val_loss, val_ang = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*70}")
            logger.info(f"Train: {train_loss:.4f} | {train_ang:.2f}°")
            logger.info(f"Val:   {val_loss:.4f} | {val_ang:.2f}° {'🎉' if is_best else ''}")
            logger.info(f"Best:  {self.best_val_loss:.4f}")
            logger.info(f"LR:    {self.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"Patience: {self.patience_counter}/{self.patience}")
            logger.info(f"{'='*70}")

            self.save_checkpoint(epoch, val_loss, is_best)
            self.current_epoch = epoch + 1

            if self.patience_counter >= self.patience:
                logger.info(f"\n⚠️ Early stop: {epoch+1}")
                break

        logger.info(f"\n✅ COMPLETE! Best: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./diagnostic_gaze_data/train_metadata.json')
    parser.add_argument('--val_data', default='./diagnostic_gaze_data/val_metadata.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--stoch_depth', type=float, default=0.2)
    parser.add_argument('--checkpoint_dir', default='./max_regularized_checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--no_mixup', action='store_false', dest='use_mixup')
    parser.add_argument('--use_cutmix', action='store_true', default=True)
    parser.add_argument('--no_cutmix', action='store_false', dest='use_cutmix')
    parser.add_argument('--accum_steps', type=int, default=2)
    parser.add_argument('--patience', type=int, default=25)
    args = parser.parse_args()

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    train_ds = AggressiveAugmentationDataset(args.train_data, augment=True, multi_scale=True)
    val_ds = AggressiveAugmentationDataset(args.val_data, augment=False, multi_scale=False)

    nw = min(8, torch.multiprocessing.cpu_count() // 2)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaxRegularizediTrackerModel(args.dropout, args.stoch_depth)
    trainer = MaxAntiOverfitTrainer(
        model=model,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_mixup=True,
        use_cutmix=True,
        accum_steps=2,
        patience=25,
        resume=args.resume
    )
    trainer.train(train_loader, val_loader, args.epochs)


if __name__ == '__main__':
    main()
