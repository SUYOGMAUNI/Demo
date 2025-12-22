"""
PART 1: Models and Core Components
===================================
Contains: Model architecture, extractors, workers
Save as: gaze_models.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import psycopg2
from datetime import datetime
import time
import logging
import mediapipe as mp
import queue
from threading import Thread, Lock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class StochasticDepth(nn.Module):
    """Stochastic depth for regularization"""
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MaxRegularizediTrackerModel(nn.Module):
    """iTracker model matching training"""
    def __init__(self, dropout_rate=0.6, stochastic_depth=0.2):
        super().__init__()
        
        # Eye processing blocks
        self.eye_block1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.BatchNorm2d(96), nn.Dropout2d(0.4)
        )
        self.eye_sd1 = StochasticDepth(stochastic_depth)
        
        self.eye_block2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.BatchNorm2d(256), nn.Dropout2d(0.4)
        )
        self.eye_sd2 = StochasticDepth(stochastic_depth)
        
        self.eye_block3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        
        # Face processing blocks
        self.face_block1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.BatchNorm2d(96), nn.Dropout2d(0.4)
        )
        self.face_sd1 = StochasticDepth(stochastic_depth)
        
        self.face_block2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.BatchNorm2d(256), nn.Dropout2d(0.4)
        )
        self.face_sd2 = StochasticDepth(stochastic_depth)
        
        self.face_block3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        
        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        
        # Feature projections
        self.eye_fc = nn.Sequential(
            nn.Linear(1600, 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.face_fc = nn.Sequential(
            nn.Linear(1600, 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        
        # Final prediction layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, left, right, face, pose):
        # Left eye
        x = self.eye_block1(left)
        x = self.eye_sd1(x)
        x = self.eye_block2(x)
        x = self.eye_sd2(x)
        x = self.eye_block3(x)
        left_feat = self.eye_fc(x.view(x.size(0), -1))
        
        # Right eye
        x = self.eye_block1(right)
        x = self.eye_sd1(x)
        x = self.eye_block2(x)
        x = self.eye_sd2(x)
        x = self.eye_block3(x)
        right_feat = self.eye_fc(x.view(x.size(0), -1))
        
        # Face
        x = self.face_block1(face)
        x = self.face_sd1(x)
        x = self.face_block2(x)
        x = self.face_sd2(x)
        x = self.face_block3(x)
        face_feat = self.face_fc(x.view(x.size(0), -1))
        
        # Pose
        pose_feat = self.pose_encoder(pose)
        
        # Combine and predict
        combined = torch.cat([left_feat, right_feat, face_feat, pose_feat], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        gaze = self.fc3(x)
        
        return gaze


def load_model(path, device):
    """Load trained model"""
    model = MaxRegularizediTrackerModel().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    logger.info(f"✅ Model loaded from {path}")
    return model


# ============================================================================
# FAST EYE EXTRACTOR
# ============================================================================

# Preprocessing constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class FastEyeExtractor:
    """Optimized face/eye detector"""
    
    def __init__(self):
        # Minimal MediaPipe settings for speed
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Faster
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            static_image_mode=False
        )
        self.last_valid_result = None
        self.fail_count = 0

    def extract(self, frame):
        """Extract face, eyes, and head pose from frame"""
        try:
            # Downsample for faster processing (4x speedup)
            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            res = self.face_mesh.process(rgb)
            
            if not res.multi_face_landmarks:
                self.fail_count += 1
                logger.debug(f"No face detected, fail count: {self.fail_count}")
                # Use last valid result if recent failure
                if self.last_valid_result and self.fail_count < 2:
                    return self.last_valid_result
                return None
            
            self.fail_count = 0
            lms = res.multi_face_landmarks[0].landmark
            
            # Scale landmarks to original frame size
            h, w = frame.shape[:2]
            scale_x = w / 320
            scale_y = h / 240
            
            # Simple head pose estimation
            nose_tip = (lms[1].x * 320 * scale_x, lms[1].y * 240 * scale_y)
            yaw = (nose_tip[0] / w - 0.5) * 2  # Normalized -1 to 1
            pitch = (nose_tip[1] / h - 0.5) * 2
            pose = np.array([yaw, pitch], dtype=np.float32)
            
            # Fast face bounding box
            indices = [10, 338, 297, 332]
            xs = [int(lms[i].x * 320 * scale_x) for i in indices]
            ys = [int(lms[i].y * 240 * scale_y) for i in indices]
            
            x1 = max(0, min(xs) - 20)
            x2 = min(w, max(xs) + 20)
            y1 = max(0, min(ys) - 20)
            y2 = min(h, max(ys) + 20)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            
            # Resize face
            face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Extract eyes (fixed positions for speed)
            fh, fw = face.shape[:2]
            
            # Left eye region
            left_eye = face[
                int(fh * 0.25):int(fh * 0.60),
                int(fw * 0.55):fw
            ]
            
            # Right eye region
            right_eye = face[
                int(fh * 0.25):int(fh * 0.60),
                0:int(fw * 0.45)
            ]
            
            if left_eye.size == 0 or right_eye.size == 0:
                return None
            
            # Resize eyes
            left_eye = cv2.resize(left_eye, (224, 224), interpolation=cv2.INTER_LINEAR)
            right_eye = cv2.resize(right_eye, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            result = {
                'left_eye': left_eye,
                'right_eye': right_eye,
                'face': face,
                'head_pose': pose,
                'multiple_faces': False
            }
            
            self.last_valid_result = result
            return result
            
        except Exception as e:
            logger.debug(f"Extract error: {e}")
            return None

    def cleanup(self):
        """Release resources"""
        self.face_mesh.close()


# ============================================================================
# INFERENCE WORKER (Background Thread)
# ============================================================================

class InferenceWorker(Thread):
    """Background thread for model inference"""
    
    def __init__(self, model, device, extractor):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.extractor = extractor
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_lock = Lock()
        self.latest_result = None
        self.running = True
        
    def run(self):
        """Main inference loop"""
        while self.running:
            try:
                # Get frame (non-blocking with timeout)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Extract features
                detection = self.extractor.extract(frame)
                
                if detection is None:
                    result = (None, 'NO_FACE', None)
                elif detection.get('multiple_faces'):
                    result = (None, 'MULTIPLE_FACES', None)
                else:
                    # Preprocess images
                    try:
                        left = (cv2.cvtColor(detection['left_eye'], cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                        right = (cv2.cvtColor(detection['right_eye'], cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                        face = (cv2.cvtColor(detection['face'], cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                        
                        # Convert to tensors
                        left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        face_t = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        pose_t = torch.from_numpy(detection['head_pose']).unsqueeze(0).to(self.device)
                        
                        # Model inference
                        with torch.no_grad():
                            gaze = self.model(left_t, right_t, face_t, pose_t)
                        
                        gp = gaze.cpu().numpy()[0]
                        result = (
                            (float(np.clip(gp[0], 0, 1)), float(np.clip(gp[1], 0, 1))),
                            'SUCCESS',
                            detection['head_pose']
                        )
                    except Exception as e:
                        logger.debug(f"Preprocess error: {e}")
                        result = (None, 'PREPROC_ERR', None)
                
                # Update result (thread-safe)
                with self.result_lock:
                    self.latest_result = result
                    
            except Exception as e:
                logger.error(f"Inference worker error: {e}")
    
    def submit_frame(self, frame):
        """Submit frame for inference (non-blocking)"""
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip if queue is full
    
    def get_result(self):
        """Get latest inference result"""
        with self.result_lock:
            return self.latest_result
    
    def stop(self):
        """Stop worker thread"""
        self.running = False


# ============================================================================
# ASYNC DATABASE WRITER
# ============================================================================

class AsyncDBWriter(Thread):
    """Background thread for database writes"""
    
    def __init__(self, db_conn):
        super().__init__(daemon=True)
        self.db_conn = db_conn
        self.write_queue = queue.Queue()
        self.running = True
        
    def run(self):
        """Main write loop"""
        while self.running:
            try:
                task = self.write_queue.get(timeout=0.5)
                if task is None:
                    continue
                
                task_type, data = task
                
                if task_type == 'event':
                    self._write_event(*data)
                elif task_type == 'warning':
                    self._write_warning(*data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"DB write error: {e}")
    
    def _write_event(self, cid, sid, ts, gp, dist, viol, st, img, warn_num, pose):
        """Write gaze tracking event"""
        try:
            conn = psycopg2.connect(**self.db_conn)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO gaze_tracking_events 
                (candidate_id, exam_session_id, timestamp, gaze_x, gaze_y, baseline_distance,
                 is_off_screen, no_face_detected, multiple_faces, captured_image_path, 
                 warning_number, head_pose_yaw, head_pose_pitch)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                cid, sid, ts,
                gp[0] if gp else None,
                gp[1] if gp else None,
                dist,
                viol,
                st == 'NO_FACE',
                st == 'MULTIPLE_FACES',
                img,
                warn_num,
                pose[0] if pose else None,
                pose[1] if pose else None
            ))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Event write error: {e}")
    
    def _write_warning(self, cid, sid, warns, ts, warn_limit):
        """Write warning record"""
        try:
            conn = psycopg2.connect(**self.db_conn)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO gaze_warnings 
                (candidate_id, exam_session_id, warning_count, last_warning_time, status)
                VALUES (%s,%s,%s,%s,%s) 
                ON CONFLICT (candidate_id, exam_session_id) DO UPDATE
                SET warning_count = gaze_warnings.warning_count + 1,
                    last_warning_time = %s,
                    status = CASE WHEN gaze_warnings.warning_count + 1 >= %s 
                             THEN 'CRITICAL' ELSE 'ACTIVE' END
            """, (cid, sid, warns, ts, 'ACTIVE', ts, warn_limit))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Warning write error: {e}")
    
    def submit_event(self, cid, sid, ts, gp, dist, viol, st, img, warn_num, pose):
        """Queue event write"""
        self.write_queue.put(('event', (cid, sid, ts, gp, dist, viol, st, img, warn_num, pose)))
    
    def submit_warning(self, cid, sid, warns, ts, warn_limit):
        """Queue warning write"""
        self.write_queue.put(('warning', (cid, sid, warns, ts, warn_limit)))
    
    def stop(self):
        """Stop writer thread"""
        self.running = False
