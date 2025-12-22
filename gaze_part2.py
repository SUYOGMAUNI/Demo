"""
PART 2: Main Pipeline and Application
======================================
Contains: Pipeline, calibration, main loop
Save as: web_gaze_corrected.py

IMPORTANT: Make sure gaze_models.py is in the same directory!

Usage:
    python web_gaze_corrected.py
"""

import os
import cv2
import numpy as np
import torch
import psycopg2
from datetime import datetime
import time
import logging
from collections import deque

# Import from Part 1
from gaze_part1 import (
    load_model,
    FastEyeExtractor,
    InferenceWorker,
    AsyncDBWriter
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Calibration
CALIBRATION_DUR = 5  # seconds

# Performance
INFERENCE_FPS = 8  # Run model 5 times/second (smooth performance)
DISPLAY_FPS = 30   # Display at 30 FPS

# Detection
THRESHOLD = 0.08  # Distance threshold for violations
WARN_LIMIT = 10  # Maximum warnings before critical
VIOLATION_COOLDOWN = 0.5  # Seconds between violations

# Safe zone (normalized coordinates 0-1)
SAFE_ZONE = {
    'x_min': 0.30,  # 30% from left
    'x_max': 0.65,  # 70% from left (middle 40% of screen)
    'y_min': 0.30,  # 30% from top
    'y_max': 0.65   # 70% from top (middle 40% of screen)
}

# Database
DB_CONN = {
    'host': 'localhost',
    'database': 'Proctoring',
    'user': 'postgres',
    'password': '526183'
}

# Directories
CAPTURES_DIR = r"D:\GazeCaptures"
LOGS_DIR = r"D:\logs"
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class FastGazePipeline:
    """Ultra-fast gaze tracking pipeline"""
    
    def __init__(self, candidate_id, session_id, model_path):
        self.cid = candidate_id
        self.sid = session_id
        self.warns = 0
        self.calibrated = False
        self.baseline = None
        
        # Logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(LOGS_DIR, f"gaze_{candidate_id}_{timestamp}.txt")
        self._log("=" * 70)
        self._log(f"SESSION START | Candidate: {candidate_id} | Session: {session_id}")
        self._log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("✅ ULTRA-FAST MODE - Multi-threaded")
        self._log("=" * 70 + "\n")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = load_model(model_path, self.device)
        
        # Initialize components
        self.extractor = FastEyeExtractor()
        self.inference_worker = InferenceWorker(self.model, self.device, self.extractor)
        self.db_writer = AsyncDBWriter(DB_CONN)
        
        # State
        self.current_gaze = None
        self.current_status = None
        self.current_pose = None
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.last_violation_time = 0
        
        # Initialize database
        self._init_db()
        
        # Start workers
        self.inference_worker.start()
        self.db_writer.start()
        logger.info("✅ Multi-threaded workers started")
    
    def _init_db(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()
            
            # Events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gaze_tracking_events (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100),
                    exam_session_id VARCHAR(100),
                    timestamp TIMESTAMP,
                    gaze_x FLOAT,
                    gaze_y FLOAT,
                    baseline_distance FLOAT,
                    is_off_screen BOOLEAN,
                    no_face_detected BOOLEAN,
                    multiple_faces BOOLEAN,
                    captured_image_path TEXT,
                    warning_number INTEGER,
                    head_pose_yaw FLOAT,
                    head_pose_pitch FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Warnings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gaze_warnings (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100),
                    exam_session_id VARCHAR(100),
                    warning_count INTEGER,
                    last_warning_time TIMESTAMP,
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(candidate_id, exam_session_id)
                )
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"DB init error: {e}")
    
    def _log(self, msg):
        """Write to log file"""
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"{msg}\n")
        except:
            pass

    def _calibrate(self, cap):
        """Calibration procedure with proper worker synchronization"""
        logger.info(f"Starting calibration process...")

        # ⭐ STEP 1: Warm up the worker thread
        logger.info("Step 1/3: Warming up inference worker...")
        warmup_frames = 0
        for i in range(15):  # More warmup frames
            ret, frame = cap.read()
            if ret:
                self.inference_worker.submit_frame(frame)
                warmup_frames += 1
                time.sleep(0.1)

        logger.info(f"Submitted {warmup_frames} warmup frames")

        # ⭐ STEP 2: Wait for first valid result
        logger.info("Step 2/3: Waiting for first inference result...")
        timeout = time.time() + 5  # 5 second timeout
        first_result = None

        while time.time() < timeout:
            result = self.inference_worker.get_result()
            if result and result[1] == 'SUCCESS' and result[0]:
                first_result = result
                logger.info(f"✅ First result received: {result[0]}")
                break
            time.sleep(0.1)

        if not first_result:
            logger.error("❌ Worker not responding after warmup!")
            logger.error("Model inference is too slow or failed")
            return False

        # ⭐ STEP 3: Actual calibration
        logger.info(f"Step 3/3: Collecting calibration samples ({CALIBRATION_DUR}s)...")
        gazes = []
        start_time = time.time()
        frame_count = 0
        last_result_time = time.time()

        while time.time() - start_time < CALIBRATION_DUR:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Create display
            disp = frame.copy()
            remaining = int(CALIBRATION_DUR - (time.time() - start_time))

            # Title
            cv2.putText(disp, f"CALIBRATING... {remaining}s",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(disp, "LOOK AT CENTER OF SCREEN",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Progress
            progress = len(gazes)
            target = max(5, int(CALIBRATION_DUR * 2))  # Target: 2 samples/second
            cv2.putText(disp, f"Samples: {progress}/{target}",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Progress bar
            bar_width = 400
            bar_height = 20
            bar_x, bar_y = 50, 170
            cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
            filled = int((progress / target) * bar_width) if target > 0 else 0
            if filled > 0:
                cv2.rectangle(disp, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 255, 0), -1)

            # Draw center target (bigger and more visible)
            h, w = disp.shape[:2]
            center = (w // 2, h // 2)
            cv2.circle(disp, center, 30, (0, 255, 255), 4)
            cv2.circle(disp, center, 10, (0, 255, 255), -1)

            # Draw crosshair
            cv2.line(disp, (center[0] - 40, center[1]), (center[0] + 40, center[1]), (0, 255, 255), 2)
            cv2.line(disp, (center[0], center[1] - 40), (center[0], center[1] + 40), (0, 255, 255), 2)

            cv2.imshow('Calibration', disp)
            cv2.waitKey(1)

            # Submit frame every 3rd frame (reduce load)
            if frame_count % 3 == 0:
                self.inference_worker.submit_frame(frame)

            # Check for new results more frequently
            result = self.inference_worker.get_result()

            if result and result != last_result_time:  # New result
                gaze_point, status, pose = result

                if status == 'SUCCESS' and gaze_point:
                    gazes.append(gaze_point)
                    last_result_time = result

                    # Log progress
                    if len(gazes) % 3 == 0:
                        logger.info(f"Progress: {len(gazes)} samples collected")

            # Small delay
            time.sleep(0.05)

        # Close calibration window
        cv2.destroyWindow('Calibration')

        # ⭐ STEP 4: Calculate baseline
        logger.info(f"\nCalibration Results:")
        logger.info(f"  Duration: {CALIBRATION_DUR}s")
        logger.info(f"  Frames shown: {frame_count}")
        logger.info(f"  Valid samples: {len(gazes)}")

        # Lower threshold - need at least 3 samples
        if len(gazes) >= 3:
            self.baseline = np.mean(gazes, axis=0)
            self.calibrated = True

            # Calculate variance to assess quality
            if len(gazes) > 1:
                variance = np.var(gazes, axis=0)
                logger.info(f"  Sample variance: ({variance[0]:.4f}, {variance[1]:.4f})")

            logger.info(f"\n✅ CALIBRATION SUCCESSFUL!")
            logger.info(f"   Baseline: ({self.baseline[0]:.3f}, {self.baseline[1]:.3f})")
            logger.info(f"   Samples used: {len(gazes)}")

            if len(gazes) < 5:
                logger.warning(f"⚠️ Only {len(gazes)} samples - calibration may be less accurate")

            return True
        else:
            logger.error(f"\n❌ CALIBRATION FAILED")
            logger.error(f"   Only collected {len(gazes)} samples (need at least 3)")
            logger.error("\n💡 Troubleshooting:")
            logger.error("   1. Model inference is too slow for real-time")
            logger.error("   2. Try closing other applications")
            logger.error("   3. Check if GPU is being used (CUDA available?)")
            logger.error(f"   4. Current device: {self.device}")

            return False

    def _check_violation(self, gaze_point):
        """Check if gaze is a violation"""
        if not gaze_point or not self.calibrated:
            return False, 0.0

        # Check cooldown
        now = time.time()
        if now - self.last_violation_time < VIOLATION_COOLDOWN:
            return False, 0.0

        # Calculate distance from baseline
        dist = np.linalg.norm(np.array(gaze_point) - self.baseline)

        # Check safe zone
        x, y = gaze_point
        in_safe_zone = (
                SAFE_ZONE['x_min'] <= x <= SAFE_ZONE['x_max'] and
                SAFE_ZONE['y_min'] <= y <= SAFE_ZONE['y_max']
        )

        # Determine violation
        is_violation = (dist > THRESHOLD or not in_safe_zone)

        if is_violation:
            logger.warning(
                f"🚨 VIOLATION: gaze=({x:.3f},{y:.3f}) "
                f"baseline=({self.baseline[0]:.3f},{self.baseline[1]:.3f}) "
                f"dist={dist:.3f}"
            )

        return is_violation, dist
    
    def run(self):
        """Main tracking loop"""
        try:
            # Open camera
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.error("Cannot open camera")
                return
            
            logger.info("Camera opened successfully")
            
            # Calibration
            if not self._calibrate(cap):
                logger.error("Calibration failed, exiting...")
                cap.release()
                return
            
            logger.info("\n" + "=" * 70)
            logger.info("TRACKING STARTED")
            logger.info("=" * 70 + "\n")
            
            last_inference_submit = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                now = time.time()
                
                # Calculate FPS
                fps = 1.0 / (now - self.last_frame_time) if (now - self.last_frame_time) > 0 else 0
                self.fps_queue.append(fps)
                self.last_frame_time = now
                avg_fps = np.mean(self.fps_queue)
                
                # Submit frame for inference at specified interval
                if now - last_inference_submit >= 1.0 / INFERENCE_FPS:
                    self.inference_worker.submit_frame(frame)
                    last_inference_submit = now

                # Get latest inference result
                result = self.inference_worker.get_result()
                if result:
                    gaze_point, status, pose = result
                    self.current_gaze = gaze_point
                    self.current_status = status
                    self.current_pose = pose

                    # ⭐ NEW: Check for NO_FACE violations FIRST
                    if status == 'NO_FACE':
                        # Check cooldown
                        if now - self.last_violation_time >= VIOLATION_COOLDOWN:
                            self.warns += 1
                            self.last_violation_time = now

                            # Save violation image
                            timestamp = datetime.utcnow()
                            filename = f"{self.cid}_{timestamp.strftime('%Y%m%d_%H%M%S')}_noface.jpg"
                            filepath = os.path.join(CAPTURES_DIR, filename)

                            try:
                                cv2.imwrite(filepath, frame)
                            except:
                                filepath = None

                            # Log
                            msg = f"VIOLATION #{self.warns}: NO FACE DETECTED"
                            self._log(msg)
                            logger.warning(f"🚨 {msg}")

                            # Async database write
                            self.db_writer.submit_event(
                                self.cid, self.sid, timestamp, None, 0.0,
                                True, status, filepath, self.warns, None
                            )
                            self.db_writer.submit_warning(
                                self.cid, self.sid, self.warns, timestamp, WARN_LIMIT
                            )

                    # Check for gaze violations (existing code)
                    elif status == 'SUCCESS' and gaze_point:
                        is_violation, distance = self._check_violation(gaze_point)

                        if is_violation:
                            self.warns += 1
                            self.last_violation_time = now
                            
                            # Save violation image
                            timestamp = datetime.utcnow()
                            filename = f"{self.cid}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                            filepath = os.path.join(CAPTURES_DIR, filename)
                            
                            try:
                                cv2.imwrite(filepath, frame)
                            except:
                                filepath = None
                            
                            # Log
                            x, y = gaze_point
                            msg = f"VIOLATION #{self.warns}: gaze=({x:.3f},{y:.3f}) dist={distance:.3f}"
                            self._log(msg)
                            
                            # Async database write
                            self.db_writer.submit_event(
                                self.cid, self.sid, timestamp, gaze_point, distance,
                                True, status, filepath, self.warns, pose
                            )
                            self.db_writer.submit_warning(
                                self.cid, self.sid, self.warns, timestamp, WARN_LIMIT
                            )
                
                # Create display frame
                disp = self._create_display(frame, avg_fps)
                
                # Show frame
                cv2.imshow('Gaze Tracker [Q=Quit]', disp)
                
                # Handle key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            cap.release()
            self.extractor.cleanup()
            self.inference_worker.stop()
            self.db_writer.stop()
            cv2.destroyAllWindows()
            
            # Final log
            self._log("\n" + "=" * 70)
            self._log(f"SESSION END | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"Total Warnings: {self.warns}")
            if self.baseline is not None:
                self._log(f"Baseline: ({self.baseline[0]:.3f}, {self.baseline[1]:.3f})")
            self._log("=" * 70)
            
            logger.info(f"\n{'=' * 70}")
            logger.info("SESSION COMPLETE")
            logger.info(f"{'=' * 70}")
            logger.info(f"Candidate: {self.cid}")
            logger.info(f"Total Warnings: {self.warns}")
            logger.info(f"Log: {self.log_path}")
            logger.info(f"{'=' * 70}")
    
    def _create_display(self, frame, avg_fps):
        """Create display frame with overlay"""
        disp = frame.copy()
        h, w = disp.shape[:2]
        
        # Draw safe zone
        x1 = int(SAFE_ZONE['x_min'] * w)
        y1 = int(SAFE_ZONE['y_min'] * h)
        x2 = int(SAFE_ZONE['x_max'] * w)
        y2 = int(SAFE_ZONE['y_max'] * h)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw center
        center = (w // 2, h // 2)
        cv2.circle(disp, center, 5, (0, 255, 255), -1)
        
        # FPS
        cv2.putText(disp, f"FPS: {avg_fps:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Candidate ID
        cv2.putText(disp, f"ID: {self.cid}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Warnings
        warn_color = (0, 255, 0) if self.warns < WARN_LIMIT else (0, 0, 255)
        cv2.putText(disp, f"Warnings: {self.warns}/{WARN_LIMIT}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warn_color, 2)
        
        # Status
        if self.current_status == 'NO_FACE':
            cv2.putText(disp, "NO FACE DETECTED",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.current_status == 'SUCCESS' and self.current_gaze:
            x, y = self.current_gaze
            
            # Calculate distance
            if self.baseline is not None:
                dist = np.linalg.norm(np.array(self.current_gaze) - self.baseline)
                is_violation = dist > THRESHOLD
            else:
                dist = 0
                is_violation = False
            
            # Color based on violation
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            
            # Display gaze coordinates
            cv2.putText(disp, f"Gaze: ({x:.2f}, {y:.2f})",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(disp, f"Distance: {dist:.3f}",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw gaze point
            gaze_x = int(x * w)
            gaze_y = int(y * h)
            cv2.circle(disp, (gaze_x, gaze_y), 12, color, 2)
            cv2.circle(disp, (gaze_x, gaze_y), 3, color, -1)
            
            # Draw line from center to gaze
            cv2.line(disp, center, (gaze_x, gaze_y), color, 1)
        
        # Critical warning overlay
        if self.warns >= WARN_LIMIT:
            cv2.putText(disp, "CRITICAL ALERT!",
                       (w // 2 - 150, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return disp


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("ULTRA-FAST GAZE TRACKING SYSTEM")
    print("=" * 60)
    print("\n💡 OPTIMIZATIONS:")
    print("  • Multi-threaded processing")
    print("  • 30 FPS display, 5 FPS inference")
    print("  • Smart violation detection")
    print("  • Async database writes")
    print("\n" + "=" * 60)
    
    # Get inputs
    cid = input("\nCandidate ID: ").strip()
    if not cid:
        cid = "test_candidate"
        print(f"Using default: {cid}")
    
    sid = input("Exam Session ID: ").strip()
    if not sid:
        sid = "test_session"
        print(f"Using default: {sid}")
    
    mpath = input("Model path [./max_regularized_checkpoints/best_max_regularized.pth]: ").strip()
    if not mpath:
        mpath = "./max_regularized_checkpoints/best_max_regularized.pth"
    
    # Check model exists
    if not os.path.exists(mpath):
        print(f"\n❌ Model not found: {mpath}")
        print("Please provide a valid model path.")
        return
    
    print("\n" + "=" * 60)
    print("STARTING PIPELINE...")
    print(f"Candidate: {cid}")
    print(f"Session: {sid}")
    print(f"Model: {mpath}")
    print("=" * 60 + "\n")
    
    try:
        pipeline = FastGazePipeline(cid, sid, mpath)
        pipeline.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
