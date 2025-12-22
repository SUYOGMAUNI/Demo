"""
Enhanced Face Match Pipeline with Multi-Person Detection and Faster FPS
Detects and alerts when multiple people are in frame during exam
"""

import cv2
import numpy as np
import dlib
import psycopg2
from datetime import datetime
import time
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parameters
FACE_MATCH_THRESHOLD = 0.6
CHECK_INTERVAL = 5  # Face match check every 5 seconds
WARNING_THRESHOLD = 3
MULTI_PERSON_WARNING_THRESHOLD = 2

# FPS optimization
FRAME_SKIP = 2  # Process every Nth frame for face detection (speeds up display)
DETECTION_SCALE = 0.5  # Downscale frame for face detection (speeds up processing)

# Database config
DB_CONN = {
    'host': 'localhost',
    'database': 'Proctoring',
    'user': 'postgres',
    'password': '526183'
}

# Directories
REGISTERED_IMAGES_DIR = r"D:\Saved Images"
MISMATCH_CAPTURES_DIR = r"D:\MismatchCaptures"
MULTI_PERSON_CAPTURES_DIR = r"D:\MultiPersonCaptures"
LOGS_DIR = r"D:\logs"

os.makedirs(MISMATCH_CAPTURES_DIR, exist_ok=True)
os.makedirs(MULTI_PERSON_CAPTURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class FaceMatchPipeline:
    """Enhanced pipeline with multi-person detection and faster FPS"""

    def __init__(self, candidate_id, exam_session_id=None):
        self.candidate_id = candidate_id
        self.exam_session_id = exam_session_id
        self.warning_count = 0
        self.multi_person_warning_count = 0
        self.is_monitoring = False
        self.frame_count = 0  # For frame skipping

        # Initialize log file
        self.log_file_path = os.path.join(
            LOGS_DIR,
            f"facematch_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self._write_log("=" * 70)
        self._write_log("ENHANCED FACE MATCH PIPELINE - SESSION STARTED")
        self._write_log(f"Candidate ID: {candidate_id}")
        self._write_log(f"Multi-Person Detection: ENABLED")
        self._write_log(f"FPS Optimization: ENABLED (Frame Skip={FRAME_SKIP})")
        self._write_log("=" * 70 + "\n")

        # Initialize dlib models
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(
                'models/shape_predictor_68_face_landmarks.dat'
            )
            self.face_recognizer = dlib.face_recognition_model_v1(
                'models/dlib_face_recognition_resnet_model_v1.dat'
            )
            logger.info("Dlib models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading dlib models: {e}")
            raise

        # Load registered face
        self.registered_embedding = None
        self.registered_image_path = None
        self._load_registered_face()

        # Initialize database
        self._initialize_database()

        # Webcam
        self.cap = None

        # Cache for continuous display
        self.cached_num_faces = 0
        self.cached_faces = []

    def _initialize_database(self):
        """Create necessary database tables"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            # Enhanced face_match_events table with multi-person detection
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_match_events (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) NOT NULL,
                    exam_session_id VARCHAR(100),
                    timestamp TIMESTAMP NOT NULL,
                    match_status VARCHAR(20),
                    euclidean_distance FLOAT,
                    captured_image_path TEXT,
                    warning_number INTEGER,
                    num_faces_detected INTEGER,
                    violation_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Multi-person violations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS multi_person_violations (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) NOT NULL,
                    exam_session_id VARCHAR(100),
                    timestamp TIMESTAMP NOT NULL,
                    num_persons_detected INTEGER,
                    captured_image_path TEXT,
                    warning_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Face match warnings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_match_warnings (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) NOT NULL,
                    exam_session_id VARCHAR(100),
                    warning_count INTEGER,
                    multi_person_warning_count INTEGER,
                    last_warning_time TIMESTAMP,
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(candidate_id, exam_session_id)
                )
            """)

            conn.commit()
            cur.close()
            conn.close()
            logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def _load_registered_face(self):
        """Load registered student face"""
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for ext in possible_extensions:
            image_path = os.path.join(REGISTERED_IMAGES_DIR, f"{self.candidate_id}{ext}")
            if os.path.exists(image_path):
                self.registered_image_path = image_path
                break

        if self.registered_image_path is None:
            raise FileNotFoundError(f"Registered image not found for {self.candidate_id}")

        self.registered_embedding = self._get_face_embedding(self.registered_image_path)

        if self.registered_embedding is None:
            raise ValueError("Could not extract face from registered image")

        logger.info(f"Registered face embedding loaded for: {self.candidate_id}")

    def _write_log(self, message):
        """Write message to text log file"""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")

    def detect_all_faces(self, frame):
        """
        Detect ALL faces in frame (optimized for speed)
        Returns: (num_faces, embeddings_list, faces_list)
        """
        try:
            # Downscale for faster detection
            small_frame = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            rgb_img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect all faces (upsample=1 for speed, can reduce to 0 for even faster)
            faces = self.face_detector(rgb_img, 0)
            num_faces = len(faces)

            # Scale face coordinates back to original size
            scale_factor = 1.0 / DETECTION_SCALE
            scaled_faces = []
            for face in faces:
                scaled_face = dlib.rectangle(
                    int(face.left() * scale_factor),
                    int(face.top() * scale_factor),
                    int(face.right() * scale_factor),
                    int(face.bottom() * scale_factor)
                )
                scaled_faces.append(scaled_face)

            return num_faces, scaled_faces

        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return 0, []

    def _get_face_embedding(self, image_path_or_frame):
        """Extract face embedding from first detected face"""
        try:
            if isinstance(image_path_or_frame, str):
                img = cv2.imread(image_path_or_frame)
                if img is None:
                    return None
            else:
                img = image_path_or_frame

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(rgb_img, 1)

            if len(faces) == 0:
                return None

            face = faces[0]
            shape = self.shape_predictor(rgb_img, face)
            face_embedding = self.face_recognizer.compute_face_descriptor(rgb_img, shape)

            return np.array(face_embedding)

        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None

    def compare_faces(self, live_embedding):
        """Compare live face embedding with registered embedding"""
        if self.registered_embedding is None or live_embedding is None:
            return "ERROR", 999.0

        distance = np.linalg.norm(self.registered_embedding - live_embedding)

        if distance <= FACE_MATCH_THRESHOLD:
            match_status = "MATCH"
        else:
            match_status = "MISMATCH"

        return match_status, float(distance)

    def handle_multi_person_violation(self, frame, num_persons, timestamp):
        """Handle case when multiple people detected"""
        self.multi_person_warning_count += 1

        # Save capture
        filename = f"{self.candidate_id}_multiperson_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(MULTI_PERSON_CAPTURES_DIR, filename)

        # Draw warning on frame
        warning_frame = frame.copy()
        cv2.putText(
            warning_frame,
            f"VIOLATION: {num_persons} PERSONS DETECTED",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        cv2.putText(
            warning_frame,
            f"WARNING #{self.multi_person_warning_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        cv2.imwrite(filepath, warning_frame)

        # Log to text file
        log_message = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
        log_message += f"⚠️ MULTI-PERSON VIOLATION | {num_persons} persons | "
        log_message += f"WARNING #{self.multi_person_warning_count} | Image: {filepath}"
        self._write_log(log_message)

        logger.warning(
            f"Multi-person violation: {num_persons} persons detected "
            f"(Warning #{self.multi_person_warning_count})"
        )

        # Store in database
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO multi_person_violations 
                (candidate_id, exam_session_id, timestamp, num_persons_detected, 
                 captured_image_path, warning_number)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                self.candidate_id,
                self.exam_session_id,
                timestamp,
                num_persons,
                filepath,
                self.multi_person_warning_count
            ))

            # Update warnings table
            cur.execute("""
                INSERT INTO face_match_warnings 
                (candidate_id, exam_session_id, warning_count, multi_person_warning_count, 
                 last_warning_time, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (candidate_id, exam_session_id) 
                DO UPDATE SET 
                    multi_person_warning_count = face_match_warnings.multi_person_warning_count + 1,
                    last_warning_time = %s,
                    status = CASE 
                        WHEN face_match_warnings.multi_person_warning_count + 1 >= %s THEN 'CRITICAL'
                        ELSE 'ACTIVE'
                    END
            """, (
                self.candidate_id,
                self.exam_session_id,
                self.warning_count,
                self.multi_person_warning_count,
                timestamp,
                'ACTIVE',
                timestamp,
                MULTI_PERSON_WARNING_THRESHOLD
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing multi-person violation: {e}")

        # Check threshold
        if self.multi_person_warning_count >= MULTI_PERSON_WARNING_THRESHOLD:
            self._write_log("\n" + "!" * 70)
            self._write_log(f"🚨 CRITICAL: {self.multi_person_warning_count} multi-person violations!")
            self._write_log("!" * 70 + "\n")
            self.raise_critical_alert('MULTI_PERSON')

        return filepath

    def raise_critical_alert(self, alert_type='FACE_MISMATCH'):
        """Raise critical alert"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            # Create critical_alerts table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS critical_alerts (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) NOT NULL,
                    exam_session_id VARCHAR(100),
                    alert_type VARCHAR(50),
                    alert_message TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            if alert_type == 'MULTI_PERSON':
                message = f'Multiple persons detected: {self.multi_person_warning_count} violations'
            else:
                message = f'Face mismatches detected: {self.warning_count} warnings'

            cur.execute("""
                INSERT INTO critical_alerts 
                (candidate_id, exam_session_id, alert_type, alert_message, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                self.candidate_id,
                self.exam_session_id,
                alert_type,
                message,
                datetime.utcnow()
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.critical(f"Critical alert raised: {alert_type}")

        except Exception as e:
            logger.error(f"Error raising critical alert: {e}")

    def start_monitoring(self):
        """Start face match monitoring"""
        try:
            self.cap = cv2.VideoCapture(0)

            # Set camera properties for better FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                logger.error("Could not open webcam")
                return

            self.is_monitoring = True
            logger.info(f"Enhanced monitoring started for: {self.candidate_id}")

            self._monitoring_loop()

        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            self.is_monitoring = False

    def _monitoring_loop(self):
        """Main monitoring loop with multi-person detection and faster FPS"""
        last_check_time = time.time()
        fps_start_time = time.time()
        fps_counter = 0

        try:
            while self.is_monitoring:
                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Could not read frame")
                    time.sleep(0.1)
                    continue

                self.frame_count += 1
                display_frame = frame.copy()
                current_time = time.time()

                # OPTIMIZED: Only detect faces every Nth frame for display
                if self.frame_count % FRAME_SKIP == 0:
                    self.cached_num_faces, self.cached_faces = self.detect_all_faces(frame)

                # Draw rectangles around ALL detected faces (using cached results)
                for face in self.cached_faces:
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()

                    # Color based on number of faces
                    color = (0, 255, 0) if self.cached_num_faces == 1 else (0, 0, 255)
                    thickness = 2 if self.cached_num_faces == 1 else 3
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # Display status with better visibility
                status_color = (0, 0, 255) if self.cached_num_faces > 1 else (0, 255, 0)

                # Background for text (for better readability)
                cv2.rectangle(display_frame, (5, 5), (400, 120), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (5, 5), (400, 120), status_color, 2)

                cv2.putText(
                    display_frame,
                    f"Faces: {self.cached_num_faces}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Mismatch: {self.warning_count} | Multi-Person: {self.multi_person_warning_count}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

                # Calculate and display FPS
                fps_counter += 1
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                else:
                    fps = fps_counter / max(current_time - fps_start_time, 0.001)

                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1
                )

                # PERIODIC face match verification - Every 5 seconds
                if current_time - last_check_time >= CHECK_INTERVAL:
                    logger.info("Performing periodic verification...")
                    timestamp = datetime.utcnow()

                    # Get fresh face detection for verification
                    num_faces, faces = self.detect_all_faces(frame)

                    # Check for multiple persons FIRST (PRIORITY CHECK)
                    if num_faces > 1:
                        logger.warning(f"⚠️ VIOLATION: {num_faces} persons detected!")
                        self.handle_multi_person_violation(frame, num_faces, timestamp)

                    elif num_faces == 1:
                        # Normal face verification
                        live_embedding = self._get_face_embedding(frame)

                        if live_embedding is not None:
                            match_status, distance = self.compare_faces(live_embedding)

                            if match_status == "MISMATCH":
                                self.warning_count += 1
                                logger.warning(f"Face mismatch! Distance: {distance:.4f}")

                                # Save capture
                                filename = f"{self.candidate_id}_mismatch_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                                filepath = os.path.join(MISMATCH_CAPTURES_DIR, filename)

                                mismatch_frame = frame.copy()
                                cv2.putText(
                                    mismatch_frame,
                                    f"MISMATCH - Distance: {distance:.3f}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2
                                )
                                cv2.imwrite(filepath, mismatch_frame)

                                self._write_log(
                                    f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"MISMATCH | Distance: {distance:.4f} | "
                                    f"WARNING #{self.warning_count}"
                                )

                                # Check threshold
                                if self.warning_count >= WARNING_THRESHOLD:
                                    self.raise_critical_alert('FACE_MISMATCH')
                            else:
                                logger.info(f"Face match verified - Distance: {distance:.4f}")

                    elif num_faces == 0:
                        logger.warning("No face detected")

                    last_check_time = current_time

                # Show frame
                cv2.imshow('Face Match Monitoring (Press Q to quit)', display_frame)

                # Exit on 'q' key (reduced wait time for faster response)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Monitoring stopped by user")
                    break

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        self._write_log("\n" + "=" * 70)
        self._write_log("SESSION ENDED")
        self._write_log(f"Face Mismatch Warnings: {self.warning_count}")
        self._write_log(f"Multi-Person Warnings: {self.multi_person_warning_count}")
        self._write_log("=" * 70)

        logger.info(f"Monitoring stopped. Log: {self.log_file_path}")


# =============================================================================
# Student Registration Module
# =============================================================================

class StudentRegistration:
    """Handle student registration with face capture"""

    def __init__(self):
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(
                'models/shape_predictor_68_face_landmarks.dat'
            )
            self.face_recognizer = dlib.face_recognition_model_v1(
                'models/dlib_face_recognition_resnet_model_v1.dat'
            )
            logger.info("Registration module initialized")
        except Exception as e:
            logger.error(f"Error loading dlib models: {e}")
            raise

    def capture_registration_photo(self, candidate_id, name=None, email=None):
        """
        Capture photo from webcam for student registration
        Saves to D:\Saved Images\{candidate_id}.jpg
        """
        logger.info(f"Starting registration photo capture for: {candidate_id}")

        cap = cv2.VideoCapture(0)
        captured = False
        captured_frame = None

        print("\n" + "=" * 70)
        print("STUDENT REGISTRATION - PHOTO CAPTURE")
        print("=" * 70)
        print(f"Candidate ID: {candidate_id}")
        if name:
            print(f"Name: {name}")
        if email:
            print(f"Email: {email}")
        print("\nInstructions:")
        print("  1. Position your face in the center of the frame")
        print("  2. Ensure good lighting on your face")
        print("  3. Look directly at the camera")
        print("  4. Remove glasses if possible")
        print("  5. Press SPACE to capture photo")
        print("  6. Press Q to cancel")
        print("=" * 70 + "\n")

        try:
            while not captured:
                ret, frame = cap.read()

                if not ret:
                    logger.error("Could not read from webcam")
                    break

                display_frame = frame.copy()

                # Detect faces
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_detector(rgb_frame, 1)

                # Draw rectangles around detected faces
                for face in faces:
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        "Face Detected - Press SPACE to capture",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                # Display status messages
                if len(faces) == 0:
                    cv2.putText(
                        display_frame,
                        "NO FACE DETECTED - Position your face in frame",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                elif len(faces) > 1:
                    cv2.putText(
                        display_frame,
                        "MULTIPLE FACES - Only one person allowed",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "READY - Press SPACE to capture",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow('Student Registration', display_frame)

                key = cv2.waitKey(1) & 0xFF

                # Capture on SPACE key
                if key == ord(' '):
                    if len(faces) == 1:
                        captured_frame = frame.copy()
                        captured = True
                        logger.info("Photo captured successfully")
                        print("\n✓ Photo captured!")
                    else:
                        print("\n✗ Cannot capture - ensure exactly ONE face is visible")
                        logger.warning("Capture failed - wrong number of faces")

                # Quit on Q key
                elif key == ord('q'):
                    logger.info("Registration cancelled by user")
                    print("\nRegistration cancelled.")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if captured and captured_frame is not None:
            # Save image to local directory
            os.makedirs(REGISTERED_IMAGES_DIR, exist_ok=True)
            image_path = os.path.join(REGISTERED_IMAGES_DIR, f"{candidate_id}.jpg")
            cv2.imwrite(image_path, captured_frame)
            logger.info(f"Registration photo saved: {image_path}")

            # Generate face embedding
            embedding = self._get_face_embedding(captured_frame)

            if embedding is not None:
                # Save to database
                self._save_registration_to_db(
                    candidate_id, name, email, image_path, embedding
                )

                print("\n" + "=" * 70)
                print("✅ REGISTRATION SUCCESSFUL")
                print("=" * 70)
                print(f"Candidate ID: {candidate_id}")
                print(f"Image saved: {image_path}")
                print(f"Database: Updated")
                print("=" * 70 + "\n")
                return True
            else:
                logger.error("Could not extract face embedding from captured photo")
                print("\n✗ Error: Could not extract face features. Please try again.")
                return False

        return False

    def _get_face_embedding(self, frame):
        """Extract face embedding from frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(rgb_frame, 1)

            if len(faces) == 0:
                return None

            face = faces[0]
            shape = self.shape_predictor(rgb_frame, face)
            embedding = self.face_recognizer.compute_face_descriptor(rgb_frame, shape)

            return np.array(embedding)

        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None

    def _save_registration_to_db(self, candidate_id, name, email, image_path, embedding):
        """Save registration info to database"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS student_registration (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) UNIQUE NOT NULL,
                    name VARCHAR(200),
                    email VARCHAR(200),
                    registered_image_path TEXT,
                    face_embedding BYTEA,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Convert embedding to bytes
            embedding_bytes = psycopg2.Binary(embedding.tobytes())

            # Insert or update registration
            cur.execute("""
                INSERT INTO student_registration 
                (candidate_id, name, email, registered_image_path, face_embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (candidate_id) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    registered_image_path = EXCLUDED.registered_image_path,
                    face_embedding = EXCLUDED.face_embedding,
                    registration_date = CURRENT_TIMESTAMP
            """, (candidate_id, name, email, image_path, embedding_bytes))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Registration saved to database for {candidate_id}")

        except Exception as e:
            logger.error(f"Error saving registration to database: {e}")
            raise


# =============================================================================
# Main Menu Interface
# =============================================================================

def main_menu():
    """Interactive menu for registration and monitoring"""

    while True:
        print("\n" + "=" * 70)
        print("AI-BASED EXAM PROCTORING SYSTEM")
        print("Enhanced Face Match Pipeline with Multi-Person Detection")
        print("=" * 70)
        print("\nMAIN MENU:")
        print("  1. Register New Student")
        print("  2. Start Face Match Monitoring")
        print("  3. View Registered Students")
        print("  4. Exit")
        print("=" * 70)

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            # Student Registration
            print("\n" + "-" * 70)
            print("STUDENT REGISTRATION")
            print("-" * 70)

            candidate_id = input("Enter Candidate ID (required): ").strip()

            if not candidate_id:
                print("✗ Error: Candidate ID cannot be empty")
                continue

            name = input("Enter Full Name (optional): ").strip() or None
            email = input("Enter Email (optional): ").strip() or None

            try:
                registration = StudentRegistration()
                success = registration.capture_registration_photo(candidate_id, name, email)

                if success:
                    print("\nStudent registered successfully!")
                else:
                    print("\nRegistration failed. Please try again.")

            except Exception as e:
                logger.error(f"Registration error: {e}")
                print(f"\n✗ Error during registration: {e}")

        elif choice == '2':
            # Start Monitoring
            print("\n" + "-" * 70)
            print("START FACE MATCH MONITORING")
            print("-" * 70)

            candidate_id = input("Enter Candidate ID: ").strip()

            if not candidate_id:
                print("✗ Error: Candidate ID cannot be empty")
                continue

            exam_session_id = input("Enter Exam Session ID: ").strip() or f"exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            try:
                print("\nInitializing monitoring system...")
                pipeline = FaceMatchPipeline(candidate_id, exam_session_id)

                print("\n" + "=" * 70)
                print("MONITORING STARTED")
                print("=" * 70)
                print(f"Candidate: {candidate_id}")
                print(f"Session: {exam_session_id}")
                print("\nPress 'Q' in the video window to stop monitoring")
                print("=" * 70 + "\n")

                pipeline.start_monitoring()

                # Show summary after monitoring stops
                summary = pipeline.get_session_summary()
                if summary:
                    print("\n" + "=" * 70)
                    print("SESSION SUMMARY")
                    print("=" * 70)
                    print(f"Total Checks: {summary.get('total_checks', 0)}")
                    print(f"Face Mismatches: {summary.get('mismatches', 0)}")
                    print(f"Multi-Person Warnings: {pipeline.multi_person_warning_count}")
                    print(f"No Face Detected: {summary.get('no_face', 0)}")
                    print(f"Session Log: {pipeline.log_file_path}")
                    print("=" * 70)

            except FileNotFoundError as e:
                print(f"\n✗ Error: {e}")
                print(f"Please register student '{candidate_id}' first (Option 1)")
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                print(f"\n✗ Error during monitoring: {e}")

        elif choice == '3':
            # View Registered Students
            print("\n" + "-" * 70)
            print("REGISTERED STUDENTS")
            print("-" * 70)

            try:
                conn = psycopg2.connect(**DB_CONN)
                cur = conn.cursor()

                cur.execute("""
                    SELECT candidate_id, name, email, registration_date 
                    FROM student_registration 
                    ORDER BY registration_date DESC
                """)

                students = cur.fetchall()
                cur.close()
                conn.close()

                if students:
                    print(f"\nTotal Registered Students: {len(students)}\n")
                    for i, (cid, name, email, reg_date) in enumerate(students, 1):
                        print(f"{i}. Candidate ID: {cid}")
                        if name:
                            print(f"   Name: {name}")
                        if email:
                            print(f"   Email: {email}")
                        print(f"   Registered: {reg_date}")
                        print()
                else:
                    print("\nNo students registered yet.")

            except Exception as e:
                logger.error(f"Error fetching students: {e}")
                print(f"\n✗ Error: {e}")

        elif choice == '4':
            # Exit
            print("\n" + "=" * 70)
            print("Thank you for using the AI-Based Exam Proctoring System")
            print("=" * 70 + "\n")
            break

        else:
            print("\n✗ Invalid choice. Please enter 1, 2, 3, or 4.")


# Usage
if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n✗ Fatal error: {e}")