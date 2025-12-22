"""
Enhanced Audio Pipeline for AI-Based Exam Proctoring System
Integrates with your existing code and adds warning tracking,
database schema, and better error handling
"""

import os
import time
import wave
import numpy as np
import pyaudio
import psycopg2
from datetime import datetime
import threading
import queue
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Audio and Detection Parameters ---
SAMPLE_RATE = 16000       # Hz
CHUNK_SIZE = 1024         # samples per frame
THRESHOLD = 0.01          # Amplitude threshold (adjust as needed)
TRIGGER_WINDOW = 0.5      # seconds above threshold to trigger capture
CAPTURE_DURATION = 30     # seconds to record after trigger

# Warning system parameters
WARNING_THRESHOLD = 3     # Number of warnings before escalation
COOLDOWN_PERIOD = 60      # Seconds before warnings can trigger again

# --- PostgreSQL connection params ---
DB_CONN = {
    'host': 'localhost',
    'database': 'Proctoring',
    'user': 'postgres',
    'password': '526183'
}

# --- Local storage parameters ---
LOCAL_AUDIO_DIR = r"D:\CapturedAudio"
os.makedirs(LOCAL_AUDIO_DIR, exist_ok=True)

# --- Local text logs directory ---
LOGS_DIR = r"D:\logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class AudioPipeline:
    """
    Enhanced Audio Pipeline with warning tracking and session management
    Matches your documentation's Parallel Audio Pipeline architecture
    """

    def __init__(self, candidate_id, exam_session_id=None):
        self.candidate_id = candidate_id
        self.exam_session_id = exam_session_id
        self.warning_count = 0
        self.is_monitoring = False
        self.last_trigger_time = None

        # Initialize text log file
        self.log_file_path = os.path.join(
            LOGS_DIR,
            f"audio_{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self._write_log("="*70)
        self._write_log("AUDIO PIPELINE - SESSION STARTED")
        self._write_log(f"Candidate ID: {candidate_id}")
        self._write_log(f"Exam Session ID: {exam_session_id}")
        self._write_log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log("="*70 + "\n")

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()

        # Initialize database
        self._initialize_database()

    def _initialize_database(self):
        """Create necessary database tables if they don't exist"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            # Create suspicious_audio_events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS suspicious_audio_events (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) NOT NULL,
                    exam_session_id VARCHAR(100),
                    timestamp TIMESTAMP NOT NULL,
                    audio_data BYTEA,
                    audio_file_path TEXT,
                    amplitude_peak FLOAT,
                    duration_seconds INTEGER,
                    warning_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create audio_warnings table for tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audio_warnings (
                    id SERIAL PRIMARY KEY,
                    candidate_id VARCHAR(100) NOT NULL,
                    exam_session_id VARCHAR(100),
                    warning_count INTEGER,
                    last_warning_time TIMESTAMP,
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            cur.close()
            conn.close()
            logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def _write_log(self, message):
        """Write message to text log file"""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")

    def get_amplitude(self, segment):
        """Calculate max normalized amplitude of audio chunk."""
        audio_np = segment.astype(np.float32) / np.iinfo(np.int16).max
        return np.max(np.abs(audio_np))

    def capture_segment(self, stream, duration):
        """Capture and return audio segment from stream for given duration."""
        frames = []
        num_chunks = int(SAMPLE_RATE / CHUNK_SIZE * duration)

        for _ in range(num_chunks):
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                logger.warning(f"Error reading audio chunk: {e}")
                continue

        if frames:
            segment = np.concatenate(frames)
            return segment
        return None

    def save_audio_locally(self, audio_np_segment, timestamp, amplitude_peak):
        """Save audio numpy segment as a WAV file."""
        filename = f"{self.candidate_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(LOCAL_AUDIO_DIR, filename)

        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_np_segment.tobytes())

            logger.info(f"Saved audio locally: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving audio locally: {e}")
            return None

    def store_audio_in_db(self, audio_bytes, filepath, timestamp, amplitude_peak):
        """Store audio information in PostgreSQL database and log to file"""

        # Write to text log
        self.warning_count += 1
        log_message = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
        log_message += f"SUSPICIOUS AUDIO DETECTED | WARNING #{self.warning_count}"
        log_message += f" | Peak Amplitude: {amplitude_peak:.4f}"
        log_message += f" | Duration: {CAPTURE_DURATION}s"
        if filepath:
            log_message += f" | Audio File: {filepath}"

        self._write_log(log_message)

        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            # Insert audio event (warning_count already incremented above)
            cur.execute("""
                INSERT INTO suspicious_audio_events 
                (candidate_id, exam_session_id, timestamp, audio_data, audio_file_path, 
                 amplitude_peak, duration_seconds, warning_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                self.candidate_id,
                self.exam_session_id,
                timestamp,
                psycopg2.Binary(audio_bytes),
                filepath,
                float(amplitude_peak),
                CAPTURE_DURATION,
                self.warning_count
            ))

            event_id = cur.fetchone()[0]

            # Update warning count
            cur.execute("""
                INSERT INTO audio_warnings 
                (candidate_id, exam_session_id, warning_count, last_warning_time, status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (candidate_id, exam_session_id) 
                DO UPDATE SET 
                    warning_count = audio_warnings.warning_count + 1,
                    last_warning_time = %s,
                    status = CASE 
                        WHEN audio_warnings.warning_count + 1 >= %s THEN 'CRITICAL'
                        ELSE 'ACTIVE'
                    END
            """, (
                self.candidate_id,
                self.exam_session_id,
                self.warning_count,
                timestamp,
                'ACTIVE',
                timestamp,
                WARNING_THRESHOLD
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Stored audio event #{event_id} in database")
            return event_id

        except Exception as e:
            logger.error(f"Error storing audio in DB: {e}")
            return None

    def check_warning_threshold(self):
        """Check if warning threshold exceeded (N warnings)"""
        if self.warning_count >= WARNING_THRESHOLD:
            alert_msg = (
                f"⚠️ WARNING THRESHOLD EXCEEDED for {self.candidate_id}: "
                f"{self.warning_count} warnings!"
            )
            logger.critical(alert_msg)
            self._write_log("\n" + "!"*70)
            self._write_log(alert_msg)
            self._write_log("!"*70 + "\n")
            self.raise_critical_alert()
            return True
        return False

    def raise_critical_alert(self):
        """Raise critical alert when threshold exceeded"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            # Log critical alert
            cur.execute("""
                INSERT INTO critical_alerts 
                (candidate_id, exam_session_id, alert_type, alert_message, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                self.candidate_id,
                self.exam_session_id,
                'AUDIO_VIOLATION',
                f'Multiple suspicious audio events detected: {self.warning_count} warnings',
                datetime.utcnow()
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.critical(f"Critical alert raised for {self.candidate_id}")

        except Exception as e:
            logger.error(f"Error raising critical alert: {e}")

    def is_cooldown_active(self):
        """Check if in cooldown period after last trigger"""
        if self.last_trigger_time is None:
            return False

        elapsed = (datetime.utcnow() - self.last_trigger_time).total_seconds()
        return elapsed < COOLDOWN_PERIOD

    def start_monitoring(self):
        """Start the audio monitoring pipeline"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )

            self.is_monitoring = True
            logger.info(f"Audio monitoring started for candidate: {self.candidate_id}")

            self._monitoring_loop()

        except Exception as e:
            logger.error(f"Error starting audio monitoring: {e}")
            self.is_monitoring = False

    def _monitoring_loop(self):
        """Main monitoring loop - matches your documentation's continuous monitoring"""
        trigger_counter = 0
        sustained_trig_needed = int(SAMPLE_RATE / CHUNK_SIZE * TRIGGER_WINDOW)
        peak_amplitude = 0

        try:
            while self.is_monitoring:
                # Read audio chunk
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                segment = np.frombuffer(data, dtype=np.int16)
                amplitude = self.get_amplitude(segment)

                # Track peak amplitude during trigger window
                if amplitude > THRESHOLD:
                    trigger_counter += 1
                    peak_amplitude = max(peak_amplitude, amplitude)
                    logger.debug(
                        f"Trigger +1 ({trigger_counter}/{sustained_trig_needed}), "
                        f"amplitude={amplitude:.3f}"
                    )
                else:
                    if trigger_counter > 0:
                        logger.debug(f"Trigger reset (was {trigger_counter})")
                    trigger_counter = 0
                    peak_amplitude = 0

                # Sustained trigger detected
                if trigger_counter >= sustained_trig_needed:
                    # Check cooldown
                    if self.is_cooldown_active():
                        logger.info("Cooldown active, skipping capture")
                        trigger_counter = 0
                        continue

                    logger.warning("** Suspicious audio detected! Capturing segment... **")

                    # Capture 30-second segment
                    captured_segment = self.capture_segment(self.stream, CAPTURE_DURATION)

                    if captured_segment is not None:
                        audio_bytes = captured_segment.tobytes()
                        timestamp = datetime.utcnow()

                        # Save locally
                        filepath = self.save_audio_locally(
                            captured_segment, timestamp, peak_amplitude
                        )

                        # Save to database
                        self.store_audio_in_db(
                            audio_bytes, filepath, timestamp, peak_amplitude
                        )

                        logger.warning(
                            f"Audio segment captured at {timestamp} "
                            f"(Warning #{self.warning_count})"
                        )

                        # Update last trigger time
                        self.last_trigger_time = timestamp

                        # Check warning threshold
                        self.check_warning_threshold()

                    trigger_counter = 0
                    peak_amplitude = 0

                time.sleep(CHUNK_SIZE / SAMPLE_RATE)

        except KeyboardInterrupt:
            logger.info("Audio monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop the audio monitoring pipeline"""
        self.is_monitoring = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()

        # Write session summary to log
        self._write_log("\n" + "="*70)
        self._write_log("AUDIO PIPELINE - SESSION ENDED")
        self._write_log(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"Total Warnings: {self.warning_count}")
        self._write_log("="*70)

        logger.info(f"Audio monitoring stopped for candidate: {self.candidate_id}")
        logger.info(f"Session log saved to: {self.log_file_path}")

    def get_session_summary(self):
        """Get summary of audio events for this session"""
        try:
            conn = psycopg2.connect(**DB_CONN)
            cur = conn.cursor()

            cur.execute("""
                SELECT 
                    COUNT(*) as total_events,
                    MAX(amplitude_peak) as max_amplitude,
                    MIN(timestamp) as first_event,
                    MAX(timestamp) as last_event
                FROM suspicious_audio_events
                WHERE candidate_id = %s AND exam_session_id = %s
            """, (self.candidate_id, self.exam_session_id))

            result = cur.fetchone()
            cur.close()
            conn.close()

            return {
                'total_events': result[0] or 0,
                'max_amplitude': result[1],
                'first_event': result[2],
                'last_event': result[3],
                'warning_count': self.warning_count
            }

        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return None


# =============================================================================
# Integration with Main Proctoring System
# =============================================================================

class IntegratedProctoringSystem:
    """
    Integrated system combining Audio, Webcam, and Face Match pipelines
    """

    def __init__(self, candidate_id, exam_session_id):
        self.candidate_id = candidate_id
        self.exam_session_id = exam_session_id

        # Initialize pipelines
        self.audio_pipeline = AudioPipeline(candidate_id, exam_session_id)
        # self.webcam_pipeline = WebcamPipeline(candidate_id, exam_session_id)  # Your code
        # self.face_match_pipeline = FaceMatchPipeline(candidate_id, exam_session_id)  # Your code

        self.monitoring_threads = []

    def start_proctoring(self):
        """Start all monitoring pipelines in parallel"""
        logger.info(f"Starting proctoring for candidate: {self.candidate_id}")

        # Start audio pipeline in separate thread
        audio_thread = threading.Thread(
            target=self.audio_pipeline.start_monitoring,
            daemon=True
        )
        audio_thread.start()
        self.monitoring_threads.append(audio_thread)

        # TODO: Start webcam pipeline
        # webcam_thread = threading.Thread(
        #     target=self.webcam_pipeline.start_monitoring,
        #     daemon=True
        # )
        # webcam_thread.start()
        # self.monitoring_threads.append(webcam_thread)

        # TODO: Start face match pipeline
        # face_thread = threading.Thread(
        #     target=self.face_match_pipeline.start_monitoring,
        #     daemon=True
        # )
        # face_thread.start()
        # self.monitoring_threads.append(face_thread)

        logger.info("All monitoring pipelines started")

    def stop_proctoring(self):
        """Stop all monitoring pipelines"""
        logger.info("Stopping all pipelines...")

        self.audio_pipeline.stop_monitoring()
        # self.webcam_pipeline.stop_monitoring()
        # self.face_match_pipeline.stop_monitoring()

        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5)

        logger.info("All pipelines stopped")

    def generate_final_report(self):
        """Generate comprehensive proctoring report"""
        audio_summary = self.audio_pipeline.get_session_summary()

        report = {
            'candidate_id': self.candidate_id,
            'exam_session_id': self.exam_session_id,
            'audio_violations': audio_summary,
            # 'webcam_violations': self.webcam_pipeline.get_summary(),
            # 'face_match_violations': self.face_match_pipeline.get_summary(),
            'timestamp': datetime.utcnow().isoformat()
        }

        return report


# =============================================================================
# Usage Examples
# =============================================================================

def main():
    """Example usage of enhanced audio pipeline"""

    # Single pipeline usage
    print("="*60)
    print("Example 1: Standalone Audio Pipeline")
    print("="*60)

    pipeline = AudioPipeline(
        candidate_id="candidate_123",
        exam_session_id="exam_2025_01"
    )

    try:
        pipeline.start_monitoring()
    except KeyboardInterrupt:
        pipeline.stop_monitoring()

    # Get summary
    summary = pipeline.get_session_summary()
    print("\nSession Summary:")
    print(f"  Total Events: {summary['total_events']}")
    print(f"  Max Amplitude: {summary['max_amplitude']}")
    print(f"  Warning Count: {summary['warning_count']}")

    print("\n" + "="*60)
    print("Example 2: Integrated Proctoring System")
    print("="*60)

    # Integrated system usage
    system = IntegratedProctoringSystem(
        candidate_id="candidate_456",
        exam_session_id="exam_2025_02"
    )

    try:
        system.start_proctoring()

        # Keep monitoring until Ctrl+C
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        system.stop_proctoring()

        # Generate final report
        report = system.generate_final_report()
        print("\nFinal Report:")
        print(report)


if __name__ == "__main__":
    main()