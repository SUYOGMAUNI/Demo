import numpy as np
import cv2
from scipy.io import loadmat
from pathlib import Path
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedMPIIGazePreprocessor:
    """
    Improved preprocessor that:
    1. Uses annotation.txt for better face extraction
    2. Loads camera calibration data
    3. Properly extracts face regions from original images
    4. Uses manual annotations when available
    """

    def __init__(self, data_dir, output_dir, image_size=224, use_annotations=True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.use_annotations = use_annotations
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.original_dir = self.data_dir / "Original"
        self.normalized_dir = self.data_dir / "Normalized"
        self.annotation_subset_dir = self.data_dir.parent / "Annotation Subset"
        
        # Failure tracking
        self.failure_reasons = defaultdict(int)
        self.stats = defaultdict(int)

        # Validate directories
        if not self.normalized_dir.exists():
            raise FileNotFoundError(f"Normalized directory not found: {self.normalized_dir}")
        if not self.original_dir.exists():
            raise FileNotFoundError(f"Original directory not found: {self.original_dir}")

        # Load manual annotations if available
        self.manual_annotations = self._load_manual_annotations()
        
        logger.info("=" * 80)
        logger.info("IMPROVED MPIIGAZE PREPROCESSOR")
        logger.info("=" * 80)
        logger.info(f"Data directory:    {self.data_dir}")
        logger.info(f"Output directory:  {self.output_dir}")
        logger.info(f"Image size:        {self.image_size}×{self.image_size}")
        logger.info(f"Use annotations:   {self.use_annotations}")
        logger.info(f"Manual annotations: {len(self.manual_annotations)} samples")
        logger.info("=" * 80)

    def _load_manual_annotations(self):
        """Load manual annotations from Annotation Subset"""
        manual_annots = {}
        
        if not self.annotation_subset_dir.exists():
            logger.warning("Annotation Subset directory not found")
            return manual_annots
        
        try:
            # Look for annotation files
            for annot_file in self.annotation_subset_dir.glob("*.txt"):
                with open(annot_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # Format: person_id day_name image_idx ...
                            key = f"{parts[0]}_{parts[1]}_{parts[2]}"
                            # Store facial landmarks and pupil centers
                            manual_annots[key] = {
                                'landmarks': [float(x) for x in parts[3:15]],  # 6 landmarks (x,y)
                                'pupils': [float(x) for x in parts[15:19]]  # 2 pupils (x,y)
                            }
            
            logger.info(f"Loaded {len(manual_annots)} manual annotations")
        except Exception as e:
            logger.warning(f"Could not load manual annotations: {e}")
        
        return manual_annots

    def _load_camera_calibration(self, person_dir):
        """Load camera calibration parameters"""
        calib_dir = person_dir / "Calibration"
        calibration = {}
        
        try:
            # Load camera intrinsics
            camera_file = calib_dir / "Camera.mat"
            if camera_file.exists():
                cam_data = loadmat(str(camera_file))
                calibration['camera_matrix'] = cam_data.get('cameraMatrix', None)
                calibration['dist_coeffs'] = cam_data.get('distCoeffs', None)
            
            # Load monitor pose
            monitor_file = calib_dir / "monitorPose.mat"
            if monitor_file.exists():
                monitor_data = loadmat(str(monitor_file))
                calibration['monitor_rvecs'] = monitor_data.get('rvecs', None)
                calibration['monitor_tvecs'] = monitor_data.get('tvecs', None)
            
            # Load screen size
            screen_file = calib_dir / "screenSize.mat"
            if screen_file.exists():
                screen_data = loadmat(str(screen_file))
                calibration['screen_size'] = {
                    'width_pixel': screen_data.get('width_pixel', [[0]])[0][0],
                    'height_pixel': screen_data.get('height_pixel', [[0]])[0][0],
                    'width_mm': screen_data.get('width_mm', [[0]])[0][0],
                    'height_mm': screen_data.get('height_mm', [[0]])[0][0]
                }
        except Exception as e:
            logger.debug(f"Could not load calibration: {e}")
        
        return calibration if calibration else None

    def process_dataset(self):
        """Main processing pipeline"""
        logger.info("\nStarting preprocessing...\n")

        all_samples = []
        failed_samples = 0
        total_attempted = 0

        person_dirs = sorted([d for d in self.normalized_dir.iterdir()
                              if d.is_dir() and d.name.startswith('p')])

        if len(person_dirs) == 0:
            raise ValueError(f"No person directories found in {self.normalized_dir}")

        logger.info(f"Found {len(person_dirs)} persons\n")

        for person_idx, person_dir in enumerate(person_dirs, 1):
            person_id = person_dir.name

            logger.info(f"[{person_idx}/{len(person_dirs)}] Processing {person_id}...")
            start_time = time.time()

            try:
                # Load calibration for this person
                original_person_dir = self.original_dir / person_id
                calibration = self._load_camera_calibration(original_person_dir)
                
                samples, failed, attempted = self._process_person(
                    person_dir, person_id, calibration
                )
                all_samples.extend(samples)
                failed_samples += failed
                total_attempted += attempted

                elapsed = time.time() - start_time
                success_rate = 100 * len(samples) / attempted if attempted > 0 else 0

                logger.info(
                    f"  ✓ {person_id}: {len(samples):4d} valid / {attempted:4d} total "
                    f"({success_rate:.1f}% success) - {elapsed:.1f}s"
                )

            except Exception as e:
                logger.error(f"  ✗ {person_id}: Error - {e}")
                continue

        self._save_metadata(all_samples)
        self._print_statistics(total_attempted, len(all_samples))

        logger.info("\n" + "=" * 80)
        logger.info("✅ PREPROCESSING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total attempted: {total_attempted}")
        logger.info(f"Valid samples:   {len(all_samples)} ({100 * len(all_samples) / total_attempted:.1f}%)")
        logger.info(f"Failed samples:  {failed_samples} ({100 * failed_samples / total_attempted:.1f}%)")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)

        return all_samples

    def _process_person(self, person_dir, person_id, calibration):
        """Process all days for one person"""
        mat_files = sorted(person_dir.glob('*.mat'))

        if len(mat_files) == 0:
            logger.warning(f"  ⚠️  {person_id}: No .mat files found")
            return [], 0, 0

        samples = []
        failed = 0
        attempted = 0

        original_person_dir = self.original_dir / person_id

        for mat_file in tqdm(mat_files, desc=f"  {person_id} files", leave=False):
            try:
                day_name = mat_file.stem
                
                # Load annotation.txt for this day
                annotations = self._load_day_annotations(
                    original_person_dir / day_name / "annotation.txt"
                )
                
                s, f, a = self._process_mat_file(
                    mat_file, person_id, day_name, annotations, calibration
                )
                samples.extend(s)
                failed += f
                attempted += a
            except Exception as e:
                logger.debug(f"    Error in {mat_file.name}: {e}")
                continue

        return samples, failed, attempted

    def _load_day_annotations(self, annotation_file):
        """
        Load annotation.txt for a specific day
        Format per line (44 dimensions):
        - 1-24: Eye landmarks (24 values)
        - 25-26: On-screen gaze target (x, y)
        - 27-29: 3D gaze target
        - 30-35: 3D head pose (rotation + translation)
        - 36-38: 3D right eye center
        - 39-41: 3D left eye center
        """
        annotations = []
        
        if not annotation_file.exists():
            return annotations
        
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    if len(values) >= 41:
                        annotations.append({
                            'eye_landmarks': np.array(values[0:24]).reshape(12, 2),
                            'gaze_target_2d': np.array(values[24:26]),
                            'gaze_target_3d': np.array(values[26:29]),
                            'head_pose_rot': np.array(values[29:32]),
                            'head_pose_trans': np.array(values[32:35]),
                            'right_eye_3d': np.array(values[35:38]),
                            'left_eye_3d': np.array(values[38:41])
                        })
        except Exception as e:
            logger.debug(f"Could not load annotations from {annotation_file}: {e}")
        
        return annotations

    def _process_mat_file(self, mat_file, person_id, day_name, annotations, calibration):
        """Process single .mat file with annotations"""
        try:
            data = loadmat(str(mat_file))
        except Exception:
            self.failure_reasons['mat_load_error'] += 1
            return [], 0, 0

        try:
            left_data = data['data'][0, 0]['left'][0, 0]
        except (KeyError, IndexError):
            try:
                left_data = data['data']['left'][0, 0]
            except Exception:
                self.failure_reasons['mat_structure_error'] += 1
                return [], 0, 0

        try:
            images = left_data['image']
            gazes = left_data['gaze']
            poses = left_data['pose']
        except (KeyError, ValueError):
            self.failure_reasons['mat_data_missing'] += 1
            return [], 0, 0

        # Load original images for this day
        original_day_dir = self.original_dir / person_id / day_name
        
        num_samples = len(images)
        samples = []
        failed = 0

        for i in range(num_samples):
            try:
                # Get annotation if available
                annotation = annotations[i] if i < len(annotations) else None
                
                # Load original image
                original_img = self._load_original_image(original_day_dir, i)
                
                sample = self._process_sample(
                    images[i], gazes[i], poses[i],
                    person_id, day_name, i,
                    original_img, annotation, calibration
                )

                if sample:
                    samples.append(sample)
                    self.stats['successful'] += 1
                else:
                    failed += 1

            except Exception as e:
                self.failure_reasons['processing_exception'] += 1
                failed += 1

        return samples, failed, num_samples

    def _load_original_image(self, day_dir, idx):
        """Load original image for given index"""
        if not day_dir.exists():
            return None
        
        # Original images are typically named as frame_XXXX_face.jpg or similar
        # Try different naming patterns
        patterns = [
            f"{idx:04d}.jpg",
            f"frame_{idx:04d}.jpg",
            f"frame_{idx:04d}_face.jpg",
            f"{idx}.jpg"
        ]
        
        for pattern in patterns:
            img_path = day_dir / pattern
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue
        
        return None

    def _process_sample(self, image, gaze, pose, person_id, day_name, idx,
                       original_image=None, annotation=None, calibration=None):
        """Process single sample with improved face extraction"""

        # Convert normalized eye image to uint8
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Validate size
        if image.shape[0] < 20 or image.shape[1] < 40:
            self.failure_reasons['image_too_small'] += 1
            return None

        # Split eyes from normalized image
        h, w = image.shape[:2]
        mid = w // 2
        left_eye = image[:, mid:].copy()
        right_eye = image[:, :mid].copy()

        if left_eye.shape[0] < 10 or left_eye.shape[1] < 10:
            self.failure_reasons['left_eye_too_small'] += 1
            return None
        if right_eye.shape[0] < 10 or right_eye.shape[1] < 10:
            self.failure_reasons['right_eye_too_small'] += 1
            return None

        # Resize eyes
        try:
            left_eye = cv2.resize(left_eye, (self.image_size, self.image_size))
            right_eye = cv2.resize(right_eye, (self.image_size, self.image_size))
        except Exception:
            self.failure_reasons['resize_failed'] += 1
            return None

        # Extract face using annotations if available
        if original_image is not None and annotation is not None:
            face = self._extract_face_with_annotations(original_image, annotation)
            self.stats['face_from_annotations'] += 1
        elif original_image is not None:
            face = self._extract_face_from_original(original_image)
            self.stats['face_from_original'] += 1
        else:
            face = self._extract_face_fallback(image)
            self.stats['face_from_fallback'] += 1

        if face is None:
            self.failure_reasons['face_extraction_failed'] += 1
            return None

        try:
            face = cv2.resize(face, (self.image_size, self.image_size))
        except Exception:
            self.failure_reasons['face_resize_failed'] += 1
            return None

        # Convert gaze with screen calibration if available
        gaze_angles = self._validate_gaze(gaze)
        if gaze_angles is None:
            self.failure_reasons['invalid_gaze'] += 1
            return None

        # OPTIONAL: compute screen coords for metadata
        screen_gaze = None
        if annotation is not None and calibration is not None:
            screen_gaze = self._gaze_from_annotation(annotation, calibration)
            self.stats['screen_coords_computed'] += 1

        # Validate pose
        pose_2d = self._validate_pose(pose)
        if pose_2d is None:
            self.failure_reasons['invalid_pose'] += 1
            return None

        # Quality check
        quality_result = self._quality_check(left_eye, right_eye, face)
        if quality_result != 'pass':
            self.failure_reasons[f'quality_{quality_result}'] += 1
            return None

        # Save files
        save_dir = self.output_dir / person_id / day_name
        save_dir.mkdir(parents=True, exist_ok=True)

        sample_id = f"{person_id}_{day_name}_{idx:04d}"

        try:
            left_path = save_dir / f"{sample_id}_left.jpg"
            right_path = save_dir / f"{sample_id}_right.jpg"
            face_path = save_dir / f"{sample_id}_face.jpg"

            cv2.imwrite(str(left_path), cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(right_path), cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(face_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            sample_data = {
                'sample_id': sample_id,
                'person_id': person_id,
                'day': day_name,
                'left_eye_path': str(left_path.relative_to(self.output_dir)),
                'right_eye_path': str(right_path.relative_to(self.output_dir)),
                'face_path': str(face_path.relative_to(self.output_dir)),
                'gaze': gaze_angles.tolist(),  # PRIMARY: radians [pitch, yaw]
                'head_pose': pose_2d.tolist(),
                'screen_gaze': screen_gaze.tolist() if screen_gaze is not None else None,  # Optional
            }
            
            # Add annotation data if available
            if annotation is not None:
                sample_data['has_annotation'] = True
                sample_data['gaze_target_2d'] = annotation['gaze_target_2d'].tolist()
            
            return sample_data

        except Exception:
            self.failure_reasons['file_save_failed'] += 1
            return None

    def _extract_face_with_annotations(self, original_image, annotation):
        """Extract face using eye landmarks from annotations"""
        try:
            # Get eye landmarks (4 eye corners)
            eye_landmarks = annotation['eye_landmarks']
            
            # Calculate bounding box from eye landmarks with margin
            x_coords = eye_landmarks[:, 0]
            y_coords = eye_landmarks[:, 1]
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Add margin for full face (eyes are typically in upper 1/3 of face)
            eye_width = x_max - x_min
            eye_height = y_max - y_min
            
            # Expand to full face size
            face_width = eye_width * 2.5
            face_height = eye_height * 3.5
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Adjust center slightly down from eyes
            center_y += eye_height * 1.2
            
            x1 = int(max(0, center_x - face_width / 2))
            x2 = int(min(original_image.shape[1], center_x + face_width / 2))
            y1 = int(max(0, center_y - face_height / 2))
            y2 = int(min(original_image.shape[0], center_y + face_height / 2))
            
            face = original_image[y1:y2, x1:x2].copy()
            
            if face.shape[0] < 50 or face.shape[1] < 50:
                return None
            
            return face
            
        except Exception:
            return None

    def _extract_face_from_original(self, original_image):
        """Extract face from original image (center crop)"""
        try:
            h, w = original_image.shape[:2]
            
            # Take center 70% as face region
            face_size = int(min(h, w) * 0.7)
            center_y, center_x = h // 2, w // 2

            y1 = max(0, center_y - face_size // 2)
            y2 = min(h, center_y + face_size // 2)
            x1 = max(0, center_x - face_size // 2)
            x2 = min(w, center_x + face_size // 2)

            face = original_image[y1:y2, x1:x2].copy()

            if face.shape[0] < 50 or face.shape[1] < 50:
                return None

            return face

        except Exception:
            return None

    def _extract_face_fallback(self, normalized_image):
        """Fallback: use normalized image as face"""
        try:
            return normalized_image.copy()
        except Exception:
            return None

    def _gaze_from_annotation(self, annotation, calibration):
        """Convert gaze from annotation (more accurate than angles)"""
        try:
            gaze_2d = annotation['gaze_target_2d']
            
            # Normalize to [0, 1] using screen size
            if calibration and 'screen_size' in calibration:
                screen_w = calibration['screen_size']['width_pixel']
                screen_h = calibration['screen_size']['height_pixel']
                
                if screen_w > 0 and screen_h > 0:
                    x = gaze_2d[0] / screen_w
                    y = gaze_2d[1] / screen_h
                    
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    
                    return np.array([x, y], dtype=np.float32)
            
            # Fallback: assume already normalized or use simple normalization
            return np.array(gaze_2d, dtype=np.float32)
            
        except Exception:
            return None

    def _gaze_to_screen_coords(self, gaze_angles):
        """
        DEPRECATED: Do not use for training labels!

        This method attempts to convert camera-space angular gaze to screen
        coordinates, but MPIIGaze does not provide the necessary calibration
        parameters for accurate conversion. Use annotation-based screen coords
        when available, or keep gaze in angular form for training.

        This method is kept only for legacy compatibility.
        """
        logger.warning(
            "DEPRECATED: _gaze_to_screen_coords called. "
            "Screen coordinate conversion should not be used for MPIIGaze training. "
            "Gaze labels should remain in angular form (radians)."
        )
        return None

    def _validate_pose(self, head_pose):
        """
        Validate head pose - returns [pitch, yaw] to match gaze format

        MPIIGaze .mat files store pose as [pitch, yaw] in radians
        We maintain this order for consistency with gaze labels
        """
        try:
            pitch = float(head_pose[0])
            yaw = float(head_pose[1])

            if np.isnan(pitch) or np.isnan(yaw):
                return None
            if np.isinf(pitch) or np.isinf(yaw):
                return None

            return np.array([pitch, yaw], dtype=np.float32)

        except Exception:
            return None

        # ADD NEW METHOD:
    def _validate_gaze(self, gaze_angles):
        """Validate gaze angles - keep in radians"""
        try:
            pitch = float(gaze_angles[0])
            yaw = float(gaze_angles[1])

            if np.isnan(pitch) or np.isnan(yaw):
                return None
            if np.isinf(pitch) or np.isinf(yaw):
                return None

            return np.array([pitch, yaw], dtype=np.float32)
        except Exception:
            return None

    def _quality_check(self, left_eye, right_eye, face):
        """Quality check for images"""
        for img, name in [(left_eye, 'left_eye'), (right_eye, 'right_eye'), (face, 'face')]:
            if img is None or img.size == 0:
                return f'{name}_null'

            brightness = np.mean(img)
            if brightness < 5:
                return f'{name}_too_dark'
            if brightness > 250:
                return f'{name}_too_bright'

            if np.std(img) < 2:
                return f'{name}_low_variance'

        return 'pass'

    def _print_statistics(self, total_attempted, valid_samples):
        """Print statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 PROCESSING STATISTICS")
        logger.info("=" * 80)
        
        if self.stats:
            logger.info("\nData sources:")
            for key, value in sorted(self.stats.items()):
                logger.info(f"  {key:30s}: {value:6d}")
        
        if self.failure_reasons:
            logger.info("\nFailure breakdown:")
            total_failures = sum(self.failure_reasons.values())
            for reason, count in sorted(self.failure_reasons.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                percentage = 100 * count / total_failures
                logger.info(f"  {reason:30s}: {count:6d} ({percentage:5.1f}%)")
        
        logger.info("=" * 80)

    def _save_metadata(self, samples):
        """Save metadata"""
        metadata = {
            'total_samples': len(samples),
            'image_size': self.image_size,
            'has_face_images': True,
            'has_annotations': self.use_annotations,

            # Coordinate system documentation
            'coordinate_system': 'angles',
            'gaze_format': 'radians',
            'gaze_description': '[pitch, yaw] in radians',
            'head_pose_format': 'radians',
            'head_pose_description': '[pitch, yaw] in radians - matches gaze format',
            'label_order': 'pitch_first',

            # Version and fixes
            'preprocessing_version': '6.2_corrected_pose_order',  # SINGLE VERSION
            'critical_fix': 'Head pose order corrected to [pitch, yaw]',

            # Statistics
            'statistics': dict(self.stats),
            'failure_breakdown': dict(self.failure_reasons),

            # Data
            'samples': samples
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n💾 Metadata saved: {metadata_path}")

        # Validation warning
        logger.info("\n" + "⚠️ " * 20)
        logger.info("CRITICAL: Verify label consistency")
        logger.info("=" * 80)
        logger.info("Both gaze and head_pose use [pitch, yaw] order in radians")
        logger.info("  - gaze[0] = pitch (vertical angle)")
        logger.info("  - gaze[1] = yaw (horizontal angle)")
        logger.info("  - head_pose[0] = pitch")
        logger.info("  - head_pose[1] = yaw")
        logger.info("Training script MUST use same order!")
        logger.info("=" * 80)


def split_by_person(metadata_path, output_dir, val_size=0.1, test_size=0.15):
    """Split dataset by person"""
    logger.info("\n" + "=" * 80)
    logger.info("PERSON-BASED DATASET SPLITTING")
    logger.info("=" * 80)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    samples = metadata['samples']
    person_samples = defaultdict(list)

    for sample in samples:
        person_samples[sample['person_id']].append(sample)

    persons = sorted(person_samples.keys())

    logger.info(f"\nTotal persons: {len(persons)}")
    logger.info(f"Total samples: {len(samples)}")

    train_persons, temp = train_test_split(
        persons, test_size=val_size + test_size, random_state=42
    )
    val_persons, test_persons = train_test_split(
        temp, test_size=test_size / (val_size + test_size), random_state=42
    )

    train_samples = []
    val_samples = []
    test_samples = []

    for person in train_persons:
        train_samples.extend(person_samples[person])
    for person in val_persons:
        val_samples.extend(person_samples[person])
    for person in test_persons:
        test_samples.extend(person_samples[person])

    logger.info("\nSplit results:")
    logger.info(f"  Train: {len(train_samples):5d} samples from {len(train_persons):2d} persons")
    logger.info(f"  Val:   {len(val_samples):5d} samples from {len(val_persons):2d} persons")
    logger.info(f"  Test:  {len(test_samples):5d} samples from {len(test_persons):2d} persons")

    output_dir = Path(output_dir)

    for split_name, split_samples, split_persons in [
        ('train', train_samples, train_persons),
        ('val', val_samples, val_persons),
        ('test', test_samples, test_persons)
    ]:
        split_metadata = {
            'total_samples': len(split_samples),
            'num_persons': len(split_persons),
            'person_ids': sorted(split_persons),
            'image_size': metadata['image_size'],
            'has_face_images': metadata.get('has_face_images', False),
            'samples': split_samples
        }

        split_path = output_dir / f'{split_name}_metadata.json'
        with open(split_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)

        logger.info(f"  💾 Saved: {split_path}")

    logger.info("=" * 80)
    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(
        description='Improved MPIIGaze Preprocessing with Annotations'
    )

    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./MPIIGaze/Data',
        help='Path to MPIIGaze/Data directory'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./improved_gaze_data',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--image_size', 
        type=int, 
        default=224,
        help='Output image size'
    )
    parser.add_argument(
        '--no_annotations',
        action='store_true',
        help='Disable use of annotation files'
    )

    args = parser.parse_args()

    logger.info("\n🔍 IMPROVED MPIIGAZE PREPROCESSING\n")

    preprocessor = ImprovedMPIIGazePreprocessor(
        args.data_dir,
        args.output_dir,
        args.image_size,
        use_annotations=not args.no_annotations
    )

    samples = preprocessor.process_dataset()

    if len(samples) == 0:
        logger.error("\n❌ No valid samples generated!")
        return 1

    logger.info("\n📊 STEP 2: DATASET SPLITTING\n")

    split_by_person(
        Path(args.output_dir) / 'metadata.json',
        args.output_dir
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL DONE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info("Check metadata.json for detailed statistics")
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}")
        logger.exception("Full traceback:")
        exit(1)
