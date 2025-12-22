"""
FIXED Diagnostic MPIIGaze Preprocessing
========================================
Fixed version with proper validation (no strict pose range checks)
"""

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


class DiagnosticMPIIGazePreprocessor:
    """Preprocessor with failure diagnostics"""

    def __init__(self, original_data_dir, normalized_data_dir, output_dir, image_size=224):
        self.original_dir = Path(original_data_dir)
        self.normalized_dir = Path(normalized_data_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Failure tracking
        self.failure_reasons = defaultdict(int)

        if not self.normalized_dir.exists():
            raise FileNotFoundError(f"Normalized directory not found: {normalized_data_dir}")

        if not self.original_dir.exists():
            logger.warning(f"Original directory not found: {original_data_dir}")
            logger.warning("Will use normalized images for face (less accurate)")
            self.use_original = False
        else:
            self.use_original = True

        logger.info("=" * 80)
        logger.info("FIXED DIAGNOSTIC MPIIGAZE PREPROCESSOR")
        logger.info("=" * 80)
        logger.info(f"Normalized (eyes): {self.normalized_dir}")
        logger.info(f"Original (face):   {self.original_dir}")
        logger.info(f"Output:            {self.output_dir}")
        logger.info(f"Image size:        {self.image_size}×{self.image_size}")
        logger.info(f"Use original face: {self.use_original}")
        logger.info("\n🔍 DIAGNOSTIC MODE: Will track all failure reasons")
        logger.info("✅ FIXED: No strict pose range checks (trusts MPIIGaze)")
        logger.info("=" * 80)

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
                samples, failed, attempted = self._process_person(person_dir, person_id)
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
        self._print_failure_report(total_attempted, len(all_samples))

        logger.info("\n" + "=" * 80)
        logger.info("✅ PREPROCESSING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total attempted: {total_attempted}")
        logger.info(f"Valid samples:   {len(all_samples)} ({100 * len(all_samples) / total_attempted:.1f}%)")
        logger.info(f"Failed samples:  {failed_samples} ({100 * failed_samples / total_attempted:.1f}%)")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)

        return all_samples

    def _process_person(self, person_dir, person_id):
        """Process all .mat files for one person"""
        mat_files = sorted(person_dir.glob('*.mat'))

        if len(mat_files) == 0:
            logger.warning(f"  ⚠️  {person_id}: No .mat files found")
            return [], 0, 0

        samples = []
        failed = 0
        attempted = 0

        for mat_file in tqdm(mat_files, desc=f"  {person_id} files", leave=False):
            try:
                s, f, a = self._process_mat_file(mat_file, person_id)
                samples.extend(s)
                failed += f
                attempted += a
            except Exception as e:
                logger.debug(f"    Error in {mat_file.name}: {e}")
                continue

        return samples, failed, attempted

    def _process_mat_file(self, mat_file, person_id):
        """Process single .mat file"""
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

        # Try to load original images
        original_images = None
        if self.use_original:
            try:
                original_mat = self.original_dir / person_id / mat_file.name
                if original_mat.exists():
                    original_data = loadmat(str(original_mat))
                    try:
                        original_images = original_data['data'][0, 0]['left'][0, 0]['image']
                    except:
                        pass
            except Exception:
                pass

        num_samples = len(images)
        day_name = mat_file.stem

        samples = []
        failed = 0

        for i in range(num_samples):
            try:
                original_img = original_images[i] if original_images is not None else None

                sample = self._process_sample(
                    images[i], gazes[i], poses[i],
                    person_id, day_name, i, original_img
                )

                if sample:
                    samples.append(sample)
                else:
                    failed += 1

            except Exception:
                self.failure_reasons['processing_exception'] += 1
                failed += 1

        return samples, failed, num_samples

    def _process_sample(self, image, gaze, pose, person_id, day_name, idx, original_image=None):
        """Process single sample with failure tracking"""

        # Convert to uint8
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

        # Split eyes
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

        # Extract face
        if original_image is not None:
            face = self._extract_face_from_original(original_image)
        else:
            face = self._extract_face_from_normalized(image)

        if face is None:
            self.failure_reasons['face_extraction_failed'] += 1
            return None

        try:
            face = cv2.resize(face, (self.image_size, self.image_size))
        except Exception:
            self.failure_reasons['face_resize_failed'] += 1
            return None

        # Convert gaze
        gaze_2d = self._gaze_to_screen_coords(gaze)
        if gaze_2d is None:
            self.failure_reasons['invalid_gaze'] += 1
            return None

        # Validate pose (FIXED - no range check)
        pose_2d = self._validate_pose(pose)
        if pose_2d is None:
            self.failure_reasons['invalid_pose'] += 1
            return None

        # Quality check
        quality_result = self._quality_check_detailed(left_eye, right_eye, face)
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
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(right_path), cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(face_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            return {
                'sample_id': sample_id,
                'person_id': person_id,
                'day': day_name,
                'left_eye_path': str(left_path),
                'right_eye_path': str(right_path),
                'face_path': str(face_path),
                'gaze': gaze_2d.tolist(),
                'gaze_angles': gaze.tolist(),
                'head_pose': pose_2d.tolist()
            }

        except Exception:
            self.failure_reasons['file_save_failed'] += 1
            try:
                left_path.unlink(missing_ok=True)
                right_path.unlink(missing_ok=True)
                face_path.unlink(missing_ok=True)
            except:
                pass
            return None

    def _extract_face_from_original(self, original_image):
        """Extract face from original MPIIGaze image"""
        try:
            if original_image.dtype != np.uint8:
                original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)

            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            elif original_image.shape[2] == 1:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

            h, w = original_image.shape[:2]
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

    def _extract_face_from_normalized(self, normalized_image):
        """Fallback: extract face from normalized image"""
        try:
            return normalized_image.copy()
        except Exception:
            return None

    def _gaze_to_screen_coords(self, gaze_angles):
        """Convert gaze angles to screen coordinates"""
        try:
            theta = float(gaze_angles[0])
            phi = float(gaze_angles[1])

            if np.isnan(theta) or np.isnan(phi) or np.isinf(theta) or np.isinf(phi):
                return None

            fov = 1.2
            x = (np.sin(theta) / np.sin(fov) + 1) / 2
            y = (np.sin(phi) / np.sin(fov) + 1) / 2

            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            result = np.array([x, y], dtype=np.float32)

            if np.isnan(result).any() or np.isinf(result).any():
                return None

            return result

        except Exception:
            return None

    def _validate_pose(self, head_pose):
        """
        FIXED: Validate head pose - only check for NaN/Inf
        NO range checks - trust MPIIGaze normalized data
        """
        try:
            yaw = float(head_pose[0])
            pitch = float(head_pose[1])

            # Only check for invalid values, NOT range!
            if np.isnan(yaw) or np.isnan(pitch):
                return None
            if np.isinf(yaw) or np.isinf(pitch):
                return None

            # NO RANGE CHECK - MPIIGaze normalized data can have values beyond ±2.0
            return np.array([yaw, pitch], dtype=np.float32)

        except Exception:
            return None

    def _quality_check_detailed(self, left_eye, right_eye, face):
        """Very lenient quality check"""
        for idx, (img, name) in enumerate([(left_eye, 'left_eye'),
                                           (right_eye, 'right_eye'),
                                           (face, 'face')]):
            if img is None or img.size == 0:
                return f'{name}_null'

            brightness = np.mean(img)
            if brightness < 2:  # Very lenient
                return f'{name}_too_dark'
            if brightness > 253:  # Very lenient
                return f'{name}_too_bright'

            if np.std(img) < 1:  # Very lenient
                return f'{name}_low_variance'

        return 'pass'

    def _print_failure_report(self, total_attempted, valid_samples):
        """Print detailed failure analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 FAILURE ANALYSIS REPORT")
        logger.info("=" * 80)

        if not self.failure_reasons:
            logger.info("✅ No failures detected!")
            return

        sorted_failures = sorted(self.failure_reasons.items(),
                                 key=lambda x: x[1],
                                 reverse=True)

        total_failures = sum(self.failure_reasons.values())

        logger.info(f"\nTotal failures: {total_failures}")
        logger.info(f"Success rate:   {100 * valid_samples / total_attempted:.1f}%\n")

        logger.info("Failure breakdown:")
        logger.info("-" * 80)

        for reason, count in sorted_failures:
            percentage = 100 * count / total_failures
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            logger.info(f"  {reason:30s} {count:6d} ({percentage:5.1f}%) {bar}")

        logger.info("-" * 80)

        logger.info("\n💡 ANALYSIS:")

        top_reason, top_count = sorted_failures[0]

        if 'quality_' in top_reason:
            logger.info("  ⚠️  Most failures are from quality checks")
            logger.info("  → Images have extreme brightness values")
            logger.info("  → This is normal for varied lighting conditions")

        if 'face_extraction_failed' in top_reason:
            logger.info("  ⚠️  Many face extractions failing")
            logger.info("  → Check if original images are available")

        if 'invalid_gaze' in top_reason or 'invalid_pose' in top_reason:
            logger.info("  ⚠️  Invalid gaze/pose detected")
            logger.info("  → This should be RARE in normalized data (<1%)")
            logger.info("  → If high, check if .mat files are corrupted")

        logger.info("=" * 80)

    def _save_metadata(self, samples):
        """Save metadata"""
        metadata = {
            'total_samples': len(samples),
            'image_size': self.image_size,
            'has_face_images': True,
            'has_face_grid': False,
            'fov': 1.2,
            'preprocessing_version': '5.1_diagnostic_fixed',
            'validation': 'lenient_no_range_checks',
            'failure_breakdown': dict(self.failure_reasons),
            'samples': samples
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n💾 Metadata saved: {metadata_path}")


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

    train_persons, temp = train_test_split(persons, test_size=val_size + test_size, random_state=42)
    val_persons, test_persons = train_test_split(temp, test_size=test_size / (val_size + test_size), random_state=42)

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
            'has_face_grid': metadata.get('has_face_grid', False),
            'fov': metadata['fov'],
            'samples': split_samples
        }

        split_path = output_dir / f'{split_name}_metadata.json'
        with open(split_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)

        logger.info(f"  💾 Saved: {split_path}")

    logger.info("=" * 80)
    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description='Fixed Diagnostic MPIIGaze Preprocessing')

    parser.add_argument('--original_dir', type=str, default='./MPIIGaze/Data/Original')
    parser.add_argument('--normalized_dir', type=str, default='./MPIIGaze/Data/Normalized')
    parser.add_argument('--output_dir', type=str, default='./diagnostic_gaze_data')
    parser.add_argument('--image_size', type=int, default=224)

    args = parser.parse_args()

    logger.info("\n🔍 STEP 1: DIAGNOSTIC PREPROCESSING (FIXED)\n")

    preprocessor = DiagnosticMPIIGazePreprocessor(
        args.original_dir,
        args.normalized_dir,
        args.output_dir,
        args.image_size
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
    logger.info("Expected success rate: 98-99% (with fixed validation)")
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
