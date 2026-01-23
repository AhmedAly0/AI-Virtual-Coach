"""
Pose estimation utilities for exercise video analysis.

Extracts MediaPipe 3D pose landmarks from videos and computes normalized joint angles.
Provides both static (aggregated) and temporal (time-series) feature representations.
Includes tempo preservation via FPS-normalized duration features.

Note: Uses 3D coordinates (x, y, z) where z represents depth from camera plane.
"""

import os
import sys
import re
import logging
import warnings
from typing import List, Tuple, Dict, Optional
from contextlib import contextmanager

# Suppress TensorFlow/MediaPipe C++ warnings before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ warnings
os.environ['GLOG_minloglevel'] = '3'  # Suppress Google logging

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from scipy import interpolate

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
warnings.filterwarnings('ignore', category=UserWarning, module='google')


@contextmanager
def suppress_stderr():
    """Context manager to temporarily suppress stderr output."""
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stderr = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(save_stderr, 2)
        os.close(null_fd)

# Import subject ID normalization from data_loader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_loader import _normalize_subject_id

logger = logging.getLogger(__name__)

# MediaPipe Pose setup using Tasks API
# Download model from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'pose_landmarker_full.task')

# Initialize with stderr suppression to hide TFLite C++ warnings
try:
    with suppress_stderr():
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=python.BaseOptions.Delegate.CPU  # pip package only supports CPU
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,  # Lowered from 0.5 for more lenient detection
            min_tracking_confidence=0.3          # Lowered from 0.5 for more lenient tracking
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(options)
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe PoseLandmarker: {e}")
    logger.error(f"Please download the model file to: {MODEL_PATH}")
    logger.error("Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    raise


def calculate_angle(a, b, c):
    """Calculate angle at joint b formed by points a-b-c in 3D space.
    
    Uses the dot product formula: cos(θ) = (v1·v2) / (|v1||v2|)
    where v1 = vector from b to a, v2 = vector from b to c
    
    Args:
        a, b, c: MediaPipe landmark objects with .x, .y, and .z attributes
        
    Returns:
        float: Angle in degrees (0-180)
    """
    # Create 3D vectors
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    # Vectors from joint b to points a and c
    ba = a - b
    bc = c - b
    
    # Compute angle using dot product
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if magnitude_ba < 1e-6 or magnitude_bc < 1e-6:
        return 0.0
    
    # Compute cosine and clip to [-1, 1] for numerical stability
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def _normalize_landmarks(landmarks):
    """Normalize pose landmarks using pelvis center and torso length in 3D space.
    
    Args:
        landmarks: MediaPipe pose landmarks
        
    Returns:
        list: Normalized landmark coordinates as [(x, y, z), ...] or None if normalization fails
    """
    try:
        # Calculate pelvis center (mean of left and right hip) in 3D
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        pelvis_x = (left_hip.x + right_hip.x) / 2.0
        pelvis_y = (left_hip.y + right_hip.y) / 2.0
        pelvis_z = (left_hip.z + right_hip.z) / 2.0
        
        # Calculate mid-shoulder point in 3D
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        mid_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2.0
        
        # Calculate torso length in 3D (Euclidean distance)
        torso_length = np.sqrt(
            (mid_shoulder_x - pelvis_x)**2 + 
            (mid_shoulder_y - pelvis_y)**2 + 
            (mid_shoulder_z - pelvis_z)**2
        )
        
        if torso_length < 1e-6:
            return None
        
        # Normalize all landmarks in 3D
        normalized = []
        for lm in landmarks:
            norm_x = (lm.x - pelvis_x) / torso_length
            norm_y = (lm.y - pelvis_y) / torso_length
            norm_z = (lm.z - pelvis_z) / torso_length
            normalized.append(type('Landmark', (), {'x': norm_x, 'y': norm_y, 'z': norm_z})())
        
        return normalized
    except Exception as e:
        return None


def extract_features_from_video(video_path: str) -> Optional[Tuple[np.ndarray, int, float]]:
    """Extract normalized 3D joint angles from a single video with tempo metadata.
    
    Processes each frame to extract 9 core joint angles in 3D space after normalizing
    pose landmarks by pelvis center and torso length. Uses MediaPipe's 3D landmarks
    (x, y, z) where z represents depth from the camera plane.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        Tuple[np.ndarray, int, float]: 
            - features: Shape (num_valid_frames, 9) containing 3D angles in degrees
            - total_frames: Total number of frames in video (including failed detections)
            - fps: Frames per second of the video
        Returns None if processing fails
    """
    # Create a fresh PoseLandmarker instance for this video
    # This prevents timestamp state contamination between videos in VIDEO mode
    try:
        with suppress_stderr():
            base_options = python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=python.BaseOptions.Delegate.CPU  # pip package only supports CPU
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            local_pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        logger.error(f"Failed to create PoseLandmarker for {video_path}: {e}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return None
    
    # Get video metadata for tempo features
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Default fallback
    
    frame_features_list = []
    frame_count = 0
    frames_with_pose = 0
    frames_with_failed_normalization = 0
    frames_with_failed_angles = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Use frame index as timestamp for VIDEO mode
        # MediaPipe VIDEO mode only requires monotonically increasing timestamps
        # Using frame index (0, 1, 2, 3...) eliminates collision/rounding issues
        timestamp_ms = frame_count
        
        # Detect pose landmarks using the local instance
        detection_result = local_pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        frame_count += 1
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            frames_with_pose += 1
            lm = detection_result.pose_landmarks[0]  # Get first person's landmarks
            
            # Normalize landmarks before computing angles
            normalized_lm = _normalize_landmarks(lm)
            
            if normalized_lm is None:
                frames_with_failed_normalization += 1
                continue
            
            try:
                # Compute 8 core joint angles on normalized landmarks
                left_elbow = calculate_angle(normalized_lm[11], normalized_lm[13], normalized_lm[15])
                right_elbow = calculate_angle(normalized_lm[12], normalized_lm[14], normalized_lm[16])
                
                left_shoulder = calculate_angle(normalized_lm[13], normalized_lm[11], normalized_lm[23])
                right_shoulder = calculate_angle(normalized_lm[14], normalized_lm[12], normalized_lm[24])
                
                left_hip = calculate_angle(normalized_lm[11], normalized_lm[23], normalized_lm[25])
                right_hip = calculate_angle(normalized_lm[12], normalized_lm[24], normalized_lm[26])
                
                left_knee = calculate_angle(normalized_lm[23], normalized_lm[25], normalized_lm[27])
                right_knee = calculate_angle(normalized_lm[24], normalized_lm[26], normalized_lm[28])
                
                # NEW: Ankle angles (for Calf Raises detection)
                # Landmarks: knee → ankle → heel
                left_ankle = calculate_angle(normalized_lm[25], normalized_lm[27], normalized_lm[29])
                right_ankle = calculate_angle(normalized_lm[26], normalized_lm[28], normalized_lm[30])
                
                # NEW: Wrist angles (for arm curl exercises)
                # Landmarks: elbow → wrist → pinky
                left_wrist = calculate_angle(normalized_lm[13], normalized_lm[15], normalized_lm[17])
                right_wrist = calculate_angle(normalized_lm[14], normalized_lm[16], normalized_lm[18])
                
                # Compute torso lean angle (angle between torso and vertical axis) in 3D
                # Torso vector: from pelvis (hip midpoint) to mid-shoulder
                # Use original landmarks to get actual positions
                mid_shoulder_x = (lm[11].x + lm[12].x) / 2.0
                mid_shoulder_y = (lm[11].y + lm[12].y) / 2.0
                mid_shoulder_z = (lm[11].z + lm[12].z) / 2.0
                pelvis_x = (lm[23].x + lm[24].x) / 2.0
                pelvis_y = (lm[23].y + lm[24].y) / 2.0
                pelvis_z = (lm[23].z + lm[24].z) / 2.0
                
                # Torso vector in 3D (pointing up from pelvis to shoulders)
                torso_vec = np.array([
                    mid_shoulder_x - pelvis_x,
                    mid_shoulder_y - pelvis_y,
                    mid_shoulder_z - pelvis_z
                ])
                # Vertical vector in 3D (pointing up: negative y in image coordinates, z=0)
                vertical_vec = np.array([0, -1, 0])
                
                # Calculate angle between torso and vertical in 3D
                dot_product = np.dot(torso_vec, vertical_vec)
                torso_mag = np.linalg.norm(torso_vec)
                if torso_mag > 1e-6:
                    cos_angle = dot_product / torso_mag
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                    torso_lean = np.degrees(np.arccos(cos_angle))
                else:
                    torso_lean = 0.0
                
                # 13 angles total: 9 original + 4 new (ankle and wrist)
                frame_features = [left_elbow, right_elbow, left_shoulder, right_shoulder,
                                 left_hip, right_hip, left_knee, right_knee, torso_lean,
                                 left_ankle, right_ankle, left_wrist, right_wrist]
                
                frame_features_list.append(frame_features)
                
            except Exception as e:
                frames_with_failed_angles += 1
                continue
    
    cap.release()
    
    total_frames = frame_count  # Total frames in video (including failed detections)
    
    if len(frame_features_list) == 0:
        return None
    
    features = np.array(frame_features_list, dtype=np.float32)
    return features, total_frames, fps


def process_video_list(video_files: List[str]) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Process multiple video files and return separate sequences per rep with tempo metadata.
    
    Args:
        video_files (List[str]): List of video file paths
        
    Returns:
        Tuple containing:
            - rep_sequences (List[np.ndarray]): List of sequences, each with shape (num_frames, 9)
            - total_frame_counts (List[int]): Total frames per video
            - fps_values (List[float]): FPS per video
    """
    rep_sequences = []
    total_frame_counts = []
    fps_values = []
    
    for video_file in video_files:
        result = extract_features_from_video(video_file)
        if result is not None:
            features, total_frames, fps = result
            if len(features) > 0:
                rep_sequences.append(features)
                total_frame_counts.append(total_frames)
                fps_values.append(fps)
    
    return rep_sequences, total_frame_counts, fps_values


def build_static_rep_features(rep_sequences: List[np.ndarray]) -> np.ndarray:
    """Compute static statistical features for each rep.
    
    For each rep and each angle, computes: mean, std, min, max, range
    
    Args:
        rep_sequences (List[np.ndarray]): List of sequences, each shape (Ti, A)
        
    Returns:
        np.ndarray: Shape (num_reps, A * 5) containing aggregated statistics
    """
    if not rep_sequences:
        return np.array([]).reshape(0, 65)  # 13 angles * 5 stats = 65
    
    num_reps = len(rep_sequences)
    num_angles = rep_sequences[0].shape[1]
    feature_vectors = []
    
    for seq in rep_sequences:
        # Compute statistics per angle
        rep_features = []
        for angle_idx in range(num_angles):
            angle_series = seq[:, angle_idx]
            
            mean_val = np.mean(angle_series)
            std_val = np.std(angle_series)
            min_val = np.min(angle_series)
            max_val = np.max(angle_series)
            range_val = max_val - min_val
            
            rep_features.extend([mean_val, std_val, min_val, max_val, range_val])
        
        feature_vectors.append(rep_features)
    
    return np.array(feature_vectors, dtype=np.float32)


def build_temporal_rep_features(rep_sequences: List[np.ndarray], T_fixed: int = 50) -> np.ndarray:
    """Resample each rep to fixed length using linear interpolation.
    
    Args:
        rep_sequences (List[np.ndarray]): List of sequences, each shape (Ti, A)
        T_fixed (int): Target fixed length for all sequences
        
    Returns:
        np.ndarray: Shape (num_reps, T_fixed, A) with resampled sequences
    """
    if not rep_sequences:
        return np.array([]).reshape(0, T_fixed, 13)
    
    num_reps = len(rep_sequences)
    num_angles = rep_sequences[0].shape[1]
    resampled_sequences = []
    
    for seq in rep_sequences:
        T_orig = seq.shape[0]
        
        if T_orig == 0:
            # Handle empty sequences - fill with zeros
            resampled = np.zeros((T_fixed, num_angles), dtype=np.float32)
        elif T_orig == 1:
            # Single frame - replicate across time
            resampled = np.tile(seq[0], (T_fixed, 1))
        else:
            # Interpolate each angle independently
            orig_time = np.linspace(0, 1, T_orig)
            target_time = np.linspace(0, 1, T_fixed)
            
            resampled_angles = []
            for angle_idx in range(num_angles):
                angle_series = seq[:, angle_idx]
                f = interpolate.interp1d(orig_time, angle_series, kind='linear')
                resampled_angle = f(target_time)
                resampled_angles.append(resampled_angle)
            
            resampled = np.column_stack(resampled_angles).astype(np.float32)
        
        resampled_sequences.append(resampled)
    
    return np.array(resampled_sequences, dtype=np.float32)


def _find_videos_by_view(volunteer_folder: str, view: str) -> List[str]:
    """Find all videos matching the specified view in a volunteer folder.
    
    Handles case variations: 'Front view', 'Front View', 'FRONT', etc.
    Checks both root folder and view-specific subfolders.
    
    Args:
        volunteer_folder (str): Path to volunteer folder
        view (str): 'front' or 'side'
        
    Returns:
        List[str]: List of absolute paths to matching video files (sorted)
    """
    video_paths = []
    view_lower = view.lower()
    
    if not os.path.isdir(volunteer_folder):
        return video_paths
    
    # PRIORITY 1: Check root folder for files with view in name (main case)
    try:
        for filename in os.listdir(volunteer_folder):
            filepath = os.path.join(volunteer_folder, filename)
            if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
                if view_lower in filename.lower():
                    video_paths.append(filepath)
    except Exception as e:
        logger.warning(f"Error scanning root folder {volunteer_folder}: {e}")
    
    # PRIORITY 2: Check for view-specific subfolder (backup case)
    try:
        view_subfolder = os.path.join(volunteer_folder, view_lower)
        if os.path.isdir(view_subfolder):
            for filename in os.listdir(view_subfolder):
                filepath = os.path.join(view_subfolder, filename)
                if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
                    video_paths.append(filepath)
    except Exception as e:
        logger.warning(f"Error scanning subfolder {view_lower}: {e}")
    
    return sorted(video_paths)


def _scan_exercise_folder(exercise_path: str, exercise_name: str, view: str) -> List[Tuple[str, List[str], str]]:
    """Scan an exercise folder for volunteer videos.
    
    Args:
        exercise_path (str): Path to exercise folder
        exercise_name (str): Name of the exercise
        view (str): 'front' or 'side'
        
    Returns:
        List[Tuple[str, List[str], str]]: List of (exercise_name, video_paths, subject_id) tuples
    """
    samples = []
    
    if not os.path.isdir(exercise_path):
        logger.warning(f"Exercise folder not found: {exercise_path}")
        return samples
    
    for folder_name in os.listdir(exercise_path):
        volunteer_folder = os.path.join(exercise_path, folder_name)
        
        if not os.path.isdir(volunteer_folder):
            continue
        
        # Normalize subject ID using data_loader utility
        subject_id = _normalize_subject_id(folder_name)
        
        # Find videos for this view
        video_paths = _find_videos_by_view(volunteer_folder, view)
        
        if video_paths:
            samples.append((exercise_name, video_paths, subject_id))
    
    return samples


def scan_clips_directory(clips_path: str, view: str) -> Tuple[List[Tuple[str, List[str], str]], Dict]:
    """Scan the entire Clips directory structure.
    
    Args:
        clips_path (str): Path to Clips folder
        view (str): 'front' or 'side'
        
    Returns:
        Tuple containing:
            - List of (exercise_name, video_paths, subject_id) tuples
            - Dictionary with scanning statistics
    """
    if not os.path.isdir(clips_path):
        raise ValueError(f"Clips directory not found: {clips_path}")
    
    all_samples = []
    exercise_folders = sorted(os.listdir(clips_path))
    
    for exercise_folder_name in tqdm(exercise_folders, desc=f"Scanning {view} videos"):
        exercise_path = os.path.join(clips_path, exercise_folder_name)
        
        if not os.path.isdir(exercise_path):
            continue
        
        # Extract clean exercise name (remove number prefix if present)
        exercise_name = re.sub(r'^\d+\)\s*', '', exercise_folder_name).strip()
        
        samples = _scan_exercise_folder(exercise_path, exercise_name, view)
        all_samples.extend(samples)
    
    # Compute statistics
    unique_subjects = set(item[2] for item in all_samples)
    unique_exercises = set(item[0] for item in all_samples)
    total_videos = sum(len(item[1]) for item in all_samples)
    
    stats = {
        'view': view,
        'total_samples': len(all_samples),
        'total_videos': total_videos,
        'unique_subjects': len(unique_subjects),
        'unique_exercises': len(unique_exercises),
        'exercise_names': sorted(unique_exercises),
        'subject_ids': sorted(unique_subjects),
    }
    
    logger.info(
        f"[scan_clips_directory] Scanned {view} view: {stats['total_samples']} samples, "
        f"{stats['total_videos']} videos, {stats['unique_subjects']} subjects, "
        f"{stats['unique_exercises']} exercises"
    )
    
    return all_samples, stats


def extract_pose_estimates(
    clips_path: str,
    view: str,
    T_fixed: int = 50,
    output_path: Optional[str] = None,
    version_tag: str = "v1"
) -> Tuple[Dict[str, np.ndarray], Dict, List[Dict]]:
    """Extract pose estimates from all videos in clips directory with tempo preservation.
    
    Args:
        clips_path (str): Path to Clips folder
        view (str): 'front' or 'side'
        T_fixed (int): Fixed length for temporal sequences (default: 50)
        output_path (str, optional): Path to save NPZ file. If None, only returns data.
        version_tag (str): Version identifier (e.g., 'v1', 'v2'). Default: 'v1'
        
    Returns:
        Tuple containing:
            - dataset (Dict): Contains 'X_static', 'X_temporal', 'exercise_names', 'subject_ids',
                             'tempo_duration_sec', 'tempo_frame_count', 'tempo_fps'
            - statistics (Dict): Processing statistics
            - failed_videos (List[Dict]): List of failed video information
    """
    # Scan directory structure
    logger.info(f"Scanning clips directory for {view} view...")
    samples, scan_stats = scan_clips_directory(clips_path, view)
    
    if not samples:
        raise ValueError(f"No samples found in {clips_path} for view '{view}'")
    
    logger.info(f"Found {len(samples)} sample(s) to process")
    
    # Process videos and extract features
    all_rep_sequences = []
    all_exercise_names = []
    all_subject_ids = []
    all_tempo_frame_counts = []  # Total frames per rep (all frames, not just valid)
    all_tempo_fps = []  # FPS per rep
    failed_videos = []
    
    total_videos_processed = 0
    total_frames_extracted = 0  # Valid pose frames
    
    for exercise_name, video_paths, subject_id in tqdm(samples, desc=f"Extracting {view} poses"):
        logger.info(f"\nProcessing: {exercise_name} / {subject_id}")
        try:
            rep_sequences, total_frame_counts, fps_values = process_video_list(video_paths)
            
            if not rep_sequences:
                error_msg = f"No valid sequences extracted from {len(video_paths)} video(s)"
                failed_videos.append({
                    'exercise': exercise_name,
                    'subject': subject_id,
                    'videos': video_paths,
                    'error': error_msg
                })
                logger.warning(f"  ⚠️ {error_msg}")
                logger.warning(f"    Videos: {[os.path.basename(v) for v in video_paths]}")
                continue
            
            logger.info(f"  ✓ Extracted {len(rep_sequences)} rep(s)")
            
            # Each rep gets its own entry
            for i, rep_seq in enumerate(rep_sequences):
                all_rep_sequences.append(rep_seq)
                all_exercise_names.append(exercise_name)
                all_subject_ids.append(subject_id)
                all_tempo_frame_counts.append(total_frame_counts[i])
                all_tempo_fps.append(fps_values[i])
                total_frames_extracted += len(rep_seq)  # Valid pose frames
            
            total_videos_processed += len(video_paths)
            
        except Exception as e:
            error_msg = str(e)
            failed_videos.append({
                'exercise': exercise_name,
                'subject': subject_id,
                'videos': video_paths,
                'error': error_msg
            })
            logger.warning(f"  ✗ Exception: {error_msg}")
    
    if not all_rep_sequences:
        raise ValueError(
            f"No valid rep sequences extracted from any videos. "
            f"Failed samples: {len(failed_videos)}. "
            f"Check logs for details."
        )
    
    logger.info(f"\n✓ Successfully extracted {len(all_rep_sequences)} reps from {total_videos_processed} videos")
    
    # Build static and temporal features
    logger.info(f"Building temporal features (T_fixed={T_fixed})...")
    X_temporal = build_temporal_rep_features(all_rep_sequences, T_fixed=T_fixed)
    
    # Build tempo features (FPS-normalized)
    tempo_frame_counts = np.array(all_tempo_frame_counts, dtype=np.int32)
    tempo_fps = np.array(all_tempo_fps, dtype=np.float32)
    tempo_duration_sec = tempo_frame_counts / tempo_fps  # Duration in seconds
    
    logger.info(f"Tempo features: frame_count range={tempo_frame_counts.min()}-{tempo_frame_counts.max()}, "
                f"fps range={tempo_fps.min():.1f}-{tempo_fps.max():.1f}, "
                f"duration range={tempo_duration_sec.min():.2f}s-{tempo_duration_sec.max():.2f}s")
    
    # Convert subject IDs to integers (extract number from volunteer_XXX)
    subject_id_ints = []
    for subject_id in all_subject_ids:
        match = re.search(r'(\d+)', subject_id)
        if match:
            subject_id_ints.append(int(match.group(1)))
        else:
            subject_id_ints.append(0)  # Unknown
    
    dataset = {
        'X_temporal': X_temporal,
        'exercise_names': np.array(all_exercise_names, dtype=object),
        'subject_ids': np.array(subject_id_ints, dtype=np.int32),
        'tempo_duration_sec': tempo_duration_sec,  # FPS-normalized duration
        'tempo_frame_count': tempo_frame_counts,    # Raw frame count
        'tempo_fps': tempo_fps,                     # Original FPS
    }
    
    statistics = {
        'view': view,
        'total_reps': len(all_rep_sequences),
        'total_videos_processed': total_videos_processed,
        'total_frames_extracted': total_frames_extracted,
        'unique_subjects': len(set(all_subject_ids)),
        'unique_exercises': len(set(all_exercise_names)),
        'failed_videos': len(failed_videos),
        'temporal_shape': X_temporal.shape if len(X_temporal) > 0 else (0, 0, 0),
        'angle_names': ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                       'left_hip', 'right_hip', 'left_knee', 'right_knee', 'torso_lean',
                       'left_ankle', 'right_ankle', 'left_wrist', 'right_wrist'],
        'tempo_stats': {
            'duration_mean': float(np.mean(tempo_duration_sec)),
            'duration_median': float(np.median(tempo_duration_sec)),
            'duration_std': float(np.std(tempo_duration_sec)),
            'frame_count_mean': float(np.mean(tempo_frame_counts)),
            'frame_count_median': float(np.median(tempo_frame_counts)),
            'fps_mean': float(np.mean(tempo_fps)),
            'fps_unique': list(np.unique(tempo_fps)),
        }
    }
    
    logger.info(
        f"\n{'='*60}\n[extract_pose_estimates] Completed {view} view:\n"
        f"  - Total reps: {statistics['total_reps']}\n"
        f"  - Videos processed: {statistics['total_videos_processed']}\n"
        f"  - Frames extracted: {statistics['total_frames_extracted']}\n"
        f"  - Unique subjects: {statistics['unique_subjects']}\n"
        f"  - Unique exercises: {statistics['unique_exercises']}\n"
        f"  - Failed videos: {statistics['failed_videos']}\n"
        f"  - Temporal features: {X_temporal.shape}\n"
        f"  - Tempo duration (mean): {statistics['tempo_stats']['duration_mean']:.2f}s\n"
        f"  - Tempo frame count (median): {statistics['tempo_stats']['frame_count_median']:.0f}\n"
        f"  - FPS values found: {statistics['tempo_stats']['fps_unique']}\n"
        f"{'='*60}"
    )
    
    # Save to NPZ file if output path provided
    if output_path:
        # Determine base path and create versioned file paths
        base_path = os.path.splitext(output_path)[0]  # Remove .npz extension
        temporal_path = f"{base_path}_temporal_{version_tag}.npz"
        
        # Save temporal features
        logger.info(f"Saving temporal features to {temporal_path}...")
        np.savez(
            temporal_path,
            X_temporal=X_temporal,
            exercise_names=dataset['exercise_names'],
            subject_ids=dataset['subject_ids'],
            tempo_duration_sec=tempo_duration_sec,
            tempo_frame_count=tempo_frame_counts,
            tempo_fps=tempo_fps,
            view=view,
            T_fixed=T_fixed,
            angle_names=statistics['angle_names'],
        )
        logger.info(f"✓ Temporal features saved ({X_temporal.shape}) with tempo metadata")
        
        # Update statistics with file paths
        statistics['temporal_file'] = temporal_path
    
    return dataset, statistics, failed_videos