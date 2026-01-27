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


# ============================================================================
# LANDMARK INDICES (MediaPipe Pose)
# ============================================================================
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Key landmark indices for feature extraction
LM_LEFT_EAR = 7
LM_RIGHT_EAR = 8
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_PINKY = 17
LM_RIGHT_PINKY = 18
LM_LEFT_INDEX = 19
LM_RIGHT_INDEX = 20
LM_LEFT_THUMB = 21
LM_RIGHT_THUMB = 22
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28
LM_LEFT_HEEL = 29
LM_RIGHT_HEEL = 30


def calculate_angle(a, b, c):
    """Calculate angle at joint b formed by points a-b-c in 3D space.
    
    Convenience wrapper for MediaPipe landmark objects. Delegates to 
    calculate_angle_from_coords() after converting landmarks to numpy arrays.
    
    Args:
        a, b, c: MediaPipe landmark objects with .x, .y, and .z attributes
        
    Returns:
        float: Angle in degrees (0-180)
    """
    a_coords = np.array([a.x, a.y, a.z])
    b_coords = np.array([b.x, b.y, b.z])
    c_coords = np.array([c.x, c.y, c.z])
    return calculate_angle_from_coords(a_coords, b_coords, c_coords)


def calculate_angle_from_coords(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at joint b formed by points a-b-c in 3D space.
    
    Args:
        a, b, c: numpy arrays of shape (3,) representing [x, y, z] coordinates
        
    Returns:
        float: Angle in degrees (0-180)
    """
    ba = a - b
    bc = c - b
    
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    if magnitude_ba < 1e-6 or magnitude_bc < 1e-6:
        return 0.0
    
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points.
    
    Args:
        p1, p2: numpy arrays of shape (3,) representing [x, y, z] coordinates
        
    Returns:
        float: Euclidean distance (already normalized if landmarks are normalized)
    """
    return np.linalg.norm(p1 - p2)


def calculate_vertical_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate signed vertical (y-axis) distance between two points.
    
    In image coordinates, y increases downward, so negative values mean p1 is above p2.
    
    Args:
        p1, p2: numpy arrays of shape (3,) representing [x, y, z] coordinates
        
    Returns:
        float: Signed vertical distance (p1.y - p2.y)
    """
    return p1[1] - p2[1]


def _normalize_landmarks(landmarks):
    """Normalize pose landmarks using pelvis center and torso length in 3D space.
    
    Note: This is a legacy wrapper for backward compatibility with code that expects
    landmark-like objects. New code should use _normalize_landmarks_to_array() directly.
    
    Args:
        landmarks: MediaPipe pose landmarks
        
    Returns:
        list: Normalized landmark coordinates as pseudo-landmark objects with .x, .y, .z
              attributes, or None if normalization fails
    """
    normalized_array = _normalize_landmarks_to_array(landmarks)
    if normalized_array is None:
        return None
    
    # Convert numpy array back to landmark-like objects for backward compatibility
    normalized = []
    for coords in normalized_array:
        normalized.append(type('Landmark', (), {'x': coords[0], 'y': coords[1], 'z': coords[2]})())
    
    return normalized


def _normalize_landmarks_to_array(landmarks) -> Optional[np.ndarray]:
    """Normalize pose landmarks and return as numpy array.
    
    Args:
        landmarks: MediaPipe pose landmarks (33 landmarks)
        
    Returns:
        np.ndarray: Shape (33, 3) with normalized [x, y, z] coordinates, or None if fails
    """
    try:
        # Calculate pelvis center (mean of left and right hip) in 3D
        left_hip = landmarks[LM_LEFT_HIP]
        right_hip = landmarks[LM_RIGHT_HIP]
        pelvis = np.array([
            (left_hip.x + right_hip.x) / 2.0,
            (left_hip.y + right_hip.y) / 2.0,
            (left_hip.z + right_hip.z) / 2.0
        ])
        
        # Calculate mid-shoulder point in 3D
        left_shoulder = landmarks[LM_LEFT_SHOULDER]
        right_shoulder = landmarks[LM_RIGHT_SHOULDER]
        mid_shoulder = np.array([
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
            (left_shoulder.z + right_shoulder.z) / 2.0
        ])
        
        # Calculate torso length in 3D (Euclidean distance)
        torso_length = np.linalg.norm(mid_shoulder - pelvis)
        
        if torso_length < 1e-6:
            return None
        
        # Normalize all 33 landmarks in 3D
        normalized = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            normalized[i, 0] = (lm.x - pelvis[0]) / torso_length
            normalized[i, 1] = (lm.y - pelvis[1]) / torso_length
            normalized[i, 2] = (lm.z - pelvis[2]) / torso_length
        
        return normalized
    except Exception as e:
        return None


# ============================================================================
# FEATURE COMPUTATION FROM NORMALIZED LANDMARKS
# ============================================================================

def compute_angles_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Compute 13 joint angles from normalized landmark array.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (13,) containing angles in degrees
    """
    # Elbow angles
    left_elbow = calculate_angle_from_coords(
        landmarks[LM_LEFT_SHOULDER], landmarks[LM_LEFT_ELBOW], landmarks[LM_LEFT_WRIST])
    right_elbow = calculate_angle_from_coords(
        landmarks[LM_RIGHT_SHOULDER], landmarks[LM_RIGHT_ELBOW], landmarks[LM_RIGHT_WRIST])
    
    # Shoulder angles
    left_shoulder = calculate_angle_from_coords(
        landmarks[LM_LEFT_ELBOW], landmarks[LM_LEFT_SHOULDER], landmarks[LM_LEFT_HIP])
    right_shoulder = calculate_angle_from_coords(
        landmarks[LM_RIGHT_ELBOW], landmarks[LM_RIGHT_SHOULDER], landmarks[LM_RIGHT_HIP])
    
    # Hip angles
    left_hip = calculate_angle_from_coords(
        landmarks[LM_LEFT_SHOULDER], landmarks[LM_LEFT_HIP], landmarks[LM_LEFT_KNEE])
    right_hip = calculate_angle_from_coords(
        landmarks[LM_RIGHT_SHOULDER], landmarks[LM_RIGHT_HIP], landmarks[LM_RIGHT_KNEE])
    
    # Knee angles
    left_knee = calculate_angle_from_coords(
        landmarks[LM_LEFT_HIP], landmarks[LM_LEFT_KNEE], landmarks[LM_LEFT_ANKLE])
    right_knee = calculate_angle_from_coords(
        landmarks[LM_RIGHT_HIP], landmarks[LM_RIGHT_KNEE], landmarks[LM_RIGHT_ANKLE])
    
    # Ankle angles (for Calf Raises)
    left_ankle = calculate_angle_from_coords(
        landmarks[LM_LEFT_KNEE], landmarks[LM_LEFT_ANKLE], landmarks[LM_LEFT_HEEL])
    right_ankle = calculate_angle_from_coords(
        landmarks[LM_RIGHT_KNEE], landmarks[LM_RIGHT_ANKLE], landmarks[LM_RIGHT_HEEL])
    
    # Wrist angles (for arm curls)
    left_wrist = calculate_angle_from_coords(
        landmarks[LM_LEFT_ELBOW], landmarks[LM_LEFT_WRIST], landmarks[LM_LEFT_PINKY])
    right_wrist = calculate_angle_from_coords(
        landmarks[LM_RIGHT_ELBOW], landmarks[LM_RIGHT_WRIST], landmarks[LM_RIGHT_PINKY])
    
    # Torso lean angle (from vertical)
    mid_shoulder = (landmarks[LM_LEFT_SHOULDER] + landmarks[LM_RIGHT_SHOULDER]) / 2.0
    pelvis = (landmarks[LM_LEFT_HIP] + landmarks[LM_RIGHT_HIP]) / 2.0
    torso_vec = mid_shoulder - pelvis
    vertical_vec = np.array([0, -1, 0])  # Pointing up in image coordinates
    
    torso_mag = np.linalg.norm(torso_vec)
    if torso_mag > 1e-6:
        cos_angle = np.dot(torso_vec, vertical_vec) / torso_mag
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        torso_lean = np.degrees(np.arccos(cos_angle))
    else:
        torso_lean = 0.0
    
    return np.array([
        left_elbow, right_elbow, left_shoulder, right_shoulder,
        left_hip, right_hip, left_knee, right_knee, torso_lean,
        left_ankle, right_ankle, left_wrist, right_wrist
    ], dtype=np.float32)


def compute_distances_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Compute distance-based features from normalized landmark array.
    
    These features are specifically designed to help discriminate:
    - Shrugs: shoulder elevation (ear-to-shoulder vertical distance)
    - Curl variants: arm position and extension
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (6,) containing normalized distances
    """
    # === SHOULDER ELEVATION FEATURES (for Shrugs) ===
    # Vertical distance from ear to shoulder (smaller = shoulders raised)
    # In normalized coordinates, this captures shoulder shrug motion
    left_ear_shoulder_dist = calculate_vertical_distance(
        landmarks[LM_LEFT_EAR], landmarks[LM_LEFT_SHOULDER])
    right_ear_shoulder_dist = calculate_vertical_distance(
        landmarks[LM_RIGHT_EAR], landmarks[LM_RIGHT_SHOULDER])
    
    # === ARM POSITION FEATURES (for Curl variants) ===
    # Wrist-to-shoulder distance: captures arm extension
    left_wrist_shoulder_dist = calculate_distance(
        landmarks[LM_LEFT_WRIST], landmarks[LM_LEFT_SHOULDER])
    right_wrist_shoulder_dist = calculate_distance(
        landmarks[LM_RIGHT_WRIST], landmarks[LM_RIGHT_SHOULDER])
    
    # Elbow-to-hip horizontal distance: captures if arms are tucked vs extended
    # Seated curls have arms more tucked, hammer curls neutral, EZ bar in front
    left_elbow_hip_dist = calculate_distance(
        landmarks[LM_LEFT_ELBOW], landmarks[LM_LEFT_HIP])
    right_elbow_hip_dist = calculate_distance(
        landmarks[LM_RIGHT_ELBOW], landmarks[LM_RIGHT_HIP])
    
    return np.array([
        left_ear_shoulder_dist, right_ear_shoulder_dist,
        left_wrist_shoulder_dist, right_wrist_shoulder_dist,
        left_elbow_hip_dist, right_elbow_hip_dist
    ], dtype=np.float32)


def compute_all_features_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Compute combined angle + distance features from normalized landmarks.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (19,) containing 13 angles + 6 distances
    """
    angles = compute_angles_from_landmarks(landmarks)  # 13 features
    distances = compute_distances_from_landmarks(landmarks)  # 6 features
    return np.concatenate([angles, distances])


# Feature name constants
ANGLE_NAMES = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'torso_lean',
    'left_ankle', 'right_ankle', 'left_wrist', 'right_wrist'
]

DISTANCE_NAMES = [
    'left_ear_shoulder_vert', 'right_ear_shoulder_vert',
    'left_wrist_shoulder_dist', 'right_wrist_shoulder_dist',
    'left_elbow_hip_dist', 'right_elbow_hip_dist'
]

ALL_FEATURE_NAMES = ANGLE_NAMES + DISTANCE_NAMES

# Additional landmark indices for specialized features
LM_LEFT_FOOT_INDEX = 31
LM_RIGHT_FOOT_INDEX = 32
LM_NOSE = 0

# =============================================================================
# SPECIALIZED FEATURE NAMES (Phase 2 - Confusion Cluster Discrimination)
# =============================================================================

# Group 1: Curl variant discrimination features (8 features)
CURL_FEATURE_NAMES = [
    'left_forearm_supination',      # Forearm rotation estimate
    'right_forearm_supination',     # Forearm rotation estimate
    'left_upper_arm_vertical',      # Upper arm angle from vertical
    'right_upper_arm_vertical',     # Upper arm angle from vertical
    'inter_wrist_distance',         # Distance between wrists (EZ bar = fixed width)
    'wrist_centerline_offset',      # How far wrists are from body centerline (X-axis)
    'left_elbow_body_dist',         # Left elbow distance from torso centerline
    'right_elbow_body_dist',        # Right elbow distance from torso centerline
]

# Group 2: Deadlift vs Rows discrimination features (4 features)
HINGE_FEATURE_NAMES = [
    'shoulder_width_ratio',         # Shoulder retraction indicator
    'left_wrist_hip_vertical',      # Wrist height relative to hip
    'right_wrist_hip_vertical',     # Wrist height relative to hip
    'hip_depth_ratio',              # Hip hinge depth indicator
]

# Group 3: Kickback vs Rows discrimination features (2 features)
KICKBACK_FEATURE_NAMES = [
    'left_wrist_posterior',         # How far wrist is behind hip
    'right_wrist_posterior',        # How far wrist is behind hip
]

# Group 4: Shrugs vs Calf Raises - vertical displacement features (4 features)
ELEVATION_FEATURE_NAMES = [
    'left_heel_elevation',          # Heel height relative to foot index
    'right_heel_elevation',         # Heel height relative to foot index
    'shoulder_center_y',            # Shoulder center vertical position
    'ankle_center_y',               # Ankle center vertical position
]

# Combined specialized features (18 total: 8 curl + 4 hinge + 2 kickback + 4 elevation)
SPECIALIZED_FEATURE_NAMES = (
    CURL_FEATURE_NAMES +
    HINGE_FEATURE_NAMES +
    KICKBACK_FEATURE_NAMES +
    ELEVATION_FEATURE_NAMES
)

# Full extended feature set (base 19 + specialized 18 = 37 total)
ALL_EXTENDED_FEATURE_NAMES = ALL_FEATURE_NAMES + SPECIALIZED_FEATURE_NAMES


# =============================================================================
# SIDE-VIEW SPECIALIZED FEATURE NAMES (Phase 2 - Side View Discrimination)
# =============================================================================

# Group 1: Vertical displacement features (4 features) - Shrugs vs Calf Raises
SIDE_VERTICAL_FEATURE_NAMES = [
    'shoulder_elevation_y',         # Shoulder center Y position
    'heel_ground_clearance',        # Heel height above foot index
    'shoulder_hip_y_ratio',         # Shoulder height / hip height ratio
    'ear_shoulder_compression',     # Ear-to-shoulder vertical gap
]

# Group 2: Overhead arm position features (4 features) - Overhead Triceps Extension
SIDE_OVERHEAD_FEATURE_NAMES = [
    'elbow_above_shoulder',         # Elbow Y relative to shoulder Y
    'wrist_above_elbow',            # Wrist Y relative to elbow Y
    'upper_arm_vertical_angle_side', # Upper arm angle from vertical
    'forearm_vertical_angle_side',  # Forearm angle from vertical
]

# Group 3: Sagittal arm trajectory features (4 features) - Curls, Presses
SIDE_SAGITTAL_FEATURE_NAMES = [
    'wrist_forward_of_shoulder',    # Wrist Z relative to shoulder Z
    'elbow_forward_of_hip',         # Elbow Z relative to hip Z
    'arm_reach_forward',            # Wrist forward of torso centerline
    'elbow_tuck_side',              # Elbow proximity to torso (Z-axis)
]

# Group 4: Hip hinge profile features (4 features) - Deadlift/Rows/Kickbacks
SIDE_HINGE_FEATURE_NAMES = [
    'torso_angle_from_vertical',    # Torso lean angle
    'hip_behind_ankle',             # Hip Z relative to ankle Z
    'shoulder_forward_of_hip',      # Shoulder Z relative to hip Z
    'knee_hip_alignment_z',         # Knee-hip Z alignment
]

# Group 5: Postural stability features (2 features) - General context
SIDE_STABILITY_FEATURE_NAMES = [
    'stance_width_normalized',      # Horizontal ankle distance
    'center_of_mass_y',             # Approximate COM height
]

# Combined side-view specialized features (18 total: 4+4+4+4+2)
SIDE_SPECIALIZED_FEATURE_NAMES = (
    SIDE_VERTICAL_FEATURE_NAMES +
    SIDE_OVERHEAD_FEATURE_NAMES +
    SIDE_SAGITTAL_FEATURE_NAMES +
    SIDE_HINGE_FEATURE_NAMES +
    SIDE_STABILITY_FEATURE_NAMES
)

# Side-view full extended feature set (base 19 + side specialized 18 = 37 total)
SIDE_ALL_EXTENDED_FEATURE_NAMES = ALL_FEATURE_NAMES + SIDE_SPECIALIZED_FEATURE_NAMES


# =============================================================================
# SPECIALIZED FEATURE COMPUTATION (Phase 2 - Front View)
# =============================================================================

def compute_curl_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute features to discriminate curl variants (Hammer, EZ Bar, Seated).
    
    Key discriminators:
    - Forearm supination angle: EZ Bar (~30-45°), Hammer (~0°), Seated (~90°)
    - Upper arm vertical angle: Seated curls often have arms behind torso
    - Inter-wrist distance: EZ Bar has fixed bar width, Hammer/Seated can vary
    - Wrist centerline offset: EZ Bar wrists centered, Hammer at sides
    - Elbow body distance: EZ Bar elbows tucked, Hammer elbows can flare
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (8,) curl discrimination features
    """
    # Left arm landmarks
    left_wrist = landmarks[LM_LEFT_WRIST]
    left_index = landmarks[LM_LEFT_INDEX]
    left_thumb = landmarks[LM_LEFT_THUMB]
    left_elbow = landmarks[LM_LEFT_ELBOW]
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    
    # Hand plane normal (approximates palm direction)
    left_hand_v1 = left_index - left_wrist
    left_hand_v2 = left_thumb - left_wrist
    left_hand_normal = np.cross(left_hand_v1, left_hand_v2)
    left_hand_normal_norm = np.linalg.norm(left_hand_normal)
    
    # Forearm axis
    left_forearm = left_wrist - left_elbow
    left_forearm_norm = np.linalg.norm(left_forearm)
    
    if left_hand_normal_norm > 1e-6 and left_forearm_norm > 1e-6:
        left_hand_normal_unit = left_hand_normal / left_hand_normal_norm
        left_supination = np.degrees(np.arcsin(np.clip(left_hand_normal_unit[1], -1, 1)))
    else:
        left_supination = 0.0
    
    # Right arm landmarks
    right_wrist = landmarks[LM_RIGHT_WRIST]
    right_index = landmarks[LM_RIGHT_INDEX]
    right_thumb = landmarks[LM_RIGHT_THUMB]
    right_elbow = landmarks[LM_RIGHT_ELBOW]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    
    right_hand_v1 = right_index - right_wrist
    right_hand_v2 = right_thumb - right_wrist
    right_hand_normal = np.cross(right_hand_v1, right_hand_v2)
    right_hand_normal_norm = np.linalg.norm(right_hand_normal)
    
    right_forearm = right_wrist - right_elbow
    right_forearm_norm = np.linalg.norm(right_forearm)
    
    if right_hand_normal_norm > 1e-6 and right_forearm_norm > 1e-6:
        right_hand_normal_unit = right_hand_normal / right_hand_normal_norm
        right_supination = np.degrees(np.arcsin(np.clip(right_hand_normal_unit[1], -1, 1)))
    else:
        right_supination = 0.0
    
    # Upper arm vertical angle
    vertical = np.array([0, 1, 0])
    
    left_upper_arm = left_elbow - left_shoulder
    left_upper_arm_norm = np.linalg.norm(left_upper_arm)
    if left_upper_arm_norm > 1e-6:
        cos_angle = np.dot(left_upper_arm, vertical) / left_upper_arm_norm
        left_upper_arm_vertical = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    else:
        left_upper_arm_vertical = 0.0
    
    right_upper_arm = right_elbow - right_shoulder
    right_upper_arm_norm = np.linalg.norm(right_upper_arm)
    if right_upper_arm_norm > 1e-6:
        cos_angle = np.dot(right_upper_arm, vertical) / right_upper_arm_norm
        right_upper_arm_vertical = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    else:
        right_upper_arm_vertical = 0.0
    
    # Hip landmarks for body centerline reference
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    
    # Inter-wrist distance
    inter_wrist_distance = calculate_distance(left_wrist, right_wrist)
    
    # Body centerline
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    body_centerline_x = (shoulder_center_x + hip_center_x) / 2
    
    # Wrist centerline offset
    wrist_center_x = (left_wrist[0] + right_wrist[0]) / 2
    wrist_centerline_offset = abs(wrist_center_x - body_centerline_x)
    
    # Elbow-body distance
    left_elbow_body_dist = abs(left_elbow[0] - body_centerline_x)
    right_elbow_body_dist = abs(right_elbow[0] - body_centerline_x)
    
    return np.array([
        left_supination, right_supination,
        left_upper_arm_vertical, right_upper_arm_vertical,
        inter_wrist_distance, wrist_centerline_offset,
        left_elbow_body_dist, right_elbow_body_dist
    ], dtype=np.float32)


def compute_hinge_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute features to discriminate deadlift vs rows.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (4,) hinge movement discrimination features
    """
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    left_wrist = landmarks[LM_LEFT_WRIST]
    right_wrist = landmarks[LM_RIGHT_WRIST]
    left_knee = landmarks[LM_LEFT_KNEE]
    right_knee = landmarks[LM_RIGHT_KNEE]
    
    # Shoulder width ratio
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    hip_width = calculate_distance(left_hip, right_hip)
    shoulder_width_ratio = shoulder_width / hip_width if hip_width > 1e-6 else 1.0
    
    # Wrist-hip vertical distance
    left_wrist_hip_vertical = left_hip[1] - left_wrist[1]
    right_wrist_hip_vertical = right_hip[1] - right_wrist[1]
    
    # Hip depth ratio
    mid_hip_y = (left_hip[1] + right_hip[1]) / 2
    mid_knee_y = (left_knee[1] + right_knee[1]) / 2
    hip_depth_ratio = mid_hip_y / mid_knee_y if abs(mid_knee_y) > 1e-6 else 1.0
    
    return np.array([
        shoulder_width_ratio,
        left_wrist_hip_vertical,
        right_wrist_hip_vertical,
        hip_depth_ratio
    ], dtype=np.float32)


def compute_kickback_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute features to discriminate triceps kickbacks vs rows.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (2,) kickback discrimination features
    """
    left_wrist = landmarks[LM_LEFT_WRIST]
    right_wrist = landmarks[LM_RIGHT_WRIST]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    
    # Wrist posterior position (z-axis)
    left_wrist_posterior = left_wrist[2] - left_hip[2]
    right_wrist_posterior = right_wrist[2] - right_hip[2]
    
    return np.array([
        left_wrist_posterior,
        right_wrist_posterior
    ], dtype=np.float32)


def compute_elevation_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute features to discriminate shrugs vs calf raises.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (4,) elevation discrimination features
    """
    left_heel = landmarks[LM_LEFT_HEEL]
    right_heel = landmarks[LM_RIGHT_HEEL]
    left_foot_idx = landmarks[LM_LEFT_FOOT_INDEX]
    right_foot_idx = landmarks[LM_RIGHT_FOOT_INDEX]
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_ankle = landmarks[LM_LEFT_ANKLE]
    right_ankle = landmarks[LM_RIGHT_ANKLE]
    
    # Heel elevation
    left_heel_elevation = left_foot_idx[1] - left_heel[1]
    right_heel_elevation = right_foot_idx[1] - right_heel[1]
    
    # Vertical position tracking
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
    
    return np.array([
        left_heel_elevation,
        right_heel_elevation,
        shoulder_center_y,
        ankle_center_y
    ], dtype=np.float32)


def compute_specialized_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute all specialized features from a single frame.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (18,) all specialized features
    """
    curl_features = compute_curl_features(landmarks)          # 8 features
    hinge_features = compute_hinge_features(landmarks)        # 4 features
    kickback_features = compute_kickback_features(landmarks)  # 2 features
    elevation_features = compute_elevation_features(landmarks) # 4 features
    
    return np.concatenate([
        curl_features,
        hinge_features,
        kickback_features,
        elevation_features
    ])


def compute_specialized_features_sequence(landmarks: np.ndarray) -> np.ndarray:
    """Compute specialized features for a temporal sequence of landmarks.
    
    Args:
        landmarks: Shape (T, 33, 3) normalized landmark sequence
        
    Returns:
        np.ndarray: Shape (T, 18) specialized features per frame
    """
    T = landmarks.shape[0]
    features = np.zeros((T, 18), dtype=np.float32)
    
    for t in range(T):
        features[t] = compute_specialized_features(landmarks[t])
    
    return features


def compute_all_specialized_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute specialized features for a batch of landmark sequences.
    
    Args:
        landmarks: Shape (N, T, 33, 3) batch of landmark sequences
        
    Returns:
        np.ndarray: Shape (N, T, 18) specialized features
    """
    N, T = landmarks.shape[0], landmarks.shape[1]
    num_specialized = len(SPECIALIZED_FEATURE_NAMES)
    
    X_specialized = np.zeros((N, T, num_specialized), dtype=np.float32)
    
    for n in range(N):
        X_specialized[n] = compute_specialized_features_sequence(landmarks[n])
    
    return X_specialized


# =============================================================================
# SIDE-VIEW SPECIALIZED FEATURE COMPUTATION (Phase 2 - Side View)
# =============================================================================

def compute_side_vertical_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute vertical displacement features for side-view discrimination.
    
    Targets: Shrugs vs Calf Raises vs stationary exercises.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (4,) vertical displacement features
    """
    # Get landmarks
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    left_ear = landmarks[LM_LEFT_EAR]
    right_ear = landmarks[LM_RIGHT_EAR]
    left_heel = landmarks[LM_LEFT_HEEL]
    right_heel = landmarks[LM_RIGHT_HEEL]
    left_foot_idx = landmarks[LM_LEFT_FOOT_INDEX]
    right_foot_idx = landmarks[LM_RIGHT_FOOT_INDEX]
    
    # Feature 1: Shoulder center Y position
    shoulder_elevation_y = (left_shoulder[1] + right_shoulder[1]) / 2
    
    # Feature 2: Heel ground clearance (plantar flexion indicator)
    left_clearance = left_foot_idx[1] - left_heel[1]
    right_clearance = right_foot_idx[1] - right_heel[1]
    heel_ground_clearance = (left_clearance + right_clearance) / 2
    
    # Feature 3: Shoulder/hip Y ratio
    hip_center_y = (left_hip[1] + right_hip[1]) / 2
    if abs(hip_center_y) < 1e-6:
        shoulder_hip_y_ratio = 1.0
    else:
        shoulder_hip_y_ratio = shoulder_elevation_y / hip_center_y
    
    # Feature 4: Ear-shoulder compression
    left_ear_dist = left_ear[1] - left_shoulder[1]
    right_ear_dist = right_ear[1] - right_shoulder[1]
    ear_shoulder_compression = (left_ear_dist + right_ear_dist) / 2
    
    return np.array([
        shoulder_elevation_y,
        heel_ground_clearance,
        shoulder_hip_y_ratio,
        ear_shoulder_compression
    ], dtype=np.float32)


def compute_side_overhead_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute overhead arm position features for side-view discrimination.
    
    Targets: Overhead Triceps Extension vs other arm exercises.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (4,) overhead arm features
    """
    # Get landmarks
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_elbow = landmarks[LM_LEFT_ELBOW]
    right_elbow = landmarks[LM_RIGHT_ELBOW]
    left_wrist = landmarks[LM_LEFT_WRIST]
    right_wrist = landmarks[LM_RIGHT_WRIST]
    
    # Feature 1: Elbow above shoulder
    left_elbow_diff = left_elbow[1] - left_shoulder[1]
    right_elbow_diff = right_elbow[1] - right_shoulder[1]
    elbow_above_shoulder = (left_elbow_diff + right_elbow_diff) / 2
    
    # Feature 2: Wrist above elbow
    left_wrist_diff = left_wrist[1] - left_elbow[1]
    right_wrist_diff = right_wrist[1] - right_elbow[1]
    wrist_above_elbow = (left_wrist_diff + right_wrist_diff) / 2
    
    # Feature 3: Upper arm vertical angle (from sagittal plane)
    left_upper_arm = left_elbow - left_shoulder
    right_upper_arm = right_elbow - right_shoulder
    avg_upper_arm = (left_upper_arm + right_upper_arm) / 2
    vertical = np.array([0, 1, 0])
    
    norm = np.linalg.norm(avg_upper_arm)
    if norm < 1e-6:
        upper_arm_vertical_angle_side = 0.0
    else:
        cos_angle = np.dot(avg_upper_arm, vertical) / norm
        upper_arm_vertical_angle_side = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    # Feature 4: Forearm vertical angle
    left_forearm = left_wrist - left_elbow
    right_forearm = right_wrist - right_elbow
    avg_forearm = (left_forearm + right_forearm) / 2
    
    norm = np.linalg.norm(avg_forearm)
    if norm < 1e-6:
        forearm_vertical_angle_side = 0.0
    else:
        cos_angle = np.dot(avg_forearm, vertical) / norm
        forearm_vertical_angle_side = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    return np.array([
        elbow_above_shoulder,
        wrist_above_elbow,
        upper_arm_vertical_angle_side,
        forearm_vertical_angle_side
    ], dtype=np.float32)


def compute_side_sagittal_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute sagittal plane arm trajectory features for side-view.
    
    Targets: Curls, pressing movements, front raises.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (4,) sagittal arm features
    """
    # Get landmarks
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_elbow = landmarks[LM_LEFT_ELBOW]
    right_elbow = landmarks[LM_RIGHT_ELBOW]
    left_wrist = landmarks[LM_LEFT_WRIST]
    right_wrist = landmarks[LM_RIGHT_WRIST]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    
    # Feature 1: Wrist forward of shoulder (Z-axis)
    # Note: In MediaPipe, smaller Z = closer to camera = more forward
    left_wrist_forward = left_shoulder[2] - left_wrist[2]
    right_wrist_forward = right_shoulder[2] - right_wrist[2]
    wrist_forward_of_shoulder = (left_wrist_forward + right_wrist_forward) / 2
    
    # Feature 2: Elbow forward of hip
    left_elbow_forward = left_hip[2] - left_elbow[2]
    right_elbow_forward = right_hip[2] - right_elbow[2]
    elbow_forward_of_hip = (left_elbow_forward + right_elbow_forward) / 2
    
    # Feature 3: Arm reach forward (wrist forward of torso centerline)
    torso_z = (left_shoulder[2] + right_shoulder[2] + left_hip[2] + right_hip[2]) / 4
    wrist_z = (left_wrist[2] + right_wrist[2]) / 2
    arm_reach_forward = torso_z - wrist_z
    
    # Feature 4: Elbow tuck (how close elbows are to torso Z-axis)
    left_tuck = abs(left_elbow[2] - left_hip[2])
    right_tuck = abs(right_elbow[2] - right_hip[2])
    elbow_tuck_side = (left_tuck + right_tuck) / 2
    
    return np.array([
        wrist_forward_of_shoulder,
        elbow_forward_of_hip,
        arm_reach_forward,
        elbow_tuck_side
    ], dtype=np.float32)


def compute_side_hinge_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute hip hinge profile features for side-view discrimination.
    
    Targets: Deadlift vs Rows vs Kickbacks.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (4,) hip hinge features
    """
    # Get landmarks
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    left_knee = landmarks[LM_LEFT_KNEE]
    right_knee = landmarks[LM_RIGHT_KNEE]
    left_ankle = landmarks[LM_LEFT_ANKLE]
    right_ankle = landmarks[LM_RIGHT_ANKLE]
    
    # Feature 1: Torso angle from vertical
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_vector = shoulder_center - hip_center
    vertical = np.array([0, 1, 0])
    
    norm = np.linalg.norm(torso_vector)
    if norm < 1e-6:
        torso_angle_from_vertical = 0.0
    else:
        cos_angle = np.dot(torso_vector, vertical) / norm
        torso_angle_from_vertical = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    # Feature 2: Hip behind ankle (Z-axis)
    hip_z = (left_hip[2] + right_hip[2]) / 2
    ankle_z = (left_ankle[2] + right_ankle[2]) / 2
    hip_behind_ankle = hip_z - ankle_z
    
    # Feature 3: Shoulder forward of hip
    shoulder_z = (left_shoulder[2] + right_shoulder[2]) / 2
    shoulder_forward_of_hip = hip_z - shoulder_z
    
    # Feature 4: Knee-hip Z alignment
    knee_z = (left_knee[2] + right_knee[2]) / 2
    knee_hip_alignment_z = hip_z - knee_z
    
    return np.array([
        torso_angle_from_vertical,
        hip_behind_ankle,
        shoulder_forward_of_hip,
        knee_hip_alignment_z
    ], dtype=np.float32)


def compute_side_stability_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute postural stability features for side-view.
    
    Provides general body position context.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (2,) stability features
    """
    # Get landmarks
    left_shoulder = landmarks[LM_LEFT_SHOULDER]
    right_shoulder = landmarks[LM_RIGHT_SHOULDER]
    left_hip = landmarks[LM_LEFT_HIP]
    right_hip = landmarks[LM_RIGHT_HIP]
    left_ankle = landmarks[LM_LEFT_ANKLE]
    right_ankle = landmarks[LM_RIGHT_ANKLE]
    
    # Feature 1: Stance width (X-axis distance between ankles)
    stance_width_normalized = abs(left_ankle[0] - right_ankle[0])
    
    # Feature 2: Approximate center of mass Y position
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    center_of_mass_y = (shoulder_y + hip_y) / 2
    
    return np.array([
        stance_width_normalized,
        center_of_mass_y
    ], dtype=np.float32)


def compute_side_specialized_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute all side-view specialized features from a single frame.
    
    Args:
        landmarks: Shape (33, 3) normalized landmark coordinates
        
    Returns:
        np.ndarray: Shape (18,) all side-view specialized features
    """
    vertical_features = compute_side_vertical_features(landmarks)    # 4 features
    overhead_features = compute_side_overhead_features(landmarks)    # 4 features
    sagittal_features = compute_side_sagittal_features(landmarks)    # 4 features
    hinge_features = compute_side_hinge_features(landmarks)          # 4 features
    stability_features = compute_side_stability_features(landmarks)  # 2 features
    
    return np.concatenate([
        vertical_features,
        overhead_features,
        sagittal_features,
        hinge_features,
        stability_features
    ])


def compute_side_specialized_features_sequence(landmarks: np.ndarray) -> np.ndarray:
    """Compute side-view specialized features for a temporal sequence.
    
    Args:
        landmarks: Shape (T, 33, 3) normalized landmark sequence
        
    Returns:
        np.ndarray: Shape (T, 18) side-view specialized features per frame
    """
    T = landmarks.shape[0]
    features = np.zeros((T, 18), dtype=np.float32)
    
    for t in range(T):
        features[t] = compute_side_specialized_features(landmarks[t])
    
    return features


def compute_all_side_specialized_features(landmarks: np.ndarray) -> np.ndarray:
    """Compute side-view specialized features for a batch of landmark sequences.
    
    Args:
        landmarks: Shape (N, T, 33, 3) batch of landmark sequences
        
    Returns:
        np.ndarray: Shape (N, T, 18) side-view specialized features
    """
    N, T = landmarks.shape[0], landmarks.shape[1]
    num_specialized = len(SIDE_SPECIALIZED_FEATURE_NAMES)
    
    X_side_specialized = np.zeros((N, T, num_specialized), dtype=np.float32)
    
    for n in range(N):
        X_side_specialized[n] = compute_side_specialized_features_sequence(landmarks[n])
    
    return X_side_specialized


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


def extract_raw_landmarks_from_video(video_path: str) -> Optional[Tuple[np.ndarray, int, float]]:
    """Extract raw normalized 3D landmarks from a single video.
    
    This is the new two-stage pipeline: extract and save raw landmarks first,
    then compute features from them later. This allows experimenting with
    different feature combinations without re-processing videos.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        Tuple[np.ndarray, int, float]: 
            - landmarks: Shape (num_valid_frames, 33, 3) normalized landmark coordinates
            - total_frames: Total number of frames in video (including failed detections)
            - fps: Frames per second of the video
        Returns None if processing fails
    """
    # Create a fresh PoseLandmarker instance for this video
    try:
        with suppress_stderr():
            base_options = python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=python.BaseOptions.Delegate.CPU
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
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
    
    frame_landmarks_list = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = frame_count
        
        detection_result = local_pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_count += 1
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            lm = detection_result.pose_landmarks[0]
            
            # Normalize landmarks to array format
            normalized_lm = _normalize_landmarks_to_array(lm)
            
            if normalized_lm is not None:
                frame_landmarks_list.append(normalized_lm)
    
    cap.release()
    
    total_frames = frame_count
    
    if len(frame_landmarks_list) == 0:
        return None
    
    landmarks = np.array(frame_landmarks_list, dtype=np.float32)
    return landmarks, total_frames, fps


def compute_features_from_landmark_sequence(
    landmarks: np.ndarray,
    feature_type: str = 'all'
) -> np.ndarray:
    """Compute features from a sequence of normalized landmarks.
    
    Args:
        landmarks: Shape (T, 33, 3) normalized landmark sequence
        feature_type: One of 'angles', 'distances', 'all'
            - 'angles': 13 joint angles only
            - 'distances': 6 distance features only
            - 'all': 19 features (13 angles + 6 distances)
        
    Returns:
        np.ndarray: Shape (T, num_features) feature sequence
    """
    T = landmarks.shape[0]
    
    if feature_type == 'angles':
        features = np.zeros((T, 13), dtype=np.float32)
        for t in range(T):
            features[t] = compute_angles_from_landmarks(landmarks[t])
    elif feature_type == 'distances':
        features = np.zeros((T, 6), dtype=np.float32)
        for t in range(T):
            features[t] = compute_distances_from_landmarks(landmarks[t])
    elif feature_type == 'all':
        features = np.zeros((T, 19), dtype=np.float32)
        for t in range(T):
            features[t] = compute_all_features_from_landmarks(landmarks[t])
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Use 'angles', 'distances', or 'all'")
    
    return features


def process_video_list_raw(video_files: List[str]) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Process multiple video files and return raw landmark sequences.
    
    Args:
        video_files (List[str]): List of video file paths
        
    Returns:
        Tuple containing:
            - landmark_sequences (List[np.ndarray]): List of sequences, each shape (T, 33, 3)
            - total_frame_counts (List[int]): Total frames per video
            - fps_values (List[float]): FPS per video
    """
    landmark_sequences = []
    total_frame_counts = []
    fps_values = []
    
    for video_file in video_files:
        result = extract_raw_landmarks_from_video(video_file)
        if result is not None:
            landmarks, total_frames, fps = result
            if len(landmarks) > 0:
                landmark_sequences.append(landmarks)
                total_frame_counts.append(total_frames)
                fps_values.append(fps)
    
    return landmark_sequences, total_frame_counts, fps_values


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


def _resample_sequence(seq: np.ndarray, T_fixed: int) -> np.ndarray:
    """Resample a single sequence to fixed length using linear interpolation.
    
    Generic helper function that handles sequences of any shape (T, ...).
    
    Args:
        seq: Input sequence with shape (T_orig, ...) where ... can be any trailing dimensions
        T_fixed: Target fixed length
        
    Returns:
        np.ndarray: Resampled sequence with shape (T_fixed, ...)
    """
    T_orig = seq.shape[0]
    trailing_shape = seq.shape[1:]
    
    if T_orig == 0:
        return np.zeros((T_fixed,) + trailing_shape, dtype=np.float32)
    elif T_orig == 1:
        return np.tile(seq[0:1], (T_fixed,) + (1,) * len(trailing_shape))
    else:
        orig_time = np.linspace(0, 1, T_orig)
        target_time = np.linspace(0, 1, T_fixed)
        
        # Flatten trailing dimensions for interpolation
        flat_seq = seq.reshape(T_orig, -1)
        num_features = flat_seq.shape[1]
        
        resampled_flat = np.zeros((T_fixed, num_features), dtype=np.float32)
        for i in range(num_features):
            f = interpolate.interp1d(orig_time, flat_seq[:, i], kind='linear')
            resampled_flat[:, i] = f(target_time)
        
        return resampled_flat.reshape((T_fixed,) + trailing_shape)


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
    
    resampled_sequences = [_resample_sequence(seq, T_fixed) for seq in rep_sequences]
    return np.array(resampled_sequences, dtype=np.float32)


def build_temporal_landmark_features(
    landmark_sequences: List[np.ndarray], 
    T_fixed: int = 50
) -> np.ndarray:
    """Resample raw landmark sequences to fixed length using linear interpolation.
    
    Args:
        landmark_sequences (List[np.ndarray]): List of sequences, each shape (Ti, 33, 3)
        T_fixed (int): Target fixed length for all sequences
        
    Returns:
        np.ndarray: Shape (num_reps, T_fixed, 33, 3) with resampled landmark sequences
    """
    if not landmark_sequences:
        return np.array([]).reshape(0, T_fixed, 33, 3)
    
    resampled_sequences = [_resample_sequence(seq, T_fixed) for seq in landmark_sequences]
    return np.array(resampled_sequences, dtype=np.float32)


def compute_features_from_resampled_landmarks(
    landmarks: np.ndarray,
    feature_type: str = 'all'
) -> np.ndarray:
    """Compute features from resampled landmark array.
    
    Args:
        landmarks: Shape (N, T, 33, 3) resampled landmark sequences
        feature_type: One of 'angles', 'distances', 'all'
        
    Returns:
        np.ndarray: Shape (N, T, num_features) feature sequences
    """
    N, T = landmarks.shape[0], landmarks.shape[1]
    
    if feature_type == 'angles':
        num_features = 13
    elif feature_type == 'distances':
        num_features = 6
    elif feature_type == 'all':
        num_features = 19
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    features = np.zeros((N, T, num_features), dtype=np.float32)
    
    for n in range(N):
        features[n] = compute_features_from_landmark_sequence(landmarks[n], feature_type)
    
    return features


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


def extract_raw_pose_landmarks(
    clips_path: str,
    view: str,
    T_fixed: int = 50,
    output_path: Optional[str] = None,
    version_tag: str = "v2_landmarks"
) -> Tuple[Dict[str, np.ndarray], Dict, List[Dict]]:
    """Extract raw normalized landmarks from all videos (two-stage pipeline).
    
    This is the first stage of the new pipeline:
    1. Extract raw landmarks and save to NPZ (this function)
    2. Compute features from landmarks using compute_features_from_resampled_landmarks()
    
    Storing raw landmarks allows experimenting with different feature combinations
    (angles only, angles + distances, etc.) without re-processing videos.
    
    Args:
        clips_path (str): Path to Clips folder
        view (str): 'front' or 'side'
        T_fixed (int): Fixed length for temporal sequences (default: 50)
        output_path (str, optional): Path to save NPZ file. If None, only returns data.
        version_tag (str): Version identifier. Default: 'v2_landmarks'
        
    Returns:
        Tuple containing:
            - dataset (Dict): Contains 'X_landmarks', 'X_angles', 'X_distances', 'X_all_features',
                             'exercise_names', 'subject_ids', tempo metadata
            - statistics (Dict): Processing statistics
            - failed_videos (List[Dict]): List of failed video information
    """
    logger.info(f"Scanning clips directory for {view} view...")
    samples, scan_stats = scan_clips_directory(clips_path, view)
    
    if not samples:
        raise ValueError(f"No samples found in {clips_path} for view '{view}'")
    
    logger.info(f"Found {len(samples)} sample(s) to process")
    
    # Process videos and extract raw landmarks
    all_landmark_sequences = []
    all_exercise_names = []
    all_subject_ids = []
    all_tempo_frame_counts = []
    all_tempo_fps = []
    failed_videos = []
    
    total_videos_processed = 0
    total_frames_extracted = 0
    
    for exercise_name, video_paths, subject_id in tqdm(samples, desc=f"Extracting {view} landmarks"):
        logger.info(f"\nProcessing: {exercise_name} / {subject_id}")
        try:
            landmark_sequences, total_frame_counts, fps_values = process_video_list_raw(video_paths)
            
            if not landmark_sequences:
                error_msg = f"No valid sequences extracted from {len(video_paths)} video(s)"
                failed_videos.append({
                    'exercise': exercise_name,
                    'subject': subject_id,
                    'videos': video_paths,
                    'error': error_msg
                })
                logger.warning(f"  ⚠️ {error_msg}")
                continue
            
            logger.info(f"  ✓ Extracted {len(landmark_sequences)} rep(s)")
            
            for i, lm_seq in enumerate(landmark_sequences):
                all_landmark_sequences.append(lm_seq)
                all_exercise_names.append(exercise_name)
                all_subject_ids.append(subject_id)
                all_tempo_frame_counts.append(total_frame_counts[i])
                all_tempo_fps.append(fps_values[i])
                total_frames_extracted += len(lm_seq)
            
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
    
    if not all_landmark_sequences:
        raise ValueError(
            f"No valid landmark sequences extracted. "
            f"Failed samples: {len(failed_videos)}."
        )
    
    logger.info(f"\n✓ Successfully extracted {len(all_landmark_sequences)} reps from {total_videos_processed} videos")
    
    # Build resampled landmarks
    logger.info(f"Building temporal landmarks (T_fixed={T_fixed})...")
    X_landmarks = build_temporal_landmark_features(all_landmark_sequences, T_fixed=T_fixed)
    
    # Compute all feature types from landmarks
    logger.info("Computing feature representations from landmarks...")
    X_angles = compute_features_from_resampled_landmarks(X_landmarks, feature_type='angles')
    X_distances = compute_features_from_resampled_landmarks(X_landmarks, feature_type='distances')
    X_all_features = compute_features_from_resampled_landmarks(X_landmarks, feature_type='all')
    
    logger.info(f"  - Angles: {X_angles.shape} (13 features)")
    logger.info(f"  - Distances: {X_distances.shape} (6 features)")
    logger.info(f"  - All features: {X_all_features.shape} (19 features)")
    
    # Compute specialized features (Phase 2 - confusion cluster discrimination)
    logger.info("Computing specialized features...")
    try:
        X_specialized = compute_all_specialized_features(X_landmarks)
        
        logger.info(f"  - Specialized: {X_specialized.shape} ({len(SPECIALIZED_FEATURE_NAMES)} features)")
        
        specialized_features_available = True
    except Exception as e:
        logger.warning(f"  ⚠️ Could not compute specialized features: {e}")
        X_specialized = None
        specialized_features_available = False
    
    # Build tempo features
    tempo_frame_counts = np.array(all_tempo_frame_counts, dtype=np.int32)
    tempo_fps = np.array(all_tempo_fps, dtype=np.float32)
    tempo_duration_sec = tempo_frame_counts / tempo_fps
    
    # Convert subject IDs to integers
    subject_id_ints = []
    for subject_id in all_subject_ids:
        match = re.search(r'(\d+)', subject_id)
        if match:
            subject_id_ints.append(int(match.group(1)))
        else:
            subject_id_ints.append(0)
    
    dataset = {
        'X_landmarks': X_landmarks,  # Shape: (N, T, 33, 3) - raw normalized landmarks
        'X_angles': X_angles,        # Shape: (N, T, 13) - angle features only
        'X_distances': X_distances,  # Shape: (N, T, 6) - distance features only
        'X_all_features': X_all_features,  # Shape: (N, T, 19) - angles + distances
        'exercise_names': np.array(all_exercise_names, dtype=object),
        'subject_ids': np.array(subject_id_ints, dtype=np.int32),
        'tempo_duration_sec': tempo_duration_sec,
        'tempo_frame_count': tempo_frame_counts,
        'tempo_fps': tempo_fps,
    }
    
    # Add specialized features if available
    if specialized_features_available:
        dataset['X_specialized'] = X_specialized  # Shape: (N, T, 18)
    
    statistics = {
        'view': view,
        'total_reps': len(all_landmark_sequences),
        'total_videos_processed': total_videos_processed,
        'total_frames_extracted': total_frames_extracted,
        'unique_subjects': len(set(all_subject_ids)),
        'unique_exercises': len(set(all_exercise_names)),
        'failed_videos': len(failed_videos),
        'landmarks_shape': X_landmarks.shape,
        'angles_shape': X_angles.shape,
        'distances_shape': X_distances.shape,
        'all_features_shape': X_all_features.shape,
        'angle_names': ANGLE_NAMES,
        'distance_names': DISTANCE_NAMES,
        'all_feature_names': ALL_FEATURE_NAMES,
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
    
    # Add specialized feature statistics if available
    if specialized_features_available:
        statistics['specialized_shape'] = X_specialized.shape
        statistics['specialized_feature_names'] = list(SPECIALIZED_FEATURE_NAMES)
    
    # Build log message
    log_msg = (
        f"\n{'='*60}\n[extract_raw_pose_landmarks] Completed {view} view:\n"
        f"  - Total reps: {statistics['total_reps']}\n"
        f"  - Videos processed: {statistics['total_videos_processed']}\n"
        f"  - Frames extracted: {statistics['total_frames_extracted']}\n"
        f"  - Unique subjects: {statistics['unique_subjects']}\n"
        f"  - Unique exercises: {statistics['unique_exercises']}\n"
        f"  - Failed videos: {statistics['failed_videos']}\n"
        f"  - Landmarks: {X_landmarks.shape}\n"
        f"  - Angles (13): {X_angles.shape}\n"
        f"  - Distances (6): {X_distances.shape}\n"
        f"  - All features (19): {X_all_features.shape}\n"
    )
    if specialized_features_available:
        log_msg += f"  - Specialized ({len(SPECIALIZED_FEATURE_NAMES)}): {X_specialized.shape}\n"
    log_msg += f"{'='*60}"
    logger.info(log_msg)
    
    # Save to NPZ file if output path provided
    if output_path:
        base_path = os.path.splitext(output_path)[0]
        
        # Save landmarks + all feature types
        landmarks_path = f"{base_path}_{version_tag}.npz"
        logger.info(f"Saving to {landmarks_path}...")
        
        # Build save dict
        save_dict = {
            # Raw landmarks (for future feature experiments)
            'X_landmarks': X_landmarks,
            # Pre-computed feature sets
            'X_angles': X_angles,
            'X_distances': X_distances,
            'X_all_features': X_all_features,
            # Metadata
            'exercise_names': dataset['exercise_names'],
            'subject_ids': dataset['subject_ids'],
            'tempo_duration_sec': tempo_duration_sec,
            'tempo_frame_count': tempo_frame_counts,
            'tempo_fps': tempo_fps,
            'view': view,
            'T_fixed': T_fixed,
            'angle_names': ANGLE_NAMES,
            'distance_names': DISTANCE_NAMES,
            'all_feature_names': ALL_FEATURE_NAMES,
        }
        
        # Add specialized features if available
        if specialized_features_available:
            save_dict['X_specialized'] = X_specialized
            save_dict['specialized_feature_names'] = list(SPECIALIZED_FEATURE_NAMES)
        
        np.savez_compressed(landmarks_path, **save_dict)
        logger.info(f"✓ Saved landmarks and features to {landmarks_path}")
        
        statistics['output_file'] = landmarks_path
    
    return dataset, statistics, failed_videos


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