"""
Frame distribution analysis for exercise videos.

Analyzes raw video frame counts, FPS, and durations across all exercises
to help optimize the T_fixed parameter for temporal feature extraction.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_loader import _normalize_subject_id


def get_video_metadata(video_path: str) -> dict:
    """Extract metadata from a single video file.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video metadata with keys: total_frames, fps, duration_sec, width, height
              Returns None if video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps == 0 or fps is None:
            fps = 30.0  # Default fallback
        
        duration_sec = total_frames / fps if fps > 0 else 0.0
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration_sec': duration_sec,
            'width': width,
            'height': height,
        }
    finally:
        cap.release()


def analyze_exercise_folder(
    exercise_path: str,
    exercise_name: str,
    view: str
) -> pd.DataFrame:
    """Analyze all videos in an exercise folder.
    
    Args:
        exercise_path (str): Path to exercise folder
        exercise_name (str): Name of the exercise
        view (str): 'front' or 'side'
        
    Returns:
        pd.DataFrame: Video metadata for all videos in this exercise
    """
    records = []
    view_lower = view.lower()
    
    if not os.path.isdir(exercise_path):
        return pd.DataFrame(records)
    
    for folder_name in os.listdir(exercise_path):
        volunteer_folder = os.path.join(exercise_path, folder_name)
        
        if not os.path.isdir(volunteer_folder):
            continue
        
        # Normalize subject ID
        subject_id = _normalize_subject_id(folder_name)
        
        # Find videos for this view
        for filename in os.listdir(volunteer_folder):
            filepath = os.path.join(volunteer_folder, filename)
            
            if not os.path.isfile(filepath) or not filename.lower().endswith('.mp4'):
                continue
            
            if view_lower not in filename.lower():
                continue
            
            # Extract metadata
            metadata = get_video_metadata(filepath)
            
            if metadata is not None:
                record = {
                    'video_path': filepath,
                    'video_name': filename,
                    'exercise': exercise_name,
                    'subject_id': subject_id,
                    'view': view,
                    **metadata
                }
                records.append(record)
    
    return pd.DataFrame(records)


def analyze_all_videos(clips_path: str, view: str = 'front') -> pd.DataFrame:
    """Analyze all videos in the Clips directory.
    
    Args:
        clips_path (str): Path to Clips folder
        view (str): 'front' or 'side'
        
    Returns:
        pd.DataFrame: Complete video metadata for all videos
    """
    if not os.path.isdir(clips_path):
        raise ValueError(f"Clips directory not found: {clips_path}")
    
    all_records = []
    exercise_folders = sorted(os.listdir(clips_path))
    
    for exercise_folder_name in tqdm(exercise_folders, desc=f"Analyzing {view} videos"):
        exercise_path = os.path.join(clips_path, exercise_folder_name)
        
        if not os.path.isdir(exercise_path):
            continue
        
        # Extract clean exercise name
        exercise_name = re.sub(r'^\d+\)\s*', '', exercise_folder_name).strip()
        
        df_exercise = analyze_exercise_folder(exercise_path, exercise_name, view)
        all_records.append(df_exercise)
    
    df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    return df


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute comprehensive statistics from video metadata.
    
    Args:
        df (pd.DataFrame): Video metadata DataFrame
        
    Returns:
        dict: Statistics summary
    """
    if len(df) == 0:
        return {}
    
    stats = {
        'overall': {
            'count': len(df),
            'frame_count': {
                'min': int(df['total_frames'].min()),
                'max': int(df['total_frames'].max()),
                'mean': float(df['total_frames'].mean()),
                'median': float(df['total_frames'].median()),
                'std': float(df['total_frames'].std()),
                'p25': float(df['total_frames'].quantile(0.25)),
                'p75': float(df['total_frames'].quantile(0.75)),
                'p90': float(df['total_frames'].quantile(0.90)),
                'p95': float(df['total_frames'].quantile(0.95)),
            },
            'duration_sec': {
                'min': float(df['duration_sec'].min()),
                'max': float(df['duration_sec'].max()),
                'mean': float(df['duration_sec'].mean()),
                'median': float(df['duration_sec'].median()),
                'std': float(df['duration_sec'].std()),
            },
            'fps': {
                'unique_values': sorted(df['fps'].unique().tolist()),
                'mean': float(df['fps'].mean()),
                'mode': float(df['fps'].mode()[0]) if len(df['fps'].mode()) > 0 else None,
            }
        },
        'per_exercise': {}
    }
    
    # Per-exercise statistics
    for exercise in df['exercise'].unique():
        df_ex = df[df['exercise'] == exercise]
        stats['per_exercise'][exercise] = {
            'count': len(df_ex),
            'frame_count_mean': float(df_ex['total_frames'].mean()),
            'frame_count_median': float(df_ex['total_frames'].median()),
            'frame_count_std': float(df_ex['total_frames'].std()),
            'duration_mean': float(df_ex['duration_sec'].mean()),
            'duration_median': float(df_ex['duration_sec'].median()),
        }
    
    return stats


def plot_distributions(df: pd.DataFrame, output_dir: str, view: str):
    """Generate distribution plots.
    
    Args:
        df (pd.DataFrame): Video metadata DataFrame
        output_dir (str): Directory to save plots
        view (str): 'front' or 'side'
    """
    if len(df) == 0:
        print("No data to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Overall frame count histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['total_frames'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(50, color='red', linestyle='--', linewidth=2, label='Current T_fixed=50')
    plt.axvline(df['total_frames'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median={df["total_frames"].median():.0f}')
    plt.xlabel('Frame Count')
    plt.ylabel('Number of Videos')
    plt.title(f'Frame Count Distribution ({view} view)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(df['duration_sec'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Videos')
    plt.title(f'Duration Distribution ({view} view)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_{view}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot per exercise
    plt.figure(figsize=(14, 6))
    df_sorted = df.sort_values('total_frames', ascending=False)
    exercise_order = df_sorted.groupby('exercise')['total_frames'].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x='exercise', y='total_frames', order=exercise_order, palette='Set2')
    plt.axhline(50, color='red', linestyle='--', linewidth=1.5, label='T_fixed=50')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Exercise')
    plt.ylabel('Frame Count')
    plt.title(f'Frame Count Distribution by Exercise ({view} view)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_per_exercise_{view}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of median frame counts
    pivot = df.groupby(['exercise', 'view'])['total_frames'].median().unstack(fill_value=0)
    if len(pivot) > 0:
        plt.figure(figsize=(6, 10))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Median Frame Count'})
        plt.title(f'Median Frame Count Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{view}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Violin plot comparing front vs side (if both views exist)
    if len(df['view'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='view', y='total_frames', palette='muted')
        plt.axhline(50, color='red', linestyle='--', linewidth=1.5, label='T_fixed=50')
        plt.ylabel('Frame Count')
        plt.title('Frame Count Distribution: Front vs Side Views')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'violin_front_vs_side.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Plots saved to {output_dir}")


def print_summary(stats: dict, view: str):
    """Print statistics summary to console.
    
    Args:
        stats (dict): Statistics dictionary
        view (str): 'front' or 'side'
    """
    if not stats:
        print("No statistics available")
        return
    
    overall = stats['overall']
    frame_stats = overall['frame_count']
    
    print(f"\n{'='*70}")
    print(f"FRAME DISTRIBUTION ANALYSIS - {view.upper()} VIEW")
    print(f"{'='*70}")
    
    print(f"\n📊 Overall Statistics ({overall['count']} videos):")
    print(f"  Frame Count:")
    print(f"    Min:    {frame_stats['min']}")
    print(f"    Max:    {frame_stats['max']}")
    print(f"    Mean:   {frame_stats['mean']:.1f}")
    print(f"    Median: {frame_stats['median']:.1f}  ⭐ (Recommended T_fixed)")
    print(f"    Std:    {frame_stats['std']:.1f}")
    print(f"    P25:    {frame_stats['p25']:.1f}")
    print(f"    P75:    {frame_stats['p75']:.1f}")
    print(f"    P90:    {frame_stats['p90']:.1f}")
    print(f"    P95:    {frame_stats['p95']:.1f}")
    
    print(f"\n  Duration:")
    print(f"    Min:    {overall['duration_sec']['min']:.2f}s")
    print(f"    Max:    {overall['duration_sec']['max']:.2f}s")
    print(f"    Mean:   {overall['duration_sec']['mean']:.2f}s")
    print(f"    Median: {overall['duration_sec']['median']:.2f}s")
    
    print(f"\n  FPS:")
    print(f"    Unique: {overall['fps']['unique_values']}")
    print(f"    Mean:   {overall['fps']['mean']:.1f}")
    print(f"    Mode:   {overall['fps']['mode']:.1f}")
    
    print(f"\n📋 Per-Exercise Statistics:")
    print(f"{'Exercise':<40} {'Count':>6} {'Mean':>7} {'Median':>7} {'Std':>7}")
    print(f"{'-'*70}")
    
    for exercise, ex_stats in sorted(stats['per_exercise'].items(), 
                                     key=lambda x: x[1]['frame_count_median'], 
                                     reverse=True):
        print(f"{exercise:<40} {ex_stats['count']:>6} "
              f"{ex_stats['frame_count_mean']:>7.1f} "
              f"{ex_stats['frame_count_median']:>7.1f} "
              f"{ex_stats['frame_count_std']:>7.1f}")
    
    print(f"\n💡 Recommendations:")
    median = frame_stats['median']
    print(f"  - Current T_fixed: 50")
    print(f"  - Median frame count: {median:.0f}")
    
    # Recommend nearest power of 2 or practical value
    if median < 40:
        rec = 32
    elif median < 55:
        rec = 50
    elif median < 72:
        rec = 64
    elif median < 96:
        rec = 80
    else:
        rec = 100
    
    if rec != 50:
        print(f"  - Recommended T_fixed: {rec} (closer to median, power of 2)")
    else:
        print(f"  - Current T_fixed=50 is reasonable ✓")
    
    print(f"\n  Test these values: {[32, 50, 64, 80, 100]}")
    print(f"{'='*70}\n")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video frame distributions')
    parser.add_argument('--clips_path', type=str, 
                       default='../../datasets/Clips',
                       help='Path to Clips directory')
    parser.add_argument('--view', type=str, default='front',
                       choices=['front', 'side', 'both'],
                       help='Which view to analyze')
    parser.add_argument('--output_dir', type=str,
                       default='../../plots/frame_distribution_analysis',
                       help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    clips_path = (script_dir / args.clips_path).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    print(f"Clips path: {clips_path}")
    print(f"Output dir: {output_dir}")
    
    views = ['front', 'side'] if args.view == 'both' else [args.view]
    
    all_dataframes = []
    
    for view in views:
        print(f"\n{'='*70}")
        print(f"Analyzing {view.upper()} view...")
        print(f"{'='*70}")
        
        # Analyze videos
        df = analyze_all_videos(str(clips_path), view)
        
        if len(df) > 0:
            all_dataframes.append(df)
            
            # Compute statistics
            stats = compute_statistics(df)
            
            # Print summary
            print_summary(stats, view)
            
            # Generate plots
            plot_distributions(df, str(output_dir), view)
            
            # Save CSV
            csv_path = output_dir / f'frame_metadata_{view}.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"✅ Metadata saved to {csv_path}")
            
            # Save JSON statistics
            json_path = output_dir / f'frame_statistics_{view}.json'
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"✅ Statistics saved to {json_path}")
    
    # Combined analysis if both views
    if len(all_dataframes) > 1:
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n{'='*70}")
        print(f"COMBINED ANALYSIS (FRONT + SIDE)")
        print(f"{'='*70}")
        stats_combined = compute_statistics(df_combined)
        print_summary(stats_combined, 'combined')
        plot_distributions(df_combined, str(output_dir), 'combined')
        
        csv_path = output_dir / 'frame_metadata_combined.csv'
        df_combined.to_csv(csv_path, index=False)
        print(f"✅ Combined metadata saved to {csv_path}")


if __name__ == '__main__':
    main()
