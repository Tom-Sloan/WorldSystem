"""
Point cloud optimizer for incremental high-quality reconstruction.

Applies confidence filtering and random downsampling to create
optimized point clouds from SLAM keyframes.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def apply_confidence_filter(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out points with low confidence scores.

    Args:
        points: (N, 3) float32 point coordinates
        colors: (N, 3) uint8 RGB colors
        confidence: (N,) float32 confidence scores
        threshold: Minimum confidence threshold (e.g., 10.0)

    Returns:
        Tuple of (filtered_points, filtered_colors, filtered_confidence)
    """
    if confidence is None or len(confidence) == 0:
        logger.warning("No confidence data available, skipping filtering")
        return points, colors, confidence

    # Apply confidence threshold
    mask = confidence >= threshold

    filtered_points = points[mask]
    filtered_colors = colors[mask]
    filtered_confidence = confidence[mask]

    removed_count = len(points) - len(filtered_points)
    removed_pct = (removed_count / len(points) * 100) if len(points) > 0 else 0

    logger.debug(
        f"Confidence filter: removed {removed_count:,} / {len(points):,} points "
        f"({removed_pct:.1f}%) below threshold {threshold}"
    )

    return filtered_points, filtered_colors, filtered_confidence


def random_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: Optional[np.ndarray],
    target_count: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Randomly sample points to target count.

    This naturally removes duplicates from overlapping regions
    and creates uniform point density, similar to offline tools.

    Args:
        points: (N, 3) float32 point coordinates
        colors: (N, 3) uint8 RGB colors
        confidence: Optional (N,) float32 confidence scores
        target_count: Target number of points

    Returns:
        Tuple of (sampled_points, sampled_colors, sampled_confidence)
    """
    current_count = len(points)

    if current_count <= target_count:
        logger.debug(f"No downsampling needed: {current_count:,} <= {target_count:,}")
        return points, colors, confidence

    # Random sampling without replacement
    indices = np.random.choice(current_count, target_count, replace=False)

    sampled_points = points[indices]
    sampled_colors = colors[indices]
    sampled_confidence = confidence[indices] if confidence is not None else None

    logger.debug(
        f"Random downsample: {current_count:,} -> {target_count:,} points "
        f"({target_count/current_count*100:.1f}% retained)"
    )

    return sampled_points, sampled_colors, sampled_confidence


def optimize_keyframes(
    keyframes: List[Dict],
    conf_threshold: float = 10.0,
    target_points: int = 1_000_000
) -> Dict:
    """
    Optimize a batch of keyframes with confidence filtering and downsampling.

    This function is designed to be picklable for ProcessPoolExecutor.

    Args:
        keyframes: List of keyframe dicts with keys:
            - keyframe_id: str
            - points: np.ndarray (N, 3) float32
            - colors: np.ndarray (N, 3) uint8
            - confidence: Optional np.ndarray (N,) float32
            - pose: np.ndarray (4, 4) float32
        conf_threshold: Minimum confidence threshold for filtering
        target_points: Target total point count after optimization

    Returns:
        Dict containing:
            - points: np.ndarray (M, 3) optimized points
            - colors: np.ndarray (M, 3) optimized colors
            - confidence: np.ndarray (M,) optimized confidence scores
            - keyframe_ids: List of processed keyframe IDs
            - stats: Dict with optimization statistics
    """
    if not keyframes:
        return {
            'points': np.array([]),
            'colors': np.array([]),
            'confidence': np.array([]),
            'keyframe_ids': [],
            'stats': {'processed': 0, 'total_input_points': 0, 'total_output_points': 0}
        }

    logger.info(f"Optimizing {len(keyframes)} keyframes...")

    # Collect all points from keyframes
    all_points = []
    all_colors = []
    all_confidence = []
    keyframe_ids = []
    total_input_points = 0

    for kf in keyframes:
        points = kf['points']
        colors = kf['colors']
        confidence = kf.get('confidence')
        keyframe_ids.append(kf['keyframe_id'])

        total_input_points += len(points)

        # Apply confidence filtering per keyframe
        if confidence is not None and len(confidence) > 0:
            filtered_pts, filtered_cols, filtered_conf = apply_confidence_filter(
                points, colors, confidence, conf_threshold
            )
        else:
            filtered_pts = points
            filtered_cols = colors
            filtered_conf = None

        if len(filtered_pts) > 0:
            all_points.append(filtered_pts)
            all_colors.append(filtered_cols)
            if filtered_conf is not None:
                all_confidence.append(filtered_conf)

    # Concatenate all filtered points
    if not all_points:
        logger.warning("No points survived confidence filtering")
        return {
            'points': np.array([]),
            'colors': np.array([]),
            'confidence': np.array([]),
            'keyframe_ids': keyframe_ids,
            'stats': {
                'processed': len(keyframes),
                'total_input_points': total_input_points,
                'total_output_points': 0,
                'removed_by_confidence': total_input_points
            }
        }

    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    combined_confidence = np.concatenate(all_confidence) if all_confidence else None

    logger.info(f"After confidence filtering: {len(combined_points):,} points "
                f"(from {total_input_points:,})")

    # Apply random downsampling to target count
    final_points, final_colors, final_confidence = random_downsample(
        combined_points, combined_colors, combined_confidence, target_points
    )

    stats = {
        'processed': len(keyframes),
        'total_input_points': total_input_points,
        'after_confidence_filter': len(combined_points),
        'total_output_points': len(final_points),
        'removed_by_confidence': total_input_points - len(combined_points),
        'removed_by_downsampling': len(combined_points) - len(final_points),
        'confidence_filter_rate': (total_input_points - len(combined_points)) / total_input_points if total_input_points > 0 else 0,
        'downsample_rate': (len(combined_points) - len(final_points)) / len(combined_points) if len(combined_points) > 0 else 0
    }

    logger.info(
        f"Optimization complete: {total_input_points:,} -> {len(final_points):,} points "
        f"({len(final_points)/total_input_points*100:.1f}% retained)"
    )

    return {
        'points': final_points,
        'colors': final_colors,
        'confidence': final_confidence,
        'keyframe_ids': keyframe_ids,
        'stats': stats
    }


def merge_with_previous(
    new_optimized: Dict,
    previous_optimized: Optional[Dict],
    max_total_points: int = 2_000_000
) -> Dict:
    """
    Merge newly optimized points with previous optimized cloud.

    Args:
        new_optimized: Dict from optimize_keyframes()
        previous_optimized: Previous optimized cloud dict (or None for first run)
        max_total_points: Maximum total points after merging

    Returns:
        Merged optimized cloud dict
    """
    if previous_optimized is None or len(previous_optimized.get('points', [])) == 0:
        logger.info("No previous optimized cloud, using new optimized cloud as-is")
        return new_optimized

    # Combine with previous
    combined_points = np.vstack([
        previous_optimized['points'],
        new_optimized['points']
    ])
    combined_colors = np.vstack([
        previous_optimized['colors'],
        new_optimized['colors']
    ])

    # Handle confidence (previous may not have it)
    prev_conf = previous_optimized.get('confidence')
    new_conf = new_optimized.get('confidence')

    combined_confidence = None
    if prev_conf is not None and new_conf is not None:
        combined_confidence = np.concatenate([prev_conf, new_conf])
    elif new_conf is not None:
        combined_confidence = new_conf

    logger.info(
        f"Merging: {len(previous_optimized['points']):,} (previous) + "
        f"{len(new_optimized['points']):,} (new) = {len(combined_points):,} total"
    )

    # Downsample if exceeding limit
    if len(combined_points) > max_total_points:
        final_points, final_colors, final_confidence = random_downsample(
            combined_points, combined_colors, combined_confidence, max_total_points
        )
        logger.info(f"Downsampled merged cloud: {len(combined_points):,} -> {len(final_points):,}")
    else:
        final_points = combined_points
        final_colors = combined_colors
        final_confidence = combined_confidence

    # Combine keyframe IDs
    prev_ids = previous_optimized.get('keyframe_ids', [])
    new_ids = new_optimized.get('keyframe_ids', [])
    all_ids = prev_ids + new_ids

    return {
        'points': final_points,
        'colors': final_colors,
        'confidence': final_confidence,
        'keyframe_ids': all_ids,
        'stats': new_optimized.get('stats', {})  # Keep stats from latest optimization
    }


# Top-level function for ProcessPoolExecutor (must be picklable)
def optimize_and_merge(
    new_keyframes: List[Dict],
    previous_optimized: Optional[Dict],
    conf_threshold: float,
    target_points: int,
    max_total_points: int = 2_000_000
) -> Dict:
    """
    Top-level function for background optimization process.

    This function is called by ProcessPoolExecutor and must be picklable.

    Args:
        new_keyframes: List of new keyframes to optimize
        previous_optimized: Previous optimized cloud (or None)
        conf_threshold: Confidence threshold for filtering
        target_points: Target points for new keyframes
        max_total_points: Maximum total points after merging

    Returns:
        Optimized and merged point cloud dict
    """
    # Optimize new keyframes
    new_optimized = optimize_keyframes(new_keyframes, conf_threshold, target_points)

    # Merge with previous
    merged = merge_with_previous(new_optimized, previous_optimized, max_total_points)

    return merged
