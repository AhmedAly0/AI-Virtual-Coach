"""
Exercise criteria loader for the Coaching Agent.

This module parses the Annotation aspects.txt file and provides
exercise-specific assessment criteria for feedback generation.
"""

from pathlib import Path
from typing import Optional


# Exercise criteria parsed from datasets/Annotation aspects.txt
# Maps exercise ID -> (exercise name, list of criteria)
EXERCISE_CRITERIA: dict[int, tuple[str, list[str]]] = {
    1: (
        "Dumbbell Shoulder Press",
        [
            "Starting position: dumbbells at shoulder height, elbows bent",
            "Top position: arms raised near vertical, slight elbow bend",
            "Elbow path: vertical and aligned with wrists",
            "Tempo: controlled movement speed",
            "Core stability: engaged throughout movement",
        ]
    ),
    2: (
        "Hammer Curls",
        [
            "Elbow position: fixed close to torso",
            "Wrist orientation: neutral (hammer grip)",
            "Movement control: no swinging",
            "Range of motion: full curl and extension",
            "Tempo: controlled movement speed",
        ]
    ),
    3: (
        "Standing Dumbbell Front Raises",
        [
            "Arm path: straight in front",
            "Raise height: to shoulder level",
            "Core engagement: stable, no back sway",
            "Wrist alignment: neutral",
            "Controlled tempo",
        ]
    ),
    4: (
        "Lateral Raises",
        [
            "Arm angle: slightly forward (30°–45°)",
            "Raise height: shoulder level",
            "Elbow position: slight bend, fixed",
            "Trap engagement: minimal",
            "No momentum: avoid swinging",
        ]
    ),
    5: (
        "Bulgarian Split Squat",
        [
            "Rear foot: elevated and stable",
            "Front knee tracking: aligned over toes",
            "Torso angle: slight forward lean",
            "Depth: front thigh parallel to floor",
            "Balance: stable and upright",
        ]
    ),
    6: (
        "EZ Bar Curls",
        [
            "Grip: neutral on angled bar",
            "Elbow position: fixed, close to sides",
            "Bar path: controlled, vertical",
            "Full ROM: contraction and extension",
            "No shoulder swing",
        ]
    ),
    7: (
        "Incline Dumbbell Bench Press",
        [
            "Bench angle: ~30–45° incline",
            "Dumbbell path: diagonally upward",
            "Elbow angle: ~75° at bottom",
            "Back position: slight arch, shoulders retracted",
            "Wrist stability: straight, aligned",
        ]
    ),
    8: (
        "Overhead Triceps Extension",
        [
            "Elbow position: close to ears",
            "ROM: full stretch behind head to lockout",
            "Wrist position: stable",
            "Core tightness: avoid arching",
            "Tempo: controlled up and down",
        ]
    ),
    9: (
        "Shrugs",
        [
            "Shoulder path: straight upward",
            "ROM: full shrug and relax",
            "Neck stability: no movement",
            "Weight control: no bouncing",
            "Pause at top",
        ]
    ),
    10: (
        "Weighted Squats",
        [
            "Stance: shoulder-width, toes slightly out",
            "Depth: thighs parallel or lower",
            "Knee tracking: in line with toes",
            "Back posture: neutral spine",
            "Weight distribution: through heels/midfoot",
        ]
    ),
    11: (
        "Seated Biceps Curls",
        [
            "Elbow position: pinned to sides",
            "Shoulder stability: no rocking",
            "ROM: full curl and extension",
            "Wrist position: neutral",
            "Back support: against bench",
        ]
    ),
    12: (
        "Triceps Kickbacks",
        [
            "Upper arm position: parallel to torso",
            "Elbow movement: only forearm moves",
            "Full extension: at end of movement",
            "No torso swing",
            "Neutral spine",
        ]
    ),
    13: (
        "Rows",
        [
            "Back angle: ~45°, flat spine",
            "Elbow path: toward waist/lower ribs",
            "Shoulder retraction: active at top",
            "Pulling motion: controlled",
            "Stable base: legs and core engaged",
        ]
    ),
    14: (
        "Deadlift",
        [
            "Spine: flat and neutral throughout",
            "Hips: hinge movement dominant",
            "Bar path: close to body",
            "Lockout: hips and knees extended",
            "Tempo: controlled movement speed",
        ]
    ),
    15: (
        "Calf Raises",
        [
            "ROM: heels below and above toes",
            "Knee status: straight or slightly bent",
            "Balance: steady, no swaying",
            "Tempo: slow and controlled",
            "Peak hold: pause at top",
        ]
    ),
}

# Name to ID mapping for lookup by name
EXERCISE_NAME_TO_ID: dict[str, int] = {
    name.lower(): id_ for id_, (name, _) in EXERCISE_CRITERIA.items()
}


def get_exercise_criteria(exercise_id: Optional[int] = None, 
                          exercise_name: Optional[str] = None) -> tuple[str, list[str]]:
    """
    Get exercise name and criteria by ID or name.
    
    Args:
        exercise_id: Exercise ID (1-15)
        exercise_name: Exercise name (case-insensitive)
        
    Returns:
        Tuple of (exercise_name, list of criteria strings)
        
    Raises:
        ValueError: If exercise not found
    """
    if exercise_id is not None:
        if exercise_id in EXERCISE_CRITERIA:
            return EXERCISE_CRITERIA[exercise_id]
        raise ValueError(f"Exercise ID {exercise_id} not found. Valid IDs: 1-15")
    
    if exercise_name is not None:
        name_lower = exercise_name.lower()
        # Try exact match first
        if name_lower in EXERCISE_NAME_TO_ID:
            return EXERCISE_CRITERIA[EXERCISE_NAME_TO_ID[name_lower]]
        # Try partial match
        for stored_name, id_ in EXERCISE_NAME_TO_ID.items():
            if name_lower in stored_name or stored_name in name_lower:
                return EXERCISE_CRITERIA[id_]
        raise ValueError(f"Exercise '{exercise_name}' not found")
    
    raise ValueError("Must provide either exercise_id or exercise_name")


def format_criteria_for_prompt(criteria: list[str]) -> str:
    """Format criteria list as a bullet-point string for prompts."""
    return "\n".join(f"• {criterion}" for criterion in criteria)


def get_all_exercises() -> list[tuple[int, str]]:
    """Get list of all exercise IDs and names."""
    return [(id_, name) for id_, (name, _) in EXERCISE_CRITERIA.items()]


# Convenience function to get criteria as formatted string
def get_formatted_criteria(exercise_id: Optional[int] = None,
                           exercise_name: Optional[str] = None) -> str:
    """Get formatted criteria string ready for prompt injection."""
    name, criteria = get_exercise_criteria(exercise_id, exercise_name)
    return format_criteria_for_prompt(criteria)
