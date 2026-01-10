"""
Agents module for AI Virtual Coach.

This module contains intelligent agents that orchestrate feedback generation
and coaching logic for exercise assessment.
"""

from .coaching_agent import CoachingAgent, FeedbackResponse
from .state import CoachingState, AssessmentInput, PerRepScore, RepTrendAnalysis
from .exercise_criteria import get_exercise_criteria, get_all_exercises

__all__ = [
    "CoachingAgent",
    "FeedbackResponse", 
    "CoachingState",
    "AssessmentInput",
    "PerRepScore",
    "RepTrendAnalysis",
    "get_exercise_criteria",
    "get_all_exercises",
]
