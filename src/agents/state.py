"""
State definitions for the Coaching Agent using LangGraph.

This module defines the TypedDict state that flows through the agent graph.
"""

from typing import TypedDict, Optional, Annotated
from operator import add


class AssessmentInput(TypedDict):
    """Input scores from the assessment model."""
    scores: dict[str, float]  # e.g., {"criterion_1": 8.5, "criterion_2": 7.0, ...}
    exercise_name: str
    exercise_id: int
    view_type: str  # "front" or "side"
    recognition_confidence: float


class CoachingState(TypedDict):
    """
    State that flows through the LangGraph coaching agent.
    
    This state is passed between nodes and accumulates information
    as the agent processes the assessment.
    """
    # Input data
    input: AssessmentInput
    
    # Exercise-specific criteria (loaded based on exercise)
    exercise_criteria: list[str]
    
    # Processed scores with analysis
    score_analysis: str
    
    # Generated feedback components
    llm_feedback: str
    detailed_feedback: list[str]
    warnings: list[str]
    
    # Final output
    final_response: Optional[dict]
    
    # Error tracking
    error: Optional[str]
