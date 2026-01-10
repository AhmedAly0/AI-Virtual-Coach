"""
State definitions for the Coaching Agent using LangGraph.

This module defines the Pydantic models for state that flows through the agent graph.
Uses Pydantic for better LangChain/LangGraph integration and validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Input Models
# ============================================================================

class PerRepScore(BaseModel):
    """Scores for a single repetition."""
    rep_number: int = Field(description="1-indexed rep number")
    scores: dict[str, float] = Field(
        description="Criterion name -> score (0-10 scale)"
    )
    
    @property
    def average(self) -> float:
        """Average score across all criteria for this rep."""
        values = list(self.scores.values())
        return sum(values) / len(values) if values else 0.0


class AssessmentInput(BaseModel):
    """Input data from the assessment model."""
    exercise_name: str = Field(description="Name of the exercise performed")
    exercise_id: int = Field(description="Exercise ID (1-15)")
    view_type: str = Field(default="front", description="Camera view: 'front' or 'side'")
    recognition_confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Exercise recognition confidence (0-1)"
    )
    
    # Per-rep assessment data
    rep_scores: list[PerRepScore] = Field(
        description="List of per-rep scores from assessment model"
    )
    
    @property
    def rep_count(self) -> int:
        """Number of reps assessed."""
        return len(self.rep_scores)
    
    @property
    def aggregated_scores(self) -> dict[str, float]:
        """Mean score per criterion across all reps."""
        if not self.rep_scores:
            return {}
        
        # Collect all scores per criterion
        criterion_scores: dict[str, list[float]] = {}
        for rep in self.rep_scores:
            for criterion, score in rep.scores.items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score)
        
        # Compute means
        return {
            criterion: sum(scores) / len(scores)
            for criterion, scores in criterion_scores.items()
        }
    
    @property
    def overall_score(self) -> float:
        """Overall mean score across all criteria and reps."""
        agg = self.aggregated_scores
        if not agg:
            return 0.0
        return sum(agg.values()) / len(agg)


# ============================================================================
# Analysis Models (computed by Python, not LLM)
# ============================================================================

class CriterionTrend(BaseModel):
    """Trend analysis for a single criterion across reps."""
    criterion: str
    mean: float
    std: float
    min_score: float
    max_score: float
    trend: str = Field(description="'improving', 'declining', or 'stable'")
    trend_magnitude: float = Field(description="Absolute change from first to last reps")
    weakest_reps: list[int] = Field(description="Rep numbers with lowest scores")


class RepTrendAnalysis(BaseModel):
    """Complete trend analysis across all reps."""
    rep_count: int
    criterion_trends: list[CriterionTrend]
    fatigue_detected: bool = Field(
        description="True if scores dropped significantly in final reps"
    )
    fatigue_details: Optional[str] = Field(
        default=None,
        description="Description of fatigue pattern if detected"
    )
    consistency_score: float = Field(
        ge=0.0, le=10.0,
        description="How consistent form was across reps (10 = very consistent)"
    )
    strongest_criterion: str
    weakest_criterion: str
    per_rep_averages: list[float] = Field(
        description="Average score for each rep (for trend visualization)"
    )


# ============================================================================
# State Model (flows through LangGraph)
# ============================================================================

class CoachingState(BaseModel):
    """
    State that flows through the LangGraph coaching agent.
    
    This state is passed between nodes and accumulates information
    as the agent processes the assessment.
    """
    # Input data
    input: AssessmentInput
    
    # Exercise-specific criteria (loaded based on exercise)
    exercise_criteria: list[str] = Field(default_factory=list)
    
    # Trend analysis (computed by Python)
    rep_analysis: Optional[RepTrendAnalysis] = None
    
    # LLM-generated feedback
    llm_feedback: str = ""
    
    # Warnings
    warnings: list[str] = Field(default_factory=list)
    
    # Final output
    final_response: Optional[dict] = None
    
    # Error tracking
    error: Optional[str] = None
    
    class Config:
        """Pydantic config for LangGraph compatibility."""
        arbitrary_types_allowed = True
