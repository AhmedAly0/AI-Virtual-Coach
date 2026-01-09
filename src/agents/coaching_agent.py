"""
Coaching Agent for AI Virtual Coach.

This agent interprets aggregated assessment scores and generates
reflective, post-exercise feedback for the user.

Based on PRD Section 6.2.6:
- Interpret aggregated scores using rule-based logic
- Generate reflective, post-exercise feedback
- Optional LLM usage for natural language generation
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AssessmentScores:
    """Container for the five aspect scores (0-10 scale)."""
    # TODO: Define the 5 assessment aspects based on your model outputs
    aspect_1: float
    aspect_2: float
    aspect_3: float
    aspect_4: float
    aspect_5: float
    
    def to_dict(self) -> dict:
        return {
            "aspect_1": self.aspect_1,
            "aspect_2": self.aspect_2,
            "aspect_3": self.aspect_3,
            "aspect_4": self.aspect_4,
            "aspect_5": self.aspect_5,
        }
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score as mean of all aspects."""
        scores = [self.aspect_1, self.aspect_2, self.aspect_3, 
                  self.aspect_4, self.aspect_5]
        return sum(scores) / len(scores)


@dataclass
class FeedbackResponse:
    """Structured feedback response to send to mobile app."""
    exercise_name: str
    recognition_confidence: float
    session_scores: AssessmentScores
    feedback_summary: str
    detailed_feedback: list[str]
    warnings: list[str]


class CoachingAgent:
    """
    Coaching Agent that generates exercise feedback.
    
    This agent takes assessment scores from the ML models and generates
    human-readable feedback using rule-based logic and optional LLM.
    
    Example usage:
        agent = CoachingAgent()
        scores = AssessmentScores(8.5, 7.0, 9.0, 6.5, 8.0)
        feedback = agent.generate_feedback(
            exercise_name="Dumbbell Shoulder Press",
            scores=scores,
            recognition_confidence=0.95
        )
    """
    
    def __init__(self, use_llm: bool = False, llm_config: Optional[dict] = None):
        """
        Initialize the Coaching Agent.
        
        Args:
            use_llm: Whether to use LLM for natural language generation.
            llm_config: Configuration for LLM (API keys, model name, etc.)
        """
        self.use_llm = use_llm
        self.llm_config = llm_config or {}
        
        # Score thresholds for feedback generation
        self.thresholds = {
            "excellent": 8.5,
            "good": 7.0,
            "needs_improvement": 5.0,
            "poor": 0.0,
        }
    
    def generate_feedback(
        self,
        exercise_name: str,
        scores: AssessmentScores,
        recognition_confidence: float,
        view_type: str = "front",
    ) -> FeedbackResponse:
        """
        Generate comprehensive feedback for an exercise session.
        
        Args:
            exercise_name: Name of the exercise performed.
            scores: Assessment scores for all five aspects.
            recognition_confidence: Confidence of exercise recognition model.
            view_type: Camera view used ("front" or "side").
            
        Returns:
            FeedbackResponse with all feedback components.
        """
        warnings = self._generate_warnings(recognition_confidence, scores)
        summary = self._generate_summary(exercise_name, scores)
        detailed = self._generate_detailed_feedback(exercise_name, scores, view_type)
        
        if self.use_llm:
            summary = self._enhance_with_llm(summary, detailed)
        
        return FeedbackResponse(
            exercise_name=exercise_name,
            recognition_confidence=recognition_confidence,
            session_scores=scores,
            feedback_summary=summary,
            detailed_feedback=detailed,
            warnings=warnings,
        )
    
    def _generate_warnings(
        self, 
        recognition_confidence: float, 
        scores: AssessmentScores
    ) -> list[str]:
        """Generate warnings for low confidence or problematic inputs."""
        warnings = []
        
        if recognition_confidence < 0.7:
            warnings.append(
                f"Low exercise recognition confidence ({recognition_confidence:.0%}). "
                "Results may be less reliable."
            )
        
        if scores.overall_score < self.thresholds["needs_improvement"]:
            warnings.append(
                "Overall form score is below average. "
                "Consider reviewing proper technique before next session."
            )
        
        return warnings
    
    def _generate_summary(
        self, 
        exercise_name: str, 
        scores: AssessmentScores
    ) -> str:
        """Generate a brief overall summary of the exercise performance."""
        overall = scores.overall_score
        
        if overall >= self.thresholds["excellent"]:
            quality = "excellent"
            message = "Great job! Your form was excellent throughout the session."
        elif overall >= self.thresholds["good"]:
            quality = "good"
            message = "Good work! Your form was solid with minor areas for improvement."
        elif overall >= self.thresholds["needs_improvement"]:
            quality = "fair"
            message = "Your form needs some attention. Focus on the feedback below."
        else:
            quality = "needs work"
            message = "Your form requires significant improvement. Consider lighter weights."
        
        return (
            f"Exercise: {exercise_name}\n"
            f"Overall Score: {overall:.1f}/10 ({quality})\n\n"
            f"{message}"
        )
    
    def _generate_detailed_feedback(
        self,
        exercise_name: str,
        scores: AssessmentScores,
        view_type: str,
    ) -> list[str]:
        """Generate detailed feedback for each assessment aspect."""
        feedback = []
        
        # TODO: Customize feedback based on actual aspect definitions
        # This is a template - replace with your specific aspects
        aspects = [
            ("Aspect 1", scores.aspect_1),
            ("Aspect 2", scores.aspect_2),
            ("Aspect 3", scores.aspect_3),
            ("Aspect 4", scores.aspect_4),
            ("Aspect 5", scores.aspect_5),
        ]
        
        for aspect_name, score in aspects:
            if score >= self.thresholds["excellent"]:
                feedback.append(f"✓ {aspect_name}: Excellent ({score:.1f}/10)")
            elif score >= self.thresholds["good"]:
                feedback.append(f"○ {aspect_name}: Good ({score:.1f}/10)")
            elif score >= self.thresholds["needs_improvement"]:
                feedback.append(f"△ {aspect_name}: Needs improvement ({score:.1f}/10)")
            else:
                feedback.append(f"✗ {aspect_name}: Needs attention ({score:.1f}/10)")
        
        return feedback
    
    def _enhance_with_llm(self, summary: str, detailed: list[str]) -> str:
        """
        Optionally enhance feedback using an LLM for more natural language.
        
        Args:
            summary: Rule-based summary text.
            detailed: List of detailed feedback points.
            
        Returns:
            Enhanced summary text from LLM.
        """
        # TODO: Implement LLM integration
        # Options: OpenAI API, local LLM, etc.
        # For now, return original summary
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Create agent
    agent = CoachingAgent(use_llm=False)
    
    # Example scores
    scores = AssessmentScores(
        aspect_1=8.5,
        aspect_2=7.2,
        aspect_3=9.1,
        aspect_4=6.8,
        aspect_5=7.5,
    )
    
    # Generate feedback
    feedback = agent.generate_feedback(
        exercise_name="Dumbbell Shoulder Press",
        scores=scores,
        recognition_confidence=0.92,
        view_type="front",
    )
    
    # Print results
    print("=" * 50)
    print(feedback.feedback_summary)
    print("=" * 50)
    print("\nDetailed Feedback:")
    for item in feedback.detailed_feedback:
        print(f"  {item}")
    
    if feedback.warnings:
        print("\nWarnings:")
        for warning in feedback.warnings:
            print(f"  ⚠ {warning}")
