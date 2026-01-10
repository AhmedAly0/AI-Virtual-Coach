"""
Coaching Agent for AI Virtual Coach - LangGraph Implementation.

This agent uses LangGraph to create a stateful workflow that:
1. Loads exercise-specific criteria
2. Analyzes per-rep assessment scores and detects trends
3. Generates personalized feedback using Gemini LLM
4. Formats the final response

Based on PRD Section 6.2.6:
- Interpret aggregated scores using rule-based logic
- Generate reflective, post-exercise feedback
- LLM usage for natural language generation
"""

from typing import Optional
import statistics
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .state import (
    CoachingState, 
    AssessmentInput, 
    PerRepScore,
    RepTrendAnalysis,
    CriterionTrend,
)
from .prompts import (
    FEEDBACK_PROMPT,
    format_per_rep_breakdown,
    format_criterion_summary,
    format_fatigue_analysis,
)
from .exercise_criteria import get_exercise_criteria, format_criteria_for_prompt
from .config import (
    GEMINI_API_KEY, 
    GEMINI_MODEL_NAME, 
    MAX_FEEDBACK_WORDS,
    SCORE_THRESHOLDS,
)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class FeedbackResponse(BaseModel):
    """Structured feedback response to send to mobile app."""
    exercise_name: str = Field(description="Name of the exercise performed")
    exercise_id: int = Field(description="ID of the exercise (1-15)")
    recognition_confidence: float = Field(description="Confidence of exercise recognition")
    rep_count: int = Field(description="Number of reps assessed")
    
    # Aggregated scores
    aggregated_scores: dict[str, float] = Field(
        description="Mean score per criterion across all reps"
    )
    overall_score: float = Field(description="Mean of all scores")
    
    # Per-rep data
    rep_scores: list[dict] = Field(
        description="Per-rep scores for detailed analysis"
    )
    
    # Trend analysis
    consistency_score: float = Field(
        description="How consistent form was (0-10)"
    )
    fatigue_detected: bool = Field(description="Whether fatigue was detected")
    trends: dict[str, str] = Field(
        description="Trend per criterion: 'improving', 'declining', 'stable'"
    )
    
    # Feedback
    feedback_summary: str = Field(description="LLM-generated coaching feedback")
    warnings: list[str] = Field(description="Any warnings about the assessment")


# ============================================================================
# Graph Nodes
# ============================================================================

def load_criteria_node(state: CoachingState) -> dict:
    """
    Node 1: Load exercise-specific criteria based on exercise ID.
    """
    try:
        exercise_id = state.input.exercise_id
        exercise_name = state.input.exercise_name
        
        # Try by ID first, then by name
        try:
            name, criteria = get_exercise_criteria(exercise_id=exercise_id)
        except ValueError:
            name, criteria = get_exercise_criteria(exercise_name=exercise_name)
        
        return {"exercise_criteria": criteria}
    
    except Exception as e:
        return {
            "exercise_criteria": [],
            "error": f"Failed to load exercise criteria: {str(e)}"
        }


def analyze_scores_node(state: CoachingState) -> dict:
    """
    Node 2: Analyze aggregated scores and generate warnings.
    """
    input_data = state.input
    aggregated = input_data.aggregated_scores
    recognition_confidence = input_data.recognition_confidence
    overall = input_data.overall_score
    
    # Generate warnings
    warnings = []
    
    if recognition_confidence < 0.7:
        warnings.append(
            f"Low exercise recognition confidence ({recognition_confidence:.0%}). "
            "Results may be less reliable."
        )
    
    if overall < SCORE_THRESHOLDS["needs_improvement"]:
        warnings.append(
            "Overall form score is below average. "
            "Consider reviewing proper technique before next session."
        )
    
    # Check for critically low aggregated scores
    for criterion, score in aggregated.items():
        if score < 3.0:
            warnings.append(
                f"'{criterion}' scored very low ({score:.1f}/10). "
                "This needs immediate attention."
            )
    
    return {"warnings": warnings}


def analyze_rep_trends_node(state: CoachingState) -> dict:
    """
    Node 3: Analyze per-rep trends (Python-based, no LLM).
    
    Computes:
    - Variance and trends per criterion
    - Fatigue detection (comparing early vs late reps)
    - Consistency score
    - Identifies weakest reps per criterion
    """
    rep_scores = state.input.rep_scores
    rep_count = state.input.rep_count
    
    if not rep_scores or rep_count < 2:
        # Not enough data for trend analysis
        return {"rep_analysis": None}
    
    # Get all criteria names from first rep
    criteria_names = list(rep_scores[0].scores.keys())
    
    criterion_trends = []
    all_stds = []
    
    for criterion in criteria_names:
        # Extract scores for this criterion across all reps
        scores = [rep.scores.get(criterion, 0) for rep in rep_scores]
        
        # Basic stats
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        min_score = min(scores)
        max_score = max(scores)
        all_stds.append(std_score)
        
        # Trend detection: compare first 3 reps vs last 3 reps
        early_reps = scores[:3]
        late_reps = scores[-3:]
        early_mean = statistics.mean(early_reps)
        late_mean = statistics.mean(late_reps)
        
        trend_diff = late_mean - early_mean
        if trend_diff > 0.5:
            trend = "improving"
        elif trend_diff < -0.5:
            trend = "declining"
        else:
            trend = "stable"
        
        # Find weakest reps (below mean - 1 std, or below 5.0)
        threshold = max(mean_score - std_score, 5.0)
        weakest_reps = [
            rep.rep_number for rep in rep_scores 
            if rep.scores.get(criterion, 0) < threshold
        ]
        # Limit to top 3 weakest
        weakest_reps = weakest_reps[:3]
        
        criterion_trends.append(CriterionTrend(
            criterion=criterion,
            mean=mean_score,
            std=std_score,
            min_score=min_score,
            max_score=max_score,
            trend=trend,
            trend_magnitude=abs(trend_diff),
            weakest_reps=weakest_reps,
        ))
    
    # Fatigue detection: check if multiple criteria declined in last 3 reps
    declining_criteria = [t for t in criterion_trends if t.trend == "declining"]
    fatigue_detected = len(declining_criteria) >= 2
    
    fatigue_details = None
    if fatigue_detected:
        declining_names = [t.criterion for t in declining_criteria]
        fatigue_details = (
            f"Form dropped in final reps for: {', '.join(declining_names)}. "
            f"Consider reducing weight or taking longer rest."
        )
    
    # Consistency score: inverse of average std (scaled to 0-10)
    avg_std = statistics.mean(all_stds) if all_stds else 0
    # Lower std = higher consistency. Max std ~3 for bad consistency
    consistency_score = max(0, min(10, 10 - (avg_std * 3)))
    
    # Find strongest and weakest criteria
    sorted_by_mean = sorted(criterion_trends, key=lambda t: t.mean, reverse=True)
    strongest = sorted_by_mean[0].criterion if sorted_by_mean else "N/A"
    weakest = sorted_by_mean[-1].criterion if sorted_by_mean else "N/A"
    
    # Per-rep averages for visualization
    per_rep_averages = [rep.average for rep in rep_scores]
    
    rep_analysis = RepTrendAnalysis(
        rep_count=rep_count,
        criterion_trends=criterion_trends,
        fatigue_detected=fatigue_detected,
        fatigue_details=fatigue_details,
        consistency_score=consistency_score,
        strongest_criterion=strongest,
        weakest_criterion=weakest,
        per_rep_averages=per_rep_averages,
    )
    
    return {"rep_analysis": rep_analysis}


def generate_llm_feedback_node(state: CoachingState) -> dict:
    """
    Node 4: Generate LLM-based coaching feedback using Gemini.
    
    Passes rich per-rep context to the LLM for insightful feedback.
    """
    if not GEMINI_API_KEY:
        return {"llm_feedback": "LLM feedback unavailable: API key not configured."}
    
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
        )
        
        # Prepare prompt variables
        input_data = state.input
        criteria = state.exercise_criteria
        rep_analysis = state.rep_analysis
        
        # Format criteria for prompt
        criteria_formatted = format_criteria_for_prompt(criteria)
        
        # Get criteria names (short form)
        criteria_names = list(input_data.rep_scores[0].scores.keys()) if input_data.rep_scores else []
        
        # Format per-rep breakdown table
        per_rep_breakdown = format_per_rep_breakdown(input_data.rep_scores, criteria_names)
        
        # Format criterion summary with trends
        criterion_summary = ""
        if rep_analysis:
            criterion_summary = format_criterion_summary(rep_analysis.criterion_trends)
        
        # Format fatigue analysis
        fatigue_analysis = ""
        if rep_analysis:
            fatigue_analysis = format_fatigue_analysis(
                rep_analysis.fatigue_detected, 
                rep_analysis.fatigue_details
            )
        
        # Build prompt
        chain = FEEDBACK_PROMPT | llm
        response = chain.invoke({
            "exercise_name": input_data.exercise_name,
            "rep_count": input_data.rep_count,
            "exercise_criteria": criteria_formatted,
            "per_rep_breakdown": per_rep_breakdown,
            "overall_score": input_data.overall_score,
            "consistency_score": rep_analysis.consistency_score if rep_analysis else 0,
            "criterion_summary": criterion_summary,
            "strongest_criterion": rep_analysis.strongest_criterion if rep_analysis else "N/A",
            "weakest_criterion": rep_analysis.weakest_criterion if rep_analysis else "N/A",
            "fatigue_analysis": fatigue_analysis,
            "max_words": MAX_FEEDBACK_WORDS,
        })
        
        return {"llm_feedback": response.content}
    
    except Exception as e:
        return {"llm_feedback": f"Error generating feedback: {str(e)}"}


def format_response_node(state: CoachingState) -> dict:
    """
    Node 5: Format the final response combining all components.
    """
    input_data = state.input
    rep_analysis = state.rep_analysis
    
    # Build trends dict
    trends = {}
    if rep_analysis:
        for ct in rep_analysis.criterion_trends:
            trends[ct.criterion] = ct.trend
    
    response = FeedbackResponse(
        exercise_name=input_data.exercise_name,
        exercise_id=input_data.exercise_id,
        recognition_confidence=input_data.recognition_confidence,
        rep_count=input_data.rep_count,
        aggregated_scores=input_data.aggregated_scores,
        overall_score=input_data.overall_score,
        rep_scores=[rep.model_dump() for rep in input_data.rep_scores],
        consistency_score=rep_analysis.consistency_score if rep_analysis else 0,
        fatigue_detected=rep_analysis.fatigue_detected if rep_analysis else False,
        trends=trends,
        feedback_summary=state.llm_feedback,
        warnings=state.warnings,
    )
    
    return {"final_response": response.model_dump()}


# ============================================================================
# Build the Graph
# ============================================================================

def build_coaching_graph() -> StateGraph:
    """Build and return the coaching agent graph."""
    
    # Create the graph with our state type
    graph = StateGraph(CoachingState)
    
    # Add nodes
    graph.add_node("load_criteria", load_criteria_node)
    graph.add_node("analyze_scores", analyze_scores_node)
    graph.add_node("analyze_rep_trends", analyze_rep_trends_node)
    graph.add_node("generate_llm", generate_llm_feedback_node)
    graph.add_node("format_response", format_response_node)
    
    # Define edges (workflow)
    # START → load_criteria → analyze_scores → analyze_rep_trends → generate_llm → format_response → END
    graph.add_edge(START, "load_criteria")
    graph.add_edge("load_criteria", "analyze_scores")
    graph.add_edge("analyze_scores", "analyze_rep_trends")
    graph.add_edge("analyze_rep_trends", "generate_llm")
    graph.add_edge("generate_llm", "format_response")
    graph.add_edge("format_response", END)
    
    return graph.compile()


# ============================================================================
# Main Agent Class
# ============================================================================

class CoachingAgent:
    """
    LangGraph-based Coaching Agent with per-rep analysis.
    
    This agent uses a state graph to process per-rep assessment scores,
    analyze trends, detect fatigue, and generate personalized feedback.
    
    Example usage:
        agent = CoachingAgent()
        
        # Per-rep scores from assessment model
        rep_scores = [
            {"rep_number": 1, "scores": {"Starting position": 8.5, "Top position": 7.0, ...}},
            {"rep_number": 2, "scores": {"Starting position": 8.0, "Top position": 7.2, ...}},
            ...
        ]
        
        response = agent.generate_feedback(
            exercise_id=1,
            exercise_name="Dumbbell Shoulder Press",
            rep_scores=rep_scores,
            recognition_confidence=0.95,
            view_type="front"
        )
    """
    
    def __init__(self):
        """Initialize the coaching agent with compiled graph."""
        self.graph = build_coaching_graph()
    
    def generate_feedback(
        self,
        exercise_id: int,
        exercise_name: str,
        rep_scores: list[dict],
        recognition_confidence: float,
        view_type: str = "front",
    ) -> FeedbackResponse:
        """
        Generate comprehensive feedback for an exercise session.
        
        Args:
            exercise_id: Exercise ID (1-15)
            exercise_name: Name of the exercise performed
            rep_scores: List of dicts with 'rep_number' and 'scores' per rep
            recognition_confidence: Confidence of exercise recognition
            view_type: Camera view used ("front" or "side")
            
        Returns:
            FeedbackResponse with all feedback components
        """
        # Convert raw dicts to PerRepScore objects
        per_rep_scores = [
            PerRepScore(rep_number=r["rep_number"], scores=r["scores"])
            for r in rep_scores
        ]
        
        # Create input
        input_data = AssessmentInput(
            exercise_name=exercise_name,
            exercise_id=exercise_id,
            view_type=view_type,
            recognition_confidence=recognition_confidence,
            rep_scores=per_rep_scores,
        )
        
        # Create initial state
        initial_state = CoachingState(
            input=input_data,
            exercise_criteria=[],
            rep_analysis=None,
            llm_feedback="",
            warnings=[],
            final_response=None,
            error=None,
        )
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Return structured response
        return FeedbackResponse(**result["final_response"])


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create agent
    agent = CoachingAgent()
    
    # 12-rep example with realistic fatigue pattern
    # Scores are strong early, drop slightly mid-set, drop more in final reps
    example_rep_scores = [
        # Early reps (1-4): Strong form
        {"rep_number": 1, "scores": {
            "Starting position": 9.0, "Top position": 8.5, 
            "Elbow path": 8.0, "Tempo": 9.0, "Core stability": 8.5
        }},
        {"rep_number": 2, "scores": {
            "Starting position": 8.8, "Top position": 8.3, 
            "Elbow path": 8.2, "Tempo": 8.8, "Core stability": 8.5
        }},
        {"rep_number": 3, "scores": {
            "Starting position": 8.5, "Top position": 8.0, 
            "Elbow path": 7.8, "Tempo": 8.5, "Core stability": 8.2
        }},
        {"rep_number": 4, "scores": {
            "Starting position": 8.5, "Top position": 8.0, 
            "Elbow path": 7.5, "Tempo": 8.5, "Core stability": 8.0
        }},
        # Mid reps (5-8): Slight fatigue, minor drops
        {"rep_number": 5, "scores": {
            "Starting position": 8.2, "Top position": 7.5, 
            "Elbow path": 7.0, "Tempo": 8.0, "Core stability": 7.8
        }},
        {"rep_number": 6, "scores": {
            "Starting position": 8.0, "Top position": 7.2, 
            "Elbow path": 6.8, "Tempo": 7.8, "Core stability": 7.5
        }},
        {"rep_number": 7, "scores": {
            "Starting position": 7.8, "Top position": 7.0, 
            "Elbow path": 6.5, "Tempo": 7.5, "Core stability": 7.5
        }},
        {"rep_number": 8, "scores": {
            "Starting position": 7.5, "Top position": 6.8, 
            "Elbow path": 6.2, "Tempo": 7.2, "Core stability": 7.2
        }},
        # Late reps (9-12): Clear fatigue, form breakdown
        {"rep_number": 9, "scores": {
            "Starting position": 7.2, "Top position": 6.5, 
            "Elbow path": 5.8, "Tempo": 6.8, "Core stability": 6.8
        }},
        {"rep_number": 10, "scores": {
            "Starting position": 7.0, "Top position": 6.0, 
            "Elbow path": 5.5, "Tempo": 6.5, "Core stability": 6.5
        }},
        {"rep_number": 11, "scores": {
            "Starting position": 6.8, "Top position": 5.8, 
            "Elbow path": 5.0, "Tempo": 6.2, "Core stability": 6.0
        }},
        {"rep_number": 12, "scores": {
            "Starting position": 6.5, "Top position": 5.5, 
            "Elbow path": 4.8, "Tempo": 6.0, "Core stability": 5.8
        }},
    ]
    
    # Generate feedback
    print("Generating feedback for 12-rep set with fatigue pattern...")
    print("=" * 70)
    
    response = agent.generate_feedback(
        exercise_id=1,
        exercise_name="Dumbbell Shoulder Press",
        rep_scores=example_rep_scores,
        recognition_confidence=0.92,
        view_type="front",
    )
    
    # Print results
    print(f"\n📋 Exercise: {response.exercise_name}")
    print(f"📊 Reps Analyzed: {response.rep_count}")
    print(f"🎯 Overall Score: {response.overall_score:.1f}/10")
    print(f"🔄 Consistency: {response.consistency_score:.1f}/10")
    print(f"😓 Fatigue Detected: {'Yes' if response.fatigue_detected else 'No'}")
    print(f"✅ Recognition Confidence: {response.recognition_confidence:.0%}")
    
    print("\n" + "=" * 70)
    print("📈 AGGREGATED SCORES (mean across all reps)")
    print("=" * 70)
    for criterion, score in response.aggregated_scores.items():
        trend = response.trends.get(criterion, "stable")
        trend_icon = {"improving": "↑", "declining": "↓", "stable": "→"}[trend]
        print(f"  {criterion}: {score:.1f}/10 {trend_icon}")
    
    print("\n" + "=" * 70)
    print("💬 COACH FEEDBACK")
    print("=" * 70)
    print(response.feedback_summary)
    
    if response.warnings:
        print("\n⚠️ Warnings:")
        for warning in response.warnings:
            print(f"  • {warning}")
