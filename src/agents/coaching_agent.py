"""
Coaching Agent for AI Virtual Coach - LangGraph Implementation.

This agent uses LangGraph to create a stateful workflow that:
1. Loads exercise-specific criteria
2. Analyzes assessment scores
3. Generates personalized feedback using Gemini LLM
4. Formats the final response

Based on PRD Section 6.2.6:
- Interpret aggregated scores using rule-based logic
- Generate reflective, post-exercise feedback
- LLM usage for natural language generation
"""

from typing import Optional, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .state import CoachingState, AssessmentInput
from .prompts import FEEDBACK_PROMPT
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
    scores: dict[str, float] = Field(description="Assessment scores per criterion")
    overall_score: float = Field(description="Mean of all scores")
    feedback_summary: str = Field(description="LLM-generated coaching feedback")
    detailed_feedback: list[str] = Field(description="Per-criterion feedback with icons")
    warnings: list[str] = Field(description="Any warnings about the assessment")


# ============================================================================
# Graph Nodes
# ============================================================================

def load_criteria_node(state: CoachingState) -> dict:
    """
    Node 1: Load exercise-specific criteria based on exercise ID.
    """
    try:
        exercise_id = state["input"]["exercise_id"]
        exercise_name = state["input"]["exercise_name"]
        
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
    Node 2: Analyze scores and generate warnings.
    """
    scores = state["input"]["scores"]
    recognition_confidence = state["input"]["recognition_confidence"]
    
    # Calculate overall score
    score_values = list(scores.values())
    overall = sum(score_values) / len(score_values) if score_values else 0
    
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
    
    # Check for critically low individual scores
    for criterion, score in scores.items():
        if score < 3.0:
            warnings.append(
                f"'{criterion}' scored very low ({score:.1f}/10). "
                "This needs immediate attention."
            )
    
    # Create score analysis summary
    analysis = f"Overall: {overall:.1f}/10. "
    excellent = [k for k, v in scores.items() if v >= SCORE_THRESHOLDS["excellent"]]
    weak = [k for k, v in scores.items() if v < SCORE_THRESHOLDS["good"]]
    
    if excellent:
        analysis += f"Strong: {', '.join(excellent)}. "
    if weak:
        analysis += f"Focus on: {', '.join(weak)}."
    
    return {
        "score_analysis": analysis,
        "warnings": warnings,
    }


def generate_detailed_feedback_node(state: CoachingState) -> dict:
    """
    Node 3: Generate detailed per-criterion feedback with icons.
    """
    scores = state["input"]["scores"]
    criteria = state["exercise_criteria"]
    
    feedback = []
    
    # Pair scores with criteria (assuming same order)
    score_items = list(scores.items())
    
    for i, (criterion_name, score) in enumerate(score_items):
        # Use criterion from loaded list if available
        display_name = criteria[i] if i < len(criteria) else criterion_name
        # Just use the first part before the colon for display
        if ":" in display_name:
            display_name = display_name.split(":")[0]
        
        if score >= SCORE_THRESHOLDS["excellent"]:
            feedback.append(f"✓ {display_name}: Excellent ({score:.1f}/10)")
        elif score >= SCORE_THRESHOLDS["good"]:
            feedback.append(f"○ {display_name}: Good ({score:.1f}/10)")
        elif score >= SCORE_THRESHOLDS["needs_improvement"]:
            feedback.append(f"△ {display_name}: Needs improvement ({score:.1f}/10)")
        else:
            feedback.append(f"✗ {display_name}: Needs attention ({score:.1f}/10)")
    
    return {"detailed_feedback": feedback}


def generate_llm_feedback_node(state: CoachingState) -> dict:
    """
    Node 4: Generate LLM-based coaching feedback using Gemini.
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
        scores = state["input"]["scores"]
        exercise_name = state["input"]["exercise_name"]
        criteria = state["exercise_criteria"]
        
        scores_formatted = "\n".join(
            f"- {k}: {v}/10" for k, v in scores.items()
        )
        overall_score = sum(scores.values()) / len(scores)
        criteria_formatted = format_criteria_for_prompt(criteria)
        
        # Generate feedback
        chain = FEEDBACK_PROMPT | llm
        response = chain.invoke({
            "exercise_name": exercise_name,
            "exercise_criteria": criteria_formatted,
            "scores_formatted": scores_formatted,
            "overall_score": overall_score,
            "max_words": MAX_FEEDBACK_WORDS,
        })
        
        return {"llm_feedback": response.content}
    
    except Exception as e:
        return {"llm_feedback": f"Error generating feedback: {str(e)}"}


def format_response_node(state: CoachingState) -> dict:
    """
    Node 5: Format the final response combining all components.
    """
    input_data = state["input"]
    scores = input_data["scores"]
    overall_score = sum(scores.values()) / len(scores)
    
    response = FeedbackResponse(
        exercise_name=input_data["exercise_name"],
        exercise_id=input_data["exercise_id"],
        recognition_confidence=input_data["recognition_confidence"],
        scores=scores,
        overall_score=overall_score,
        feedback_summary=state.get("llm_feedback", ""),
        detailed_feedback=state.get("detailed_feedback", []),
        warnings=state.get("warnings", []),
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
    graph.add_node("generate_detailed", generate_detailed_feedback_node)
    graph.add_node("generate_llm", generate_llm_feedback_node)
    graph.add_node("format_response", format_response_node)
    
    # Define edges (workflow)
    graph.add_edge(START, "load_criteria")
    graph.add_edge("load_criteria", "analyze_scores")
    graph.add_edge("analyze_scores", "generate_detailed")
    graph.add_edge("generate_detailed", "generate_llm")
    graph.add_edge("generate_llm", "format_response")
    graph.add_edge("format_response", END)
    
    return graph.compile()


# ============================================================================
# Main Agent Class
# ============================================================================

class CoachingAgent:
    """
    LangGraph-based Coaching Agent.
    
    This agent uses a state graph to process assessment scores and
    generate personalized exercise feedback.
    
    Example usage:
        agent = CoachingAgent()
        response = agent.generate_feedback(
            exercise_id=1,
            exercise_name="Dumbbell Shoulder Press",
            scores={"criterion_1": 8.5, "criterion_2": 7.0, ...},
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
        scores: dict[str, float],
        recognition_confidence: float,
        view_type: str = "front",
    ) -> FeedbackResponse:
        """
        Generate comprehensive feedback for an exercise session.
        
        Args:
            exercise_id: Exercise ID (1-15)
            exercise_name: Name of the exercise performed
            scores: Dict of criterion -> score (0-10 scale)
            recognition_confidence: Confidence of exercise recognition
            view_type: Camera view used ("front" or "side")
            
        Returns:
            FeedbackResponse with all feedback components
        """
        # Prepare input state
        input_data: AssessmentInput = {
            "scores": scores,
            "exercise_name": exercise_name,
            "exercise_id": exercise_id,
            "view_type": view_type,
            "recognition_confidence": recognition_confidence,
        }
        
        initial_state: CoachingState = {
            "input": input_data,
            "exercise_criteria": [],
            "score_analysis": "",
            "llm_feedback": "",
            "detailed_feedback": [],
            "warnings": [],
            "final_response": None,
            "error": None,
        }
        
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
    
    # Example assessment scores (matching the 5 criteria for Dumbbell Shoulder Press)
    example_scores = {
        "Starting position": 8.5,
        "Top position": 7.2,
        "Elbow path": 6.0,
        "Tempo": 9.1,
        "Core stability": 7.8,
    }
    
    # Generate feedback
    print("Generating feedback...")
    response = agent.generate_feedback(
        exercise_id=1,
        exercise_name="Dumbbell Shoulder Press",
        scores=example_scores,
        recognition_confidence=0.92,
        view_type="front",
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Exercise: {response.exercise_name}")
    print(f"Overall Score: {response.overall_score:.1f}/10")
    print(f"Recognition Confidence: {response.recognition_confidence:.0%}")
    print("=" * 60)
    
    print("\n📊 Detailed Feedback:")
    for item in response.detailed_feedback:
        print(f"  {item}")
    
    print("\n💬 Coach Feedback:")
    print(response.feedback_summary)
    
    if response.warnings:
        print("\n⚠️ Warnings:")
        for warning in response.warnings:
            print(f"  • {warning}")
