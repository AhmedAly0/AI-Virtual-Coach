"""
Prompt templates for the Coaching Agent.

This module contains all prompt templates used by the LangGraph agent
for generating exercise feedback with per-rep analysis context.
"""

from langchain_core.prompts import ChatPromptTemplate


# System prompt that defines the coach's persona
COACH_SYSTEM_PROMPT = """You are an expert AI Fitness Coach with deep knowledge of exercise biomechanics, 
proper form, and injury prevention. Your role is to provide encouraging, professional, 
and actionable feedback to help users improve their exercise technique.

Key principles:
- Always be encouraging and supportive
- Provide specific, actionable advice based on the per-rep data
- Reference specific reps when form issues occurred
- Identify patterns like fatigue-induced form breakdown
- Prioritize safety and proper form
- Keep feedback concise and conversational"""


# Main feedback generation prompt with per-rep context
FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", COACH_SYSTEM_PROMPT),
    ("human", """The user performed: {exercise_name} ({rep_count} reps)

Exercise-Specific Assessment Criteria:
{exercise_criteria}

═══════════════════════════════════════════════════════════════
PER-REP ASSESSMENT BREAKDOWN
═══════════════════════════════════════════════════════════════
{per_rep_breakdown}

═══════════════════════════════════════════════════════════════
AGGREGATED ANALYSIS (computed from all reps)
═══════════════════════════════════════════════════════════════
Overall Score: {overall_score:.1f}/10
Consistency Score: {consistency_score:.1f}/10 (how steady form was across reps)

Per-Criterion Summary:
{criterion_summary}

Trend Analysis:
- Strongest: {strongest_criterion}
- Weakest: {weakest_criterion}
{fatigue_analysis}

═══════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════
Based on this detailed per-rep assessment data:

1. Acknowledge what the user did well (reference specific strong criteria)
2. Identify the key issue(s) - be specific about WHICH reps had problems
3. If fatigue was detected, address it directly (e.g., "I noticed your form dropped on reps 10-12...")
4. Provide 2-3 specific, actionable tips for the next set
5. End with encouragement

Keep your response under {max_words} words. Be conversational, not a bullet list.""")
])


# Helper function to format per-rep breakdown for prompt
def format_per_rep_breakdown(rep_scores: list, criteria_names: list[str]) -> str:
    """
    Format per-rep scores into a readable table for the LLM.
    
    Args:
        rep_scores: List of PerRepScore objects
        criteria_names: List of criterion names (shortened for display)
        
    Returns:
        Formatted string table of all rep scores
    """
    if not rep_scores:
        return "No rep data available."
    
    lines = []
    
    # Header with shortened criteria names
    short_names = []
    for name in criteria_names:
        # Take first word or abbreviate
        if ":" in name:
            short = name.split(":")[0][:12]
        else:
            short = name[:12]
        short_names.append(short)
    
    header = "Rep  | " + " | ".join(f"{n:^8}" for n in short_names) + " | Avg"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Each rep row
    for rep in rep_scores:
        scores = [rep.scores.get(c, 0) for c in criteria_names]
        avg = sum(scores) / len(scores) if scores else 0
        
        row = f"{rep.rep_number:>3}  | "
        row += " | ".join(f"{s:^8.1f}" for s in scores)
        row += f" | {avg:.1f}"
        lines.append(row)
    
    return "\n".join(lines)


def format_criterion_summary(criterion_trends: list) -> str:
    """
    Format criterion trends for the prompt.
    
    Args:
        criterion_trends: List of CriterionTrend objects
        
    Returns:
        Formatted summary string
    """
    lines = []
    for trend in criterion_trends:
        trend_icon = {"improving": "↑", "declining": "↓", "stable": "→"}.get(trend.trend, "→")
        weak_reps = f" (weakest on reps {trend.weakest_reps})" if trend.weakest_reps else ""
        lines.append(
            f"• {trend.criterion}: {trend.mean:.1f}/10 (σ={trend.std:.2f}) {trend_icon}{weak_reps}"
        )
    return "\n".join(lines)


def format_fatigue_analysis(fatigue_detected: bool, fatigue_details: str | None) -> str:
    """Format fatigue analysis section."""
    if not fatigue_detected:
        return "- Fatigue: Not detected - form remained consistent throughout"
    return f"- ⚠️ FATIGUE DETECTED: {fatigue_details}"
