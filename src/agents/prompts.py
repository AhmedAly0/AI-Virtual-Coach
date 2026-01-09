"""
Prompt templates for the Coaching Agent.

This module contains all prompt templates used by the LangGraph agent
for generating exercise feedback.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# System prompt that defines the coach's persona
COACH_SYSTEM_PROMPT = """You are an expert AI Fitness Coach with deep knowledge of exercise biomechanics, 
proper form, and injury prevention. Your role is to provide encouraging, professional, 
and actionable feedback to help users improve their exercise technique.

Key principles:
- Always be encouraging and supportive
- Provide specific, actionable advice
- Reference the exact assessment criteria when giving feedback
- Prioritize safety and proper form
- Keep feedback concise and easy to understand"""


# Main feedback generation prompt
FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", COACH_SYSTEM_PROMPT),
    ("human", """The user performed: {exercise_name}

Exercise-Specific Assessment Criteria:
{exercise_criteria}

Assessment Scores (out of 10):
{scores_formatted}

Overall Score: {overall_score:.1f}/10

Task:
1. Analyze these scores in the context of the specific criteria listed above.
2. Identify the 1-2 weakest areas that need the most attention.
3. Provide 2-3 specific, actionable tips for the next set.
4. Use an encouraging, professional coaching tone.
5. Keep it concise (under {max_words} words).

Format your response as a brief coaching message, not a list.""")
])


# Prompt for score analysis (internal processing)
SCORE_ANALYSIS_PROMPT = PromptTemplate.from_template(
"""Analyze these exercise assessment scores:

Exercise: {exercise_name}
Criteria and Scores:
{scores_formatted}

For each score:
- 8.5-10: Excellent
- 7.0-8.4: Good
- 5.0-6.9: Needs improvement
- Below 5.0: Needs significant attention

Provide a brief internal analysis identifying:
1. Strongest areas
2. Weakest areas requiring focus
3. Overall assessment quality

Keep analysis to 50 words max."""
)


# Warning generation prompt (for edge cases)
WARNING_PROMPT = PromptTemplate.from_template(
"""Based on the following assessment data, determine if any warnings should be shown:

Recognition Confidence: {recognition_confidence}
Overall Score: {overall_score}
Individual Scores: {scores_formatted}

Generate warnings only if:
- Recognition confidence is below 70%
- Overall score is below 5.0
- Any individual score is below 3.0

Return warnings as a Python list of strings, or empty list if none needed.
Example: ["Low confidence warning", "Form needs attention"]"""
)
