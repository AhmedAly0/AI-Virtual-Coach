"""
FastAPI backend pipeline for the AI Virtual Coach mobile application.

Processes mobile pose data through a 4-stage ML pipeline:
    Stage 1: Pose Preprocessing & Feature Engineering
    Stage 2: Exercise Recognition (15-class MLP)
    Stage 3: Per-Rep Assessment (exercise-specific temporal CNN)
    Stage 4: Coaching Agent (LangGraph + Gemini LLM)
"""
