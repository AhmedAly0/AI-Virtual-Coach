# ⚠️ DEPRECATED — See [README.md](../README.md) instead

This document has been **consolidated into the unified [README.md](../README.md)** at the repository root.

## Content Moved

- **Coaching Agent Architecture** → [README.md § Coaching Agent (Deep Dive)](../README.md#coaching-agent)
- **LangGraph Workflow** → [README.md § Coaching Agent](../README.md#coaching-agent)
- **State Models** → [README.md § Coaching Agent](../README.md#coaching-agent)
- **Prompts & Config** → [README.md § Coaching Agent](../README.md#coaching-agent)
- **API Reference** → [README.md § Backend API Contract](../README.md#backend-api-contract)

## Why Consolidated?

The four separate documentation files have been merged into a single comprehensive README that serves as both a GitHub landing page and a complete technical reference.

**Please refer to [README.md](../README.md) for all information.**

---

_This file is deprecated and no longer maintained. [Click here to view the main README](../README.md)._

> **Module path:** `src/agents/`
> **Pipeline stage:** Stage 4 — Coaching Feedback Generation
> **Last updated:** February 2026

---

## Table of Contents

1.  [System Overview](#1-system-overview)
    - [1.1 Architecture Diagram](#11-architecture-diagram)
    - [1.2 Module Dependency Map](#12-module-dependency-map)
    - [1.3 Technology Stack](#13-technology-stack)
2.  [Module Reference: `__init__.py`](#2-module-reference-__init__py)
3.  [Module Reference: `state.py`](#3-module-reference-statepy)
    - [3.1 Input Models](#31-input-models)
    - [3.2 Analysis Models](#32-analysis-models)
    - [3.3 Graph State Model](#33-graph-state-model)
    - [3.4 State Lifecycle](#34-state-lifecycle)
4.  [Module Reference: `config.py`](#4-module-reference-configpy)
5.  [Module Reference: `exercise_criteria.py`](#5-module-reference-exercise_criteriapy)
6.  [Module Reference: `prompts.py`](#6-module-reference-promptspy)
7.  [Module Reference: `coaching_agent.py`](#7-module-reference-coaching_agentpy)
    - [7.1 FeedbackResponse Model](#71-feedbackresponse-model)
    - [7.2 Graph Nodes](#72-graph-nodes)
    - [7.3 Graph Construction](#73-graph-construction)
    - [7.4 CoachingAgent Class](#74-coachingagent-class)
8.  [End-to-End Walkthrough](#8-end-to-end-walkthrough)
9.  [Integration Points](#9-integration-points)
10. [Configuration Guide](#10-configuration-guide)
11. [Extensibility Guide](#11-extensibility-guide)
12. [Key Design Questions — Answered](#12-key-design-questions--answered)
13. [Troubleshooting & Common Patterns](#13-troubleshooting--common-patterns)
14. [API Quick Reference](#14-api-quick-reference)

---

## 1. System Overview

The Coaching Agent is **Stage 4** of the AI Virtual Coach pipeline — the final processing stage that transforms raw per-repetition assessment scores into personalized, actionable coaching feedback in natural language.

### Pipeline Context

```
RGB Video → Stage 1 → Stage 2 → Stage 3 → ╔═══════════════════════╗ → Mobile App
                                            ║   STAGE 4             ║
            Pose       Exercise   Per-Rep   ║   Coaching Agent      ║    Scores
            Landmarks  Recognition Aspect   ║   (LangGraph +        ║    Trends
            (MediaPipe) (CNN)      Scores   ║    Gemini LLM)        ║    Feedback
                                   (tCNN)   ╚═══════════════════════╝    Warnings
```

| Stage | Component | Output |
|-------|-----------|--------|
| 1 | MediaPipe Pose Estimation | 37 biomechanical features per frame |
| 2 | Exercise Recognition (CNN) | Exercise class (1–15) + confidence |
| 3 | Repetition Segmentation + Temporal CNN | Per-rep aspect scores (5 criteria × 0–10) |
| **4** | **Coaching Agent (this module)** | **Aggregated scores, trends, fatigue detection, NL feedback** |

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        src/agents/ Module                              │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  config.py   │    │          coaching_agent.py                   │   │
│  │              │    │                                              │   │
│  │ • API Key    │───▶│  ┌───────────────────────────────────────┐  │   │
│  │ • Model Name │    │  │      LangGraph StateGraph             │  │   │
│  │ • Thresholds │    │  │                                       │  │   │
│  └──────────────┘    │  │  START                                │  │   │
│                      │  │    │                                   │  │   │
│  ┌──────────────┐    │  │    ▼                                   │  │   │
│  │exercise_     │    │  │  ┌─────────────────┐                  │  │   │
│  │criteria.py   │───▶│  │  │ load_criteria   │ ◀── exercise_    │  │   │
│  │              │    │  │  └────────┬────────┘     criteria.py  │  │   │
│  │ • 15 exer.   │    │  │           │                           │  │   │
│  │ • 5 criteria │    │  │           ▼                           │  │   │
│  │   each       │    │  │  ┌─────────────────┐                  │  │   │
│  └──────────────┘    │  │  │ analyze_scores  │                  │  │   │
│                      │  │  └────────┬────────┘                  │  │   │
│  ┌──────────────┐    │  │           │                           │  │   │
│  │  state.py    │    │  │           ▼                           │  │   │
│  │              │    │  │  ┌─────────────────┐                  │  │   │
│  │ • Coaching   │◀──▶│  │  │analyze_rep_     │ (rule-based)    │  │   │
│  │   State      │    │  │  │  trends         │                  │  │   │
│  │ • Assessment │    │  │  └────────┬────────┘                  │  │   │
│  │   Input      │    │  │           │                           │  │   │
│  │ • PerRep     │    │  │           ▼                           │  │   │
│  │   Score      │    │  │  ┌─────────────────┐                  │  │   │
│  │ • Trends     │    │  │  │ generate_llm    │ ◀── prompts.py  │  │   │
│  └──────────────┘    │  │  │  (Gemini 2.5    │     config.py   │  │   │
│                      │  │  │   Flash)        │                  │  │   │
│  ┌──────────────┐    │  │  └────────┬────────┘                  │  │   │
│  │  prompts.py  │    │  │           │                           │  │   │
│  │              │───▶│  │           ▼                           │  │   │
│  │ • System     │    │  │  ┌─────────────────┐                  │  │   │
│  │   Prompt     │    │  │  │format_response  │                  │  │   │
│  │ • Feedback   │    │  │  └────────┬────────┘                  │  │   │
│  │   Template   │    │  │           │                           │  │   │
│  │ • Formatters │    │  │           ▼                           │  │   │
│  └──────────────┘    │  │         END → FeedbackResponse       │  │   │
│                      │  │                                       │  │   │
│                      │  └───────────────────────────────────────┘  │   │
│                      └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Dependency Map

```
__init__.py
    ├── coaching_agent.py  (CoachingAgent, FeedbackResponse)
    │       ├── state.py   (CoachingState, AssessmentInput, PerRepScore, CriterionTrend, RepTrendAnalysis)
    │       ├── prompts.py (FEEDBACK_PROMPT, format_per_rep_breakdown, format_criterion_summary, format_fatigue_analysis)
    │       ├── exercise_criteria.py (get_exercise_criteria, format_criteria_for_prompt)
    │       └── config.py  (GEMINI_API_KEY, GEMINI_MODEL_NAME, MAX_FEEDBACK_WORDS, SCORE_THRESHOLDS)
    ├── state.py           (CoachingState, AssessmentInput, PerRepScore, RepTrendAnalysis)
    └── exercise_criteria.py (get_exercise_criteria, get_all_exercises)
```

### 1.3 Technology Stack

| Component | Technology | Version / Notes |
|-----------|-----------|----------------|
| State graph framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | `StateGraph`, `START`, `END` |
| LLM integration | [LangChain Google GenAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) | `ChatGoogleGenerativeAI` |
| LLM backend | Google Gemini 2.5 Flash | Configurable via env var |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) | `BaseModel`, `Field` |
| Prompt templating | LangChain Core | `ChatPromptTemplate` |
| Environment config | `python-dotenv` | `.env` file loading |

---

## 2. Module Reference: `__init__.py`

**File:** `src/agents/__init__.py`
**Purpose:** Package initialization — defines the public API surface of the `agents` module.

### Responsibilities

- Imports and re-exports all user-facing classes and functions
- Defines `__all__` for explicit namespace control
- Acts as the single entry point for consumers

### Exported Symbols

| Symbol | Source Module | Type | Description |
|--------|-------------|------|-------------|
| `CoachingAgent` | `coaching_agent.py` | Class | Main agent orchestrator |
| `FeedbackResponse` | `coaching_agent.py` | Pydantic Model | Structured output schema |
| `CoachingState` | `state.py` | Pydantic Model | LangGraph state container |
| `AssessmentInput` | `state.py` | Pydantic Model | Input data from assessment model |
| `PerRepScore` | `state.py` | Pydantic Model | Scores for a single repetition |
| `RepTrendAnalysis` | `state.py` | Pydantic Model | Trend analysis across all reps |
| `get_exercise_criteria` | `exercise_criteria.py` | Function | Lookup criteria by ID or name |
| `get_all_exercises` | `exercise_criteria.py` | Function | List all 15 supported exercises |

### Usage

```python
from src.agents import CoachingAgent, FeedbackResponse

agent = CoachingAgent()
response: FeedbackResponse = agent.generate_feedback(
    exercise_id=1,
    exercise_name="Dumbbell Shoulder Press",
    rep_scores=[...],
    recognition_confidence=0.95,
)
```

---

## 3. Module Reference: `state.py`

**File:** `src/agents/state.py` (152 lines)
**Purpose:** Defines all Pydantic data models that flow through the LangGraph coaching agent. These models serve as the typed schema for state initialization, inter-node communication, and final output validation.

### 3.1 Input Models

#### `PerRepScore`

Represents the assessment scores for a single exercise repetition.

```python
class PerRepScore(BaseModel):
    rep_number: int       # 1-indexed repetition number
    scores: dict[str, float]  # Criterion name → score (0–10 scale)
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `rep_number` | `int` | ≥ 1 | 1-indexed repetition number |
| `scores` | `dict[str, float]` | Values in [0, 10] | Map of criterion name to score |

**Computed Properties:**

| Property | Return Type | Description |
|----------|-------------|-------------|
| `average` | `float` | Mean score across all criteria for this rep. Returns `0.0` if `scores` is empty. |

**Example:**
```python
rep = PerRepScore(
    rep_number=1,
    scores={
        "Starting position": 8.5,
        "Top position": 7.0,
        "Elbow path": 8.0,
        "Tempo": 9.0,
        "Core stability": 8.5,
    }
)
print(rep.average)  # 8.2
```

---

#### `AssessmentInput`

Container for all data arriving from Stage 3 (the assessment model) for a single exercise session/set.

```python
class AssessmentInput(BaseModel):
    exercise_name: str
    exercise_id: int              # 1–15
    view_type: str = "front"      # "front" or "side"
    recognition_confidence: float  # [0.0, 1.0]
    rep_scores: list[PerRepScore]
```

| Field | Type | Constraints | Default | Description |
|-------|------|-------------|---------|-------------|
| `exercise_name` | `str` | — | *required* | Human-readable exercise name |
| `exercise_id` | `int` | 1–15 | *required* | Exercise class ID from recognition model |
| `view_type` | `str` | `"front"` or `"side"` | `"front"` | Camera angle used for recording |
| `recognition_confidence` | `float` | `ge=0.0, le=1.0` | *required* | Softmax confidence from Stage 2 |
| `rep_scores` | `list[PerRepScore]` | — | *required* | Per-rep assessment scores from Stage 3 |

**Computed Properties:**

| Property | Return Type | Description |
|----------|-------------|-------------|
| `rep_count` | `int` | Number of repetitions assessed (`len(rep_scores)`) |
| `aggregated_scores` | `dict[str, float]` | Mean score per criterion, averaged across all reps |
| `overall_score` | `float` | Grand mean across all criteria and all reps |

**Aggregation Logic:**

The `aggregated_scores` property computes an **unweighted arithmetic mean** per criterion across all repetitions. All reps contribute equally regardless of quality:

$$\bar{s}_c = \frac{1}{N_{\text{reps}}} \sum_{r=1}^{N_{\text{reps}}} s_{r,c}$$

The `overall_score` is the mean of all criterion means:

$$\bar{S} = \frac{1}{|C|} \sum_{c \in C} \bar{s}_c$$

where $C$ is the set of criteria and $N_{\text{reps}}$ is the number of repetitions.

**Example:**
```python
input_data = AssessmentInput(
    exercise_name="Dumbbell Shoulder Press",
    exercise_id=1,
    view_type="front",
    recognition_confidence=0.92,
    rep_scores=[
        PerRepScore(rep_number=1, scores={"Tempo": 9.0, "Core": 8.5}),
        PerRepScore(rep_number=2, scores={"Tempo": 8.0, "Core": 7.5}),
    ]
)
print(input_data.rep_count)          # 2
print(input_data.aggregated_scores)  # {"Tempo": 8.5, "Core": 8.0}
print(input_data.overall_score)      # 8.25
```

---

### 3.2 Analysis Models

#### `CriterionTrend`

Trend analysis results for a single assessment criterion across all reps.

```python
class CriterionTrend(BaseModel):
    criterion: str
    mean: float
    std: float
    min_score: float
    max_score: float
    trend: str              # "improving", "declining", or "stable"
    trend_magnitude: float  # Absolute change from early to late reps
    weakest_reps: list[int] # Rep numbers with lowest scores
```

| Field | Type | Description |
|-------|------|-------------|
| `criterion` | `str` | Criterion name (e.g., `"Elbow path"`) |
| `mean` | `float` | Mean score across all reps |
| `std` | `float` | Standard deviation across reps |
| `min_score` | `float` | Lowest score observed for this criterion |
| `max_score` | `float` | Highest score observed for this criterion |
| `trend` | `str` | Direction: `"improving"` / `"declining"` / `"stable"` |
| `trend_magnitude` | `float` | `abs(late_mean - early_mean)` |
| `weakest_reps` | `list[int]` | Up to 3 rep numbers with lowest scores |

**Trend Detection Algorithm:**

The trend is determined by comparing the mean of the **first 3 reps** against the mean of the **last 3 reps**:

```
trend_diff = mean(last_3_reps) - mean(first_3_reps)

if trend_diff > 0.5  → "improving"
if trend_diff < -0.5 → "declining"
otherwise            → "stable"
```

The threshold of ±0.5 on the 0–10 scale means a 5% shift is needed to trigger a non-stable classification.

---

#### `RepTrendAnalysis`

Complete trend analysis across all repetitions for a single set.

```python
class RepTrendAnalysis(BaseModel):
    rep_count: int
    criterion_trends: list[CriterionTrend]
    fatigue_detected: bool
    fatigue_details: Optional[str] = None
    consistency_score: float        # [0.0, 10.0]
    strongest_criterion: str
    weakest_criterion: str
    per_rep_averages: list[float]
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `rep_count` | `int` | ≥ 1 | Number of reps in the set |
| `criterion_trends` | `list[CriterionTrend]` | — | Per-criterion trend analysis |
| `fatigue_detected` | `bool` | — | `True` if ≥ 2 criteria show `"declining"` trend |
| `fatigue_details` | `Optional[str]` | — | Human-readable fatigue description |
| `consistency_score` | `float` | [0, 10] | Lower variance → higher consistency |
| `strongest_criterion` | `str` | — | Criterion with highest mean score |
| `weakest_criterion` | `str` | — | Criterion with lowest mean score |
| `per_rep_averages` | `list[float]` | — | Mean score per rep (for visualization) |

**Consistency Score Formula:**

$$\text{consistency} = \max\!\Big(0,\;\min\!\big(10,\;10 - 3\,\bar{\sigma}\big)\Big)$$

where $\bar{\sigma}$ is the mean standard deviation across all criteria. A perfectly consistent set (σ = 0) scores 10; a set with $\bar{\sigma} \geq 3.33$ scores 0.

**Fatigue Detection Rule:**

```
fatigue_detected = (number of criteria with trend == "declining") >= 2
```

This multi-criteria threshold reduces false positives from single-aspect noise.

---

### 3.3 Graph State Model

#### `CoachingState`

The central state object passed between all LangGraph nodes. Each node reads from and writes to this state.

```python
class CoachingState(BaseModel):
    input: AssessmentInput
    exercise_criteria: list[str] = []
    rep_analysis: Optional[RepTrendAnalysis] = None
    llm_feedback: str = ""
    warnings: list[str] = []
    final_response: Optional[dict] = None
    error: Optional[str] = None
```

| Field | Type | Set By Node | Description |
|-------|------|-------------|-------------|
| `input` | `AssessmentInput` | *Initialization* | Immutable input data from upstream |
| `exercise_criteria` | `list[str]` | `load_criteria` | Exercise-specific criteria strings |
| `rep_analysis` | `Optional[RepTrendAnalysis]` | `analyze_rep_trends` | Computed trend analysis |
| `llm_feedback` | `str` | `generate_llm` | LLM-generated coaching text |
| `warnings` | `list[str]` | `analyze_scores` | System-level warnings |
| `final_response` | `Optional[dict]` | `format_response` | Serialized `FeedbackResponse` |
| `error` | `Optional[str]` | Any node (on failure) | Error message if a node fails |

### 3.4 State Lifecycle

```
                           INITIALIZATION
                    ┌─────────────────────────┐
                    │ input = AssessmentInput  │
                    │ exercise_criteria = []   │
                    │ rep_analysis = None      │
                    │ llm_feedback = ""        │
                    │ warnings = []            │
                    │ final_response = None    │
                    │ error = None             │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
Node 1        │ load_criteria_node                   │
              │ WRITES: exercise_criteria = [5 strs] │
              │         error (on failure)           │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
Node 2        │ analyze_scores_node                  │
              │ READS:  input.aggregated_scores      │
              │         input.recognition_confidence │
              │         input.overall_score          │
              │ WRITES: warnings = [str, ...]        │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
Node 3        │ analyze_rep_trends_node              │
              │ READS:  input.rep_scores             │
              │ WRITES: rep_analysis = RepTrend...   │
              │         (None if < 2 reps)           │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
Node 4        │ generate_llm_feedback_node           │
              │ READS:  input, exercise_criteria,    │
              │         rep_analysis                 │
              │ WRITES: llm_feedback = "text..."     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
Node 5        │ format_response_node                 │
              │ READS:  ALL state fields             │
              │ WRITES: final_response = {dict}      │
              └──────────────────┬──────────────────┘
                                 │
                           ╔═════▼═════╗
                           ║    END    ║
                           ╚═══════════╝
```

**Key observations:**
- State is **immutable from the input perspective** — `input` is set once and never modified
- Each node returns a `dict` of field updates; LangGraph merges these into state
- The graph is **linear** (no conditional branching), so all nodes always execute
- Errors are captured in the `error` field rather than raising exceptions (graceful degradation)

---

## 4. Module Reference: `config.py`

**File:** `src/agents/config.py`
**Purpose:** Centralized configuration management. Loads secrets from environment variables and defines module-wide hyperparameters.

### Environment Setup

The module uses `python-dotenv` to load a `.env` file from the project root:

```
ai-virtual-coach/
├── .env              ← Loaded automatically (git-ignored)
├── .env.example      ← Template for developers
└── src/agents/config.py
```

**`.env.example` contents:**
```dotenv
# AI Virtual Coach Environment Variables
# Copy this file to .env and fill in your values

# Gemini API Key (get yours at https://aistudio.google.com/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Override default model
# GEMINI_MODEL_NAME=gemini-2.5-flash
```

### Configuration Variables

| Variable | Type | Default | Source | Description |
|----------|------|---------|--------|-------------|
| `GEMINI_API_KEY` | `str` | `""` | `GEMINI_API_KEY` env var | Google AI Studio API key |
| `GEMINI_MODEL_NAME` | `str` | `"gemini-2.5-flash"` | `GEMINI_MODEL_NAME` env var | Gemini model identifier |
| `DEFAULT_USE_LLM` | `bool` | `True` | Hardcoded | Whether to use LLM for feedback |
| `MAX_FEEDBACK_WORDS` | `int` | `150` | Hardcoded | Word limit for LLM output |
| `SCORE_THRESHOLDS` | `dict[str, float]` | See below | Hardcoded | Score categorization boundaries |

### Score Thresholds

| Category | Threshold | Score Range |
|----------|-----------|-------------|
| `"excellent"` | `8.5` | $[8.5, 10.0]$ |
| `"good"` | `7.0` | $[7.0, 8.5)$ |
| `"needs_improvement"` | `5.0` | $[5.0, 7.0)$ |
| `"poor"` | `0.0` | $[0.0, 5.0)$ |

These thresholds align with the backend API contract defined in `src/pipelines/backend_api_contract.md`.

### Side Effects

- **On import:** Reads `.env` file from disk via `load_dotenv()`
- **Warning:** If `GEMINI_API_KEY` is empty, emits a `UserWarning` at import time (non-blocking)
- The module does **not** validate the API key format — validation happens at LLM call time

---

## 5. Module Reference: `exercise_criteria.py`

**File:** `src/agents/exercise_criteria.py` (223 lines)
**Purpose:** Domain knowledge layer that maps each of the 15 supported resistance-training exercises to their five biomechanical assessment criteria. These criteria are the same aspects scored by the Stage 3 temporal CNN assessment model.

### Data Structure

The primary data store is a module-level dictionary:

```python
EXERCISE_CRITERIA: dict[int, tuple[str, list[str]]]
```

**Schema:** `{exercise_id: (exercise_name, [criterion_1, criterion_2, ..., criterion_5])}`

### Complete Exercise Registry

| ID | Exercise Name | Criteria |
|----|--------------|----------|
| 1 | Dumbbell Shoulder Press | Starting position, Top position, Elbow path, Tempo, Core stability |
| 2 | Hammer Curls | Elbow position, Wrist orientation, Movement control, Range of motion, Tempo |
| 3 | Standing Dumbbell Front Raises | Arm path, Raise height, Core engagement, Wrist alignment, Controlled tempo |
| 4 | Lateral Raises | Arm angle, Raise height, Elbow position, Trap engagement, No momentum |
| 5 | Bulgarian Split Squat | Rear foot, Front knee tracking, Torso angle, Depth, Balance |
| 6 | EZ Bar Curls | Grip, Elbow position, Bar path, Full ROM, No shoulder swing |
| 7 | Incline Dumbbell Bench Press | Bench angle, Dumbbell path, Elbow angle, Back position, Wrist stability |
| 8 | Overhead Triceps Extension | Elbow position, ROM, Wrist position, Core tightness, Tempo |
| 9 | Shrugs | Shoulder path, ROM, Neck stability, Weight control, Pause at top |
| 10 | Weighted Squats | Stance, Depth, Knee tracking, Back posture, Weight distribution |
| 11 | Seated Biceps Curls | Elbow position, Shoulder stability, ROM, Wrist position, Back support |
| 12 | Triceps Kickbacks | Upper arm position, Elbow movement, Full extension, No torso swing, Neutral spine |
| 13 | Rows | Back angle, Elbow path, Shoulder retraction, Pulling motion, Stable base |
| 14 | Deadlift | Spine, Hips, Bar path, Lockout, Tempo |
| 15 | Calf Raises | ROM, Knee status, Balance, Tempo, Peak hold |

### Functions

#### `get_exercise_criteria(exercise_id?, exercise_name?) → (str, list[str])`

Lookup exercise criteria by ID or name.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `exercise_id` | `Optional[int]` | One of the two required | Exercise ID (1–15) |
| `exercise_name` | `Optional[str]` | One of the two required | Exercise name (case-insensitive) |

**Returns:** `(exercise_name: str, criteria: list[str])` — a tuple of the canonical exercise name and its 5 criteria strings.

**Lookup Strategy:**
1. If `exercise_id` is provided → direct dict lookup
2. If `exercise_name` is provided:
   - **Exact match** (case-insensitive) against `EXERCISE_NAME_TO_ID`
   - **Partial match** (substring containment in either direction)
3. If neither provided → raises `ValueError`
4. If no match found → raises `ValueError`

**Example:**
```python
name, criteria = get_exercise_criteria(exercise_id=10)
# ("Weighted Squats", ["Stance: shoulder-width...", "Depth: thighs parallel...", ...])

name, criteria = get_exercise_criteria(exercise_name="squat")
# ("Weighted Squats", [...])  — partial match works
```

---

#### `format_criteria_for_prompt(criteria: list[str]) → str`

Formats a list of criteria strings into a bullet-point format suitable for LLM prompt injection.

**Example:**
```python
format_criteria_for_prompt(["Stance: wide", "Depth: full"])
# "• Stance: wide\n• Depth: full"
```

---

#### `get_all_exercises() → list[tuple[int, str]]`

Returns all 15 supported exercises as `(id, name)` pairs.

```python
get_all_exercises()
# [(1, "Dumbbell Shoulder Press"), (2, "Hammer Curls"), ..., (15, "Calf Raises")]
```

---

#### `get_formatted_criteria(exercise_id?, exercise_name?) → str`

Convenience wrapper combining `get_exercise_criteria()` + `format_criteria_for_prompt()`.

---

### Internal Data Structure

```python
EXERCISE_NAME_TO_ID: dict[str, int]
# {"dumbbell shoulder press": 1, "hammer curls": 2, ...}
# Auto-generated from EXERCISE_CRITERIA at module load time (lowercase keys)
```

---

## 6. Module Reference: `prompts.py`

**File:** `src/agents/prompts.py` (136 lines)
**Purpose:** Contains all LLM prompt templates and formatting utilities. Responsible for serializing numerical assessment data into a structured context that the Gemini LLM can interpret to produce coaching feedback.

### Constants

#### `COACH_SYSTEM_PROMPT`

Defines the AI coach's persona and behavioral guidelines:

```
You are an expert AI Fitness Coach with deep knowledge of exercise
biomechanics, proper form, and injury prevention...
```

**Key behavioral directives:**
- Always encouraging and supportive
- Provide specific, actionable advice referencing per-rep data
- Reference specific rep numbers when form issues occurred
- Identify patterns like fatigue-induced form breakdown
- Prioritize safety and proper form
- Keep feedback concise and conversational

---

#### `FEEDBACK_PROMPT`

A `ChatPromptTemplate` (LangChain) with the following template variables:

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `{exercise_name}` | `str` | `AssessmentInput.exercise_name` | Exercise performed |
| `{rep_count}` | `int` | `AssessmentInput.rep_count` | Number of reps |
| `{exercise_criteria}` | `str` | `format_criteria_for_prompt()` | Bullet-point criteria |
| `{per_rep_breakdown}` | `str` | `format_per_rep_breakdown()` | Tabular rep-by-rep scores |
| `{overall_score}` | `float` | `AssessmentInput.overall_score` | Grand mean score |
| `{consistency_score}` | `float` | `RepTrendAnalysis.consistency_score` | Form consistency rating |
| `{criterion_summary}` | `str` | `format_criterion_summary()` | Per-criterion stats with trends |
| `{strongest_criterion}` | `str` | `RepTrendAnalysis.strongest_criterion` | Best-performing aspect |
| `{weakest_criterion}` | `str` | `RepTrendAnalysis.weakest_criterion` | Worst-performing aspect |
| `{fatigue_analysis}` | `str` | `format_fatigue_analysis()` | Fatigue detection summary |
| `{max_words}` | `int` | `config.MAX_FEEDBACK_WORDS` | Word limit for output |

**Prompt Structure (abridged):**
```
[System] You are an expert AI Fitness Coach...

[Human]  The user performed: {exercise_name} ({rep_count} reps)
         
         Exercise-Specific Assessment Criteria:
         {exercise_criteria}
         
         ═══ PER-REP ASSESSMENT BREAKDOWN ═══
         {per_rep_breakdown}
         
         ═══ AGGREGATED ANALYSIS ═══
         Overall Score: {overall_score}/10
         Consistency: {consistency_score}/10
         Per-Criterion Summary: {criterion_summary}
         Trend Analysis: strongest / weakest / fatigue
         
         ═══ YOUR TASK ═══
         1. Acknowledge strengths (reference criteria)
         2. Identify key issues (reference specific reps)
         3. Address fatigue if detected
         4. Provide 2–3 actionable tips
         5. End with encouragement
         Keep under {max_words} words. Conversational tone.
```

### Functions

#### `format_per_rep_breakdown(rep_scores, criteria_names) → str`

Generates a tabular display of all per-rep scores for LLM consumption.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rep_scores` | `list[PerRepScore]` | List of per-rep score objects |
| `criteria_names` | `list[str]` | Full criterion names (will be truncated to 12 chars) |

**Returns:** Formatted ASCII table string.

**Example output:**
```
Rep  | Starting pos | Top position | Elbow path   | Tempo        | Core stab    | Avg
--------------------------------------------------------------------------------------------
  1  |     9.0      |     8.5      |     8.0      |     9.0      |     8.5      | 8.6
  2  |     8.8      |     8.3      |     8.2      |     8.8      |     8.5      | 8.5
 ...
 12  |     6.5      |     5.5      |     4.8      |     6.0      |     5.8      | 5.7
```

**Criteria name truncation:** Names are shortened to the first 12 characters. If the name contains a colon (`:`), only the text before the colon is used (up to 12 chars).

---

#### `format_criterion_summary(criterion_trends) → str`

Formats trend analysis into a bullet list with statistics and direction icons.

| Parameter | Type | Description |
|-----------|------|-------------|
| `criterion_trends` | `list[CriterionTrend]` | Computed trend objects |

**Example output:**
```
• Starting position: 7.9/10 (σ=0.82) ↓ (weakest on reps [9, 10, 11])
• Elbow path: 6.8/10 (σ=1.21) ↓ (weakest on reps [10, 11, 12])
• Tempo: 7.7/10 (σ=1.08) ↓
```

**Trend icons:** `↑` improving, `↓` declining, `→` stable.

---

#### `format_fatigue_analysis(fatigue_detected, fatigue_details) → str`

Returns a one-line summary of fatigue status.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fatigue_detected` | `bool` | Whether fatigue was detected |
| `fatigue_details` | `str \| None` | Description of fatigue pattern |

**Returns:**
- If no fatigue: `"- Fatigue: Not detected - form remained consistent throughout"`
- If fatigue: `"- ⚠️ FATIGUE DETECTED: {fatigue_details}"`

---

## 7. Module Reference: `coaching_agent.py`

**File:** `src/agents/coaching_agent.py` (564 lines)
**Purpose:** Main orchestration module. Defines the LangGraph state graph, all processing nodes, and the public `CoachingAgent` class. This is the entry point for generating coaching feedback.

### 7.1 FeedbackResponse Model

The structured output schema sent to the mobile application.

```python
class FeedbackResponse(BaseModel):
    exercise_name: str
    exercise_id: int
    recognition_confidence: float
    rep_count: int
    aggregated_scores: dict[str, float]
    overall_score: float
    rep_scores: list[dict]
    consistency_score: float
    fatigue_detected: bool
    trends: dict[str, str]
    feedback_summary: str
    warnings: list[str]
```

| Field | Type | Description |
|-------|------|-------------|
| `exercise_name` | `str` | Canonical exercise name |
| `exercise_id` | `int` | Exercise class (1–15) |
| `recognition_confidence` | `float` | Recognition model confidence [0, 1] |
| `rep_count` | `int` | Number of reps analyzed |
| `aggregated_scores` | `dict[str, float]` | Mean score per criterion across all reps |
| `overall_score` | `float` | Grand mean score |
| `rep_scores` | `list[dict]` | Serialized `PerRepScore` objects (via `model_dump()`) |
| `consistency_score` | `float` | Form consistency [0, 10] |
| `fatigue_detected` | `bool` | Whether fatigue was detected |
| `trends` | `dict[str, str]` | Criterion → trend direction |
| `feedback_summary` | `str` | LLM-generated coaching text |
| `warnings` | `list[str]` | System warnings |

**Example `FeedbackResponse` (serialized):**
```json
{
    "exercise_name": "Dumbbell Shoulder Press",
    "exercise_id": 1,
    "recognition_confidence": 0.92,
    "rep_count": 12,
    "aggregated_scores": {
        "Starting position": 7.9,
        "Top position": 7.0,
        "Elbow path": 6.8,
        "Tempo": 7.7,
        "Core stability": 7.4
    },
    "overall_score": 7.4,
    "rep_scores": [
        {"rep_number": 1, "scores": {"Starting position": 9.0, ...}},
        ...
    ],
    "consistency_score": 7.2,
    "fatigue_detected": true,
    "trends": {
        "Starting position": "declining",
        "Elbow path": "declining",
        "Tempo": "declining",
        ...
    },
    "feedback_summary": "Fantastic effort on your Dumbbell Shoulder Press! ...",
    "warnings": []
}
```

---

### 7.2 Graph Nodes

The LangGraph workflow consists of 5 sequential nodes:

#### Node 1: `load_criteria_node(state) → dict`

**Purpose:** Retrieves exercise-specific assessment criteria from the domain knowledge layer.

| Reads | Writes |
|-------|--------|
| `state.input.exercise_id` | `exercise_criteria: list[str]` |
| `state.input.exercise_name` | `error: str` (on failure) |

**Logic:**
1. Try lookup by `exercise_id` first
2. If that fails, fall back to lookup by `exercise_name`
3. On any exception, set `exercise_criteria = []` and populate `error`

**Failure behavior:** Graceful — sets empty criteria list so downstream nodes can still operate (LLM receives no criteria context but can still generate generic feedback).

---

#### Node 2: `analyze_scores_node(state) → dict`

**Purpose:** Generates system-level warnings based on aggregated score analysis.

| Reads | Writes |
|-------|--------|
| `state.input.aggregated_scores` | `warnings: list[str]` |
| `state.input.recognition_confidence` | |
| `state.input.overall_score` | |

**Warning triggers:**

| Condition | Warning Message |
|-----------|----------------|
| `recognition_confidence < 0.7` | Low confidence warning — results may be less reliable |
| `overall_score < 5.0` | Below-average form — review proper technique |
| Any criterion `score < 3.0` | Critical score warning — needs immediate attention |

---

#### Node 3: `analyze_rep_trends_node(state) → dict`

**Purpose:** Rule-based trend analysis computed entirely in Python (no LLM calls). This is the core analytical engine of the coaching agent.

| Reads | Writes |
|-------|--------|
| `state.input.rep_scores` | `rep_analysis: RepTrendAnalysis` |
| `state.input.rep_count` | (or `None` if < 2 reps) |

**Short-circuit:** Returns `{"rep_analysis": None}` if `rep_count < 2` (trend analysis requires ≥ 2 data points).

**Algorithm — per criterion:**

```python
for each criterion:
    scores = [rep.scores[criterion] for rep in all_reps]
    
    # Basic statistics
    mean, std, min, max = compute_stats(scores)
    
    # Trend: compare first 3 vs last 3 reps
    early_mean = mean(scores[:3])
    late_mean  = mean(scores[-3:])
    trend_diff = late_mean - early_mean
    
    trend = "improving"  if trend_diff >  0.5
            "declining"  if trend_diff < -0.5
            "stable"     otherwise
    
    # Weakest reps: below (mean - std) or below 5.0
    threshold = max(mean - std, 5.0)
    weakest = [rep for rep in reps if rep.score < threshold][:3]
```

**Fatigue detection:**
```python
declining_count = count(criteria where trend == "declining")
fatigue_detected = declining_count >= 2
```

**Consistency score:**
```python
consistency = clamp(10 - 3 * mean(all_stds), 0, 10)
```

**Design rationale:** This node performs all computationally deterministic analysis in Python to:
1. **Reduce token usage** — the LLM receives pre-computed summaries, not raw time series
2. **Reduce latency** — rule-based logic is instant vs. LLM inference
3. **Ensure reproducibility** — same inputs always produce same trend analysis

---

#### Node 4: `generate_llm_feedback_node(state) → dict`

**Purpose:** Generates natural language coaching feedback using the Gemini LLM.

| Reads | Writes |
|-------|--------|
| `state.input` (all fields) | `llm_feedback: str` |
| `state.exercise_criteria` | |
| `state.rep_analysis` | |

**LLM Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `gemini-2.5-flash` (default) | Fast, cost-effective model |
| Temperature | `0.7` | Moderate creativity for natural-sounding feedback |
| API Key | From `config.GEMINI_API_KEY` | Required |

**Execution flow:**
1. Check if `GEMINI_API_KEY` is set — if not, return fallback message
2. Initialize `ChatGoogleGenerativeAI` with config
3. Format all prompt variables using helper functions from `prompts.py`
4. Invoke `FEEDBACK_PROMPT | llm` chain
5. Extract `response.content` as feedback text
6. On exception, return error message string (no crash)

**External API call:** This is the **only node that makes external network calls** (to Google's Gemini API).

**Fallback behavior:**
- Missing API key → `"LLM feedback unavailable: API key not configured."`
- API error → `"Error generating feedback: {error_message}"`

---

#### Node 5: `format_response_node(state) → dict`

**Purpose:** Assembles all state components into the final `FeedbackResponse` output.

| Reads | Writes |
|-------|--------|
| All state fields | `final_response: dict` |

**Logic:**
1. Extract trends from `rep_analysis.criterion_trends` into a flat `{criterion: trend}` dict
2. Construct a `FeedbackResponse` Pydantic model from all state fields
3. Serialize via `model_dump()` and store in `final_response`

---

### 7.3 Graph Construction

#### `build_coaching_graph() → StateGraph`

Constructs and compiles the LangGraph workflow.

**Graph topology (linear):**

```
START → load_criteria → analyze_scores → analyze_rep_trends → generate_llm → format_response → END
```

**Implementation:**
```python
graph = StateGraph(CoachingState)

graph.add_node("load_criteria",     load_criteria_node)
graph.add_node("analyze_scores",    analyze_scores_node)
graph.add_node("analyze_rep_trends", analyze_rep_trends_node)
graph.add_node("generate_llm",      generate_llm_feedback_node)
graph.add_node("format_response",   format_response_node)

graph.add_edge(START, "load_criteria")
graph.add_edge("load_criteria", "analyze_scores")
graph.add_edge("analyze_scores", "analyze_rep_trends")
graph.add_edge("analyze_rep_trends", "generate_llm")
graph.add_edge("generate_llm", "format_response")
graph.add_edge("format_response", END)

return graph.compile()
```

**Why linear?** The current design has no conditional routing because:
- All nodes are needed for a complete response
- Error handling is done within each node (graceful degradation)
- Future versions could add conditional edges (e.g., skip LLM if API unavailable)

---

### 7.4 CoachingAgent Class

The public-facing class that wraps the LangGraph workflow.

```python
class CoachingAgent:
    def __init__(self):
        self.graph = build_coaching_graph()
    
    def generate_feedback(
        self,
        exercise_id: int,
        exercise_name: str,
        rep_scores: list[dict],
        recognition_confidence: float,
        view_type: str = "front",
    ) -> FeedbackResponse:
        ...
```

#### `__init__(self)`

Compiles the LangGraph state graph once at construction time. The compiled graph is reusable across multiple `generate_feedback()` calls.

**Side effects:** Compiles the graph (lightweight, no I/O).

---

#### `generate_feedback(exercise_id, exercise_name, rep_scores, recognition_confidence, view_type) → FeedbackResponse`

Main entry point for generating coaching feedback.

| Parameter | Type | Constraints | Default | Description |
|-----------|------|-------------|---------|-------------|
| `exercise_id` | `int` | 1–15 | *required* | Exercise class ID |
| `exercise_name` | `str` | — | *required* | Human-readable exercise name |
| `rep_scores` | `list[dict]` | See below | *required* | Per-rep score dicts |
| `recognition_confidence` | `float` | [0.0, 1.0] | *required* | Recognition confidence |
| `view_type` | `str` | `"front"` / `"side"` | `"front"` | Camera view |

**`rep_scores` dict schema:**
```python
{
    "rep_number": int,       # 1-indexed
    "scores": {
        "criterion_name": float,  # 0–10
        ...
    }
}
```

**Returns:** `FeedbackResponse` — fully populated structured response.

**Internal flow:**
1. Convert raw `rep_scores` dicts → `PerRepScore` Pydantic objects
2. Construct `AssessmentInput` from all parameters
3. Initialize `CoachingState` with input + empty defaults
4. Invoke the compiled LangGraph (`self.graph.invoke(initial_state)`)
5. Deserialize `result["final_response"]` → `FeedbackResponse`

---

## 8. End-to-End Walkthrough

A complete walkthrough of processing a single exercise set from assessment model output to coaching feedback delivery.

### Scenario

A user performs **12 reps of Dumbbell Shoulder Press** (exercise ID 1) filmed from the front view. The exercise recognition model identified the exercise with 92% confidence. The temporal CNN assessment model produced per-repetition scores for 5 biomechanical criteria, showing a gradual decline in later reps (fatigue pattern).

### Step-by-Step

```
Step 1: Caller invokes CoachingAgent
═══════════════════════════════════
    agent = CoachingAgent()
    response = agent.generate_feedback(
        exercise_id=1,
        exercise_name="Dumbbell Shoulder Press",
        rep_scores=[
            {"rep_number": 1,  "scores": {"Starting position": 9.0, "Top position": 8.5, 
                                           "Elbow path": 8.0, "Tempo": 9.0, "Core stability": 8.5}},
            {"rep_number": 2,  "scores": {...8.8, 8.3, 8.2, 8.8, 8.5...}},
            ...
            {"rep_number": 12, "scores": {...6.5, 5.5, 4.8, 6.0, 5.8...}},
        ],
        recognition_confidence=0.92,
        view_type="front",
    )

Step 2: Input Construction (generate_feedback method)
═════════════════════════════════════════════════════
    12 raw dicts → 12 PerRepScore objects
    → AssessmentInput(exercise_id=1, rep_scores=[12 objects], ...)
    → CoachingState(input=AssessmentInput, everything_else=defaults)

Step 3: Node 1 — load_criteria
═══════════════════════════════
    get_exercise_criteria(exercise_id=1)
    → ("Dumbbell Shoulder Press", [
        "Starting position: dumbbells at shoulder height, elbows bent",
        "Top position: arms raised near vertical, slight elbow bend",
        "Elbow path: vertical and aligned with wrists",
        "Tempo: controlled movement speed",
        "Core stability: engaged throughout movement",
    ])
    State update: exercise_criteria = [5 criteria strings]

Step 4: Node 2 — analyze_scores
════════════════════════════════
    recognition_confidence = 0.92  →  ≥ 0.7, no warning
    overall_score ≈ 7.4            →  ≥ 5.0, no warning
    No criterion < 3.0              →  no critical warnings
    State update: warnings = []

Step 5: Node 3 — analyze_rep_trends
════════════════════════════════════
    For each of 5 criteria:
      "Starting position": early_mean=8.77, late_mean=6.62 → diff=-2.14 → DECLINING
      "Top position":      early_mean=8.27, late_mean=5.77 → diff=-2.50 → DECLINING
      "Elbow path":        early_mean=8.00, late_mean=5.10 → diff=-2.90 → DECLINING
      "Tempo":             early_mean=8.77, late_mean=6.23 → diff=-2.53 → DECLINING
      "Core stability":    early_mean=8.40, late_mean=6.10 → diff=-2.30 → DECLINING
    
    Fatigue: 5 declining criteria ≥ 2 → fatigue_detected = True
    Fatigue details: "Form dropped in final reps for: Starting position, 
                      Top position, Elbow path, Tempo, Core stability.
                      Consider reducing weight or taking longer rest."
    
    Consistency: avg_std ≈ 0.93 → consistency = 10 - 3(0.93) = 7.2
    Strongest: "Starting position" (highest mean)
    Weakest: "Elbow path" (lowest mean)
    
    State update: rep_analysis = RepTrendAnalysis(...)

Step 6: Node 4 — generate_llm
══════════════════════════════
    Format prompt with:
    • Per-rep table (12 rows × 5 criteria)
    • Aggregated scores per criterion
    • Trend arrows (all ↓)
    • Fatigue warning
    
    Send to Gemini 2.5 Flash (temperature=0.7)
    
    LLM returns: "Fantastic effort on your Dumbbell Shoulder Press! Your
    starting position was consistently strong throughout the set. However,
    I noticed your elbow path became less vertical as the set progressed,
    particularly from rep 9 onward. It looks like fatigue started to set
    in towards the end — your top position and core stability both dropped
    noticeably in the final reps. For your next set, consider reducing the
    weight slightly or taking a bit more rest between sets. Focus on
    keeping those elbows tracking straight up. Great work overall!"
    
    State update: llm_feedback = "Fantastic effort..."

Step 7: Node 5 — format_response
═════════════════════════════════
    Assemble FeedbackResponse from all state fields
    Serialize to dict via model_dump()
    State update: final_response = {...}

Step 8: Return to Caller
════════════════════════
    FeedbackResponse(**result["final_response"])
    → Structured object with all scores, trends, and coaching text
```

---

## 9. Integration Points

### Upstream Dependencies

| Source | Data Provided | Format |
|--------|--------------|--------|
| **Stage 2: Exercise Recognition** | `exercise_id`, `exercise_name`, `recognition_confidence` | Integer ID (1–15), string name, float [0, 1] |
| **Stage 3: Assessment Model** | Per-rep aspect scores | `list[dict]` with `rep_number` and `scores` dict (5 criteria × float 0–10) |
| **Camera View Selection** | `view_type` | `"front"` or `"side"` |

### Downstream Consumers

| Consumer | Data Consumed | Contract |
|----------|--------------|----------|
| **Mobile App** | `FeedbackResponse` (serialized JSON) | See `src/pipelines/backend_api_contract.md` |
| **Logging / Analytics** | `rep_scores`, `trends`, `warnings` | Internal use |

### Backend API Contract

The coaching agent's output aligns with the API contract at `POST /api/session/analyze`:

```json
{
    "exercise": "Dumbbell Shoulder Press",
    "reps_detected": 12,
    "scores": {
        "Starting position": 7.9,
        "Top position": 7.0,
        "Elbow path": 6.8,
        "Tempo": 7.7,
        "Core stability": 7.4
    },
    "overall_score": 7.4,
    "feedback": ["Fantastic effort on your..."],
    "warnings": []
}
```

**Error codes** the agent can trigger (from API contract):
- `UNRECOGNIZED_EXERCISE` — exercise not in criteria registry
- `NO_REPS_DETECTED` — empty `rep_scores` list
- `ANALYSIS_FAILED` — internal error during processing

### External Services

| Service | Usage | Failure Impact |
|---------|-------|----------------|
| **Google Gemini 2.5 Flash API** | Natural language feedback generation | Graceful fallback to error message string |

---

## 10. Configuration Guide

### Minimal Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Add your Gemini API key (https://aistudio.google.com/apikey)
echo 'GEMINI_API_KEY=your_key_here' >> .env
```

### Configuration Options

| Setting | Environment Variable | Default | How to Change |
|---------|---------------------|---------|---------------|
| API Key | `GEMINI_API_KEY` | `""` (warns) | `.env` file |
| LLM Model | `GEMINI_MODEL_NAME` | `"gemini-2.5-flash"` | `.env` file |
| Max feedback words | — | `150` | Edit `config.py: MAX_FEEDBACK_WORDS` |
| Score thresholds | — | See [§4](#4-module-reference-configpy) | Edit `config.py: SCORE_THRESHOLDS` |
| LLM temperature | — | `0.7` | Edit `coaching_agent.py: generate_llm_feedback_node` |
| Fatigue threshold | — | 2 declining criteria | Edit `coaching_agent.py: analyze_rep_trends_node` |
| Trend threshold | — | ±0.5 score difference | Edit `coaching_agent.py: analyze_rep_trends_node` |

### Model Alternatives

To use a different Gemini model, set the environment variable:

```bash
# For longer, more detailed feedback
GEMINI_MODEL_NAME=gemini-2.5-pro

# For fastest response times  
GEMINI_MODEL_NAME=gemini-2.0-flash-lite
```

---

## 11. Extensibility Guide

### Adding a New Exercise

To add a 16th exercise (e.g., "Lunges"):

**Step 1:** Add the exercise to `exercise_criteria.py`:

```python
EXERCISE_CRITERIA: dict[int, tuple[str, list[str]]] = {
    ...
    16: (
        "Lunges",
        [
            "Step length: appropriate for hip hinge",
            "Front knee tracking: over toes, not past",
            "Torso: upright posture throughout",
            "Rear knee: controlled descent near floor",
            "Balance: stable with no wobble",
        ]
    ),
}
```

**Step 2:** Update the exercise recognition model (Stage 2) to classify the new exercise.

**Step 3:** Train a new temporal CNN assessment model (Stage 3) for the exercise.

No changes needed to `state.py`, `prompts.py`, `coaching_agent.py`, or `config.py` — the agent auto-discovers criteria by ID lookup.

### Customizing Assessment Criteria

Each exercise must have **exactly 5 criteria** (matching the assessment model's output dimensionality). To modify criteria descriptions:

1. Edit the criteria strings in `EXERCISE_CRITERIA[exercise_id]`
2. Ensure the criterion names match those used in the assessment model's output keys
3. The prompt template will automatically include the updated descriptions

### Swapping the LLM Backend

To replace Gemini with another LangChain-compatible LLM:

**Step 1:** Update `config.py` with new provider settings.

**Step 2:** Modify the LLM initialization in `generate_llm_feedback_node`:

```python
# Example: Switch to OpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)
```

**Step 3:** The rest of the pipeline (prompts, state, formatting) remains unchanged since LangChain abstracts the LLM interface.

### Adding Conditional Routing

To add conditional edges (e.g., skip LLM if API key is missing):

```python
def should_use_llm(state: CoachingState) -> str:
    if not GEMINI_API_KEY:
        return "format_response"
    return "generate_llm"

graph.add_conditional_edges(
    "analyze_rep_trends",
    should_use_llm,
    {"generate_llm": "generate_llm", "format_response": "format_response"}
)
```

---

## 12. Key Design Questions — Answered

### Q1: How does the coaching agent handle variable repetition counts?

The agent handles any $N_{\text{reps}} \geq 0$:

| Rep Count | Behavior |
|-----------|----------|
| $N = 0$ | `aggregated_scores = {}`, `overall_score = 0.0`, `rep_analysis = None`, LLM receives "No rep data available" |
| $N = 1$ | Aggregation works (single-element means), trend analysis skipped (`rep_analysis = None` since `< 2`), no fatigue detection |
| $N \geq 2$ | Full trend analysis with early/late comparison. For $N < 6$, the "first 3" and "last 3" windows may overlap |
| $N \geq 6$ | Non-overlapping early (reps 1–3) and late (reps $N{-}2$ to $N$) windows — optimal trend detection |

The `rep_scores` list is dynamically sized — no hardcoded rep count assumptions exist in the codebase.

### Q2: What is the state lifecycle?

See [Section 3.4 — State Lifecycle](#34-state-lifecycle) for the full diagram. In summary:

1. **Initialized** by `generate_feedback()` with input data + empty defaults
2. **Progressively enriched** by each node (criteria → warnings → trends → LLM text → final response)
3. **Finalized** when `format_response_node` assembles the `final_response` dict
4. **Extracted** from the graph result by the `CoachingAgent` class

State is **not persisted** between calls — each `generate_feedback()` invocation creates fresh state.

### Q3: How are exercise-specific criteria retrieved?

`exercise_criteria.py` maintains a hardcoded `EXERCISE_CRITERIA` dictionary mapping integer IDs (1–15) to `(name, [5 criteria])` tuples. Lookup is performed by:

1. **Primary:** Direct dict access by `exercise_id` (O(1))
2. **Fallback:** Case-insensitive name matching against `EXERCISE_NAME_TO_ID`
3. **Fuzzy fallback:** Substring containment match (handles partial names like `"squat"` → `"Weighted Squats"`)

### Q4: How is fatigue detection implemented?

Fatigue detection is a **two-stage rule-based system**:

**Stage A — Per-criterion trend detection:**
$$\text{trend\_diff}_c = \bar{s}_{c,\text{last 3}} - \bar{s}_{c,\text{first 3}}$$
A criterion is "declining" if $\text{trend\_diff}_c < -0.5$.

**Stage B — Multi-criterion aggregation:**
$$\text{fatigue\_detected} = \bigl|\{c \in C : \text{trend}(c) = \text{declining}\}\bigr| \geq 2$$

The threshold of 2 declining criteria prevents false positives from single-aspect noise while catching genuine fatigue (which typically affects multiple biomechanical aspects simultaneously).

### Q5: What is the LLM prompt structure?

The prompt follows a **system + human message** structure:

1. **System message:** Persona definition (expert fitness coach) + behavioral guidelines
2. **Human message:** Structured data presentation in 3 sections:
   - **Per-rep breakdown:** ASCII table of all scores
   - **Aggregated analysis:** Overall score, consistency, per-criterion summaries with trend arrows, fatigue status
   - **Task instructions:** 5-point directive for feedback generation + word limit

The key design insight is that **deterministic analysis is done in Python** (Node 3) and the LLM receives **pre-computed summaries** rather than raw time series. This reduces token count, latency, and ensures analytical reproducibility.

### Q6: How are per-repetition scores aggregated to subject-level recommendations?

Aggregation uses **unweighted arithmetic mean** at two levels:

1. **Per-criterion:** $\bar{s}_c = \text{mean}(s_{1,c}, s_{2,c}, \ldots, s_{N,c})$ — all reps weighted equally
2. **Overall:** $\bar{S} = \text{mean}(\bar{s}_{c_1}, \bar{s}_{c_2}, \ldots, \bar{s}_{c_5})$ — all criteria weighted equally

There is **no quality-based weighting** (e.g., down-weighting fatigued reps). This is intentional — the trend analysis separately captures temporal patterns, while the mean reflects the overall session quality including fatigue effects.

### Q7: What are the failure modes?

| Failure | Cause | Handling | Impact |
|---------|-------|----------|--------|
| Unknown exercise ID | Assessment model outputs ID not in 1–15 | `load_criteria_node` catches `ValueError`, sets `error` field | Empty criteria; LLM generates generic feedback |
| Partial name match failure | Exercise name doesn't match any substring | Same as above | Same as above |
| Empty rep scores | No reps detected by segmentation | `aggregated_scores = {}`, `overall_score = 0.0`, trends skipped | Minimal but valid response |
| Missing API key | `GEMINI_API_KEY` not set | `generate_llm_feedback_node` returns fallback string | Response includes all scores/trends but no NL feedback |
| Gemini API error | Network failure, quota exceeded, invalid key | Exception caught, error message returned as feedback | Same as above |
| Single rep | Only 1 repetition detected | Trend analysis returns `None` (requires ≥ 2 reps) | No trend/fatigue data; LLM receives limited context |
| Inconsistent criterion names | Assessment model outputs different keys per rep | `scores.get(criterion, 0)` defaults missing keys to 0 | Silently treats missing criteria as 0 (may skew analysis) |

---

## 13. Troubleshooting & Common Patterns

### Common Issues

#### "LLM feedback unavailable: API key not configured"

**Cause:** `GEMINI_API_KEY` is empty or `.env` file is missing.

**Fix:**
```bash
cp .env.example .env
# Edit .env and add your key from https://aistudio.google.com/apikey
```

#### Warning at import: "GEMINI_API_KEY not set"

**Cause:** Python imported the `agents` module before `.env` was loaded, or the key is genuinely missing.

**Fix:** Ensure `.env` exists in the project root (`ai-virtual-coach/.env`) and contains a valid `GEMINI_API_KEY`.

#### Criteria lookup fails for a valid exercise

**Cause:** The `exercise_name` string from the recognition model doesn't match any name or substring in `EXERCISE_NAME_TO_ID`.

**Fix:** Use `exercise_id` (more reliable) or update `EXERCISE_CRITERIA` to include alternate spellings.

#### Fatigue not detected despite declining scores

**Cause:** Either < 2 criteria are declining (threshold is 2), or the score drop is < 0.5 points between early and late rep windows.

**Fix:** Adjust the fatigue threshold in `analyze_rep_trends_node`:
```python
# Lower threshold for more sensitive detection
fatigue_detected = len(declining_criteria) >= 1  # was >= 2
```

Or adjust the trend sensitivity:
```python
# More sensitive trend detection
if trend_diff > 0.3:   # was 0.5
    trend = "improving"
elif trend_diff < -0.3: # was -0.5
    trend = "declining"
```

#### LLM feedback is too long or too short

**Fix:** Adjust `MAX_FEEDBACK_WORDS` in `config.py`:
```python
MAX_FEEDBACK_WORDS = 200  # Default is 150
```

### Best Practices

1. **Always use `exercise_id` over `exercise_name`** for criteria lookup — it's a direct O(1) dict access vs. string matching
2. **Validate rep scores before passing to the agent** — ensure all reps have the same criterion names
3. **Monitor API latency** — the Gemini call in Node 4 is the only network-dependent step; all other nodes are sub-millisecond
4. **Test with the built-in example** — run `python -m src.agents.coaching_agent` to verify end-to-end connectivity
5. **Use the `warnings` field** in responses to surface quality issues to the user
6. **Keep criteria descriptions specific** — the LLM uses them to ground its feedback in domain knowledge

### Debugging Tips

```python
# Inspect intermediate state by running nodes manually
from src.agents.state import CoachingState, AssessmentInput, PerRepScore

state = CoachingState(input=your_assessment_input)

# Run individual nodes
from src.agents.coaching_agent import load_criteria_node, analyze_rep_trends_node

result1 = load_criteria_node(state)
print("Criteria:", result1)

# Update state and continue
state = state.model_copy(update=result1)
result3 = analyze_rep_trends_node(state)
print("Trends:", result3["rep_analysis"])
```

---

## 14. API Quick Reference

### Public API

```python
# === Main Entry Point ===
agent = CoachingAgent()
response: FeedbackResponse = agent.generate_feedback(
    exercise_id: int,          # 1–15
    exercise_name: str,        # Human-readable name
    rep_scores: list[dict],    # [{"rep_number": int, "scores": {str: float}}]
    recognition_confidence: float,  # 0.0–1.0
    view_type: str = "front",  # "front" or "side"
)

# === Exercise Lookup ===
name, criteria = get_exercise_criteria(exercise_id=1)
name, criteria = get_exercise_criteria(exercise_name="squats")
exercises = get_all_exercises()  # [(1, "Dumbbell Shoulder Press"), ...]

# === Response Fields ===
response.exercise_name          # str
response.exercise_id            # int
response.recognition_confidence # float
response.rep_count              # int
response.aggregated_scores      # dict[str, float]
response.overall_score          # float
response.rep_scores             # list[dict]
response.consistency_score      # float (0–10)
response.fatigue_detected       # bool
response.trends                 # dict[str, str]  ("improving"/"declining"/"stable")
response.feedback_summary       # str (LLM-generated)
response.warnings               # list[str]
```

### Module Imports

```python
# Full public API
from src.agents import (
    CoachingAgent,           # Main orchestrator
    FeedbackResponse,        # Output schema
    CoachingState,           # LangGraph state
    AssessmentInput,         # Input schema
    PerRepScore,             # Single-rep scores
    RepTrendAnalysis,        # Trend analysis results
    get_exercise_criteria,   # Criteria lookup
    get_all_exercises,       # List all exercises
)
```

---

*Documentation generated from source code analysis of `src/agents/` module.*
*For the research paper context, refer to the pipeline description in Section 3 of `main.tex`.*
