"""
test_agents.py
--------------
Tests for all LangGraph agents and graph routing in EduMind AI.
Covers RouterAgent, TutorAgent, QuizAgent, EvaluatorAgent, and graph_builder.
"""

import pytest
from backend.modules.agents.agent_state import AgentState
from backend.modules.agents.router_agent import RouterAgent
from backend.modules.agents.tutor_agent import TutorAgent
from backend.modules.agents.quiz_agent import QuizAgent
from backend.modules.agents.evaluator_agent import EvaluatorAgent
from backend.modules.agents.graph_builder import build_graph


# ── RouterAgent tests ──────────────────────────────────────────────

def test_router_routes_to_tutor():
    agent = RouterAgent()
    state = {"question": "explain photosynthesis"}
    result = agent.run(state)
    assert result["next_agent"] == "TutorAgent"
    assert result["agent_used"] == "RouterAgent"


def test_router_routes_to_quiz():
    agent = RouterAgent()
    state = {"question": "give me a quiz on photosynthesis"}
    result = agent.run(state)
    assert result["next_agent"] == "QuizAgent"


def test_router_routes_to_evaluator():
    agent = RouterAgent()
    state = {"question": "check my answer please"}
    result = agent.run(state)
    assert result["next_agent"] == "EvaluatorAgent"


def test_router_empty_question():
    agent = RouterAgent()
    state = {"question": ""}
    result = agent.run(state)
    assert result["next_agent"] == "TutorAgent"


# ── TutorAgent tests ───────────────────────────────────────────────

def test_tutor_builds_prompt():
    agent = TutorAgent()
    state = {
        "question": "What is photosynthesis?",
        "retrieved_docs": []
    }
    result = agent.run(state)
    assert "prompt" in result
    assert "photosynthesis" in result["prompt"]
    assert result["agent_used"] == "TutorAgent"


def test_tutor_with_docs():
    class FakeDoc:
        page_content = "Photosynthesis converts sunlight into glucose."

    agent = TutorAgent()
    state = {
        "question": "What is photosynthesis?",
        "retrieved_docs": [FakeDoc()]
    }
    result = agent.run(state)
    assert "Photosynthesis converts sunlight" in result["prompt"]


# ── QuizAgent tests ────────────────────────────────────────────────

def test_quiz_default_difficulty():
    agent = QuizAgent()
    state = {"retrieved_docs": []}
    result = agent.run(state)
    assert result["difficulty"] == "medium"
    assert "quiz_prompt" in result


def test_quiz_hard_difficulty():
    agent = QuizAgent()
    state = {"retrieved_docs": [], "difficulty": "hard"}
    result = agent.run(state)
    assert "hard" in result["quiz_prompt"]


def test_quiz_invalid_difficulty_defaults_to_medium():
    agent = QuizAgent()
    state = {"retrieved_docs": [], "difficulty": "extreme"}
    result = agent.run(state)
    assert result["difficulty"] == "medium"


# ── EvaluatorAgent tests ───────────────────────────────────────────

def test_evaluator_builds_prompt():
    agent = EvaluatorAgent()
    state = {
        "question": "What is photosynthesis?",
        "student_answer": "It is the process of making food using sunlight.",
        "correct_answer": "Photosynthesis is the process by which plants make food."
    }
    result = agent.run(state)
    assert "evaluation_prompt" in result
    assert result["agent_used"] == "EvaluatorAgent"


def test_evaluator_missing_answer():
    agent = EvaluatorAgent()
    state = {
        "question": "What is photosynthesis?",
        "student_answer": "",
        "correct_answer": "Photosynthesis is the process by which plants make food."
    }
    result = agent.run(state)
    assert "error" in result


# ── Graph builder tests ────────────────────────────────────────────

def test_graph_builds():
    graph = build_graph()
    assert graph is not None


def test_graph_routes_to_tutor():
    graph = build_graph()
    result = graph.invoke({"question": "explain gravity"})
    assert result["next_agent"] == "TutorAgent"
    assert "prompt" in result


def test_graph_routes_to_quiz():
    graph = build_graph()
    result = graph.invoke({"question": "give me a test on gravity"})
    assert result["next_agent"] == "QuizAgent"
    assert "quiz_prompt" in result