"""
agent_state.py
--------------
Defines the shared state schema for all LangGraph graphs in EduMind AI.
Using TypedDict gives us type safety + LangGraph compatibility.
Every agent reads from and writes to this state.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    Fields:
    - question        : student's original question
    - retrieved_docs  : list of LangChain Document objects from RAG
    - difficulty      : quiz difficulty — easy / medium / hard
    - student_answer  : student's answer to a quiz question
    - correct_answer  : correct answer for evaluation
    - prompt          : assembled prompt for TutorAgent
    - quiz_prompt     : assembled prompt for QuizAgent
    - evaluation_prompt: assembled prompt for EvaluatorAgent
    - next_agent      : routing decision made by RouterAgent
    - agent_used      : name of last agent that ran (for admin dashboard)
    - llm_response    : final LLM output after prompt is sent
    - error           : error message if any agent fails
    - metadata        : optional extra info (student id, session id, etc.)
    """

    question:           str
    retrieved_docs:     List[Any]
    difficulty:         str
    student_answer:     str
    correct_answer:     str
    prompt:             str
    quiz_prompt:        str
    evaluation_prompt:  str
    next_agent:         str
    agent_used:         str
    llm_response:       str
    error:              str
    metadata:           Dict[str, Any]