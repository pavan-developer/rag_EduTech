"""
router_agent.py
---------------
RouterAgent — decides which agent to call based on student intent.
This is the entry point node in every LangGraph subgraph.
Uses keyword matching strategy (upgradeable to LLM-based routing later).
"""

from typing import Any, Dict
from backend.modules.agents.base_agent import BaseAgent


class RouterAgent(BaseAgent):
    """
    Reads the student's question from state and sets 'next_agent'
    so LangGraph conditional edges know where to route next.

    Routing logic:
    - 'quiz' / 'test' / 'question'  → QuizAgent
    - 'grade' / 'check' / 'correct' → EvaluatorAgent
    - everything else                → TutorAgent
    """

    QUIZ_KEYWORDS     = {"quiz", "test", "question", "questions", "mcq"}
    EVALUATE_KEYWORDS = {"grade", "check", "correct", "evaluate", "score", "marks"}

    def __init__(self):
        super().__init__(
            name="RouterAgent",
            description="Routes student intent to the correct specialist agent"
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes state with 'question'.
        Returns state with 'next_agent' set to agent name string.
        """
        try:
            question = state.get("question", "").lower()
            words = set(question.split())

            if words & self.QUIZ_KEYWORDS:
                state["next_agent"] = "QuizAgent"
            elif words & self.EVALUATE_KEYWORDS:
                state["next_agent"] = "EvaluatorAgent"
            else:
                state["next_agent"] = "TutorAgent"

            state["agent_used"] = self.name
            return state

        except Exception as e:
            state["error"] = str(e)
            state["next_agent"] = "TutorAgent"
            return state