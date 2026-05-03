"""
quiz_agent.py
-------------
QuizAgent — generates quiz questions from retrieved document context.
Inherits from BaseAgent. Used as a LangGraph node in the quiz subgraph.
"""

from typing import Any, Dict
from backend.modules.agents.base_agent import BaseAgent


class QuizAgent(BaseAgent):
    """
    Generates multiple choice questions from educational content.
    Receives state with 'retrieved_docs' and 'difficulty',
    returns state with 'quiz_prompt' added.
    """

    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    def __init__(self):
        super().__init__(
            name="QuizAgent",
            description="Generates quiz questions from educational content"
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes state with 'retrieved_docs' and optional 'difficulty'.
        Returns state with 'quiz_prompt' added.
        """
        try:
            docs = state.get("retrieved_docs", [])
            difficulty = state.get("difficulty", "medium")

            if difficulty not in self.DIFFICULTY_LEVELS:
                difficulty = "medium"

            context = "\n\n".join([
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in docs
            ])

            quiz_prompt = (
                f"You are an expert educator. Based on the content below, "
                f"generate 5 {difficulty}-level multiple choice questions.\n\n"
                f"Each question must have:\n"
                f"- A clear question\n"
                f"- 4 options labeled A, B, C, D\n"
                f"- The correct answer indicated\n\n"
                f"Content:\n{context}\n\n"
                f"Questions:"
            )

            state["quiz_prompt"] = quiz_prompt
            state["agent_used"] = self.name
            state["difficulty"] = difficulty
            return state

        except Exception as e:
            state["error"] = str(e)
            return state