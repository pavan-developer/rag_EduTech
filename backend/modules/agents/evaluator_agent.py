"""
evaluator_agent.py
------------------
EvaluatorAgent — grades student answers and provides feedback.
Inherits from BaseAgent. Used as a LangGraph node in the evaluation subgraph.
"""

from typing import Any, Dict
from backend.modules.agents.base_agent import BaseAgent


class EvaluatorAgent(BaseAgent):
    """
    Evaluates a student's answer against the correct answer.
    Receives state with 'student_answer', 'correct_answer', and 'question'.
    Returns state with 'evaluation_prompt' added.
    """

    def __init__(self):
        super().__init__(
            name="EvaluatorAgent",
            description="Grades student answers and explains mistakes"
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes state with 'student_answer', 'correct_answer', 'question'.
        Returns state with 'evaluation_prompt' added.
        """
        try:
            question = state.get("question", "")
            student_answer = state.get("student_answer", "")
            correct_answer = state.get("correct_answer", "")

            if not student_answer:
                state["error"] = "No student answer provided"
                return state

            evaluation_prompt = (
                f"You are a patient and encouraging educator.\n\n"
                f"Question: {question}\n"
                f"Student's answer: {student_answer}\n"
                f"Correct answer: {correct_answer}\n\n"
                f"Please do the following:\n"
                f"1. State whether the student's answer is correct or incorrect\n"
                f"2. If incorrect, explain why clearly and kindly\n"
                f"3. Reinforce the correct concept in simple terms\n"
                f"4. Give a score out of 10\n\n"
                f"Feedback:"
            )

            state["evaluation_prompt"] = evaluation_prompt
            state["agent_used"] = self.name
            return state

        except Exception as e:
            state["error"] = str(e)
            return state