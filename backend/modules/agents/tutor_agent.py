"""
tutor_agent.py
--------------
TutorAgent — explains concepts to students using RAG-retrieved context.
Inherits from BaseAgent. Used as a LangGraph node in the tutor subgraph.
"""

from typing import Any, Dict
from backend.modules.agents.base_agent import BaseAgent


class TutorAgent(BaseAgent):
    """
    Explains educational concepts using retrieved document context.
    Receives state with 'question' and 'retrieved_docs', returns state
    with 'answer' added.
    """

    def __init__(self):
        super().__init__(
            name="TutorAgent",
            description="Explains concepts clearly using retrieved context"
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes state with 'question' + 'retrieved_docs'.
        Returns state with 'answer' added.
        """
        try:
            question = state.get("question", "")
            docs = state.get("retrieved_docs", [])

            context = "\n\n".join([
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in docs
            ])

            prompt = (
                f"You are a helpful tutor. Use the context below to answer "
                f"the student's question clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            state["prompt"] = prompt
            state["agent_used"] = self.name
            return state

        except Exception as e:
            state["error"] = str(e)
            return state