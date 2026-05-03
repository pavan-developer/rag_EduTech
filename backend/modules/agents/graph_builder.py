"""
graph_builder.py
----------------
Builds the main LangGraph StateGraph for EduMind AI.
Wires together RouterAgent → TutorAgent / QuizAgent / EvaluatorAgent
using conditional edges based on state['next_agent'].
"""

from langgraph.graph import StateGraph, END

from backend.modules.agents.agent_state import AgentState
from backend.modules.agents.router_agent import RouterAgent
from backend.modules.agents.tutor_agent import TutorAgent
from backend.modules.agents.quiz_agent import QuizAgent
from backend.modules.agents.evaluator_agent import EvaluatorAgent


def route_decision(state: AgentState) -> str:
    """
    Conditional edge function.
    LangGraph calls this after RouterAgent runs to decide next node.
    """
    return state.get("next_agent", "TutorAgent")


def build_graph() -> StateGraph:
    """
    Assembles and compiles the EduMind LangGraph.
    Returns a compiled graph ready to invoke.
    """

    router    = RouterAgent()
    tutor     = TutorAgent()
    quiz      = QuizAgent()
    evaluator = EvaluatorAgent()

    graph = StateGraph(AgentState)

    graph.add_node("RouterAgent",    router.run)
    graph.add_node("TutorAgent",     tutor.run)
    graph.add_node("QuizAgent",      quiz.run)
    graph.add_node("EvaluatorAgent", evaluator.run)

    graph.set_entry_point("RouterAgent")

    graph.add_conditional_edges(
        "RouterAgent",
        route_decision,
        {
            "TutorAgent":     "TutorAgent",
            "QuizAgent":      "QuizAgent",
            "EvaluatorAgent": "EvaluatorAgent",
        }
    )

    graph.add_edge("TutorAgent",     END)
    graph.add_edge("QuizAgent",      END)
    graph.add_edge("EvaluatorAgent", END)

    return graph.compile()