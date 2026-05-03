"""
base_agent.py
-------------
Abstract base class for all LangGraph agents in EduMind AI.
Follows SOLID principles — every agent must implement the `run` method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain_core.messages import BaseMessage


class BaseAgent(ABC):
    """
    Abstract base for all EduMind agents.
    Each LangGraph pattern inherits from this class.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent logic.
        Must accept a state dict and return an updated state dict.
        """
        pass

    def __repr__(self) -> str:
        return f"<Agent name={self.name}>"