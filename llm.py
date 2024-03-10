from abc import ABC, abstractmethod
from typing import List


class LLM(ABC):
    """Base class for a Large Language Model (LLM)."""

    @abstractmethod
    def tokenize(self, text: str) -> List[float]:
        """Abstract method to tokenize input text into a list of floats.
        Subclasses must implement this method to provide tokenization functionality.
        Args:
            text (str): The input text to be tokenized.
        Returns:
            List[float]: The list of floats representing the tokenized form of the input text.
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    def answer(self, query: str) -> str:
        """Abstract method to generate a model response for a given query.
        Subclasses must implement this method to provide a response for a given query.
        Args:
            query (str): The query for which the model should generate a response.
        Returns:
            str: The model-generated response to the input query.
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()
