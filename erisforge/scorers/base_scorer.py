from abc import (
    ABC,
    abstractmethod,
)


class BaseScorer(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def score(self, user_query: str, model_response: str) -> float:
        pass
