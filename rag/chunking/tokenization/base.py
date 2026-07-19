from abc import ABC, abstractmethod


class BaseTokenCounter(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def count(self, text: str) -> int:
        pass