from abc import ABC, abstractmethod


class BaseCleaningStep(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def clean(self, text: str) -> str:
        pass