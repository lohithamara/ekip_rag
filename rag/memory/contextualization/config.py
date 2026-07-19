from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextualizerConfig:

    enabled: bool = True

    temperature: float = 0.0

    max_tokens: int = 128

    max_history_turns: int = 3

    max_history_characters: int = 4000