from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:

    enabled: bool = True

    max_turns: int = 5

    max_characters: int = 6000