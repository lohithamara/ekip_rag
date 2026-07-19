import re

from docx import text

from rag.chunking.tokenization.base import (
    BaseTokenCounter,
)


class WhitespaceTokenCounter(BaseTokenCounter):

    @property
    def name(self) -> str:
        return "whitespace"

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))
    
class CharacterCounter(BaseTokenCounter):

    @property
    def name(self) -> str:
        return "characters"

    def count(self, text: str) -> int:
        return len(text)