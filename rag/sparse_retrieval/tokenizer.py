import re


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:

    return TOKEN_PATTERN.findall(
        text.lower()
    )