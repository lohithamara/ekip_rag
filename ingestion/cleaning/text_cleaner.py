import re
import unicodedata

from ingestion.cleaning.base import BaseCleaningStep


class UnicodeNormalizationStep(BaseCleaningStep):

    @property
    def name(self) -> str:
        return "unicode_normalization_nfkc"

    def clean(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)


class LineEndingNormalizationStep(BaseCleaningStep):

    @property
    def name(self) -> str:
        return "line_ending_normalization"

    def clean(self, text: str) -> str:
        return (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )


class ControlCharacterRemovalStep(BaseCleaningStep):

    @property
    def name(self) -> str:
        return "control_character_removal"

    def clean(self, text: str) -> str:

        allowed_characters = {
            "\n",
            "\t",
        }

        return "".join(
            character
            for character in text
            if (
                character in allowed_characters
                or unicodedata.category(character)
                not in {"Cc", "Cf"}
            )
        )


class HorizontalWhitespaceNormalizationStep(
    BaseCleaningStep
):

    @property
    def name(self) -> str:
        return "horizontal_whitespace_normalization"

    def clean(self, text: str) -> str:

        lines = text.split("\n")

        cleaned_lines = []

        for line in lines:

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            )

            cleaned_lines.append(
                line.strip()
            )

        return "\n".join(cleaned_lines)


class ExcessiveBlankLineRemovalStep(
    BaseCleaningStep
):

    MAX_CONSECUTIVE_NEWLINES = 2

    @property
    def name(self) -> str:
        return "excessive_blank_line_removal"

    def clean(self, text: str) -> str:

        return re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )


class OuterWhitespaceRemovalStep(
    BaseCleaningStep
):

    @property
    def name(self) -> str:
        return "outer_whitespace_removal"

    def clean(self, text: str) -> str:
        return text.strip()