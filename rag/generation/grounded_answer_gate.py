class GroundedAnswerGate:

    ABSTENTION_TEXT = (
        "I do not know based on the provided "
        "context."
    )

    def apply(
        self,
        answer: str,
        validated_sources: tuple[
            str,
            ...,
        ],
    ) -> str:

        cleaned_answer = (
            answer.strip()
        )

        if not cleaned_answer:
            return self.ABSTENTION_TEXT

        if self._is_abstention(
            cleaned_answer
        ):
            return self.ABSTENTION_TEXT

        if validated_sources:
            return cleaned_answer

        return self.ABSTENTION_TEXT

    def _is_abstention(
        self,
        answer: str,
    ) -> bool:

        return (
            answer.lower()
            == self.ABSTENTION_TEXT.lower()
        )