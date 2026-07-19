import re


class CitationValidator:

    CITATION_PATTERN = re.compile(
        r"\[([^\[\]]+)\]"
    )

    def validate(
        self,
        answer: str,
        allowed_sources: set[str],
    ) -> list[str]:

        citations = self.CITATION_PATTERN.findall(
            answer
        )

        validated = []
        seen = set()

        for citation in citations:

            source = citation.strip()

            if source not in allowed_sources:
                continue

            if source in seen:
                continue

            seen.add(source)
            validated.append(source)

        return validated