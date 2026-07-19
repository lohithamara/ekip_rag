import re

from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True,
)
class CitationCompletenessResult:

    factual_units: int

    cited_units: int

    uncited_units: tuple[str, ...]

    completeness_score: float


class CitationCompletenessEvaluator:

    CITATION_PATTERN = re.compile(
        r"\[([^\[\]]+)\]"
    )

    CITATION_ONLY_PATTERN = re.compile(
        r"^(?:\s*\[[^\[\]]+\]\s*)+$"
    )

    BULLET_PREFIX_PATTERN = re.compile(
        r"^\s*[-*]\s+"
    )

    def evaluate(
        self,
        answer: str,
        allowed_sources: set[str],
    ) -> CitationCompletenessResult:

        units = self._extract_units(
            answer
        )

        if not units:
            return (
                CitationCompletenessResult(
                    factual_units=0,
                    cited_units=0,
                    uncited_units=(),
                    completeness_score=1.0,
                )
            )

        factual_units = []

        for unit in units:

            if self._is_introductory_unit(
                unit
            ):
                continue

            if self._is_citation_only(
                unit
            ):
                if factual_units:

                    previous_text, previous_cited = (
                        factual_units[-1]
                    )

                    citation_valid = (
                        self._has_valid_citation(
                            unit=unit,
                            allowed_sources=(
                                allowed_sources
                            ),
                        )
                    )

                    factual_units[-1] = (
                        previous_text,
                        (
                            previous_cited
                            or citation_valid
                        ),
                    )

                continue

            factual_units.append(
                (
                    unit,
                    self._has_valid_citation(
                        unit=unit,
                        allowed_sources=(
                            allowed_sources
                        ),
                    ),
                )
            )

        if not factual_units:
            return (
                CitationCompletenessResult(
                    factual_units=0,
                    cited_units=0,
                    uncited_units=(),
                    completeness_score=1.0,
                )
            )

        cited_units = sum(
            1
            for _, cited in factual_units
            if cited
        )

        uncited_units = tuple(
            unit
            for unit, cited in factual_units
            if not cited
        )

        completeness_score = (
            cited_units
            / len(factual_units)
        )

        return CitationCompletenessResult(
            factual_units=len(
                factual_units
            ),
            cited_units=cited_units,
            uncited_units=uncited_units,
            completeness_score=(
                completeness_score
            ),
        )

    def _extract_units(
        self,
        answer: str,
    ) -> tuple[str, ...]:

        normalized = answer.strip()

        if not normalized:
            return ()

        if self._is_abstention(
            normalized
        ):
            return ()

        units = []

        for line in normalized.splitlines():

            cleaned = line.strip()

            if not cleaned:
                continue

            cleaned = (
                self.BULLET_PREFIX_PATTERN
                .sub(
                    "",
                    cleaned,
                )
                .strip()
            )

            if not cleaned:
                continue

            sentences = re.split(
                r"(?<=[.!?])\s+",
                cleaned,
            )

            for sentence in sentences:

                sentence = (
                    sentence.strip()
                )

                if sentence:
                    units.append(
                        sentence
                    )

        return tuple(units)

    def _has_valid_citation(
        self,
        unit: str,
        allowed_sources: set[str],
    ) -> bool:

        citations = {
            citation.strip()
            for citation
            in self.CITATION_PATTERN.findall(
                unit
            )
        }

        return bool(
            citations
            & allowed_sources
        )

    def _is_citation_only(
        self,
        unit: str,
    ) -> bool:

        return bool(
            self.CITATION_ONLY_PATTERN
            .fullmatch(
                unit.strip()
            )
        )

    @staticmethod
    def _is_introductory_unit(
        unit: str,
    ) -> bool:

        return unit.rstrip().endswith(
            ":"
        )

    @staticmethod
    def _is_abstention(
        answer: str,
    ) -> bool:

        normalized = " ".join(
            answer.lower().split()
        )

        return (
            "i do not know based on the "
            "provided context"
            in normalized
        )