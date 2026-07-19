import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClaimCitationPair:

    claim: str

    citations: tuple[
        str,
        ...,
    ]


class ClaimCitationExtractor:

    CITATION_PATTERN = re.compile(
        r"\[([^\[\]]+)\]"
    )

    LIST_PREFIX_PATTERN = re.compile(
        r"^\s*(?:[-*•]|\d+[.)])\s*"
    )

    def extract(
        self,
        answer: str,
        allowed_sources: set[str],
    ) -> tuple[
        ClaimCitationPair,
        ...,
    ]:

        if not answer.strip():
            return ()

        units = self._split_units(
            answer
        )

        pairs = []

        pending_prefix = ""

        for unit in units:

            cleaned_unit = unit.strip()

            if not cleaned_unit:
                continue

            citations = (
                self._extract_citations(
                    cleaned_unit,
                    allowed_sources,
                )
            )

            claim = self._remove_citations(
                cleaned_unit
            )

            claim = self._remove_list_prefix(
                claim
            )

            claim = self._clean_claim(
                claim
            )

            if not citations:

                if self._can_be_prefix(
                    claim
                ):
                    pending_prefix = claim

                continue

            if not claim:
                continue

            if (
                pending_prefix
                and self._needs_prefix(
                    claim
                )
            ):
                claim = self._combine(
                    pending_prefix,
                    claim,
                )

            pairs.append(
                ClaimCitationPair(
                    claim=claim,
                    citations=citations,
                )
            )

        return tuple(pairs)

    def _split_units(
        self,
        answer: str,
    ) -> tuple[str, ...]:

        normalized = answer.replace(
            "\r\n",
            "\n",
        )

        units = []

        for line in normalized.split(
            "\n"
        ):

            line = line.strip()

            if not line:
                continue

            sentence_units = re.split(
                r"(?<=[.!?])\s+",
                line,
            )

            for unit in sentence_units:

                unit = unit.strip()

                if unit:
                    units.append(unit)

        return tuple(units)

    def _extract_citations(
        self,
        unit: str,
        allowed_sources: set[str],
    ) -> tuple[str, ...]:

        citations = []

        seen = set()

        for raw_source in (
            self.CITATION_PATTERN.findall(
                unit
            )
        ):

            source = raw_source.strip()

            if source not in allowed_sources:
                continue

            if source in seen:
                continue

            seen.add(source)
            citations.append(source)

        return tuple(citations)

    def _remove_citations(
        self,
        unit: str,
    ) -> str:

        return self.CITATION_PATTERN.sub(
            "",
            unit,
        )

    def _remove_list_prefix(
        self,
        claim: str,
    ) -> str:

        return self.LIST_PREFIX_PATTERN.sub(
            "",
            claim,
        )

    @staticmethod
    def _clean_claim(
        claim: str,
    ) -> str:

        claim = " ".join(
            claim.split()
        )

        claim = claim.strip()

        claim = claim.rstrip(
            " ,;:"
        )

        return claim

    @staticmethod
    def _can_be_prefix(
        claim: str,
    ) -> bool:

        if not claim:
            return False

        words = claim.split()

        lowered = claim.lower()

        prefix_markers = (
            "as follows",
            "but the exact",
            "the following",
            "different limits",
            "conflicting",
            "required for requests",
        )

        return (
            len(words) >= 4
            or any(
                marker in lowered
                for marker in prefix_markers
            )
        )

    @staticmethod
    def _needs_prefix(
        claim: str,
    ) -> bool:

        words = claim.split()

        if len(words) <= 6:
            return True

        lowered = claim.lower()

        continuation_starts = (
            "and ",
            "or ",
            "but ",
            "however ",
            "while ",
            "including ",
            "such as ",
        )

        return lowered.startswith(
            continuation_starts
        )

    @staticmethod
    def _combine(
        prefix: str,
        claim: str,
    ) -> str:

        prefix = prefix.rstrip(
            " ,;:"
        )

        claim = claim.lstrip(
            " ,;:"
        )

        return (
            f"{prefix} {claim}"
        )