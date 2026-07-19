from ingestion.cleaning.pipeline import (
    TextCleaningPipeline,
)
from ingestion.cleaning.text_cleaner import (
    ControlCharacterRemovalStep,
    ExcessiveBlankLineRemovalStep,
    HorizontalWhitespaceNormalizationStep,
    LineEndingNormalizationStep,
    OuterWhitespaceRemovalStep,
    UnicodeNormalizationStep,
)


def create_default_cleaning_pipeline(
) -> TextCleaningPipeline:

    return TextCleaningPipeline(
        steps=[
            UnicodeNormalizationStep(),
            LineEndingNormalizationStep(),
            ControlCharacterRemovalStep(),
            HorizontalWhitespaceNormalizationStep(),
            ExcessiveBlankLineRemovalStep(),
            OuterWhitespaceRemovalStep(),
        ]
    )