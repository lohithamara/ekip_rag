from ingestion.cleaning.base import BaseCleaningStep


class TextCleaningPipeline:

    def __init__(
        self,
        steps: list[BaseCleaningStep],
    ):
        self.steps = steps

    def clean(self, text: str) -> tuple[str, list[str]]:

        current_text = text

        applied_steps = []

        for step in self.steps:

            current_text = step.clean(
                current_text
            )

            applied_steps.append(
                step.name
            )

        return current_text, applied_steps
    
    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]