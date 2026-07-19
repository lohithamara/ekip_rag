from rag.memory.schemas import (
    ConversationTurn,
)


class ConversationQueryContextualizer:

    @staticmethod
    def build_prompt(
        query: str,
        turns: tuple[
            ConversationTurn,
            ...,
        ],
    ) -> tuple[str, str]:

        system_prompt = (
            "You rewrite conversational follow-up "
            "questions into standalone retrieval "
            "queries.\n\n"

            "Your only job is to resolve references "
            "and omitted conversational context.\n\n"

            "RULES:\n"
            "- Preserve the user's original intent "
            "exactly.\n"

            "- Resolve pronouns, references, ellipsis, "
            "and omitted subjects only when supported "
            "by the conversation history.\n"

            "- If the current query asks to regenerate, "
            "repeat, retry, or answer a previous question "
            "again, rewrite it as the most recent relevant "
            "user question itself.\n"

            "- For regeneration or repetition requests, "
            "remove meta-instructions such as 'regenerate', "
            "'answer again', 'repeat the answer', 'retry', "
            "or 'try again' from the retrieval query.\n"

            "- If the current query asks to explain, expand, "
            "summarize, simplify, or give examples about a "
            "previous topic, preserve that requested operation "
            "while explicitly including the referenced topic.\n"

            "- Add only the minimum information needed "
            "to make the current query standalone.\n"

            "EXAMPLES:\n"
            "Previous user question: What are the main "
            "responsibilities of the engineering team?\n"
            "Current query: Regenerate the answer again.\n"
            "Output: What are the main responsibilities "
            "of the engineering team?\n\n"

            "Previous user question: What is the deployment "
            "process?\n"
            "Current query: Summarize it.\n"
            "Output: Summarize the deployment process.\n\n"
        )

        history_parts = []

        for index, turn in enumerate(
            turns,
            start=1,
        ):

            history_parts.append(
                (
                    f"<turn id=\"{index}\">\n"
                    "<user_message>\n"
                    f"{turn.user_message}\n"
                    "</user_message>\n"
                    "<assistant_message>\n"
                    f"{turn.assistant_message}\n"
                    "</assistant_message>\n"
                    "</turn>"
                )
            )

        history_text = "\n\n".join(
            history_parts
        )

        user_prompt = (
            "<task>\n"
            "Rewrite the current query as a "
            "standalone retrieval query using only "
            "the minimum conversational context "
            "needed to resolve references or omitted "
            "details.\n"
            "</task>\n\n"

            "<conversation_history>\n"
            f"{history_text}\n"
            "</conversation_history>\n\n"

            "<current_query>\n"
            f"{query}\n"
            "</current_query>\n\n"

            "<output_requirements>\n"
            "Return exactly one standalone retrieval "
            "query.\n"
            "Preserve the meaning and requested "
            "relationship of the current query.\n"
            "Do not answer the query.\n"
            "Do not add explanations, labels, quotes, "
            "or formatting.\n"
            "</output_requirements>"
        )

        return (
            system_prompt,
            user_prompt,
        )