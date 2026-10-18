"""Turn a follow-up question into a standalone question, using the conversation history.

"Și dacă e noaptea?" cannot be searched on its own. Before retrieval we ask the LLM to rewrite it,
e.g. into "Trebuie să circul cu luminile de întâlnire aprinse noaptea?". The rewritten question is used
for retrieval and for the answer; the history itself is never given to the answering prompt as a source.
"""

from src.generation.llm import LLM

MAX_TURNS = 6  # only the most recent messages matter for resolving "it", "and if", ...
MAX_CHARS_PER_MESSAGE = 600  # long answers are cut; the start holds the topic

REWRITE_PROMPT = """You rewrite follow-up questions for a search engine over Romanian traffic law.

Given the conversation and the user's latest question, write the latest question as ONE standalone \
question in Romanian that can be understood without the conversation: replace pronouns and references \
("it", "that fine", "and if...") with what they refer to. Keep the user's meaning; do not answer it and \
do not add facts. If the question is already standalone, return it unchanged.

Reply with the question only, nothing else."""


def format_history(history: list[dict]) -> str:
    """Render the last messages as "Utilizator: ..." / "Asistent: ..." lines."""
    lines = []
    for message in history[-MAX_TURNS:]:
        speaker = "Utilizator" if message["role"] == "user" else "Asistent"
        content = message["content"]
        if len(content) > MAX_CHARS_PER_MESSAGE:
            content = content[:MAX_CHARS_PER_MESSAGE] + " [...]"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def rewrite_question(question: str, history: list[dict], llm: LLM) -> str:
    """Return a standalone version of `question`. Without history, the question is returned as is."""
    if not history:
        return question
    user = f"<conversation>\n{format_history(history)}\n</conversation>\n\nLatest question: {question}"
    rewritten = llm.generate(REWRITE_PROMPT, user).strip().strip('"')
    return rewritten or question
