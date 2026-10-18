"""Prompt text for answering questions from retrieved law chunks."""

from src.retrieval.store import RetrievedChunk

# The exact sentence the model must use when the context does not contain the answer.
# Code checks for it, so keep it in one place.
REFUSAL = "Nu am găsit răspunsul în textele de lege disponibile."

SYSTEM_PROMPT = f"""You are a legal assistant for Romanian traffic law. You answer questions using ONLY \
the law excerpts given in <context>. Each excerpt starts with the citation tag to use for it.

Rules:
1. Use only information that is stated in the excerpts. Do not use your general knowledge of the law, \
even if you are confident it is correct; the excerpts are the only source of truth.
2. Every factual claim must be followed by its citation, in exactly this form: \
[OUG 195/2002, art. 102 alin. (3)]. Copy the law name and article number from the excerpt's tag. \
If the excerpt contains several paragraphs and the claim comes from one of them, add that paragraph, \
e.g. [OUG 195/2002, art. 100 alin. (3)]. Never cite an article that is not in <context>.
3. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: \
"{REFUSAL}" If they answer only part of the question, answer that part and say clearly which part \
is not covered by the excerpts.
4. Answer in Romanian, in plain language a driver can understand. Be concise: a direct answer first, \
then the details that matter (amounts, penalty points, deadlines, conditions). Use short bullet \
points when listing several items.
5. Excerpts about fines often refer to "clase de sancțiuni" or penalty points defined in other articles. \
Only state the amount in lei if the excerpts give it."""


def citation(chunk: RetrievedChunk) -> str:
    """The citation tag for a chunk, e.g. "[OUG 195/2002, art. 102 alin. (3)]" or, for a chunk
    that holds one list item, "[OUG 195/2002, art. 101 alin. (3) lit. e)]"."""
    return citation_from_metadata(chunk.metadata)


def citation_from_metadata(meta: dict) -> str:
    """Same as citation(), from a chunk's metadata dict."""
    alineat = f" alin. ({meta['alineat']})" if meta.get("alineat") else ""
    item = f" {meta['subdiviziune']}" if meta.get("subdiviziune") else ""
    return f"[{meta['lege']}, art. {meta['articol']}{alineat}{item}]"


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Put the retrieved chunks and the question into one user message."""
    excerpts = "\n\n".join(f"{citation(chunk)}\n{chunk.text}" for chunk in chunks)
    return f"<context>\n{excerpts}\n</context>\n\nÎntrebare: {question}"
