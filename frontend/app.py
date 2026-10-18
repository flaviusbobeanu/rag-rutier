"""Streamlit chat UI for the traffic-law assistant. It only talks to the FastAPI backend over HTTP.

Run (two terminals):
    uvicorn src.api.main:app
    streamlit run frontend/app.py

Streamlit reruns this whole script on every interaction; st.session_state keeps the conversation between runs.
"""

import sys
from pathlib import Path

import httpx
import streamlit as st

# `streamlit run` puts frontend/ on the import path, not the project root, so `src` would not be found.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings

DISCLAIMER = "Acesta nu este un sfat juridic și nu înlocuiește un avocat."
TIMEOUT = httpx.Timeout(120.0)  # an answer means retrieval + one or two LLM calls

st.set_page_config(page_title="RAG Rutier", page_icon="🚦")


def api_error(error: httpx.HTTPError) -> str:
    """A readable message for a failed request (the API puts its reason in `detail`)."""
    if isinstance(error, httpx.HTTPStatusError):
        try:
            return f"Eroare {error.response.status_code}: {error.response.json()['detail']}"
        except (ValueError, KeyError):
            return f"Eroare {error.response.status_code}"
    return f"Nu pot contacta serverul la {settings.api_url}. Ai pornit `uvicorn src.api.main:app`?"


@st.cache_data(ttl=60)
def health() -> dict | None:
    try:
        return httpx.get(f"{settings.api_url}/health", timeout=10).raise_for_status().json()
    except httpx.HTTPError:
        return None


def show_details(message: dict) -> None:
    """Under an answer: the law versions used and the retrieved sources."""
    if message.get("standalone_question"):
        st.caption(f"Căutat ca: {message['standalone_question']}")
    for lege, versiune in message.get("versions", {}).items():
        st.caption(f"Versiune: {lege}, forma consolidată la {versiune}")
    if message.get("sources"):
        with st.expander(f"Surse ({len(message['sources'])})"):
            for source in message["sources"]:
                st.markdown(f"**{source['citation']}** · scor {source['score']:.3f}")
                st.text(source["text"])


def history_for_api() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", and for answers: sources, versions, ...}
    st.session_state.uploaded_file_id = None

# Sidebar: backend status, settings, fine upload
with st.sidebar:
    st.header("RAG Rutier")
    status = health()
    if status:
        st.success(f"Server OK · {status['chunks']} fragmente · {status['llm']}")
        for lege, versiune in status["law_versions"].items():
            st.caption(f"{lege}: forma consolidată la {versiune}")
    else:
        st.error(f"Serverul nu răspunde la {settings.api_url}")

    k = st.slider("Fragmente de lege căutate (k)", 1, 20, 5)
    if st.button("Conversație nouă"):
        st.session_state.messages = []
        st.session_state.uploaded_file_id = None
        st.rerun()

    st.divider()
    st.subheader("Analizează un proces-verbal")
    st.caption("PDF-ul este procesat doar în memorie și nu este salvat.")
    uploaded = st.file_uploader("Încarcă PDF", type="pdf")

st.title("🚦 Asistent legislație rutieră")
st.warning(DISCLAIMER, icon="⚖️")

# An uploaded fine is analyzed once (not on every rerun), and the explanation joins the conversation
# so follow-up questions about it work.
if uploaded is not None and uploaded.file_id != st.session_state.uploaded_file_id:
    st.session_state.uploaded_file_id = uploaded.file_id
    with st.spinner("Citesc procesul-verbal..."):
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            result = httpx.post(f"{settings.api_url}/upload", files=files, timeout=TIMEOUT)
            result = result.raise_for_status().json()
        except httpx.HTTPError as error:
            st.error(api_error(error))
            result = None
    if result:
        facts = result["facts"]
        summary = (
            "📄 Proces-verbal încărcat. Articole: "
            + (", ".join(facts["articles"]) or "-")
            + f" · amendă: {facts['amount_lei'] or '-'} lei · puncte: {facts['penalty_points'] or '-'}"
            + f" · data: {facts['date'] or '-'} · text citit prin: {facts['text_method']}"
        )
        st.session_state.messages.append({"role": "user", "content": summary})
        explanation = result["explanation"]
        if explanation:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": explanation["answer"],
                    "sources": explanation["sources"],
                    "versions": explanation["versions"],
                }
            )
        else:
            st.session_state.messages.append({"role": "assistant", "content": result["message"]})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_details(message)

if question := st.chat_input("Întreabă despre legislația rutieră..."):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Caut în lege..."):
            try:
                body = {"question": question, "history": history_for_api(), "k": k}
                response = httpx.post(f"{settings.api_url}/ask", json=body, timeout=TIMEOUT)
                answer = response.raise_for_status().json()
            except httpx.HTTPError as error:
                st.error(api_error(error))
                st.stop()  # keep the question out of the history, so the user can simply ask again
        message = {
            "role": "assistant",
            "content": answer["answer"],
            "sources": answer["sources"],
            "versions": answer["versions"],
            # only shown when query rewriting actually changed the question
            "standalone_question": answer["standalone_question"]
            if answer["standalone_question"] != question
            else "",
        }
        st.markdown(message["content"])
        show_details(message)
    st.session_state.messages += [{"role": "user", "content": question}, message]
