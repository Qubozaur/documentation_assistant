"""
Streamlit UI for the LangChain Documentation Assistant.

This app:
- Renders a simple chat interface.
- Sends user questions to the backend LLM (`run_llm`).
- Displays answers along with the document sources used.
"""

from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Extract and normalize `source` values from a list of context documents.

    Each document is expected to have a `metadata` attribute (as LangChain `Document` does).
    If no source is present, it falls back to `"Unknown"`.
    """
    formatted_sources: List[str] = []
    for doc in context_docs or []:
        # Safely get a metadata dict from the document
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or "Unknown"
        formatted_sources.append(str(source))
    return formatted_sources


def _render_sidebar() -> None:
    """Render the sidebar with session-level controls (e.g., clearing the chat)."""
    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear chat", use_container_width=True):
            # Remove stored messages and restart the app to reflect the change.
            st.session_state.pop("messages", None)
            st.rerun()


def _init_session_state() -> None:
    """Initialize the chat message history in Streamlit session state.

    We keep a list of message dicts, each with:
    - role: "user" or "assistant"
    - content: text of the message
    - sources: list of document source strings (assistant only)
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me anything about LangChain docs. "
                    "I’ll retrieve relevant context and cite sources."
                ),
                "sources": [],
            }
        ]


def _render_message_history() -> None:
    """Render the full conversation history (messages + optional sources)."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.markdown(f"- {source}")


def _handle_user_prompt(prompt: str) -> None:
    """Process a new user prompt: display it, call the LLM, and show the answer.

    This function also updates the `st.session_state.messages` history.
    """
    # Store and display the user's message
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

            # Save the assistant reply back into the session history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:  # noqa: BLE001
            # Surface a friendly error message plus the stack trace for debugging.
            st.error("Failed to generate a response.")
            st.exception(e)


def main() -> None:
    """Entrypoint for the Streamlit app."""
    st.set_page_config(
        page_title="LangChain Documentation Assistant",
        layout="centered",
    )
    st.title("LangChain Documentation Assistant")

    _render_sidebar()
    _init_session_state()
    _render_message_history()

    # Chat input is rendered at the bottom of the page
    prompt = st.chat_input("Ask a question about LangChain…")
    if prompt:
        _handle_user_prompt(prompt)


if __name__ == "__main__":
    main()
