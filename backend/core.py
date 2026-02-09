"""
Core backend logic for the LangChain Documentation Assistant.

This module wires together:
- Pinecone vector store (for retrieval over LangChain docs).
- OpenAI embeddings (must match the index dimension used during ingestion).
- A simple LangChain agent that can call a `retrieve_context` tool.
"""

from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load API keys and other configuration from the environment (.env file).
load_dotenv()

# Embedding model used for retrieval. This must match the model (and dimension)
# that was used to populate the Pinecone index in the ingestion script.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store wrapping a Pinecone index that stores the LangChain docs.
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings,
)

# Chat model used by the agent to reason over retrieved context and user queries.
model = init_chat_model("gpt-3.5-turbo", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant LangChain documentation from Pinecone for a user query.

    Returns:
        A tuple of:
        - serialized: human-readable text describing each retrieved chunk,
                      including its source URL.
        - retrieved_docs: the raw LangChain `Document` objects so the UI can
                          display sources and other metadata.
    """
    # Use the vector store retriever to find the top-k most relevant chunks.
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize docs into a single text blob that the LLM can easily consume.
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run the LangChain agent with the `retrieve_context` tool on a user query.

    The agent:
    - Uses `retrieve_context` to pull in relevant documentation.
    - Answers the user's question using that context.
    - Implicitly cites sources in the answer text.

    Returns:
        A dict with:
        - "answer": the final answer string from the assistant.
        - "context": a list of LangChain `Document` objects used as context.
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # Create a simple tools-enabled agent backed by the OpenAI chat model.
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Start the conversation with a single user message.
    messages = [{"role": "user", "content": query}]

    # Invoke the agent; the result contains the intermediate tool messages.
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    # Collect any context documents that were returned as tool artifacts.
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs,
    }


if __name__ == "__main__":
    # Simple manual test for running this module directly.
    result = run_llm(query="what are deep agents?")
    print(result)