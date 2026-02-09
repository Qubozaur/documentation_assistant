import asyncio
import logging
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings,
)

tavily_extract = TavilyExtract()

tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)

tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    logger.info("Starting indexing %s documents in %s batches", len(documents), len(batches))

    for batch_num, batch in enumerate(batches, start=1):
        try:
            logger.info("Indexing batch %s with %s documents", batch_num, len(batch))
            vectorstore.add_documents(batch)
        except Exception as e:
            logger.error("Failed to index batch %s: %s", batch_num, e, exc_info=True)
            raise

    logger.info("Finished indexing documents into Pinecone")        


async def main():
    logger.info("Starting Tavily crawl")
    tavily_crawl_response = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )

    results = tavily_crawl_response.get("results", [])
    logger.info("Tavily crawl returned %s results", len(results))

    all_docs = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result.get("url", "unknown")},
        )
        for result in results
        if result.get("raw_content")
    ]

    logger.info("Built %s documents from crawl results", len(all_docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    logger.info("Split into %s chunks", len(splitted_docs))

    await index_documents_async(splitted_docs, batch_size=500)


if __name__ == "__main__":
    asyncio.run(main())