from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.chroma import ChromaVectorStore


def setup_indexes(config, logger) -> list[BaseIndex]:
    logger.info(f"Load configurations")

    logger.info("Setup connection with the VectorDB")
    client = chromadb.PersistentClient(
        path="D:/python_projects/llm-ollama-llamaindex-bootstrap-ui/backend/vector_db",
    )

    logger.info("Loading embedding model")
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDINGS
    )

    logger.info("Loading indexes from VectorDB collection")
    pdf_collection = client.get_or_create_collection("pdf_data")
    html_collection = client.get_or_create_collection("html_data")
    pdf_vector_store = ChromaVectorStore(chroma_collection=pdf_collection)
    html_vector_store = ChromaVectorStore(chroma_collection=html_collection)
    pdf_index = VectorStoreIndex.from_vector_store(
        pdf_vector_store,
        embed_model=embed_model,
    )
    html_index = VectorStoreIndex.from_vector_store(
        html_vector_store,
        embed_model=embed_model,
    )
    return [pdf_index, html_index]


def setup_tools(indexes: list[BaseIndex]) -> list[QueryEngineTool]:
    pdf_engine = indexes[0].as_query_engine(llm=None, similarity_top_k=3)
    html_engine = indexes[1].as_query_engine(llm=None, similarity_top_k=3)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=pdf_engine,
            metadata=ToolMetadata(
                name="BME_Rulebook_Data_Store",
                description=(
                    "This tool is a data store, containing chunks of legal regulations and rules that specify how BME, "
                    "a Hungarian University functions."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=html_engine,
            metadata=ToolMetadata(
                name="BME_FAQ_Website_Data_Store",
                description=(
                    "This tool is a data store, containing frequently asked questions and their answers from the "
                    "website of the Central Study Office of BME, a Hungarian university. "
                    "The data contains information about the use of university systems, "
                    "academic matters, finances, student ID card creation and student loans."
                ),
            ),
        ),
    ]
    return query_engine_tools


def setup_llm(config) -> OpenAI:
    llm = OpenAI(model="gpt-4o-mini", system_prompt=config.SYSTEM_PROMPT, api_key=config.OPENAI_API_KEY)
    return llm


def setup_memory() -> ChatMemoryBuffer:
    memory = ChatMemoryBuffer(token_limit=8000)
    return memory

def setup_agent(config, logger) -> OpenAIAgent:
    indexes = setup_indexes(config, logger)
    llm = setup_llm(config)
    tools = setup_tools(indexes)
    memory = setup_memory()

    agent = OpenAIAgent(
        tools=tools,
        llm=llm,
        memory=memory,
        prefix_messages=[],
        verbose=True
    )
    return agent
