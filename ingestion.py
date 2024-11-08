# Import libraries to manage environment variables from a .env file
from dotenv import load_dotenv  # For loading environment variables from the .env file
import os  # To access operating system functionalities (e.g., environment variables)

# Import the Pinecone library and its classes to interact with Pinecone's vector database service
import pinecone  # Pinecone API for working with vector stores (used in RAG pipelines)
from pinecone import (
    Pinecone,
    ServerlessSpec,
)  # Pinecone class for managing the vector database and ServerlessSpec for specifying the cloud environment

# llama_index is an external library for creating document indexing systems
# It is used to break down documents into smaller components (nodes) for processing and querying with LLMs
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
)  # SimpleDirectoryReader for reading files from a directory and Settings for managing configuration
from llama_index.core.node_parser import (
    SimpleNodeParser,
)  # SimpleNodeParser for parsing document nodes into chunks or sentences

# Import OpenAI LLM and embedding models, which are used in the RAG process for generating responses and embedding the documents into vector space
from llama_index.llms.openai import (
    OpenAI,
)  # OpenAI LLM class for querying GPT models (used for generating responses in RAG)
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
)  # OpenAI Embedding class for generating text embeddings from documents (used for chunking)

# Import vector store classes to store and retrieve data efficiently during querying
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)  # VectorStoreIndex for creating an index and StorageContext for managing vector store operations
from llama_index.vector_stores.pinecone import (
    PineconeVectorStore,
)  # PineconeVectorStore allows the interaction between Pinecone and the indexing system

# Import service context to provide a consistent context for LLM services during RAG
from llama_index.core.service_context import (
    ServiceContext,
)  # Manages the context (e.g., embeddings, LLM) used throughout indexing

# Import function to load a document reader for unstructured data (e.g., HTML files)
from llama_index.core import (
    download_loader,
)  # For downloading specific loaders for unstructured data (e.g., HTML, PDFs)

# Import for sentence chunking, which splits text into manageable pieces for embedding or processing
from llama_index.core.node_parser import (
    SentenceSplitter,
)  # SentenceSplitter splits documents into chunks (for example, chunks of 512 tokens)

# Import NLTK (Natural Language Toolkit) for natural language processing
import nltk  # NLTK is a toolkit for working with human language data (text) in Python

# Download necessary NLTK resource for part-of-speech tagging
# This helps in processing and understanding text by breaking it down into its grammatical components (nouns, verbs, etc.)
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

# Load environment variables (e.g., API keys) from a .env file into the system's environment
load_dotenv()


# ================== Pinecone Initialization ==================

# Initialize Pinecone using the API key from environment variables
# Pinecone is used to store document vectors (embeddings) for fast retrieval during querying
pc = Pinecone(
    api_key=os.environ.get(
        "PINECONE_API_KEY"
    )  # Fetch the Pinecone API key from the environment
)

# Set the index name for Pinecone, which will hold the document vectors
index_name = "sas"  # A unique identifier for this specific vector store

# Check if the Pinecone index exists
# If the index does not exist, create a new one with the specified dimension and similarity metric
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # The dimensionality of the document embeddings (depends on the model, e.g., OpenAI's embeddings use 1536)
        metric="cosine",  # Cosine similarity is commonly used for comparing vectors
        spec=ServerlessSpec(
            cloud="aws",  # Pinecone is hosted on AWS
            region=os.environ.get(
                "PINECONE_ENVIRONMENT"
            ),  # Region where the Pinecone environment is hosted
        ),
    )


# ================== Document Ingestion ==================

if __name__ == "__main__":
    print("Going to ingest pinecone documentation...")
    print(f"{os.environ['PINECONE_API_KEY']}")  # Print API key (for debugging purposes)

    # Download and initialize the UnstructuredReader class
    # UnstructuredReader allows reading of unstructured document formats such as HTML files
    UnstructuredReader = download_loader("UnstructuredReader")

    # Read all the documents from the specified directory (./llamaindex-docs)
    # SimpleDirectoryReader loads documents from a local directory and extracts the text
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",  # The directory where the documents are located
        file_extractor={
            ".html": UnstructuredReader()
        },  # Extract HTML files using the UnstructuredReader
    )
    documents = dir_reader.load_data()  # Load all the data (documents) into the system

    # ================== LLM and Embeddings Configuration ==================

    # Initialize LLM (Large Language Model) for querying the data
    # The model 'gpt-4o-mini' will be used for generating text responses during data querying
    llm = OpenAI(
        model="gpt-4o-mini", temperature=0
    )  # A GPT model from OpenAI with a deterministic output (temperature=0)

    # Initialize the embedding model for converting document chunks into vector space
    # 'text-embedding-ada-002' is a popular model for generating 1536-dimensional embeddings for text data
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )  # Converts text chunks into vector embeddings

    # SentenceSplitter splits the documents into smaller chunks for efficient processing and querying
    node_parser = SentenceSplitter(
        chunk_size=512, chunk_overlap=20
    )  # Breaks the text into chunks of 512 tokens, with 20 tokens overlapping between chunks (ensuring that the transition between chunks remains coherent)

    # ================== Vector Store and Indexing ==================

    # Create an index in Pinecone to store document embeddings
    pinecone_index = pc.Index(
        index_name
    )  # Use the previously created index for storing and querying embeddings

    # VectorStore provides an interface to store and search embeddings efficiently
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )  # Initialize the vector store with Pinecone

    # StorageContext manages the storage, ensuring that documents are ingested and queried correctly
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )  # Use the default storage context for managing vector operations

    # ================== Create Vector Index ==================

    # Create a vector index with the documents
    # The LLM, embedding model, node parser, and storage context are all combined to create a searchable index
    index = VectorStoreIndex.from_documents(
        documents=documents,  # The documents to be ingested into the index
        storage_context=storage_context,  # The storage context to manage where embeddings are stored
        llm=llm,  # Use the LLM for processing queries
        embed_model=embed_model,  # Use the embedding model for converting documents into vector space
        node_parser=node_parser,  # Use the node parser to chunk the documents into manageable pieces
        show_progress=True,  # Show progress of the ingestion process
    )

    print("Finished ingesting...")
