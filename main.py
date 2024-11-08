# Import required libraries

from dotenv import load_dotenv  # For loading environment variables from .env file
import os  # For accessing operating system functionalities, like environment variables
import streamlit as st  # Streamlit for building and deploying interactive web apps
from pinecone import (
    Pinecone,
    ServerlessSpec,
)  # Pinecone class and ServerlessSpec for managing Pinecone vector store service

# LlamaIndex libraries for managing debugging, callbacks, embeddings, LLMs, and indexing
from llama_index.core.callbacks import (
    LlamaDebugHandler,
    CallbackManager,
)  # Debugging and callback handling
from llama_index.core.settings import Settings  # Global settings for LlamaIndex
from llama_index.core.chat_engine.types import (
    ChatMode,
)  # Different chat modes for interacting with the LLM

from llama_index.embeddings.openai import (
    OpenAIEmbedding,
)  # OpenAI Embeddings for converting text into vector space
from llama_index.llms.openai import (
    OpenAI,
)  # OpenAI Large Language Model (LLM) for querying documents and answering questions
from llama_index.core.indices.postprocessor import (
    SentenceEmbeddingOptimizer,
)  # Optimizes sentence embeddings for query accuracy

# from node_postprocessors.duplicate_postprocessing import DuplicateRemoverNodePostprocessor
from llama_index.core import (
    VectorStoreIndex,
)  # Manages the document embeddings as a vector store index
from llama_index.vector_stores.pinecone import (
    PineconeVectorStore,
)  # Interface for working with Pinecone vector store

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# ================== Debugging and Callbacks ==================

# Initialize the LlamaDebugHandler to handle debugging events and print traces when the process ends
llama_debug = LlamaDebugHandler(print_trace_on_end=True)

# Create a CallbackManager to manage different callback handlers (e.g., debugging events)
callback_manager = CallbackManager(handlers=[llama_debug])

# ================== LLM and Embeddings Setup ==================

# Initialize the embedding model from OpenAI (e.g., Ada model for embedding chunks of text)
embed_model = OpenAIEmbedding()

# Initialize the OpenAI LLM for generating responses (e.g., GPT models)
llm = OpenAI(model="gpt-4o-mini", temperature=0) 
# llm = OpenAI() default is often GPT-3.5, usually referenced as "gpt-3.5-turbo"
#

# ================== Global Settings ==================

# Set the global LlamaIndex settings to use the initialized LLM and embedding model
Settings.llm = llm  # Configure the LLM globally
Settings.embed_model = embed_model  # Set the embedding model globally
Settings.callback_manager = (
    callback_manager  # Assign the callback manager globally for logging and debugging
)

# ================== Vector Index Creation ==================


# Streamlit function for caching the vector store index
# The @st.cache_resource decorator ensures that the vector store index is cached so it doesn't get recomputed multiple times
@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    print("RAG...")  # Print a message for debugging
    # Initialize Pinecone using the API key from environment variables
    pc = Pinecone(
        api_key=os.environ.get(
            "PINECONE_API_KEY"
        )  # Load the Pinecone API key from environment variables
    )

    # Set the index name to 'sas' for the vector store
    index_name = "sas"

    # Check if the Pinecone index exists
    # If it doesn't exist, create a new one with the given name, dimensions, and metric (cosine similarity)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,  # Name of the index
            dimension=1536,  # Dimensionality of the embeddings (1536 is standard for OpenAI's embedding model)
            metric="cosine",  # Metric to measure the similarity between embeddings (cosine similarity)
            spec=ServerlessSpec(
                cloud="aws",  # Cloud environment (AWS)
                region=os.environ.get(
                    "PINECONE_ENVIRONMENT"
                ),  # The region to deploy Pinecone services
            ),
        )

    # Connect to the Pinecone index
    pinecone_index = pc.Index(index_name)

    # Set up the vector store to manage document embeddings using the Pinecone vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Return a new VectorStoreIndex object that handles all interactions with Pinecone
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


# Retrieve the vector store index (cached by Streamlit)
index = get_index()

# ================== Chat Engine Setup ==================

# Check if the chat engine is not yet initialized in the Streamlit session state
if "chat_engine" not in st.session_state.keys():

    # Initialize a postprocessor to optimize sentence embeddings and remove less relevant information
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=embed_model,  # The embedding model to optimize
        percentile_cutoff=0.5,  # Cutoff threshold for removing less relevant chunks
        threshold_cutoff=0.7,  # Another threshold for filtering embeddings
    )

    # Create a chat engine to handle user interactions using the indexed documents and embeddings
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,  # Use the 'context' mode for chat (helps to maintain conversation context)
        verbose=True,  # Enable verbose mode for debugging and transparency
        # node_postprocessors=[postprocessor, DuplicateRemoverNodePostprocessor()],  # Apply postprocessors
    )

# ================== Streamlit Web Interface ==================

# Set up the Streamlit page configuration (for the chat interface)
st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",  # Title for the browser tab
    page_icon="🦙",  # Page icon (an emoji of a llama)
    layout="centered",  # Center the layout in the middle of the screen
    initial_sidebar_state="auto",  # Sidebar can be expanded or collapsed by the user
    menu_items=None,  # No custom menu items
)

# Title for the chat application displayed in the Streamlit interface
st.title("Chat with LlamaIndex docs 💬🦙")

# ================== Chat Functionality ==================

# Initialize the 'messages' session state if not already initialized
if "messages" not in st.session_state.keys():
    # Create a default assistant message to start the conversation
    st.session_state.messages = [
        {
            "role": "assistant",  # The assistant (LLM) responds
            "content": "Ask me a question about LlamaIndex's open source python library?",  # Initial message prompt
        }
    ]

# Capture the user's input in a text box
if prompt := st.chat_input("Your question"):
    # Append the user's message to the chat history in the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the chat messages in the conversation (loop through the session state's message history)
for message in st.session_state.messages:
    # For each message, create a chat bubble (assistant or user)
    with st.chat_message(message["role"]):
        st.write(message["content"])  # Display the content of each message

# If the latest message was sent by the user, generate a response from the assistant (LLM)
if st.session_state.messages[-1]["role"] != "assistant":
    # Show a spinner while the assistant is generating a response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the chat engine to generate a response based on the user's question
            response = st.session_state.chat_engine.chat(message=prompt)
            # Display the assistant's response
            st.write(response.response)
            # Append the response to the chat history
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
