import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader #Or UnstructuredPDFLoader

load_dotenv() # Load environment variables from .env file

DATA_PATH = "data/"
PDF_FILENAME = "llama2.pdf"

def load_documents():
    """
    Load documents from the specified PDF file.
    """
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    
    # Alternatively, you can use UnstructuredPDFLoader if you prefer
    # from langchain_community.document_loaders import UnstructuredPDFLoader
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {PDF_FILENAME}")
    return documents


#* Split the document into chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Split the loaded documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    
    # Split the documents into chunks
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks.")
    return all_splits

#loaded_docs = load_documents()
#chunks = split_documents(loaded_docs)

from langchain_ollama import OllamaEmbeddings
def get_embedding_function(model_name="nomic-embed-text"):
      """Initializes the Ollama embedding function."""
    # Ensure Ollama server is running (ollama serve)
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings

#embeddings = get_embedding_function()

#Set Up Local Vector Store (ChromaDB)

from langchain_community.vectorstores import Chroma

def get_vector_store(embedding_function):
    """
    Initializes the Chroma vector store with the specified embedding function.
    """
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function)
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vector_store

embedding_function = get_embedding_function()
vector_store = get_vector_store(embeddings)


#* Index Documents (Embed and Store)

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """
    Index the document chunks into the vector store.
    """
    print (f"Indexing {len(chunks)} document chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
   vector_store.persist() #ensure data is saved
   print (f"Indexing complete. Data saved to: {persist_directory}")
   return vector_store
