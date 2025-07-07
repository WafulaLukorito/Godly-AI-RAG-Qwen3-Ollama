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