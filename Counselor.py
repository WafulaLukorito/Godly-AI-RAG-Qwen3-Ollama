
import os
import time
import threading
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='app_activity.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Configuration
CONFIG = {
    "data_path": "data/",
    "bible_xml": "bible.xml",
    "chroma_path": "bible_chroma_db",
    "embedding_model": "nomic-embed-text",
    "llm_model": "qwen3:8b",
    "chunk_size": 300,
    "chunk_overlap": 50,  # Increased overlap for better verse context
    "retrieval_k": 5,
    "score_threshold": 0.3
}

def load_bible():
    """Load and parse XML Bible with enhanced logging."""
    logging.info("Starting Bible XML loading process...")
    bible_path = os.path.join(CONFIG["data_path"], CONFIG["bible_xml"])
    
    if not os.path.exists(bible_path):
        logging.error(f"Bible XML not found at {bible_path}")
        raise FileNotFoundError(f"Bible XML not found at {bible_path}")
    
    try:
        start_time = time.time()
        tree = ET.parse(bible_path)
        root = tree.getroot()
        documents = []
        verse_count = 0
        
        for book in root.findall(".//book"):
            book_name = book.get("name", f"Book_{book.get('number')}")
            logging.info(f"Processing {book_name}...")
            
            for chapter in book.findall("chapter"):
                for verse in chapter.findall("verse"):
                    verse_text = verse.text.strip()
                    metadata = {
                        "book": book_name,
                        "chapter": chapter.get("number"),
                        "verse": verse.get("number"),
                        "reference": f"{book_name} {chapter.get('number')}:{verse.get('number')}",
                        "canonical_reference": f"{book_name} {chapter_num}:{verse_num}",
                        "timestamp": datetime.now().isoformat()
                    }
                    documents.append(Document(page_content=verse_text, metadata=metadata))
                    verse_count += 1
                    
                    # Log every 100 verses for progress tracking
                    if verse_count % 100 == 0:
                        logging.info(f"Processed {verse_count} verses...")
        
        logging.info(f"Completed loading {verse_count} verses in {time.time()-start_time:.2f}s")
        return documents
    
    except Exception as e:
        logging.error(f"Error processing XML Bible: {str(e)}", exc_info=True)
        raise
    
    
def split_scriptures(documents, log_file='scripture_split_log.txt', update_interval=10):
    """Split Bible verses with context-aware chunking."""
    print("Starting scripture processing...")

    start_time = time.time()
    updates_active = True

    def periodic_update():
        while updates_active:
            elapsed = time.time() - start_time
            print(f"Processing... ({int(elapsed)}s elapsed)")
            time.sleep(update_interval)

    update_thread = threading.Thread(target=periodic_update)
    update_thread.start()

    # Verse-aware splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", ". ", "? ", "! "],  # Natural verse boundaries
        length_function=len,
        keep_separator=True,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)

    # Stop updates
    updates_active = False
    update_thread.join()

    total_time = time.time() - start_time
    print(f"Processed {len(chunks)} scripture chunks")
    print(f"Completed in {total_time:.2f} seconds")

    # Log processing details
    with open(log_file, 'a') as log:
        log.write(f"Verses processed: {len(documents)}\n")
        log.write(f"Chunks generated: {len(chunks)}\n")
        log.write(f"Chunk size: {CONFIG['chunk_size']} (overlap: {CONFIG['chunk_overlap']})\n")
        log.write(f"Time: {total_time:.2f}s | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write('-' * 40 + '\n')

    return chunks


def get_embedding_function():
    """Initialize embeddings."""
    try:
        return OllamaEmbeddings(model=CONFIG["embedding_model"])
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise


def setup_scripture_store():
    """Create/load vector store with optimized XML processing."""
    logging.info("Initializing scripture vector store...")
    try:
        embedding_function = get_embedding_function()

        if os.path.exists(CONFIG["chroma_path"]) and os.listdir(CONFIG["chroma_path"]):
            logging.info(f"Loading existing vector store from {CONFIG['chroma_path']}") # Added path to log
            return Chroma(
                persist_directory=CONFIG["chroma_path"],
                embedding_function=embedding_function
            )

        logging.info("Building new vector store from XML...")
        verses = load_bible()
        
        # Only split if any verse exceeds chunk size
        long_verses = [v for v in verses if len(v.page_content) > CONFIG["chunk_size"]]
        if long_verses:
            logging.warning(f"Found {len(long_verses)} verses exceeding chunk size, splitting...")
            verses = split_scriptures(verses)
        else:
            logging.info("No verses exceeded chunk size, skipping splitting step.")

        # --- THIS IS THE SLOW PART: ADDING PROGRESS TRACKING ---
        logging.info(f"Starting embedding process for {len(verses)} scripture chunks. This may take a while...")
        
        # Start a periodic update thread for embedding
        embedding_start_time = time.time()
        embedding_updates_active = True

        def periodic_embedding_update():
            while embedding_updates_active:
                elapsed = time.time() - embedding_start_time
                print(f"\033[3;35mEmbedding chunks in progress... ({int(elapsed)}s elapsed)\033[0m") # Added color
                time.sleep(15) # Increased interval slightly for embedding

        embedding_thread = threading.Thread(target=periodic_embedding_update)
        embedding_thread.start()

        db = Chroma.from_documents(
            documents=verses,
            embedding=embedding_function,
            persist_directory=CONFIG["chroma_path"]
        )

        # Stop embedding updates
        embedding_updates_active = False
        embedding_thread.join()
        # --- END OF SLOW PART TRACKING ---

        logging.info(f"Vector store created and embeddings completed in {time.time()-embedding_start_time:.2f}s")
        return db

    except Exception as e:
        logging.error("Failed to setup vector store", exc_info=True)
        raise
    
    
def create_counselor_chain(vector_store):
    """Create counseling chain with enhanced verse handling."""
    llm = ChatOllama(
        model=CONFIG["llm_model"],
        temperature=0.4,
        num_ctx=4096,
        num_gpu=1,  # Enable GPU if available
        timeout=300,  # 5 minute timeout
        system="""You are a compassionate Christian counselor. When responding:
        1. Acknowledge feelings first
        2. Provide 1-3 relevant Bible verses with FULL references (e.g., [John 3:16-17])
        3. Connect verses to their situation
        4. End with a short prayer"""
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": CONFIG["retrieval_k"],
            "score_threshold": CONFIG["score_threshold"],
            # REMOVE OR COMMENT OUT THIS LINE:
            # "filter": {}  # Can add metadata filters here
        }
    )

    template = """Respond as a Christian counselor using these guidelines:

    Person's Situation: {question}

    Relevant Scripture: {context}

    Structure your response:
    1. [Empathy] Acknowledge their feelings
    2. [Verses] 1-3 passages with EXACT references:
        - [Book Chapter:Verse] "quoted text"
    3. [Application] Practical wisdom from these verses and what you believe God is saying to them.
    4. [Prayer] 2-3 sentence prayer

    Always include verse references in this format: [Book Chapter:Verse] or
    [Book Chapter:Verse-Verse]"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            ref = doc.metadata.get("reference", "Unknown")
            formatted.append(f"[{ref}] {doc.page_content}")
        return "\n\n".join(formatted)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def chat_with_counselor(chain):
    """Interactive session with enhanced safeguards."""
    print("\n\033[1;36mWelcome to Biblical Counselor (XML Edition)\033[0m")
    print("\033[0;33mNote: I am an AI assistant, not a substitute for professional counseling.\033[0m")
    print("Share your concern (type 'quit' to exit):\n")
    
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.info(f"Starting counseling session {session_id}")
    
    while True:
        try:
            user_input = input("\033[1;33mYou:\033[0m ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                logging.info(f"Session {session_id} ended by user")
                print("\nMay God bless you. Remember Jeremiah 29:11 - God has plans for your welfare.")
                break
                
            if not user_input:
                print("Please share what's on your heart...")
                continue

            logging.info(f"Session {session_id} query: {user_input[:50]}...")  # Log truncated query
            start_time = time.time()
            
            print("\n\033[3;36mSearching Scripture for you...\033[0m")
            response = chain.invoke(user_input)
            
            response_time = time.time() - start_time
            logging.info(f"Response generated in {response_time:.2f}s")
            
            # Print formatted response
            print(f"\n\033[1;34mCounselor:\033[0m {response}")
            print(f"\033[0;37m(Generated in {response_time:.2f}s)\033[0m")
            
            # Log counseling interaction
            with open("counseling_logs.txt", "a") as log:
                log.write(f"\n[{datetime.now()}] Session {session_id}\n")
                log.write(f"Query: {user_input}\n")
                log.write(f"Response: {response[:500]}...\n")  # Store truncated response
                log.write("-"*50 + "\n")
            
        except KeyboardInterrupt:
            logging.info("Session interrupted by user")
            print("\n\033[0;33mPeace be with you. Come back anytime.\033[0m")
            break
        except Exception as e:
            logging.error(f"Error in session {session_id}: {str(e)}", exc_info=True)
            print("\n\033[0;31mI encountered an error. Let's try again...\033[0m")

def main():
    print("\nInitializing Biblical Counselor (XML Version)...")
    try:
        scripture_store = setup_scripture_store()
        counselor_chain = create_counselor_chain(scripture_store)
        
        print("\nCounselor ready. Share your burden:")
        chat_with_counselor(counselor_chain)
        
    except Exception as e:
        print(f"\nFailed to start: {e}")

if __name__ == "__main__":
    main()