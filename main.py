from vector_db.chroma_manager import setup_scripture_store
from chains.counselor_chain import create_counselor_chain
from interface.chat_ui import chat_with_counselor


def main():
    """Main function to run the Biblical counselor application."""
    scripture_store = setup_scripture_store()
    counselor_chain = create_counselor_chain(scripture_store)
    chat_with_counselor(counselor_chain)


if __name__ == "__main__":
    main()
