
"""
Biblical Counselor - Main Application Entry Point
"""

import argparse
import sys
import logging
from typing import Optional, Tuple

# Local imports using relative paths
from config import settings
from vector_db.chroma_manager import ChromaManager
from chains.counselor_chain import CounselorChainBuilder
from interface.chat_ui import ChatInterface
from utils.log_manager import log_manager


def parse_args() -> Tuple[argparse.Namespace, list]:
    """Parse command line arguments and unknown args for LLM."""
    parser = argparse.ArgumentParser(
        description="Biblical Counselor - Scripture-based AI assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable debug logging"
    )
    parser.add_argument(
        '--reset-db',
        action='store_true',
        help="Recreate vector database"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Override default LLM model (e.g. 'llama3:70b')"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help="Enable web interface on specified port"
    )

    return parser.parse_known_args()


def initialize_system(
    verbose: bool = False,
    reset_db: bool = False,
    model_name: Optional[str] = None,
    llm_args: Optional[list] = None
) -> Optional[ChatInterface]:
    """Initialize all application components."""
    try:
        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logger = log_manager.configure_logging(console_level=log_level)
        logger.info("Starting Biblical Counselor initialization")

        # Initialize vector store
        chroma_mgr = ChromaManager()
        if reset_db:
            logger.warning("Resetting vector database as requested")
            chroma_mgr.reset_vector_store()

        # Build counseling chain
        chain_builder = CounselorChainBuilder(
            model_name=model_name,
            extra_args=llm_args
        )
        retriever = chroma_mgr.get_retriever()
        counseling_chain = chain_builder.build_chain(retriever)

        return ChatInterface(counseling_chain)

    except Exception as e:
        logging.critical(f"Initialization failed: {str(e)}", exc_info=True)
        return None


def start_web_interface(port: int, chat_ui: ChatInterface):
    """Start Flask web interface (placeholder)."""
    print(f"\n⚠️  Web interface not yet implemented. Would run on port {port}")
    print("Using command line interface instead...\n")
    chat_ui.start_session()


def main():
    """Application entry point."""
    args, remaining_args = parse_args()

    try:
        chat_ui = initialize_system(
            verbose=args.verbose,
            reset_db=args.reset_db,
            model_name=args.model,
            llm_args=remaining_args
        )

        if not chat_ui:
            sys.exit(1)

        if args.port:
            start_web_interface(args.port, chat_ui)
        else:
            chat_ui.start_session()

    except KeyboardInterrupt:
        print("\n\nApplication terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
