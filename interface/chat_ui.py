import time
from typing import Optional
from datetime import datetime
from langchain_core.runnables import Runnable
import logging
import readline  # For better input handling

logger = logging.getLogger(__name__)


class ChatInterface:
    """Handles the interactive counseling session."""

    def __init__(self, chain: Runnable):
        self.chain = chain
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._init_logging()

    def _init_logging(self):
        """Initialize session-specific logging."""
        self.log_file = f"logs/session_{self.session_id}.log"
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def start_session(self):
        """Run the main chat loop."""
        self._print_welcome()

        while True:
            try:
                user_input = self._get_user_input()

                if self._should_exit(user_input):
                    self._end_session()
                    break

                self._process_query(user_input)

            except KeyboardInterrupt:
                self._handle_interrupt()
                break
            except Exception as e:
                self._handle_error(e)

    def _print_welcome(self):
        """Display welcome message."""
        print("\n\033[1;36mWelcome to Biblical Counselor\033[0m")
        print("\033[0;33mType 'quit' to exit. Share what's on your heart:\033[0m\n")
        logger.info(f"SESSION STARTED - ID: {self.session_id}")

    def _get_user_input(self) -> str:
        """Get and validate user input."""
        while True:
            try:
                user_input = input("\033[1;33mYou:\033[0m ").strip()
                if user_input:
                    return user_input
                print("\033[0;31mPlease share your concern...\033[0m")
            except EOFError:
                raise KeyboardInterrupt

    def _should_exit(self, input_text: str) -> bool:
        """Check for exit commands."""
        return input_text.lower() in {'quit', 'exit', 'bye'}

    def _process_query(self, query: str):
        """Handle a single user query."""
        start_time = time.time()
        logger.info(f"QUERY: {query[:200]}")  # Truncate long queries

        print("\n\033[3;36mSearching Scripture for you...\033[0m")

        try:
            response = self.chain.invoke(query)
            self._display_response(response, time.time() - start_time)
            self._log_interaction(query, response)
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            print(
                "\n\033[0;31mI encountered an error. Let's try again...\033[0m")

    def _display_response(self, response: str, response_time: float):
        """Format and display the counselor's response."""
        print(f"\n\033[1;34mCounselor:\033[0m {response}")
        print(f"\033[0;37m(Generated in {response_time:.2f}s)\033[0m")

    def _log_interaction(self, query: str, response: str):
        """Log the full interaction."""
        logger.info(
            f"RESPONSE: {response[:500]}...")  # Truncate long responses
        with open(self.log_file, 'a') as f:
            f.write(f"\nQ: {query}\nA: {response}\n{'='*50}\n")

    def _handle_interrupt(self):
        """Handle keyboard interrupts gracefully."""
        logger.info("Session interrupted by user")
        print("\n\033[0;33mPeace be with you. Come back anytime.\033[0m")

    def _handle_error(self, error: Exception):
        """Handle unexpected errors."""
        logger.error(f"Session error: {str(error)}", exc_info=True)
        print("\n\033[0;31mAn error occurred. Let's try again...\033[0m")

    def _end_session(self):
        """Clean up session ending."""
        logger.info("SESSION ENDED")
        print(
            "\n\033[0;32mMay God bless you. Remember Jeremiah 29:11. God has a plan for your welfare\033[0m")
