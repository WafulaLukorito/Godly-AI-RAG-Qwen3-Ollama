import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List
from langchain.docstore.document import Document
from config.settings import config
import logging
from utils.bible_books import get_book_name

logger = logging.getLogger(__name__)


def load_bible() -> List[Document]:
    """Load and parse XML Bible with enhanced error handling.

    Returns:
        List[Document]: Parsed Bible verses as Langchain Documents

    Raises:
        FileNotFoundError: If XML file doesn't exist
        ET.ParseError: For invalid XML structure
    """
    bible_path = os.path.join(config.data_path, config.bible_xml)
    logger.info(f"Loading Bible from {bible_path}")

    if not os.path.exists(bible_path):
        logger.error(f"XML file not found at {bible_path}")
        raise FileNotFoundError(f"Bible XML not found at {bible_path}")

    try:
        tree = ET.parse(bible_path)
        root = tree.getroot()
        documents = []

        for book in root.findall(".//book"):
        book_num = book.get("number")
        book_name = get_book_name(book_num)  # Use mapping function
        logger.debug(f"Processing {book_name}")

        for chapter in book.findall("chapter"):
            chapter_num = chapter.get("number")
            for verse in chapter.findall("verse"):
                verse_num = verse.get("number")
                verse_text = verse.text.strip() if verse.text else ""

                metadata = {
                    "book": book_name,  # Now using proper name
                    "book_number": book_num,  # Keep number for reference
                    "chapter": chapter_num,
                    "verse": verse_num,
                    "reference": f"{book_name} {chapter_num}:{verse_num}",
                    "timestamp": datetime.now().isoformat()
                }

                documents.append(
                    Document(
                        page_content=verse_text,
                        metadata=metadata
                    )
                )

        logger.info(f"Successfully loaded {len(documents)} verses")
        return documents

    except ET.ParseError as e:
        logger.error(f"XML parsing failed: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise RuntimeError(f"Bible loading failed: {e}")
