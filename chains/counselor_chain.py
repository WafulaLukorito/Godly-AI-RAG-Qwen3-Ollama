from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from langchain_core.vectorstores import VectorStoreRetriever
from config.settings import config
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CounselorChainBuilder:
    """Constructs the counseling conversation chain."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 extra_args: Optional[list] = None,
                 temperature: float = 1.2,
                 num_ctx: int = 4096):
        self.model_name = model_name or config.llm_model
        self.extra_args = extra_args or []
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._llm = None
        self._prompt_template = None

    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm

    def _initialize_llm(self) -> ChatOllama:
        """Initialize the Ollama chat model."""
        try:
            logger.info(f"Initializing LLM model: {config.llm_model}")
            return ChatOllama(
                model=config.llm_model,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                num_gpu=1,
                timeout=300,
                system=self._get_system_prompt()
            )
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize language model")

    def _get_system_prompt(self) -> str:
        """Return the system prompt template."""
        return """You are a compassionate Christian counselor. When responding:
        1. Acknowledge feelings first
        2. Provide 1-3 relevant Bible verses with FULL references (e.g., [John 3:16-17])
        3. Connect verses to their situation
        4. End with a short prayer"""

    def build_chain(self, retriever: VectorStoreRetriever):
        """Construct the complete RAG chain."""
        try:
            return self._create_chain(retriever)
        except Exception as e:
            logger.error(f"Chain construction failed: {e}", exc_info=True)
            raise RuntimeError("Failed to build counseling chain")

    def _create_chain(self, retriever: VectorStoreRetriever):
        """Create the processing pipeline."""
        prompt = self._get_chat_prompt()

        return (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_chat_prompt(self) -> ChatPromptTemplate:
        """Return the structured chat prompt."""
        return ChatPromptTemplate.from_template("""Respond as a Christian
                                                counselor using these
                                                guidelines:

Person's Situation: {question}

Relevant Scripture: {context}

Structure your response:
1. [Empathy] Acknowledge their feelings
2. [Verses] 1-3 passages with EXACT references:
    - [Book Chapter:Verse] "quoted text"
3. [Application] Practical wisdom from these verses
4. [Prayer] A short prayer in first person to help the user in the situation they're facing""")

    def _format_docs(self, docs) -> str:
        """Format retrieved documents with metadata."""
        formatted = []
        for doc in docs:
            ref = doc.metadata.get("reference", "Unknown")
            content = doc.page_content
            formatted.append(f"[{ref}] {content}")
        return "\n\n".join(formatted)
