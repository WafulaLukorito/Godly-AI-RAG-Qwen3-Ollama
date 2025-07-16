
# Biblical Counsellor 🤖✝️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An AI-powered scripture-based counselling assistant using Retrieval-Augmented Generation (RAG) to provide biblically-grounded guidance.

## Features ✨

* XML Bible processing with full verse context
* Local AI inference via Ollama
* Context-aware scripture retrieval
* Compassionate response formatting
* Interactive command-line interface
* Persistent vector knowledge base

## Disclaimer ⚠️

**This application is not a substitute for professional counselling or pastoral care.** It is an experimental AI tool that:
* Should not be used for serious mental health concerns
* May occasionally provide imperfect or incomplete guidance
* Is intended for spiritual encouragement only

Always consult qualified human counsellors for serious matters.

## Technologies Used 🛠️

### Core Stack
* **Python 3.9+** (Primary language)
* **Ollama** (Local LLM inference)
* **LangChain** (RAG framework)
* **ChromaDB** (Vector database)
* **Pydantic** (Configuration management)

### NLP Components
* **nomic-embed-text** (Embeddings)
* **Qwen/LLaMA** (Language models)
* **XML Processing** (Bible corpus)

### Infrastructure
* **Poetry** (Dependency management)
* **Logging** (Activity tracking)
* **Multi-threading** (Performance)

## Installation 🛠️

1.  **Prerequisites**:
    * Python 3.9+
    * [Ollama](https://ollama.ai/) installed and running
    * Bible XML file in `data/` directory

2.  **Set up environment**:

    ```bash
    git clone [https://github.com/WafulaLukorito/Godly-AI-RAG-Qwen3-Ollama/tree/main](https://github.com/WafulaLukorito/Godly-AI-RAG-Qwen3-Ollama/tree/main)
    cd Godly-AI-RAG-Qwen3-Ollama
    pip install -r requirements.txt
    ```

3.  **Download models**:

    ```bash
    ollama pull qwen:8b
    ollama pull nomic-embed-text
    ```

## Usage 🚀

### Command Line Interface

```bash
python -m biblical_counselor.main [OPTIONS]
````

**Options:**

  * `--verbose`, `-v` Enable debug logging
  * `--reset-db` Recreate vector database
  * `--model MODEL` Override default LLM model
  * `--port PORT` Enable web interface on specified port

**Example session:**

```text
Welcome to Biblical Counsellor!
Share your concern (type 'quit' to exit):

You: I'm feeling anxious about my future
Counsellor: I hear your anxiety about what lies ahead...
[Matthew 6:34] "Therefore do not worry about tomorrow..."
[Psalm 23:4] "Even though I walk through the darkest valley..."
Let us pray: Dear Lord, grant peace to your child...
```

## Contributing 🤝

We welcome contributions\! Here's how to help:

  * Report bugs via [Issues](https://www.google.com/search?q=https://github.com/WafulaLukorito/Godly-AI-RAG-Qwen3-Ollama/issues)
  * Suggest features in [Discussions](https://github.com/WafulaLukorito/Godly-AI-RAG-Qwen3-Ollama/discussions/1) (or create one if not available)
  * Submit Pull Requests (PRs) for:
      * Better verse retrieval
      * Improved response templates
      * Additional Bible translations
      * UX enhancements


## Roadmap 🗺️

  * Web interface (Flask/FastAPI)
  * Multiple Bible translations
  * User session history
  * Emotional tone analysis
  * Prayer journal feature

## Licence 📜

MIT Licence - See [LICENCE](https://choosealicense.com/licenses/mit/) for details.

"Your word is a lamp for my feet, a light on my path." - Psalm 119:105

```
```