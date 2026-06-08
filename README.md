# RAG System with LangChain, Chroma, and DeepEval

This repository contains a project implementing a **Retrieval-Augmented Generation (RAG) system** using LangChain, Chroma Vector Database, and DeepEval for evaluation. The project demonstrates advanced techniques in system prompting, memory integration, vector-based retrieval, and structured evaluation of LLM responses.

---

## **Project Tasks**

### **Task 0: System Prompting**
- Implemented a **system prompt** to control the role, style, and behavior of the model.
- Ensures the AI responds consistently, maintains context, and respects defined boundaries.
- Focused on **AI/ML educational content**, providing clear and structured explanations.

### **Task 1: Memory Integration**
- Integrated **conversational memory** using LangChain’s memory module.
- The model retains past interactions, tracks user inputs, and recalls important details.
- Provides **context-aware, coherent responses** across multiple conversational turns.

### **Task 2: Chroma Vector Database**
- Replaced FAISS with **Chroma** for vector storage.
- Embeddings are persisted locally, ensuring data **retention across sessions**.
- Supports **efficient retrieval** of relevant documents for RAG workflows.

### **Task 3: RAG Evaluation**
- Investigated **retrieval evaluation metrics**: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), nDCG.
- Applied one of these metrics to the RAG system to **benchmark retrieval performance**.
- Ensures that the system surfaces **relevant and accurate documents**.

### **Task 4: DeepEval Integration**
- Used **DeepEval** to evaluate the RAG system.
- Measured **accuracy, relevance, and groundedness** of generated answers.
- Provides a structured evaluation pipeline to ensure **reliable and trustworthy outputs**.

---
## **Key dependencies include:**
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `chromadb` - Vector database
- `streamlit` - Web interface
- `sentence-transformers` - Embeddings
- `wikipedia` - Data source
---
##  **Configuration:**

### Configuration File (`config.py`)

Key settings:

```python
# Topics for Wikipedia knowledge base
TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    # ... more topics
]

# Chunking parameters
CHUNK_SIZE = 1500          # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
MAX_DOCS_PER_TOPIC = 3     # Documents per topic
MAX_CHARS_PER_DOC = 40_000 # Max characters per document

# File paths
DATA_DIR = "data/"
CHROMA_DIR = "data/chroma_ai_db"
```
---

## **Application Features**

1. **Authentication**
   - Sign up with username and password
   - Login with existing credentials
   - Passwords are hashed using SHA-256

2. **Chat Interface**
   - Ask AI/ML related questions
   - System uses RAG to retrieve relevant documents
   - Responses include context from knowledge base
   - Chat history is saved per user

3. **Knowledge Base**
   - Searches 15 AI/ML Wikipedia topics
   - Returns top 4 relevant documents
   - Uses semantic search with embeddings

---

##  **Core Components:**

### `app4.py` - Main Application
Streamlit-based web interface with:
- Login/signup functionality
- Chat UI with message history
- User session management
- Integration with RAG pipeline

### `model.py` - LLM Configuration
- **LLM**: OpenAI ChatGPT-4
- **System Prompt**: AI/ML tutor specialization
- **Memory**: Conversation buffer for context
- **Response Generation**: With retrieval-augmented context

### `rag_handler.py` - RAG Pipeline
Orchestrates the retrieval-augmented generation:
1. Query vector database
2. Retrieve top-k documents (k=4)
3. Pass context to LLM
4. Generate response

### `chroma_handler.py` - Vector Database
- **Embedding Model**: BAAI/bge-base-en-v1.5
- **Persistence**: SQLite-based storage
- **Operations**: Load chunks, create/load database, query

### `auth.py` - Authentication
- SQLite database for users
- Password hashing with SHA-256
- Conversation storage per user
- Session management

### `config.py` - Configuration
Centralized configuration for:
- Topics and data sources
- File paths
- Chunking parameters
- Model parameters

### `load_data.py` - Data Loading
- Wikipedia article fetching
- Document chunking with NLTK
- JSON serialization
---
##  **System Prompt Specialization:**

The system is configured as an **AI/ML Educational Tutor** with:
- Expertise in: AI, ML, Deep Learning, NLP, Computer Vision, Data Science
- Response style: Clear, educational, practical
- Boundaries: Declines non-technical questions
- Focus: Learning and understanding

---

