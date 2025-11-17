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

## **Project Structure**

- `model.py` → Defines the LLM, system prompt, memory, and retrieval-aware question answering.  
- `chroma_handler.py` → Handles Chroma database creation, loading, and query operations.  
- `eval_test/` → Contains evaluation scripts using DeepEval and RAG metrics.  
- `config.py` → Stores configuration variables, e.g., file paths and directories.  

---
