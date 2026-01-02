# Medical Conversational RAG System

This project implements a **Conversational Retrieval-Augmented Generation (RAG)** system for medical product information using **LangChain**, **Groq LLMs**, and **vector databases**.  
It supports multi-turn conversations, semantic search, vector visualization, and a Gradio-based user interface.

---

## Project Overview

The system ingests a CSV dataset of medical products, converts text into embeddings, stores them in vector databases, retrieves relevant context for user queries, and generates grounded responses using an LLM.

### Key focus areas:
- Retrieval-Augmented Generation (RAG)
- Conversational memory
- Vector database comparison (Chroma vs FAISS)
- Visualization of embedding spaces
- Batch query support

---

## Features

- CSV-based document ingestion  
- Text chunking with overlap  
- Embeddings using HuggingFace MiniLM  
- Vector storage with **ChromaDB** and **FAISS**  
- Conversational memory for multi-turn Q&A  
- Batch query execution (parallel queries)  
- 2D & 3D vector visualization using **t-SNE**  
- Interactive **Gradio** chat interface  
- **Groq LLM** integration  

---

## Tech Stack

- **Python**
- **LangChain**
- **Groq LLM**
- **ChromaDB**
- **FAISS**
- **HuggingFace Sentence Transformers**
- **scikit-learn (t-SNE)**
- **Plotly**
- **Gradio**

---

## Repository Notes

- The main notebook (`Medical_RAG.ipynb`) is currently kept at the root level for easy access and review.
- The dataset used for ingestion is stored in the `data/` directory.
- Output artifacts such as embedding visualizations and demo videos are stored in the `outputs/` directory.
- Repository structure may be further refined as the project evolves.

---

## Demo & Visualizations

- Embedding visualizations (2D / 3D) are available in the `outputs/` directory.
- A recorded demo showcasing conversational queries and retrieval behavior is also included.

---

## Limitations

- The system retrieves semantically similar content but does not replace professional medical advice.
- Performance depends on embedding quality and chunking strategy.
- Not intended for clinical diagnosis or treatment recommendations.

---

## Learning Outcomes

This project demonstrates:
- Practical implementation of RAG pipelines
- Vector database design and comparison
- Evaluation and visualization of embedding spaces
- Building conversational LLM systems with memory
- End-to-end prototyping from data ingestion to UI
