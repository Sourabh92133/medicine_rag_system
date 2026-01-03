# Medical Conversational RAG System

This project implements a **Conversational Retrieval-Augmented Generation (RAG)** system for medical product information using **LangChain**, **Groq LLMs**, and **vector databases**.

The system supports multi-turn conversations, semantic search, vector visualization, and a Gradio-based user interface.

---

##  Demo Video
**Watch the full project demo here:**  
https://drive.google.com/file/d/1Ufvynfc0Rumnp1MUXDraArVRyxR_Ko_1/view?usp=sharing

The demo showcases:
- End-to-end RAG workflow
- Conversational queries with memory
- Vector retrieval behavior
- Gradio-based chat interface

---

## Project Overview

The system ingests a CSV dataset of medical products, converts text into embeddings, stores them in vector databases, retrieves relevant context for user queries, and generates grounded responses using an LLM.

### Key Focus Areas:
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

## ⚙️ How to Run

Follow these steps to set up and run the project locally:

### Prerequisites

*   **Python 3.8+** installed.
*   A **Groq API Key** 

### Step-by-Step Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sourabh92133/medicine_rag_system.git
    cd medicine_rag_system
    ```

2.  **Install required dependencies manually:**
    Run the following command in your terminal to install all necessary libraries:
    ```bash
    pip install langchain langchain-community langchain-groq langchain-huggingface \
                chromadb faiss-cpu sentence-transformers pandas scikit-learn \
                plotly gradio python-dotenv
    ```

3.  **Set your Groq API Key:**
    Set your API key as an environment variable in your terminal session before starting the notebook:
    ```bash
    # For Linux or Mac:
    export GROQ_API_KEY="your_api_key_here"

    # For Windows (Command Prompt):
    set GROQ_API_KEY="your_api_key_here"
    ```

4.  **Open and Run the Notebook:**
    Launch Jupyter Notebook and navigate to the main implementation file:
    ```bash
    jupyter notebook
    ```
    Inside the Jupyter file browser, open `notebooks/Medical_RAG.ipynb`.

5.  **Execute Cells:**
    Run all cells within the notebook sequentially. This process will:
    *   Ingest the data from the `data/` directory.
    *   Generate embeddings and store vectors in your chosen database (ChromaDB or FAISS).
    *   Initialize the conversational retrieval chain.
    *   Launch the **Gradio** user interface, which will provide a local URL to interact with the system.

---

## Repository Notes

- The main notebook is located in the `notebooks/` directory:
  - `notebooks/Medical_RAG.ipynb`
- The dataset used for ingestion is stored in the `data/` directory.
- Output artifacts such as embedding visualizations and demo videos are stored in the `outputs/` directory.

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
