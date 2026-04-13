# RAFT and GraphRAG: Specialized LLM Fine-Tuning and Retrieval Systems

This repository contains research documents and implementation notebooks focused on **Retrieval-Augmented Fine-Tuning (RAFT)** and **Graph-based Retrieval-Augmented Generation (GraphRAG)**. The project explores methods to enhance **Large Language Model (LLM)** performance in specialized domains by combining structured retrieval with parameter-efficient fine-tuning.

---

## Technical Overview

The system integrates two advanced methodologies for improving LLM accuracy and reasoning:

### 🔹 RAFT (Retrieval-Augmented Fine-Tuning)
A training strategy that prepares models for **"open-book" exams**. It teaches the model to extract answers from relevant documents while ignoring **distractor information**.

### 🔹 GraphRAG
A retrieval architecture that organizes document chunks into a **graph structure** based on sequential flow and semantic similarity to provide richer context to the LLM.

---

## File Descriptions

### 1. RAFT Q&A (Technical Documentation)
A comprehensive guide explaining the theoretical foundations of fine-tuning and retrieval.

- **Finetuning Methods:** Details RLHF, DPO, LoRA, and Instruction Tuning  
- **RAG vs. Fine-Tuning:** Analyzes the differences between updating model weights (Long-term memory) and providing external context (Short-term memory)  
- **Tokenization:** Explains the mathematical conversion of text into numerical sequences for model processing  

---

### 2. What is RAFT (Research Summary)
Summarizes the RAFT training recipe and its impact on model robustness.

- **Chain-of-Thought (CoT):** Explains how forcing the model to generate reasoning paths prevents overfitting and improves accuracy  
- **Data Composition:** Details the use of *Golden Documents* (relevant) and *Distractor Documents* (irrelevant) to train the model's filtering capabilities  
- **Training Dynamics:** Discusses the **"P fraction"**, where the golden document is occasionally removed to force model reliance on internal knowledge  

---

### 3. Graph_RaG.ipynb (Implementation Notebook)
A step-by-step implementation of a decentralized framework called **AuraFlow** using GraphRAG.

- **Preprocessing:** Cleaning raw text and removing noise using regular expressions  
- **Vectorization:** Utilizing spaCy for sentence chunking and SentenceTransformer for generating embeddings  
- **Graph Construction:** Building a network of nodes where edges represent sequential document flow or semantic similarity thresholds (>0.80)  
- **Reranking:** Implementing a cosine similarity-based reranker to filter the most relevant nodes before LLM generation  

---

### 4. FineTune.ipynb (Implementation Notebook)
Demonstrates the practical application of RAFT using the **TinyLlama-1.1B** model.

- **Synthetic Dataset Generation:** Creates a RAFT-style dataset with an 80/20 split of golden/distractor document presence  
- **Parameter-Efficient Fine-Tuning (PEFT):** Uses QLoRA (4-bit quantization) to fine-tune specific adapter layers, reducing hardware requirements  
- **Training Pipeline:** Utilizes the Hugging Face Trainer API with paged AdamW optimization and mixed-precision training (FP16)  

---

## Key Features

- **Multi-Document Reasoning:** Models are trained to cite specific sources using `##begin_quote##` and `##end_quote##` tags  
- **Noise Robustness:** Through RAFT, the model learns to maintain accuracy even when the retrieved context contains irrelevant data  
- **Semantic Graph Retrieval:** Unlike standard RAG, the graph structure allows the model to explore related concepts that may not be immediately adjacent in the raw text  

---

## Requirements

- Python 3.10+  
- PyTorch  
- Transformers / PEFT / Bitsandbytes  
- ChromaDB (Vector Database)  
- spaCy (NLP Processing)  
- scikit-learn (Similarity Calculations)  
- Google Generative AI (for Gemini 2.5 Flash integration)  

---

## Installation

Install core dependencies:

```bash
pip install transformers peft bitsandbytes accelerate datasets torch chromadb spacy sentence-transformers scikit-learn google-generativeai
