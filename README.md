# CHAMELEON — Generative Compression + File Comparator

This repository contains two main Streamlit applications:

- **`gpt_file3.py`** — CHAMELEON Generative Compression Engine  
- **`FILE_COMPARATOR.py`** — File & Text Comparator Tool  

These are the *only* files in the project and together form a complete suite:
- AI-powered compression using GPT-2 + arithmetic coding  
- A companion utility to compare compressed/decompressed text with originals  

---

## 🦎 CHAMELEON — Generative Compression Engine (`gpt_file3.py`)

CHAMELEON implements modern **LLM-based text compression** using:

- DistilGPT-2 language model  
- Token probability quantization to 2²⁴ integer frequencies  
- Custom 64-bit arithmetic coding  
- Binary-safe `.bin` output  
- Full decompression reversibility  

### 🔥 Features
- True generative compression — NOT gzip or heuristic compression  
- Preserves exact original text after decompression  
- Uses GPT-2’s predicted token distributions to guide the arithmetic coder  
- Streamlit UI for uploading text → compressing → downloading `.bin`  
- Safe decompression with matching model  

---

## 📗 FILE COMPARATOR (`FILE_COMPARATOR.py`)

A powerful Streamlit app for comparing *pasted paragraphs or file uploads*, supporting:

- Text files  
- PDF files (via `pdfplumber`)  
- Any UTF-8 or auto-detected encoding  
- difflib-based similarity scoring  
- Paragraph box *takes priority* over uploaded file  
- Live similarity verdict: identical / very similar / somewhat similar / different  

Useful for verifying:
- If CHAMELEON decompressed output matches original text  
- If two documents differ  
- If a PDF and a text version are the same  

---

## 📦 Requirements

Put this into `requirements.txt`:

