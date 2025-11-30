# 🦎 CHAMELEON — Generative Compression + File Comparator

This repository contains two main Streamlit applications:

- **`compressor_decompressor.py`** — CHAMELEON Generative Compression Engine  
- **`FILE_COMPARATOR.py`** — File & Text Comparator Tool  

These two tools form a complete suite:
- AI-powered text compression using GPT-2 + arithmetic coding  
- A companion utility to compare compressed/decompressed text with originals  

---

## 🦎 CHAMELEON — Generative Compression Engine (`compressor_decompressor.py`)

CHAMELEON implements modern **LLM-based text compression** using:

- DistilGPT-2 language model  
- Token probability quantization to **2²⁴ integer frequencies**  
- Custom **64-bit arithmetic coder**  
- Binary-safe `.bin` output  
- Fully lossless decompression  

### 🔥 Features
- True **generative compression** — not gzip or heuristic compression  
- Perfect restoration of original text  
- GPT-2 probabilities drive the arithmetic coder  
- Streamlit UI for:
  - Uploading or pasting text  
  - Compressing into `.bin`  
  - Decompressing back to original text  
- Full integrity even for large text input  

---

## 📗 FILE COMPARATOR (`FILE_COMPARATOR.py`)

A powerful Streamlit app for comparing pasted text or uploaded files.

### ✔️ Supports
- `.txt`, `.md`, `.json`, `.csv`, `.py`  
- **PDF files** (via `pdfplumber`)  
- UTF-8 and auto-detected encodings  
- Direct paragraph input  

### ✔️ Features
- difflib-based similarity scoring  
- Paragraph box has **priority** over uploaded file  
- Live verdict:
  - **Identical**  
  - **Very similar**  
  - **Somewhat similar**  
  - **Different**  
- Useful for:
  - Validating CHAMELEON decompressed output  
  - Text vs PDF comparison  
  - Document sanity checks  

---

## 📦 Requirements

Place this in `requirements.txt`:

```
streamlit
torch
transformers
pdfplumber
chardet
```

### Optional GPU Acceleration  
Install PyTorch with CUDA following official instructions:  
https://pytorch.org/get-started/locally/

---

## 🚀 Running Locally

### 1. Create a virtual environment
```bash
python -m venv venv
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

---

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 3. Run the Apps

#### **Start CHAMELEON Compressor**
```bash
streamlit run compressor_decompressor.py
```

#### **Start File Comparator**
```bash
streamlit run FILE_COMPARATOR.py
```

---

## 🧪 Testing

### ✔️ Test Compression + Decompression
1. Run CHAMELEON  
2. Paste text → compress → download `.bin`  
3. Upload `.bin` → decompress  
4. Compare original vs decompressed using the comparator  
5. Result should be **100% identical**

### ✔️ Test Comparator
- Paste different text on each side  
- Upload files of different formats (PDF/TXT/JSON/etc)  
- Compare extracted text  

---

## ⚠️ Important Notes

- Compression is **slow on CPU** — GPU strongly recommended  
- Must use the **exact same GPT-2 model** for decompression  
- Arithmetic coder is deterministic — even tiny model changes break compatibility  
- Comparator gracefully handles multiple encodings and PDF extraction edge cases  

---

## 🔧 Recommended Enhancements

- GPU-aware batching for faster compression  
- Support for larger LLMs (LLaMA-3 / Mistral)  
- FastAPI backend for an online API  
- Highlight character-level differences in comparator  

---

## 🙌 Credits

This project is inspired by modern neural compression research:

- Bellard’s **ts_zip**  
- **LMCompress** research papers  
- GPT-2 entropy coding experiments  

CHAMELEON integrates these concepts with a polished Streamlit interface and verification tools.
