import streamlit as st
import pdfplumber
import chardet
import base64

st.set_page_config(page_title="File Comparator", page_icon="📝", layout="wide")

# -----------------------------------------
# Helper Functions
# -----------------------------------------

def load_bytes(file):
    if file is None:
        return None
    return file.read()

def detect_encoding(raw_bytes):
    try:
        result = chardet.detect(raw_bytes)
        return result["encoding"] or "utf-8"
    except:
        return "utf-8"

def extract_text_from_any(file_bytes, filename):
    name = filename.lower()

    # PDF
    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except:
            return "[ERROR] Could not extract text from PDF"

    # Plain-text formats
    if name.endswith((".txt", ".md", ".py", ".json", ".csv")):
        enc = detect_encoding(file_bytes)
        return file_bytes.decode(enc, errors="replace")

    # Unknown: treat as binary and try fallback decode
    try:
        enc = detect_encoding(file_bytes)
        return file_bytes.decode(enc, errors="replace")
    except:
        return "[ERROR] Unsupported file format for text extraction."

def compute_similarity(text1, text2):
    if text1.strip() == "" and text2.strip() == "":
        return 100.0

    import difflib
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return ratio * 100

# -----------------------------------------
# UI
# -----------------------------------------

st.title("🔍 File Comparator")
st.caption("Compare two files of ANY type your compressor supports — text, PDF, JSON, etc.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 File A")
    file_a = st.file_uploader("Upload File A", key="file_a", type=None)
    raw_a = load_bytes(file_a) if file_a else None
    text_a = extract_text_from_any(raw_a, file_a.name) if raw_a else ""

with col2:
    st.subheader("📄 File B")
    file_b = st.file_uploader("Upload File B", key="file_b", type=None)
    raw_b = load_bytes(file_b) if file_b else None
    text_b = extract_text_from_any(raw_b, file_b.name) if raw_b else ""

st.markdown("---")

if st.button("Compare Files 🔎", type="primary"):
    if not raw_a or not raw_b:
        st.error("Please upload **both** files before comparing.")
        st.stop()

    similarity = compute_similarity(text_a, text_b)

    # Verdict
    if similarity == 100:
        st.success("✔ Files are EXACTLY the same")
    elif similarity >= 80:
        st.info(f"🟦 Files are **very similar** ({similarity:.2f}%)")
    elif similarity >= 50:
        st.warning(f"🟨 Files are **somewhat similar** ({similarity:.2f}%)")
    else:
        st.error(f"🟥 Files are **different** ({similarity:.2f}%)")

    st.markdown("### 📊 Similarity Score")
    st.metric(label="Similarity", value=f"{similarity:.2f}%")

    # Side-by-side text preview
    st.markdown("---")
    st.markdown("### 📝 Extracted Text Preview")

    col3, col4 = st.columns(2)

    with col3:
        st.text_area("Text from File A", text_a, height=300)

    with col4:
        st.text_area("Text from File B", text_b, height=300)
