# app.py — Streamlit UI for CHAMELEON COMPRESSION (real model only)
# Run: streamlit run app.py

import os
import io
import time
import struct
import gzip
import streamlit as st
from pathlib import Path

# Environment setup for stable runtime
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_num_threads(1)

# Model and arithmetic coding configuration
MODEL_NAME = "distilgpt2"
PRECISION = 64
FREQ_BITS = 24

# ------------------------------------------------------------
# Arithmetic coder components
# ------------------------------------------------------------

class ArithmeticCoder:
    # Base parameters for encoding and decoding ranges
    def __init__(self, precision=64):
        self.precision = precision
        self.MAX_CODE = (1 << precision) - 1
        self.ONE_QUARTER = 1 << (precision - 2)
        self.HALF = 1 << (precision - 1)
        self.THREE_QUARTERS = self.HALF | self.ONE_QUARTER

class Encoder(ArithmeticCoder):
    # Arithmetic encoder state transitions
    def __init__(self, precision=64, progress_cb=None):
        super().__init__(precision)
        self.low = 0
        self.high = self.MAX_CODE
        self.pending_bits = 0
        self.output = []
        self.progress_cb = progress_cb

    def encode(self, cum_freq, freq, total_freq):
        # Range update for a symbol
        if freq <= 0:
            raise ValueError(f"Negative frequency {freq}")
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * (cum_freq + freq)) // total_freq - 1

        self.low = self.low + (range_width * cum_freq) // total_freq

        # Normalization loop
        while True:
            if self.high < self.HALF:
                self.write_bit(0)
            elif self.low >= self.HALF:
                self.write_bit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.ONE_QUARTER and self.high < self.THREE_QUARTERS:
                self.pending_bits += 1
                self.low -= self.ONE_QUARTER
                self.high -= self.ONE_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def write_bit(self, bit):
        # Write current bit and flush pending ones
        self.output.append(bit)
        while self.pending_bits > 0:
            self.output.append(1 - bit)
            self.pending_bits -= 1

    def finish(self):
        # Final flush of encoding state
        self.pending_bits += 1
        self.write_bit(0 if self.low < self.ONE_QUARTER else 1)
        return self.output

class Decoder(ArithmeticCoder):
    # Arithmetic decoder implementation
    def __init__(self, bitstream, precision=64):
        super().__init__(precision)
        self.bitstream = bitstream
        self.bit_idx = 0
        self.low = 0
        self.high = self.MAX_CODE
        self.value = 0
        for _ in range(self.precision):
            self.value = (self.value << 1) | self.read_bit()

    def read_bit(self):
        # Get next bit or zero if exhausted
        if self.bit_idx < len(self.bitstream):
            bit = self.bitstream[self.bit_idx]
            self.bit_idx += 1
            return bit
        return 0

    def get_current_cum_freq(self, total_freq):
        # Determine estimated cumulative frequency for current decoding state
        range_width = self.high - self.low + 1
        return ((self.value - self.low + 1) * total_freq - 1) // range_width

    def update(self, cum_freq, freq, total_freq):
        # Update range and normalize
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * (cum_freq + freq)) // total_freq - 1
        self.low = self.low + (range_width * cum_freq) // total_freq

        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.value -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.ONE_QUARTER and self.high < self.THREE_QUARTERS:
                self.value -= self.ONE_QUARTER
                self.low -= self.ONE_QUARTER
                self.high -= self.ONE_QUARTER
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.read_bit()

# ------------------------------------------------------------
# Frequency quantization for model output
# ------------------------------------------------------------

def quantize_probs(probs, precision_bits=FREQ_BITS):
    # Convert model probability distribution to integer frequencies
    total = 1 << precision_bits
    freqs = (probs * total).long()
    freqs = torch.clamp(freqs, min=1)

    # Ensure frequencies normalize correctly
    current_sum = freqs.sum().item()
    if current_sum > total:
        diff = current_sum - total
        idx = torch.argmax(freqs)
        freqs[idx] -= diff
    elif current_sum < total:
        diff = total - current_sum
        idx = torch.argmax(freqs)
        freqs[idx] += diff

    return freqs

# ------------------------------------------------------------
# Model loader
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_name=MODEL_NAME, device=None):
    # Load GPT-2 tokenizer and model on requested device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device

# ------------------------------------------------------------
# Compression procedure
# ------------------------------------------------------------

def compress_streamlit_bytes(text, model, tokenizer, device, progress, info):
    # Tokenize
    full_tokens = tokenizer.encode(text)
    total_tokens = len(full_tokens)
    info.markdown(f"🔠 **Tokens:** {total_tokens}")

    # Encoder setup
    encoder = Encoder(precision=PRECISION)
    past = None
    start_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    current_input = torch.tensor([[start_id]], device=device)

    start_time = time.time()

    # Process each token sequentially
    for i, token_id in enumerate(full_tokens):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past)
            past = outputs.past_key_values
            probs = F.softmax(outputs.logits[0, -1, :], dim=0)

        freqs = quantize_probs(probs)
        cdf = torch.cumsum(freqs, dim=0)

        symbol_freq = int(freqs[token_id].item())
        symbol_cum_freq = int(cdf[token_id].item()) - symbol_freq
        total_freq = (1 << FREQ_BITS)

        encoder.encode(symbol_cum_freq, symbol_freq, total_freq)
        current_input = torch.tensor([[token_id]], device=device)

        # Periodic progress updates
        if (i % 1024 == 0 and i != 0) or i == total_tokens - 1:
            past = None
        if i % 8 == 0 or i == total_tokens - 1:
            elapsed = time.time() - start_time
            speed = (i + 1) / max(elapsed, 1e-6)
            progress.progress(int((i + 1) / total_tokens * 100))
            info.caption(f"Processing token {i+1}/{total_tokens} — {speed:.1f} tok/s")

    # Convert bit output to bytes
    bits = encoder.finish()
    byte_array = bytearray()
    current_byte = 0
    bit_count = 0

    for bit in bits:
        current_byte = (current_byte << 1) | bit
        bit_count += 1
        if bit_count == 8:
            byte_array.append(current_byte)
            current_byte = 0
            bit_count = 0

    if bit_count > 0:
        current_byte <<= (8 - bit_count)
        byte_array.append(current_byte)

    out = struct.pack(">I", total_tokens) + bytes(byte_array)
    if len(out) < 8:
        raise RuntimeError("Compression produced invalid output.")

    return out

# ------------------------------------------------------------
# Decompression procedure
# ------------------------------------------------------------

def decompress_streamlit_bytes(data, model, tokenizer, device, progress, info):
    # Validate header
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("Input data must be raw bytes.")
    if len(data) < 4:
        raise ValueError("Invalid file: too short.")

    total_tokens = struct.unpack(">I", data[:4])[0]

    # Convert encoded bitstream into bit list
    bits = []
    raw = data[4:]
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    # Decoder setup
    decoder = Decoder(bits, precision=PRECISION)
    past = None
    start_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    current_input = torch.tensor([[start_id]], device=device)

    decoded_tokens = []
    start_time = time.time()

    # Decode token by token
    for i in range(total_tokens):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past)
            past = outputs.past_key_values
            probs = F.softmax(outputs.logits[0, -1, :], dim=0)

        freqs = quantize_probs(probs)
        cdf = torch.cumsum(freqs, dim=0)
        total_freq = (1 << FREQ_BITS)

        target = decoder.get_current_cum_freq(total_freq)
        token_id = torch.searchsorted(cdf, target, right=True).item()
        token_id = min(token_id, freqs.size(0) - 1)

        symbol_freq = int(freqs[token_id].item())
        symbol_cum_freq = int(cdf[token_id].item()) - symbol_freq

        decoder.update(symbol_cum_freq, symbol_freq, total_freq)
        decoded_tokens.append(token_id)

        current_input = torch.tensor([[token_id]], device=device)

        # Progress tracking
        if (i % 1024 == 0 and i != 0) or i == total_tokens - 1:
            past = None
        if i % 8 == 0 or i == total_tokens - 1:
            elapsed = time.time() - start_time
            speed = (i + 1) / max(elapsed, 1e-6)
            progress.progress(int((i + 1) / total_tokens * 100))
            info.caption(f"Decoding token {i+1}/{total_tokens} — {speed:.1f} tok/s")

    # Convert tokens back to text
    text = tokenizer.decode(decoded_tokens)
    return text.encode("utf-8")

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(page_title="CHAMELEON Compression — Demo", layout="wide", page_icon="🦎")
st.title("🦎 CHAMELEON Compression — Generative AI Compressor")
st.markdown(
    """
A UI for generative compression using transformer priors.
Supports both compression and decompression workflows.
"""
)

with st.sidebar:
    st.header("Run options")
    mode = st.radio("Mode", ("Compress", "Decompress"))
    allow_gpu = st.checkbox("Allow GPU if available", value=True)
    st.markdown("---")
    st.caption("Utilizes GPT-2 and arithmetic coding for token-level compression.")

col1, col2 = st.columns([1.3, 1])

uploaded = None
uploaded_bytes = None
uploaded_name = None

# Input block
with col1:
    st.subheader("Input")
    uploaded = st.file_uploader(
        "Upload a file (text for compress / .bin for decompress)",
        type=["txt", "md", "py", "json", "csv", "bin"],
        key="uploader"
    )
    text_area = st.text_area("Or paste text here (takes precedence)", height=240)

    if uploaded is not None:
        uploaded_name = uploaded.name
        try:
            uploaded_bytes = uploaded.read()
            name_lower = uploaded_name.lower()
            if any(name_lower.endswith(ext) for ext in ('.txt', '.md', '.py', '.json', '.csv')) and text_area.strip() == "":
                try:
                    text_area = uploaded_bytes.decode("utf-8", errors="replace")
                except:
                    text_area = ""
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# Output block
with col2:
    st.subheader("Output / Controls")
    out_name = st.text_input(
        "Output filename",
        value=("compress.bin" if mode == "Compress" else "reconstructed.txt")
    )
    st.write("Selected mode:", mode)
    run_button = st.button("Run ▶️", key="run_button")
    st.write("")
    st.markdown("**Results & Downloads**")
    result_placeholder = st.empty()
    progress_bar = st.progress(0)
    info_line = st.empty()

# ------------------------------------------------------------
# Execution path
# ------------------------------------------------------------

if run_button:

    # Resolve input for chosen mode
    if mode == "Compress":
        if text_area.strip() == "" and uploaded_bytes is None:
            st.warning("Please provide data before running.")
            st.stop()

        if text_area.strip() != "":
            input_text = text_area
            input_bytes = input_text.encode("utf-8")
        else:
            try:
                input_text = uploaded_bytes.decode("utf-8", errors="replace")
                input_bytes = input_text.encode("utf-8")
            except:
                st.error("Uploaded file is not valid UTF-8 text.")
                st.stop()

    else:
        if uploaded_bytes is None:
            st.warning("Upload a .bin file for decompression.")
            st.stop()
        input_bytes = uploaded_bytes

    # Model load
    device_choice = "cuda" if (torch.cuda.is_available() and allow_gpu) else "cpu"
    info_line.info(f"Loading model ({MODEL_NAME})...")
    try:
        model, tokenizer, model_device = load_model(MODEL_NAME, device=device_choice)
        info_line.success(f"Model loaded on {model_device.upper()}")
    except Exception as e:
        st.error("Failed to load model: " + str(e))
        st.stop()

    # Compression or decompression execution
    try:
        if mode == "Compress":
            progress_bar.progress(1)
            info_line.info("Compressing...")
            out_blob = compress_streamlit_bytes(
                input_text, model, tokenizer, model_device, progress_bar, info_line
            )

            if not isinstance(out_blob, (bytes, bytearray)) or len(out_blob) < 8:
                st.error("Compression failed.")
                st.stop()

            progress_bar.progress(100)
            info_line.success("Compression complete.")

            # Compression statistics
            original_size = len(input_bytes)
            compressed_size = len(out_blob)
            percent = (1 - (compressed_size / original_size)) * 100 if original_size > 0 else 0

            st.info(
                f"📊 **Compression Statistics**\n"
                f"- Original size: **{original_size} bytes**\n"
                f"- Compressed size: **{compressed_size} bytes**\n"
                f"- Compression ratio: **{compressed_size/original_size:.3f}x**\n"
                f"- Reduction: **{percent:.2f}%**"
            )

            st.download_button(
                "Download compressed file", data=out_blob,
                file_name=out_name, mime="application/octet-stream"
            )

        else:
            progress_bar.progress(1)
            info_line.info("Decompressing...")

            if len(input_bytes) < 4:
                st.error("Invalid compressed file.")
                st.stop()

            out_text_bytes = decompress_streamlit_bytes(
                input_bytes, model, tokenizer, model_device, progress_bar, info_line
            )

            progress_bar.progress(100)
            info_line.success("Decompression complete.")
            result_placeholder.success("Decompression complete.")

            # Decompression statistics
            before = len(input_bytes)
            after = len(out_text_bytes)

            st.info(
                f"📊 **Decompression Statistics**\n"
                f"- Compressed size: **{before} bytes**\n"
                f"- Output text size: **{after} bytes**\n"
                f"- Expansion factor: **{after/before:.3f}x**"
            )

            st.download_button(
                "Download decompressed text", data=out_text_bytes,
                file_name=out_name, mime="text/plain"
            )

    except Exception as e:
        st.exception(e)

st.caption("Built for hackathon demos — CHAMELEON generative compression.")
