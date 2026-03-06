# Pixel: AI Photo Memory Manager (Streamlit + Face AI + NL Search)

An AI-powered photo management system that:

- **Recognizes faces** (local embeddings) and groups photos by people
- **Organizes photos automatically** (smart albums, tags, dedupe)
- **Searches in natural language** (Groq or Hugging Face, with fallback)
- **Shares securely** (export/share packages)
- **Stores securely** (encrypted-at-rest local storage)

## Quick start

1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### (Recommended) Enable local face recognition

Local face recognition requires ML packages that currently install cleanly on **Python 3.11/3.12**.

If you can use Python 3.11/3.12:

```bash
pip install -r requirements-face.txt
```

If you’re on Python 3.14 and `pip` fails building NumPy/Torch, the app will still run, but face detection/people clustering will be disabled.

2) Run:

```bash
streamlit run streamlit_app.py
```

## Configuration (optional but recommended)

Set one of these environment variables for natural-language parsing:

- **Groq**: `GROQ_API_KEY`
- **Hugging Face**: `HF_TOKEN`

If neither is set, the app still works with keyword search (no LLM).

## Notes

- First run downloads face model weights via `facenet-pytorch`.
- Your photos are stored encrypted under `data/` using a passphrase you provide in the app.