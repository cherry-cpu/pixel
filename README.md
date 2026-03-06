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