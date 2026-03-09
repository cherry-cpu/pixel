# Pixel — AI Photo Memory Manager

**Hackathon-ready:** AI-powered photo management with **face recognition**, **smart organization**, **natural language search**, **automated sharing**, and **secure storage**.

- **Face recognition** — Local embeddings (MTCNN + InceptionResnetV1), cluster by person, assign names, search “photos of X”.
- **Smart organization** — Auto tags (indoor/outdoor, scene), quality score, events by time, near-duplicate detection; optional Hugging Face caption/classification.
- **Natural language search** — Groq or Hugging Face LLM turns queries like “photos with Arjun at the beach” into structured filters.
- **Automated sharing** — Create encrypted share packages (token + .share file); import and view in read-only mode.
- **Secure storage** — Passphrase-derived encryption (PBKDF2 + Fernet); photos stored encrypted under `data/`.

---

## Quick start

1. **Clone and venv**

```bash
cd pixel
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

2. **Optional: face recognition** (recommended; Python 3.11/3.12 works best)

```bash
pip install -r requirements-face.txt
```

3. **API keys** (for natural language search)

Copy `.env.example` to `.env` and set at least one:

- **Groq** (recommended): [console.groq.com](https://console.groq.com/) → `GROQ_API_KEY=...`
- **Hugging Face**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → `HF_TOKEN=...`

Load env before running (or use a tool like `python-dotenv`):

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your_key"

# Or create .env and use: pip install python-dotenv then load in app if desired
```

4. **Run**

```bash
streamlit run streamlit_app.py
```

Open the URL (e.g. http://localhost:8501), set a **library passphrase** (used to encrypt photos), then:

- **Library** — Upload photos; run “Cluster unknown faces”; explore grid, timeline/events, quality & duplicates.
- **People** — Rename clustered people.
- **Search** — Use “Quick filter: photos of person” or type natural language (e.g. “photos with Arjun at the beach”).
- **Share** — Create package (select photos → token + .share file); import with token + .share file.
- **Settings** — Paths and delete cached thumbnails.

---

## Tech stack

| Layer        | Stack |
|-------------|--------|
| Frontend    | **Streamlit** (Python) |
| Backend     | SQLite, in-process services (library, search, sharing, auth) |
| Face AI     | **facenet-pytorch** (MTCNN + InceptionResnetV1), optional |
| NL search   | **Groq** (preferred) or **Hugging Face** Inference (LLM) |
| Vision (opt)| **Hugging Face** (image caption + classification) |
| Security    | **cryptography** (PBKDF2, Fernet), encrypted photo storage |

---

## Configuration

| Variable           | Purpose |
|--------------------|---------|
| `GROQ_API_KEY`     | Groq API key (NL search; preferred). |
| `HF_TOKEN`         | Hugging Face token (NL search fallback; optional caption/classification). |
| `GROQ_MODEL`       | Optional; default `llama-3.1-70b-versatile`. |
| `HF_MODEL`         | Optional; default `HuggingFaceH4/zephyr-7b-beta`. |
| `HF_CAPTION_MODEL` | Optional; default `Salesforce/blip-image-captioning-base`. |
| `HF_VISION_MODEL`  | Optional; default `google/vit-base-patch16-224`. |

---

## Notes

- First run with face support downloads model weights via `facenet-pytorch`.
- Photos are encrypted under `data/`; thumbnails are cached unencrypted (can be deleted in Settings).
- Without Groq/HF keys, search falls back to keyword-only (no natural language).
