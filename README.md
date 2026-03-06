## AI Photo Manager (Streamlit)

This project is an AI-powered photo management system built entirely with Python and Streamlit (frontend + backend).

### Features
- **Face-aware organization**: Groups and tags photos by people using image embeddings.
- **Smart photo organization**: Albums by people, time, and custom tags.
- **Natural language search**: Search photos using free-text queries (e.g. "me and John at the beach last summer").
- **Automated sharing**: Generate shareable links/albums from selected photos.
- **Secure storage**: Photos and metadata stored locally; API tokens kept out of source code.

### Tech Stack
- **Frontend & Backend**: Python + Streamlit
- **Computer Vision / Embeddings**: HuggingFace Inference API (CLIP-based models)
- **Storage**: Local folders + SQLite (via SQLAlchemy)

### Setup
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure HuggingFace token**
   - Create a HuggingFace account and generate an access token with Inference API permissions.
   - Set the token in an environment variable before running Streamlit:
     ```bash
     set HF_API_TOKEN=your_hf_token_here   # Windows (cmd)
     $env:HF_API_TOKEN="your_hf_token_here" # Windows (PowerShell)
     ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

### Notes
- No secrets are committed to the repository; tokens are loaded from environment variables or `st.secrets`.
- This code is structured so that the backend logic (storage + AI calls) lives in `backend/` and the Streamlit UI is in `app.py`.

