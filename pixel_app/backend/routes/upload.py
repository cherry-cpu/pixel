import os
from flask import Blueprint, request, jsonify
from pixel_app.core.app_state import get_app

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "uploads" # Keeping your provided target folder for raw files

@upload_bp.route("/api/upload", methods=["POST"])
def upload_photo():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # 1. Save locally exactly as requested
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # 2. Integrate with the Drishyamitra Core (AI Background Tags + Face Embeddings)
    try:
        # Read the saved bytes to ingest into the project's encrypted DB
        with open(path, "rb") as f:
            file_bytes = f.read()
            
        app_core = get_app()
        # Ensure our SQLite database tables are fully ready
        app_core.auth.ensure_initialized()
        
        # This function processes the image through MTCNN and Groq 90B Vision!
        success, msg = app_core.library.ingest(file.filename, file_bytes)

        if not success:
            return jsonify({"error": msg}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to process image AI: {str(e)}"}), 500

    # 3. Return your requested success JSON output
    return jsonify({
        "message": "Photo uploaded and processed with AI Tags and Embeddings!",
        "filename": file.filename
    })
