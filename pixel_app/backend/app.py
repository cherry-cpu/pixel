from flask import Flask
from flask_cors import CORS

from pixel_app.backend.routes.upload import upload_bp

def create_app() -> Flask:
    app = Flask(__name__)
    
    # Allow local frontend to access these APIs
    CORS(app)
    
    # Register the user's requested blueprint
    app.register_blueprint(upload_bp)

    @app.route("/", methods=["GET"])
    def health_check():
        return {"status": "ok", "message": "Drishyamitra Flask Backend is Running"}

    return app

if __name__ == "__main__":
    app = create_app()
    print("Starting Flask API. Access the Upload API at: http://localhost:5000/api/upload")
    app.run(host="0.0.0.0", port=5000, debug=True)
