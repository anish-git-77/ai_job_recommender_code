"""
Flask Web App – AI Job Recommendation System
Run:  python app.py
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from recommender import JobRecommender, extract_resume_text, extract_skills_from_text

# ─────────────────────────────────────────────
# App configuration
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]   = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB max upload
ALLOWED_EXTENSIONS = {"pdf", "txt"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the AI engine once at startup (model download happens on first run)
print("=" * 55)
print("  Starting AI Job Recommendation System")
print("=" * 55)
recommender = JobRecommender()
print("=" * 55)
print("  Server ready → http://127.0.0.1:5000")
print("=" * 55)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Landing / home page."""
    return render_template("index.html")


# ── Upload resume (PDF / TXT) ─────────────────

@app.route("/upload", methods=["POST"])
def upload_resume():
    """Handle file upload and return job recommendations."""
    top_k = int(request.form.get("top_k", 5))

    if "resume" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF and TXT files are supported"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Extract text & get profile
    resume_text = extract_resume_text(save_path)
    if not resume_text:
        return jsonify({"error": "Could not extract text from the file. Make sure it is a readable PDF or TXT."}), 400

    profile      = recommender.get_resume_profile(resume_text)
    jobs         = recommender.recommend_from_file(save_path, top_k=top_k)

    # Clean up uploaded file
    os.remove(save_path)

    return jsonify({
        "input_type":   "file",
        "profile":      profile,
        "jobs":         jobs,
        "resume_text":  resume_text[:500] + "…" if len(resume_text) > 500 else resume_text,
    })


# ── Text / skills input ───────────────────────

@app.route("/recommend-text", methods=["POST"])
def recommend_text():
    """Handle plain-text resume / skills input."""
    data      = request.get_json(silent=True) or {}
    user_text = data.get("text", "").strip()
    top_k     = int(data.get("top_k", 5))

    if not user_text:
        return jsonify({"error": "Please provide some text describing your skills or experience"}), 400

    profile = recommender.get_resume_profile(user_text)
    jobs    = recommender.recommend_from_text_input(user_text, top_k=top_k)

    return jsonify({
        "input_type": "text",
        "profile":    profile,
        "jobs":       jobs,
    })


# ── All jobs listing ──────────────────────────

@app.route("/jobs")
def list_jobs():
    """Return all available jobs as JSON (for the jobs browser page)."""
    jobs = recommender.jobs_df.to_dict(orient="records")
    # Remove the combined_text field (internal use only)
    for j in jobs:
        j.pop("combined_text", None)
    return jsonify(jobs)


# ── Health check ──────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": recommender.MODEL_NAME})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
