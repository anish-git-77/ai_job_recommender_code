"""
AI Job Recommendation Engine
Uses SentenceTransformers + FAISS for semantic job matching
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────
# 1. RESUME TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF resume."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"[ERROR] Could not read PDF: {e}")
    return text.strip()


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a plain .txt resume."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[ERROR] Could not read TXT: {e}")
        return ""


def extract_resume_text(file_path: str) -> str:
    """Auto-detect file type and extract resume text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".text"]:
        return extract_text_from_txt(file_path)
    else:
        return ""


# ─────────────────────────────────────────────
# 2. SIMPLE SKILL & KEYWORD EXTRACTOR
# ─────────────────────────────────────────────

COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "swift", "kotlin",
    "react", "angular", "vue", "node.js", "flask", "django", "spring",
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "git", "linux", "rest api", "graphql", "microservices",
    "spark", "kafka", "airflow", "tableau", "power bi",
    "html", "css", "agile", "scrum", "devops", "ci/cd",
    "blockchain", "solidity", "web3", "ethereum",
    "data analysis", "statistics", "excel", "r",
]

def extract_skills_from_text(text: str) -> list:
    """Find matching skills from the resume text."""
    text_lower = text.lower()
    found = [skill for skill in COMMON_SKILLS if skill in text_lower]
    return list(set(found))


def extract_experience_years(text: str) -> int:
    """Try to extract years of experience from resume text."""
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
        r"experience[:\s]+(\d+)\+?\s*years?",
        r"(\d+)\s*yrs?\s*(?:of\s*)?(?:work\s*)?experience",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return 0  # unknown


# ─────────────────────────────────────────────
# 3. JOB DATA LOADER
# ─────────────────────────────────────────────

def load_jobs(csv_path: str) -> pd.DataFrame:
    """Load jobs from CSV file."""
    df = pd.read_csv(csv_path)
    # Create a rich combined text for each job (used for embedding)
    df["combined_text"] = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        "Required skills: " + df["skills"].fillna("") + ". " +
        "Level: " + df["experience_level"].fillna("")
    )
    return df


# ─────────────────────────────────────────────
# 4. EMBEDDING ENGINE (SentenceTransformers)
# ─────────────────────────────────────────────

class JobRecommender:
    """
    Core AI recommendation engine.
    - Loads a SentenceTransformer model
    - Encodes job descriptions into vectors
    - Uses FAISS for fast nearest-neighbour search
    """

    MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, lightweight, good quality
    INDEX_PATH = "data/faiss_index.pkl"
    JOBS_PATH  = "data/jobs.csv"

    def __init__(self):
        print("[INFO] Loading SentenceTransformer model…")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.jobs_df = None
        self.index = None
        self.job_embeddings = None
        self._build_or_load_index()

    # ── Index management ──────────────────────

    def _build_or_load_index(self):
        """Build FAISS index from job CSV, or load cached version."""
        self.jobs_df = load_jobs(self.JOBS_PATH)

        if os.path.exists(self.INDEX_PATH):
            print("[INFO] Loading cached FAISS index…")
            with open(self.INDEX_PATH, "rb") as f:
                saved = pickle.load(f)
            self.index = saved["index"]
            self.job_embeddings = saved["embeddings"]
            print("[INFO] Index loaded ✓")
        else:
            print("[INFO] Building FAISS index (first run – may take a minute)…")
            self._build_index()

    def _build_index(self):
        """Encode all job descriptions and store in FAISS."""
        texts = self.jobs_df["combined_text"].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = normalize(embeddings)          # L2-normalize for cosine similarity

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)          # Inner-product = cosine on normalized vecs
        self.index.add(embeddings.astype("float32"))
        self.job_embeddings = embeddings

        # Cache to disk
        os.makedirs("data", exist_ok=True)
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump({"index": self.index, "embeddings": self.job_embeddings}, f)
        print("[INFO] FAISS index built and cached ✓")

    def rebuild_index(self):
        """Force rebuild (call after updating jobs.csv)."""
        if os.path.exists(self.INDEX_PATH):
            os.remove(self.INDEX_PATH)
        self._build_index()

    # ── Recommendation core ───────────────────

    def recommend(self, resume_text: str, top_k: int = 5) -> list[dict]:
        """
        Given raw resume text, return top_k matching jobs.
        Returns a list of dicts with job info + similarity score.
        """
        if not resume_text.strip():
            return []

        # Encode resume
        resume_vec = self.model.encode([resume_text], convert_to_numpy=True)
        resume_vec = normalize(resume_vec).astype("float32")

        # FAISS search
        scores, indices = self.index.search(resume_vec, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            row = self.jobs_df.iloc[idx]
            # Extract matched skills
            resume_skills = set(extract_skills_from_text(resume_text))
            job_skills    = set(s.strip().lower() for s in str(row["skills"]).split(","))
            matched = list(resume_skills & job_skills)

            results.append({
                "rank":             rank,
                "job_id":           int(row["job_id"]),
                "title":            row["title"],
                "company":          row["company"],
                "location":         row["location"],
                "description":      row["description"],
                "skills":           row["skills"],
                "experience_level": row["experience_level"],
                "salary_range":     row["salary_range"],
                "match_score":      round(float(score) * 100, 1),   # % similarity
                "matched_skills":   matched,
                "skill_match_pct":  round(len(matched) / max(len(job_skills), 1) * 100, 1),
            })

        return results

    def recommend_from_text_input(self, user_text: str, top_k: int = 5) -> list[dict]:
        """Recommend jobs from a manually typed skills/bio paragraph."""
        return self.recommend(user_text, top_k)

    def recommend_from_file(self, file_path: str, top_k: int = 5) -> list[dict]:
        """Recommend jobs from an uploaded resume file (PDF or TXT)."""
        text = extract_resume_text(file_path)
        if not text:
            return []
        return self.recommend(text, top_k)

    # ── Analytics helpers ─────────────────────

    def get_resume_profile(self, resume_text: str) -> dict:
        """Return a quick profile summary of the resume."""
        skills   = extract_skills_from_text(resume_text)
        exp_yrs  = extract_experience_years(resume_text)
        word_cnt = len(resume_text.split())
        return {
            "detected_skills":  skills,
            "experience_years": exp_yrs,
            "word_count":       word_cnt,
        }
