
import io
import re
import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

import fitz
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION AND STORAGE
# ============================================================

st.set_page_config(
    page_title="HireNexa | Recruitment Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_resume_screening_model.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

STORAGE_DIR = BASE_DIR / "HireNexa_Storage"
RESUMES_DIR = STORAGE_DIR / "resumes"
SELECTED_DIR = STORAGE_DIR / "selected_candidates"
REPORTS_DIR = STORAGE_DIR / "reports"
JD_DIR = STORAGE_DIR / "job_descriptions"
DB_PATH = STORAGE_DIR / "hirenexa.db"

for folder in [STORAGE_DIR, RESUMES_DIR, SELECTED_DIR, REPORTS_DIR, JD_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATABASE
# ============================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                domain TEXT,
                description TEXT,
                min_experience REAL,
                max_experience REAL,
                created_at TEXT,
                candidate_count INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                candidate_name TEXT,
                resume_filename TEXT,
                resume_path TEXT,
                selected_resume_path TEXT,
                prediction TEXT,
                final_score REAL,
                skill_score REAL,
                similarity_score REAL,
                experience_score REAL,
                education_score REAL,
                experience REAL,
                education TEXT,
                recruiter_decision TEXT DEFAULT 'Not Reviewed',
                recruiter_notes TEXT DEFAULT '',
                result_json TEXT,
                created_at TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
        """)

        conn.commit()


initialize_database()


def save_job(job_id, title, domain, description, min_exp, max_exp, count):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO jobs (
                job_id, title, domain, description,
                min_experience, max_experience,
                created_at, candidate_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, title, domain, description,
            min_exp, max_exp,
            datetime.now().isoformat(timespec="seconds"),
            count
        ))
        conn.commit()


def save_candidate(job_id, result, candidate_name, resume_path):
    candidate_id = str(uuid.uuid4())

    # Store the structured result, not the full resume text.
    stored_result = {
        key: value for key, value in result.items()
        if key != "resume_text"
    }

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO candidates (
                candidate_id, job_id, candidate_name,
                resume_filename, resume_path, prediction,
                final_score, skill_score, similarity_score,
                experience_score, education_score,
                experience, education, result_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate_id,
            job_id,
            candidate_name,
            Path(candidate_name).name,
            str(resume_path),
            str(result.get("prediction", "")),
            float(result.get("final_score", 0)),
            float(result.get("skill_score", 0)),
            float(result.get("similarity_score", 0)),
            float(result.get("experience_score", 0)),
            float(result.get("education_score", 0)),
            None if pd.isna(result.get("experience", np.nan))
                 else float(result["experience"]),
            str(result.get("education", "")),
            json.dumps(stored_result, default=str),
            datetime.now().isoformat(timespec="seconds")
        ))
        conn.commit()


def get_jobs():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_candidates(job_id=None):
    query = """
        SELECT c.*, j.title AS job_title
        FROM candidates c
        LEFT JOIN jobs j ON c.job_id = j.job_id
    """
    params = ()

    if job_id:
        query += " WHERE c.job_id = ?"
        params = (job_id,)

    query += " ORDER BY c.created_at DESC"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_all_candidates():
    return get_candidates()


def update_candidate_decision(candidate_id, decision, notes):
    selected_path = None

    with get_connection() as conn:
        row = conn.execute("""
            SELECT resume_path, selected_resume_path, job_id
            FROM candidates
            WHERE candidate_id = ?
        """, (candidate_id,)).fetchone()

        if row is None:
            return

        original_path = Path(row["resume_path"])
        old_selected = row["selected_resume_path"]

        if decision == "Shortlisted":
            if not original_path.exists():
                raise FileNotFoundError("Original resume file was not found.")

            destination_folder = SELECTED_DIR / row["job_id"]
            destination_folder.mkdir(parents=True, exist_ok=True)

            destination = destination_folder / original_path.name
            shutil.copy2(original_path, destination)
            selected_path = str(destination)

        else:
            if old_selected:
                old_path = Path(old_selected)
                if old_path.exists():
                    old_path.unlink()

        conn.execute("""
            UPDATE candidates
            SET recruiter_decision = ?,
                recruiter_notes = ?,
                selected_resume_path = ?
            WHERE candidate_id = ?
        """, (decision, notes, selected_path, candidate_id))

        conn.commit()


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():
    required = [MODEL_PATH, ENCODER_PATH, TFIDF_PATH]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(TFIDF_PATH)

    return model, encoder, vectorizer


try:
    model, label_encoder, tfidf = load_models()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.info(
        "Check that all three trained model files are inside the models folder."
    )
    st.stop()


# ============================================================
# 4. NLP INITIALIZATION
# ============================================================

@st.cache_resource
def initialize_nlp():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)

    words = set(stopwords.words("english"))
    words -= {"no", "nor", "not"}

    return words, WordNetLemmatizer()


try:
    stop_words, lemmatizer = initialize_nlp()
except Exception as error:
    st.error(f"NLP initialization failed: {error}")
    st.stop()


# ============================================================
# 5. SKILL DICTIONARY
# ============================================================

SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript",
    "r", "sql", "mysql", "postgresql", "mongodb", "excel",
    "power bi", "tableau", "machine learning", "deep learning",
    "artificial intelligence", "nlp", "natural language processing",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
    "numpy", "matplotlib", "seaborn", "data analysis", "data science",
    "data visualization", "statistics", "aws", "azure", "google cloud",
    "docker", "kubernetes", "git", "github", "flask", "django",
    "fastapi", "streamlit", "langchain", "langgraph",
    "generative ai", "rag", "llm", "node.js", "html", "css",
    "react", "angular", "communication", "project management",
    "agile", "scrum"
]

SKILL_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(
        re.escape(skill)
        for skill in sorted(SKILLS, key=len, reverse=True)
    )
    + r")(?![a-z0-9])",
    re.IGNORECASE
)


# ============================================================
# 6. RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )
        text = "\n".join(page.get_text() for page in document)
        document.close()
        return text

    if filename.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        content = [p.text for p in document.paragraphs]

        for table in document.tables:
            for row in table.rows:
                content.extend(cell.text for cell in row.cells)

        return "\n".join(content)

    raise ValueError("Only PDF and DOCX files are supported.")


def safe_filename(filename):
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:150] or "resume"


# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " ",
        text
    )
    text = re.sub(r"\+?\d[\d\s()-]{8,}\d", " ", text)

    tokens = re.findall(
        r"(?<!\w)\.net\b|c\+\+|c#|"
        r"[a-z0-9]+(?:[.#][a-z0-9]+)*",
        text
    )

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(
        lemmatizer.lemmatize(word) for word in tokens
    )


# ============================================================
# 8. INFORMATION EXTRACTION
# ============================================================

def extract_skills(text):
    return sorted(set(
        match.lower() for match in SKILL_PATTERN.findall(str(text))
    ))


def extract_education(text):
    text = str(text).lower()

    patterns = {
        "PhD": r"\b(ph\.?d|doctorate)\b",
        "Master": r"\b(master'?s|m\.?tech|m\.?e|mba|mca|m\.?sc)\b",
        "Bachelor": r"\b(bachelor|b\.?tech|b\.?e|bca|b\.?sc|undergraduate)\b",
        "Diploma": r"\bdiploma\b"
    }

    for level, pattern in patterns.items():
        if re.search(pattern, text):
            return level

    return "Not Mentioned"


EDUCATION_RANK = {
    "Not Mentioned": 0,
    "Diploma": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4
}


def extract_experience(text):
    """
    Extract candidate experience from resume wording.
    This is an estimate based on explicit year statements.
    """
    text = str(text).lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
        r"(?:\s+of)?\s+(?:experience|work|professional)",

        r"(?:experience|worked|working)\s+(?:for\s+)?"
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
    ]

    values = []

    for pattern in patterns:
        for value in re.findall(pattern, text):
            try:
                values.append(float(value))
            except ValueError:
                pass

    return max(values) if values else np.nan


def extract_required_experience_range(text):
    """
    Supports:
      0-4 years
      0 to 4 years
      2–5 yrs
      3+ years
      at least 2 years
      minimum 2 years
      5 years of experience

    Returns (minimum, maximum).
    Maximum is None for open-ended requirements.
    """
    text = str(text).lower()

    range_pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:-|–|—|to)\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:years?|yrs?)"
    )

    matches = re.findall(range_pattern, text)

    if matches:
        low, high = max(
            ((float(a), float(b)) for a, b in matches),
            key=lambda pair: pair[0]
        )
        return min(low, high), max(low, high)

    plus_pattern = (
        r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)"
    )
    plus_matches = re.findall(plus_pattern, text)

    if plus_matches:
        return max(map(float, plus_matches)), None

    minimum_pattern = (
        r"(?:at least|minimum|min\.?)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
    )
    minimum_matches = re.findall(minimum_pattern, text)

    if minimum_matches:
        return max(map(float, minimum_matches)), None

    single_pattern = (
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
        r"(?:\s+of)?\s+(?:experience|work|professional)"
    )
    single_matches = re.findall(single_pattern, text)

    if single_matches:
        value = max(map(float, single_matches))
        return value, None

    return None, None


def extract_certifications(text):
    known = [
        "aws certified",
        "azure certified",
        "google professional",
        "pmp",
        "cissp",
        "comptia",
        "tensorflow certificate",
        "google data analytics",
        "microsoft certified"
    ]

    text = str(text).lower()
    return [item for item in known if item in text]


def infer_domain(text):
    text = str(text).lower()

    domains = {
        "data": [
            "data analyst", "data science",
            "machine learning", "data engineer"
        ],
        "software": [
            "software developer", "web developer",
            "backend", "frontend", "full stack"
        ],
        "finance": [
            "accountant", "financial analyst", "finance"
        ],
        "marketing": [
            "marketing", "seo", "social media"
        ],
        "hr": [
            "human resources", "recruitment",
            "talent acquisition"
        ]
    }

    for domain, keywords in domains.items():
        if any(keyword in text for keyword in keywords):
            return domain

    return "Unknown"


# ============================================================
# 9. EXPERIENCE MATCHING
# ============================================================

def calculate_experience_score(candidate_exp, required_min):
    if required_min is None or required_min <= 0:
        return 100.0

    if pd.isna(candidate_exp):
        return 0.0

    if candidate_exp >= required_min:
        return 100.0

    return round(
        max(0, candidate_exp / required_min * 100),
        2
    )


# ============================================================
# 10. MACHINE LEARNING SCREENING
# ============================================================

def screen_candidate(
    resume_text,
    jd,
    job_domain,
    required_min=None,
    required_max=None
):
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(jd)

    if not resume_clean or not jd_clean:
        raise ValueError("Resume or job description has no readable text.")

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd)

    matched = sorted(set(resume_skills) & set(jd_skills))
    missing = sorted(set(jd_skills) - set(resume_skills))

    skill_score = (
        len(matched) / len(jd_skills) * 100
        if jd_skills else 0.0
    )

    resume_vector = tfidf.transform([resume_clean])
    jd_vector = tfidf.transform([jd_clean])

    similarity = float(
        cosine_similarity(resume_vector, jd_vector)[0][0]
    )
    similarity_score = similarity * 100

    candidate_experience = extract_experience(resume_text)

    if required_min is None:
        parsed_min, parsed_max = extract_required_experience_range(jd)
        required_min = parsed_min
        required_max = parsed_max

    experience_score = calculate_experience_score(
        candidate_experience,
        required_min
    )

    education = extract_education(resume_text)
    required_education = extract_education(jd)

    if required_education == "Not Mentioned":
        education_score = 100.0
    else:
        education_score = (
            100.0
            if EDUCATION_RANK[education] >= EDUCATION_RANK[required_education]
            else 0.0
        )

    final_score = (
        0.50 * skill_score
        + 0.30 * similarity_score
        + 0.15 * experience_score
        + 0.05 * education_score
    )

    certifications = extract_certifications(resume_text)

    features = {
        "resume_skill_count": len(resume_skills),
        "jd_skill_count": len(jd_skills),
        "education_score": EDUCATION_RANK[education],
        "experience_years": (
            0.0 if pd.isna(candidate_experience)
            else float(candidate_experience)
        ),
        "certification_count": len(certifications),
        "resume_word_count": len(resume_text.split()),
        "jd_word_count": len(jd.split()),
        "matched_skill_count": len(matched),
        "skill_match_percentage": skill_score,
        "skill_jaccard_similarity": (
            len(matched) / len(set(resume_skills) | set(jd_skills))
            if set(resume_skills) | set(jd_skills) else 0.0
        ),
        "tfidf_similarity": similarity,
        "education_level": education,
        "resume_domain": infer_domain(resume_text),
        "jd_domain": str(job_domain).lower()
    }

    # Keep the trained model's original feature order.
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = list(features.keys())

    missing_features = [
        name for name in feature_names if name not in features
    ]
    if missing_features:
        raise ValueError(
            "The trained model expects features not created by this app: "
            + ", ".join(missing_features)
        )

    X = pd.DataFrame(
        [{name: features[name] for name in feature_names}]
    )

    prediction = model.predict(X)[0]

    # Support models trained with encoded integer target labels.
    try:
        prediction_label = label_encoder.inverse_transform(
            [int(prediction)]
        )[0]
    except (ValueError, TypeError):
        prediction_label = prediction

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", range(len(probability_values)))

        try:
            decoded_classes = label_encoder.inverse_transform(
                np.asarray(classes, dtype=int)
            )
        except Exception:
            decoded_classes = classes

        probabilities = {
            str(label): float(probability)
            for label, probability in zip(
                decoded_classes, probability_values
            )
        }

    return {
        "prediction": str(prediction_label),
        "probabilities": probabilities,
        "final_score": round(float(final_score), 2),
        "skill_score": round(float(skill_score), 2),
        "similarity_score": round(float(similarity_score), 2),
        "experience_score": round(float(experience_score), 2),
        "education_score": round(float(education_score), 2),
        "experience": (
            None if pd.isna(candidate_experience)
            else float(candidate_experience)
        ),
        "education": education,
        "resume_skills": resume_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "certifications": certifications,
        "resume_text": resume_text,
        "required_min_experience": required_min,
        "required_max_experience": required_max
    }


# ============================================================
# 11. REPORT GENERATION
# ============================================================

def create_report_dataframe(rows):
    records = []

    for row in rows:
        item = dict(row)

        try:
            result = json.loads(item.get("result_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            result = {}

        records.append({
            "Candidate": item.get("candidate_name"),
            "Job Title": item.get("job_title"),
            "AI Classification": item.get("prediction"),
            "Match Score": item.get("final_score"),
            "Skill Score": item.get("skill_score"),
            "Similarity Score": item.get("similarity_score"),
            "Experience": item.get("experience"),
            "Education": item.get("education"),
            "Matched Skills": ", ".join(result.get("matched_skills", [])),
            "Missing Skills": ", ".join(result.get("missing_skills", [])),
            "Recruiter Decision": item.get("recruiter_decision"),
            "Recruiter Notes": item.get("recruiter_notes"),
            "Screened At": item.get("created_at")
        })

    return pd.DataFrame(records)


def save_report(job_id=None):
    rows = get_candidates(job_id)

    if not rows:
        return None

    report = create_report_dataframe(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"HireNexa_Report_{job_id or 'All'}_{timestamp}.csv"
    report_path = REPORTS_DIR / filename

    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    return report_path


# ============================================================
# 12. UI HELPERS AND STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F5F7FB;
    color: #24344D;
    font-family: Inter, sans-serif;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 2.2rem 3rem;
}

h1, h2, h3, h4 {
    color: #142B4A !important;
}

[data-testid="stSidebar"] {
    background: #142B4A;
}

[data-testid="stSidebar"] * {
    color: #F7FAFF !important;
}

.brand-logo {
    font-size: 28px;
    font-weight: 800;
    color: white;
}

.brand-caption {
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #B6C7DD;
    margin-top: 5px;
}

.page-heading {
    font-size: 29px;
    font-weight: 800;
    color: #142B4A;
    margin-bottom: 5px;
}

.page-description {
    font-size: 14px;
    color: #718096;
    margin-bottom: 24px;
}

.section-heading {
    font-size: 19px;
    font-weight: 700;
    color: #142B4A;
    margin: 20px 0 12px;
}

.metric-card {
    background: white;
    border: 1px solid #E4EAF2;
    border-radius: 13px;
    padding: 18px;
    min-height: 115px;
}

.metric-label {
    color: #718096;
    font-size: 11px;
    font-weight: 700;
}

.metric-value {
    color: #142B4A;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #8290A4;
    font-size: 11px;
    margin-top: 5px;
}

.skill-pill, .missing-pill {
    display: inline-block;
    padding: 6px 10px;
    margin: 3px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}

.skill-pill {
    background: #EAF2FF;
    color: #2855BD;
}

.missing-pill {
    background: #FFF0EF;
    color: #B93B3B;
}

.candidate-card {
    background: white;
    border: 1px solid #E4EAF2;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #8794A7;
    font-size: 11px;
    padding: 25px 0 5px;
}

.stButton > button {
    border-radius: 8px;
    min-height: 40px;
    font-weight: 600;
}

div[data-testid="stExpander"] {
    border: 1px solid #E4EAF2;
    border-radius: 10px;
    background: white;
}
</style>
""", unsafe_allow_html=True)


def page_header(title, description):
    st.markdown(
        f'<div class="page-heading">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="page-description">{description}</div>',
        unsafe_allow_html=True
    )


def section_title(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def metric_card(col, label, value, note):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)


def render_skills(skills, missing=False):
    if not skills:
        st.caption("No skills identified.")
        return

    css_class = "missing-pill" if missing else "skill-pill"
    html = "".join(
        f'<span class="{css_class}">{skill}</span>'
        for skill in skills
    )
    st.markdown(html, unsafe_allow_html=True)


def get_result_for_candidate(row):
    try:
        return json.loads(row.get("result_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand-logo">✦ HireNexa</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-caption">RECRUITMENT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "New Screening",
            "Candidate Explorer",
            "Shortlisted Candidates",
            "Compare Candidates",
            "Recruitment Pipeline",
            "Reports & History",
            "Storage Overview",
            "About HireNexa"
        ],
        key="main_navigation"
    )

    st.divider()

    st.markdown("**PLATFORM STATUS**")

    st.markdown("""
    <div class="status-card">
        <span class="status-icon">✓</span>
        <span>ML model loaded</span>
    </div>
    <div class="status-card">
        <span class="status-icon">✓</span>
        <span>NLP engine ready</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("HireNexa · Recruiter Workspace")


# ============================================================
# 14. DASHBOARD
# ============================================================

if page == "Dashboard":
    page_header(
        "Recruitment Dashboard",
        "Monitor screening activity, candidate decisions and recruitment performance."
    )

    jobs = get_jobs()
    candidates = get_all_candidates()

    total_jobs = len(jobs)
    total_candidates = len(candidates)

    shortlisted = sum(
        row["recruiter_decision"] == "Shortlisted"
        for row in candidates
    )
    on_hold = sum(
        row["recruiter_decision"] == "On Hold"
        for row in candidates
    )
    pending = sum(
        row["recruiter_decision"] == "Not Reviewed"
        for row in candidates
    )

    average_score = (
        np.mean([row["final_score"] or 0 for row in candidates])
        if candidates else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "TOTAL JOBS", total_jobs, "Job openings created")
    metric_card(c2, "TOTAL CANDIDATES", total_candidates, "Resumes screened")
    metric_card(c3, "SHORTLISTED", shortlisted, "Recruiter-selected candidates")
    metric_card(c4, "AVERAGE MATCH", f"{average_score:.1f}%", "Weighted match indicator")

    section_title("Recruitment Overview")

    left, right = st.columns(2)

    with left:
        if candidates:
            decision_df = pd.DataFrame([
                {"Decision": row["recruiter_decision"]}
                for row in candidates
            ])
            counts = decision_df["Decision"].value_counts().reset_index()
            counts.columns = ["Decision", "Candidates"]

            fig = px.pie(
                counts,
                names="Decision",
                values="Candidates",
                hole=0.55,
                title="Candidate Decision Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candidate records yet.")

    with right:
        if candidates:
            score_df = pd.DataFrame([
                {
                    "Candidate": row["candidate_name"],
                    "Match Score": row["final_score"] or 0
                }
                for row in candidates
            ]).nlargest(10, "Match Score")

            fig = px.bar(
                score_df.sort_values("Match Score"),
                x="Match Score",
                y="Candidate",
                orientation="h",
                title="Top 10 Candidate Match Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Screen candidates to view score analytics.")

    section_title("Recent Screening Activity")

    if candidates:
        recent_df = pd.DataFrame([
            {
                "Candidate": row["candidate_name"],
                "Job": row["job_title"],
                "Match Score": row["final_score"],
                "Decision": row["recruiter_decision"],
                "Date": row["created_at"]
            }
            for row in candidates[:10]
        ])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("Start a new screening session to populate your dashboard.")


# ============================================================
# 15. NEW SCREENING
# ============================================================

if page == "New Screening":
    page_header(
        "New Screening Session",
        "Create a job opening and upload multiple candidate resumes."
    )

    with st.form("new_screening_form"):
        section_title("01 · Job Information")

        job_title = st.text_input(
            "Job Title",
            placeholder="Example: Data Scientist"
        )

        col1, col2 = st.columns(2)

        with col1:
            job_domain = st.selectbox(
                "Department / Domain",
                ["data", "software", "finance", "marketing", "hr", "Unknown"]
            )

        with col2:
            use_manual_experience = st.checkbox(
                "Set experience range manually",
                value=False
            )

        exp1, exp2 = st.columns(2)

        with exp1:
            min_experience = st.number_input(
                "Minimum Experience (Years)",
                min_value=0.0,
                max_value=40.0,
                value=0.0,
                step=0.5,
                disabled=not use_manual_experience
            )

        with exp2:
            max_experience = st.number_input(
                "Maximum Experience (Years)",
                min_value=0.0,
                max_value=40.0,
                value=4.0,
                step=0.5,
                disabled=not use_manual_experience
            )

        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder=(
                "Enter responsibilities, required skills, "
                "education and experience requirements..."
            )
        )

        section_title("02 · Resume Upload")

        uploaded_files = st.file_uploader(
            "Upload candidate resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        submitted = st.form_submit_button(
            "Start AI Screening",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not job_title.strip():
            st.error("Please enter a job title.")
        elif not job_description.strip():
            st.error("Please enter a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
        elif (
            use_manual_experience
            and max_experience < min_experience
        ):
            st.error("Maximum experience cannot be lower than minimum experience.")
        else:
            if use_manual_experience:
                required_min = float(min_experience)
                required_max = float(max_experience)
            else:
                required_min, required_max = extract_required_experience_range(
                    job_description
                )

            job_id = str(uuid.uuid4())[:12]

            jd_path = JD_DIR / f"{job_id}_job_description.txt"
            jd_path.write_text(job_description, encoding="utf-8")

            progress = st.progress(0)
            status = st.empty()

            successful_results = []
            errors = []

            for index, uploaded_file in enumerate(uploaded_files):
                status.write(f"Processing {uploaded_file.name}...")

                try:
                    resume_text = extract_resume_text(uploaded_file)

                    if not resume_text.strip():
                        raise ValueError(
                            "No readable text found. Scanned PDFs may need OCR."
                        )

                    result = screen_candidate(
                        resume_text,
                        job_description,
                        job_domain,
                        required_min,
                        required_max
                    )

                    resume_folder = RESUMES_DIR / job_id
                    resume_folder.mkdir(parents=True, exist_ok=True)

                    unique_name = (
                        f"{uuid.uuid4().hex[:8]}_"
                        f"{safe_filename(uploaded_file.name)}"
                    )
                    resume_path = resume_folder / unique_name
                    resume_path.write_bytes(uploaded_file.getvalue())

                    save_candidate(
                        job_id,
                        result,
                        uploaded_file.name,
                        resume_path
                    )

                    successful_results.append(result)

                except Exception as error:
                    errors.append((uploaded_file.name, str(error)))

                progress.progress((index + 1) / len(uploaded_files))

            status.empty()

            if successful_results:
                save_job(
                    job_id,
                    job_title,
                    job_domain,
                    job_description,
                    required_min,
                    required_max,
                    len(successful_results)
                )

                report_path = save_report(job_id)

                st.success(
                    f"Screening completed: {len(successful_results)} "
                    f"resume(s) processed successfully."
                )

                st.info(f"Job ID: {job_id}")

                if required_min is not None:
                    st.write(
                        f"Experience requirement detected: "
                        f"{required_min} years minimum"
                        + (
                            f" to {required_max} years maximum"
                            if required_max is not None else "+"
                        )
                    )

                if report_path:
                    st.write(f"Initial report saved: `{report_path}`")

            if errors:
                with st.expander("Files that could not be processed"):
                    for filename, error in errors:
                        st.warning(f"{filename}: {error}")


# ============================================================
# 16. CANDIDATE EXPLORER
# ============================================================

if page == "Candidate Explorer":
    page_header(
        "Candidate Explorer",
        "Search candidate profiles, inspect screening results and manage recruiter decisions."
    )

    jobs = get_jobs()

    if not jobs:
        st.info("No screening sessions found. Create a new screening first.")
    else:
        job_options = {
            f"{job['title']} · {job['created_at'][:10]} · {job['job_id']}": job["job_id"]
            for job in jobs
        }

        selected_job_label = st.selectbox(
            "Select Job Opening",
            list(job_options.keys())
        )
        selected_job_id = job_options[selected_job_label]

        candidates = get_candidates(selected_job_id)

        if not candidates:
            st.info("No candidates are stored for this job.")
        else:
            section_title("Candidate Filters")

            f1, f2, f3 = st.columns(3)

            with f1:
                minimum_score = st.slider(
                    "Minimum Match Score",
                    0, 100, 0
                )

            with f2:
                decisions = st.multiselect(
                    "Recruiter Decision",
                    ["Not Reviewed", "Shortlisted", "On Hold", "Rejected"],
                    default=["Not Reviewed", "Shortlisted", "On Hold", "Rejected"]
                )

            with f3:
                search_text = st.text_input(
                    "Search Candidate",
                    placeholder="Candidate name or filename"
                )

            filtered = []

            for row in candidates:
                if (row["final_score"] or 0) < minimum_score:
                    continue
                if row["recruiter_decision"] not in decisions:
                    continue
                if search_text and search_text.lower() not in row["candidate_name"].lower():
                    continue
                filtered.append(row)

            st.caption(f"{len(filtered)} candidate(s) match your filters.")

            for row in filtered:
                result = get_result_for_candidate(row)

                with st.expander(
                    f"{row['candidate_name']}  ·  "
                    f"{(row['final_score'] or 0):.1f}% match  ·  "
                    f"{row['recruiter_decision']}"
                ):
                    left, right = st.columns([2, 1])

                    with left:
                        st.write(f"**AI Classification:** {row['prediction']}")
                        st.write(f"**Experience:** {row['experience'] if row['experience'] is not None else 'Not detected'}")
                        st.write(f"**Education:** {row['education']}")
                        st.write(f"**Match Score:** {row['final_score']:.1f}%")

                        st.write("**Matched Skills**")
                        render_skills(result.get("matched_skills", []))

                        st.write("**Missing Skills**")
                        render_skills(
                            result.get("missing_skills", []),
                            missing=True
                        )

                        st.write("**Certifications**")
                        st.write(", ".join(result.get("certifications", [])) or "Not detected")

                    with right:
                        st.metric("Skill Match", f"{row['skill_score']:.1f}%")
                        st.metric("Text Similarity", f"{row['similarity_score']:.1f}%")
                        st.metric("Experience Score", f"{row['experience_score']:.1f}%")
                        st.metric("Education Score", f"{row['education_score']:.1f}%")

                    resume_path = Path(row["resume_path"])

                    if resume_path.exists():
                        st.download_button(
                            "Download Original Resume",
                            data=resume_path.read_bytes(),
                            file_name=row["resume_filename"],
                            mime=(
                                "application/pdf"
                                if resume_path.suffix.lower() == ".pdf"
                                else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            ),
                            key=f"download_original_{row['candidate_id']}"
                        )
                    else:
                        st.warning("Original resume file is missing.")

                    st.divider()
                    st.write("**Recruiter Review**")

                    decision_key = f"decision_{row['candidate_id']}"
                    notes_key = f"notes_{row['candidate_id']}"

                    new_decision = st.selectbox(
                        "Recruiter Decision",
                        ["Not Reviewed", "Shortlisted", "On Hold", "Rejected"],
                        index=[
                            "Not Reviewed", "Shortlisted", "On Hold", "Rejected"
                        ].index(row["recruiter_decision"]),
                        key=decision_key
                    )

                    new_notes = st.text_area(
                        "Recruiter Notes",
                        value=row["recruiter_notes"] or "",
                        key=notes_key
                    )

                    if st.button(
                        "Save Recruiter Review",
                        key=f"save_review_{row['candidate_id']}"
                    ):
                        try:
                            update_candidate_decision(
                                row["candidate_id"],
                                new_decision,
                                new_notes
                            )
                            st.success("Recruiter decision and notes saved.")
                            st.rerun()
                        except Exception as error:
                            st.error(f"Could not save review: {error}")


# ============================================================
# 17. SHORTLISTED CANDIDATES
# ============================================================

if page == "Shortlisted Candidates":
    page_header(
        "Shortlisted Candidates",
        "Access resumes selected by the recruiter."
    )

    shortlisted_rows = [
        row for row in get_all_candidates()
        if row["recruiter_decision"] == "Shortlisted"
    ]

    st.metric("Total Shortlisted Candidates", len(shortlisted_rows))

    if not shortlisted_rows:
        st.info("No candidates have been shortlisted yet.")
    else:
        shortlist_df = pd.DataFrame([
            {
                "Candidate": row["candidate_name"],
                "Job": row["job_title"],
                "Match Score": row["final_score"],
                "Experience": row["experience"],
                "Education": row["education"],
                "Selected At": row["created_at"]
            }
            for row in shortlisted_rows
        ])

        st.dataframe(shortlist_df, use_container_width=True, hide_index=True)

        for row in shortlisted_rows:
            with st.expander(row["candidate_name"]):
                selected_path = row["selected_resume_path"]

                if selected_path and Path(selected_path).exists():
                    path = Path(selected_path)
                    st.download_button(
                        "Download Shortlisted Resume",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"shortlist_download_{row['candidate_id']}"
                    )
                else:
                    st.warning("Shortlisted resume copy is not available.")


# ============================================================
# 18. COMPARE CANDIDATES
# ============================================================

if page == "Compare Candidates":
    page_header(
        "Compare Candidates",
        "Compare candidate screening indicators side by side."
    )

    candidates = get_all_candidates()

    if len(candidates) < 2:
        st.info("At least two screened candidates are required.")
    else:
        candidate_options = {
            f"{row['candidate_name']} · {row['job_title']} · {row['candidate_id'][:6]}": row
            for row in candidates
        }

        selected_labels = st.multiselect(
            "Select candidates to compare",
            list(candidate_options.keys()),
            max_selections=5
        )

        if len(selected_labels) >= 2:
            selected_rows = [
                candidate_options[label] for label in selected_labels
            ]

            comparison = pd.DataFrame([
                {
                    "Candidate": row["candidate_name"],
                    "Job": row["job_title"],
                    "AI Classification": row["prediction"],
                    "Match Score": row["final_score"],
                    "Skill Score": row["skill_score"],
                    "Similarity Score": row["similarity_score"],
                    "Experience Score": row["experience_score"],
                    "Education Score": row["education_score"],
                    "Experience": row["experience"],
                    "Decision": row["recruiter_decision"]
                }
                for row in selected_rows
            ])

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                comparison,
                x="Candidate",
                y=["Skill Score", "Similarity Score", "Experience Score", "Education Score"],
                barmode="group",
                title="Candidate Score Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 19. RECRUITMENT PIPELINE
# ============================================================

if page == "Recruitment Pipeline":
    page_header(
        "Recruitment Pipeline",
        "Track candidate progress through recruiter-managed stages."
    )

    candidates = get_all_candidates()

    statuses = [
        "Not Reviewed",
        "Shortlisted",
        "On Hold",
        "Rejected"
    ]

    if not candidates:
        st.info("No candidate activity available.")
    else:
        cols = st.columns(4)

        for col, status in zip(cols, statuses):
            count = sum(
                row["recruiter_decision"] == status
                for row in candidates
            )
            metric_card(col, status.upper(), count, "Candidates")

        section_title("Candidates by Recruitment Stage")

        for status in statuses:
            subset = [
                row for row in candidates
                if row["recruiter_decision"] == status
            ]

            with st.expander(f"{status} · {len(subset)} candidates"):
                if not subset:
                    st.caption("No candidates in this stage.")
                else:
                    for row in subset:
                        st.write(
                            f"**{row['candidate_name']}** · "
                            f"{row['job_title']} · "
                            f"{row['final_score']:.1f}% match"
                        )
                        if row["recruiter_notes"]:
                            st.caption(row["recruiter_notes"])


# ============================================================
# 20. REPORTS AND HISTORY
# ============================================================

if page == "Reports & History":
    page_header(
        "Reports & Screening History",
        "Review previous screening sessions and save updated recruitment reports."
    )

    jobs = get_jobs()

    if not jobs:
        st.info("No screening history available.")
    else:
        jobs_df = pd.DataFrame([
            {
                "Job ID": job["job_id"],
                "Job Title": job["title"],
                "Domain": job["domain"],
                "Candidates": job["candidate_count"],
                "Created At": job["created_at"]
            }
            for job in jobs
        ])

        st.dataframe(jobs_df, use_container_width=True, hide_index=True)

        selected_job = st.selectbox(
            "Choose a job for its report",
            ["All Jobs"] + [
                f"{job['title']} · {job['job_id']}" for job in jobs
            ]
        )

        job_id = None
        if selected_job != "All Jobs":
            job_id = selected_job.split(" · ")[-1]

        rows = get_candidates(job_id)
        report_df = create_report_dataframe(rows)

        if not report_df.empty:
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            csv_data = report_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Download CSV Report",
                data=csv_data,
                file_name="HireNexa_Recruitment_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

            if st.button("Save Report to Reports Folder"):
                report_path = save_report(job_id)
                if report_path:
                    st.success(f"Report saved: {report_path}")
        else:
            st.info("No candidate records for this selection.")

    section_title("Previously Saved Reports")

    report_files = sorted(
        REPORTS_DIR.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not report_files:
        st.caption("No saved reports yet.")
    else:
        for report_file in report_files:
            col1, col2 = st.columns([4, 1])

            with col1:
                st.write(f"📄 {report_file.name}")

            with col2:
                st.download_button(
                    "Download",
                    data=report_file.read_bytes(),
                    file_name=report_file.name,
                    mime="text/csv",
                    key=f"report_{report_file.name}"
                )


# ============================================================
# 21. STORAGE OVERVIEW
# ============================================================

if page == "Storage Overview":
    page_header(
        "Storage Overview",
        "Inspect where HireNexa stores candidate files, shortlisted resumes and reports."
    )

    jobs = get_jobs()
    candidates = get_all_candidates()

    original_files = list(RESUMES_DIR.rglob("*"))
    original_files = [p for p in original_files if p.is_file()]

    selected_files = list(SELECTED_DIR.rglob("*"))
    selected_files = [p for p in selected_files if p.is_file()]

    report_files = list(REPORTS_DIR.glob("*.csv"))

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "JOB OPENINGS", len(jobs), "Saved jobs")
    metric_card(c2, "ORIGINAL RESUMES", len(original_files), "Uploaded resumes")
    metric_card(c3, "SHORTLISTED COPIES", len(selected_files), "Selected resumes")
    metric_card(c4, "REPORTS", len(report_files), "Saved CSV reports")

    section_title("Storage Locations")

    st.code(
        f"""HireNexa_Storage/
├── hirenexa.db
├── resumes/
├── selected_candidates/
├── job_descriptions/
└── reports/

Database: {DB_PATH}
"""
    )

    st.caption(
        "These folders are created automatically beside your app.py file."
    )


# ============================================================
# 22. ABOUT
# ============================================================

if page == "About HireNexa":
    page_header(
        "About HireNexa",
        "AI-assisted recruitment intelligence for recruiter-led candidate evaluation."
    )

    st.markdown("""
    ### Our Mission

    HireNexa helps recruiters organize resume screening, identify relevant
    skills, compare candidate information against job requirements and
    maintain a structured recruitment pipeline.

    ### Core Features

    - Multiple PDF and DOCX resume uploads
    - Resume text extraction and NLP preprocessing
    - Technical skill extraction
    - Experience range extraction from job descriptions
    - TF-IDF similarity analysis
    - LightGBM model classification
    - Candidate score indicators
    - Recruiter decisions and notes
    - Separate shortlisted resume storage
    - Persistent SQLite database
    - CSV recruitment reports
    - Candidate comparison and pipeline analytics

    ### Technology Stack

    Python · Streamlit · Pandas · NumPy · Scikit-learn · LightGBM ·
    NLTK · TF-IDF · PyMuPDF · SQLite · Plotly

    ### Responsible Recruitment

    HireNexa is a decision-support tool. AI classification and match scores
    are indicators, not guarantees of job performance or hiring success.
    Recruiters must review candidate information and make final decisions.
    """)


# ============================================================
# 23. FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <b>HireNexa</b><br>
    AI-Powered Recruitment Intelligence<br>
    Built with Python · Machine Learning · NLP · Streamlit
</div>
""", unsafe_allow_html=True)
