"""
Streamlit entry point for the Career Path Prediction & Guidance System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.exceptions import PredictorError
from src.recommender import get_recommendations
from src.validator import validate_student_input

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load Predictor at startup — cached so it is only loaded once per session.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_predictor():
    """Load and cache the Predictor.  Returns None if artifacts are missing."""
    from src.predictor import Predictor  # import here to keep top-level clean

    return Predictor()


predictor = None
try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        "⚠️ Model artifacts not found. "
        "Please run the training pipeline first (`python train_pipeline.py --data <path>`) "
        "to generate the required artifacts before using this application."
    )

# ---------------------------------------------------------------------------
# Option lists for multiselect widgets
# ---------------------------------------------------------------------------

TECHNICAL_SKILLS_OPTIONS: list[str] = [
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "SQL",
    "R",
    "MATLAB",
    "Machine Learning",
    "Deep Learning",
    "Data Analysis",
    "Data Visualization",
    "Cloud Computing (AWS/GCP/Azure)",
    "Linux/Unix",
    "Git",
    "Docker",
    "Kubernetes",
    "Networking",
    "Cybersecurity",
    "Database Management",
    "Web Development",
    "Mobile Development",
    "DevOps",
    "Embedded Systems",
    "Computer Vision",
    "Natural Language Processing",
]

INTERESTS_OPTIONS: list[str] = [
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science",
    "Software Development",
    "Web Development",
    "Mobile Development",
    "Cybersecurity",
    "Cloud Computing",
    "Networking",
    "Database Administration",
    "Business Intelligence",
    "UX / Product Design",
    "Research",
    "Finance & FinTech",
    "Healthcare Technology",
    "Game Development",
    "Robotics",
    "IoT",
    "Blockchain",
    "Project Management",
]

SOFT_SKILLS_OPTIONS: list[str] = [
    "Communication",
    "Teamwork",
    "Leadership",
    "Problem Solving",
    "Critical Thinking",
    "Time Management",
    "Creativity",
    "Adaptability",
    "Attention to Detail",
    "Conflict Resolution",
    "Presentation Skills",
    "Negotiation",
    "Empathy",
    "Decision Making",
]

# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------

st.title("🎓 Career Path Predictor")
st.markdown(
    "Enter your academic profile below and click **Predict Career** to receive "
    "a personalised career path recommendation along with curated course suggestions."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form(key="student_input_form"):
    st.subheader("Your Academic Profile")

    gpa = st.number_input(
        label="GPA / Overall Score",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        format="%.2f",
        help="Enter your cumulative GPA or overall academic score. "
             "Accepted range: 0.0 – 100.0 (use your institution's scale).",
        placeholder="e.g. 3.8 or 85.5",
    )

    technical_skills = st.multiselect(
        label="Technical Skills",
        options=TECHNICAL_SKILLS_OPTIONS,
        help="Select all technical skills you are proficient in. "
             "You can also type to search for a skill not listed.",
        placeholder="e.g. Python, SQL, Machine Learning …",
    )

    interests = st.multiselect(
        label="Areas of Interest",
        options=INTERESTS_OPTIONS,
        help="Select the professional or academic areas you are most passionate about.",
        placeholder="e.g. Data Science, Web Development …",
    )

    soft_skills = st.multiselect(
        label="Soft Skills (optional)",
        options=SOFT_SKILLS_OPTIONS,
        help="Select soft skills that describe your working style and interpersonal strengths.",
        placeholder="e.g. Communication, Leadership …",
    )

    extracurricular = st.text_input(
        label="Extracurricular Activities (optional)",
        help="Briefly describe clubs, hackathons, internships, or other activities you participate in.",
        placeholder="e.g. Robotics club, open-source contributor, student council …",
    )

    submitted = st.form_submit_button(label="Predict Career", type="primary")

# ---------------------------------------------------------------------------
# Form processing
# ---------------------------------------------------------------------------

if submitted:
    # Convert multiselect lists to comma-separated strings for the model
    raw_input = {
        "gpa": gpa,
        "technical_skills": ", ".join(technical_skills) if technical_skills else "",
        "interests": ", ".join(interests) if interests else "",
    }

    # --- Validation ---
    is_valid, errors = validate_student_input(raw_input)

    if not is_valid:
        for field_name, error_msg in errors.items():
            # Map internal field names to display-friendly labels
            display_labels = {
                "gpa": "GPA / Overall Score",
                "technical_skills": "Technical Skills",
                "interests": "Areas of Interest",
                "soft_skills": "Soft Skills",
                "extracurricular": "Extracurricular Activities",
            }
            label = display_labels.get(field_name, field_name.replace("_", " ").title())
            st.warning(f"**{label}**: {error_msg}")

    elif predictor is None:
        # Artifacts were missing at startup; prevent a confusing secondary error.
        st.error(
            "The predictor could not be loaded. "
            "Please run the training pipeline and restart the application."
        )

    else:
        # --- Prediction ---
        career_label: str | None = None
        confidence: float | None = None

        try:
            career_label, confidence = predictor.predict(raw_input)
        except PredictorError:
            st.error("An unexpected error occurred. Please try again.")

        if career_label is not None and confidence is not None:
            st.divider()
            st.subheader("Your Career Prediction")

            st.success(f"🏆 Recommended Career Path: **{career_label}**")

            st.metric(
                label="Confidence Score",
                value=f"{confidence * 100:.1f}%",
                help="How confident the model is in this prediction (0 – 100%).",
            )

            # --- Course recommendations ---
            courses = None
            try:
                courses = get_recommendations(career_label)
            except PredictorError:
                st.error("An unexpected error occurred. Please try again.")

            if courses:
                st.subheader("📚 Recommended Courses")
                st.markdown(
                    "Here are curated courses to help you build skills for your predicted career:"
                )
                for course in courses:
                    st.markdown(
                        f"- **[{course.title}]({course.url})** — *{course.provider}*"
                    )
