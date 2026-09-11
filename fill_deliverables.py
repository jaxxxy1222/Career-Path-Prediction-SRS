"""
Script to fill all Excel deliverable files for the
Career Path Prediction & Guidance System project.
Run: python fill_deliverables.py
"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import date

FOLDER = "src/Deliverables"

# ── helpers ──────────────────────────────────────────────────────────────────

def w(ws, row, col, value, bold=False, wrap=False, bg=None):
    cell = ws.cell(row=row, column=col, value=value)
    if bold:
        cell.font = Font(bold=True)
    if wrap:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    if bg:
        cell.fill = PatternFill("solid", fgColor=bg)
    return cell

def hdr(ws, row, col, value):
    w(ws, row, col, value, bold=True, bg="1F4E79")
    ws.cell(row=row, column=col).font = Font(bold=True, color="FFFFFF")


# ── 1. PROJECT CHARTER ───────────────────────────────────────────────────────

def fill_project_charter():
    path = f"{FOLDER}/Project Charter.xlsx"
    wb = openpyxl.load_workbook(path)
    ws = wb["GPI - Project Charter"]

    data = [
        (4,  1, "PROJECT NAME",          2, "Career Path Prediction & Guidance System"),
        (5,  1, "PROJECT SPONSOR",       2, "IAC Global Professional Internship"),
        (6,  1, "PROJECT MANAGER",       2, "Intern / Developer"),
        (7,  1, "PROJECT START DATE",    2, "2025-07-01"),
        (8,  1, "PROJECT END DATE",      2, "2025-08-12"),
        (9,  1, "PROJECT ID",            2, "CPP-2025-001"),
        (10, 1, "VERSION",               2, "1.0"),
    ]
    for row, c1, label, c2, value in data:
        w(ws, row, c1, label, bold=True)
        w(ws, row, c2, value)

    # Purpose
    w(ws, 12, 1, "PURPOSE / OBJECTIVE", bold=True)
    w(ws, 13, 1,
      "To develop a standalone web application that uses machine learning and deep learning "
      "models to predict suitable career paths for students based on their academic profile, "
      "technical skills, and areas of interest. The system provides personalized career "
      "guidance and curated course recommendations.", wrap=True)

    # Scope
    w(ws, 15, 1, "IN SCOPE", bold=True)
    in_scope = (
        "1. Load and preprocess student datasets (CSV/Excel)\n"
        "2. Exploratory Data Analysis (EDA) with automated visualizations\n"
        "3. Train 4 classical ML models: Random Forest, SVM, KNN, Logistic Regression\n"
        "4. Train deep learning model (Keras neural network)\n"
        "5. Auto-select best model by weighted F1-score\n"
        "6. Serialize model artifacts using Joblib\n"
        "7. Streamlit web UI for student input and career prediction\n"
        "8. Course recommendations for 10 career labels\n"
        "9. Input validation with descriptive error messages\n"
        "10. Offline training pipeline script (train_pipeline.py)"
    )
    w(ws, 16, 1, in_scope, wrap=True)

    w(ws, 18, 1, "OUT OF SCOPE", bold=True)
    out_scope = (
        "1. Real-time dataset updates or live data ingestion\n"
        "2. User authentication or account management\n"
        "3. Mobile application development\n"
        "4. Cloud deployment or CI/CD pipeline\n"
        "5. Multi-language support"
    )
    w(ws, 19, 1, out_scope, wrap=True)

    # Stakeholders
    w(ws, 21, 1, "STAKEHOLDERS", bold=True)
    stakeholders = [
        ("Project Sponsor", "IAC Global", "Approves scope and resources"),
        ("Project Manager / Developer", "Intern", "Implements and maintains the system"),
        ("Primary End User", "Student", "Submits profile, receives career guidance"),
    ]
    w(ws, 22, 1, "Role", bold=True); w(ws, 22, 2, "Name / Group", bold=True)
    for i, (role, name, resp) in enumerate(stakeholders, start=23):
        w(ws, i, 1, f"{role} — {resp}")
        w(ws, i, 2, name)

    # Tech stack
    w(ws, 27, 1, "TECHNOLOGY STACK", bold=True)
    tech = (
        "Python 3.9+ | Streamlit 1.35+ | scikit-learn 1.5.0 | TensorFlow 2.16.1 | "
        "Pandas 2.2.2 | NumPy 1.26.4 | Matplotlib 3.9.0 | Seaborn 0.13.2 | "
        "Joblib 1.4.2 | pytest 8.2.2 | hypothesis 6.103.1"
    )
    w(ws, 28, 1, tech, wrap=True)

    # Success criteria
    w(ws, 30, 1, "SUCCESS CRITERIA", bold=True)
    criteria = (
        "1. All 4 classical ML models train without errors\n"
        "2. Best model selected by weighted F1-score and saved as artifact\n"
        "3. Streamlit UI serves predictions within 3 seconds\n"
        "4. Input validation catches empty/invalid fields before prediction\n"
        "5. At least 3 course recommendations returned per career label\n"
        "6. 59+ automated tests passing (0 failures)"
    )
    w(ws, 31, 1, criteria, wrap=True)

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 60
    ws.row_dimensions[13].height = 80
    ws.row_dimensions[16].height = 120
    ws.row_dimensions[19].height = 80
    ws.row_dimensions[28].height = 50
    ws.row_dimensions[31].height = 100

    wb.save(path)
    print(f"✅ Filled: Project Charter.xlsx")


# ── 2. LESSONS LEARNT LOG ────────────────────────────────────────────────────

def fill_lessons_learnt():
    path = f"{FOLDER}/Lessons Learnt Log.xlsx"
    wb = openpyxl.load_workbook(path)
    ws = wb["LESSONS LEARNED"]

    # Project info rows
    w(ws, 4, 3, "Career Path Prediction & Guidance System")   # PROJECT NAME
    w(ws, 5, 3, "CPP-2025-001")                               # PROJECT ID
    w(ws, 6, 3, "1.0")                                        # VERSION
    w(ws, 7, 3, str(date.today()))                            # DATE

    # Headers row — find or use row 9
    headers = ["ID", "Category", "Phase", "What Happened",
               "Root Cause", "Lesson Learned", "Action Taken", "Status"]
    for col, h in enumerate(headers, start=2):
        hdr(ws, 9, col, h)

    lessons = [
        ("LL-001", "Technical / Environment", "Setup",
         "Installed scikit-learn 1.9.0 which requires NumPy 2.x, but NumPy 1.26.4 was pinned. "
         "Caused: AttributeError: module 'numpy' has no attribute 'long'",
         "requirements.txt pinned scikit-learn==1.5.0 but pip installed 1.9.0 due to environment conflict.",
         "Always use a fresh virtual environment. Pin exact versions you tested with.",
         "Pinned scikit-learn==1.5.0, upgraded Streamlit to 1.58.0. Added venv instructions to README.",
         "Closed"),
        ("LL-002", "Technical / ML", "Integration",
         "Artifacts serialized with scikit-learn 1.9.0 loaded with 1.5.0. InconsistentVersionWarning — predictions failed.",
         "Artifacts generated in a different environment state before version was locked.",
         "Whenever you change a library version, delete old artifacts and retrain from scratch.",
         "Documented: 'run train_pipeline.py after any dependency change'. Retrained all artifacts.",
         "Closed"),
        ("LL-003", "Technical / ML", "Demo",
         "Streamlit form sent 5 columns to predictor but model trained on only 3. Shape mismatch → PredictorError.",
         "Form raw_input dict passed directly to predictor without filtering to trained columns.",
         "Predictor input must exactly match columns used during training. Document the input schema.",
         "Updated app.py to only pass gpa, technical_skills, interests to the predictor.",
         "Closed"),
        ("LL-004", "Process / Environment", "Setup",
         "scipy 1.18.0 in system environment required NumPy 2.x — conflicted with project NumPy 1.26.4 pin.",
         "Project installed into system Python instead of isolated virtual environment.",
         "Always create a project-specific venv before installing dependencies.",
         "Documented venv setup steps in project README and Project Report.",
         "Closed"),
        ("LL-005", "Technical / Architecture", "Implementation",
         "TensorFlow listed as required but many environments don't have it. Risk of pipeline crash.",
         "TensorFlow import was at module level — would fail immediately on import.",
         "Wrap optional heavy dependencies in lazy imports. Use try/except ImportError.",
         "TensorFlow imported lazily inside train_deep_learning_model(). Pipeline logs WARNING and continues.",
         "Closed"),
        ("LL-006", "Process", "All Phases",
         "Having a detailed spec (requirements.md, design.md, tasks.md) before coding reduced rework significantly.",
         "N/A — positive lesson.",
         "Invest time in clear requirements and design documents before coding. Acceptance criteria map to tests.",
         "Continue using spec-driven development for future projects.",
         "Closed"),
        ("LL-007", "Technical / ML", "Implementation",
         "Running EDA first revealed class imbalance in sample dataset — explained inflated accuracy on random data.",
         "Sample dataset used random career label assignments, causing trivially learnable patterns.",
         "Always inspect class balance and feature distributions before training.",
         "EDA integrated as step 3 in training pipeline, before any model training begins.",
         "Closed"),
    ]

    for i, (lid, cat, phase, what, cause, lesson, action, status) in enumerate(lessons, start=10):
        row_data = [lid, cat, phase, what, cause, lesson, action, status]
        for col, val in enumerate(row_data, start=2):
            w(ws, i, col, val, wrap=True)

    # Column widths
    widths = {"B": 10, "C": 22, "D": 18, "E": 40, "F": 40, "G": 40, "H": 40, "I": 12}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    wb.save(path)
    print(f"✅ Filled: Lessons Learnt Log.xlsx")


# ── 3. RAID LOG ──────────────────────────────────────────────────────────────

def fill_raid_log():
    path = f"{FOLDER}/RAID Log.xlsx"
    wb = openpyxl.load_workbook(path)
    ws = wb["RAID LOG"]

    # Project header fields
    w(ws, 3, 2, "Career Path Prediction & Guidance System")  # Project title
    w(ws, 3, 5, "CPP-2025-001")                              # Project ID
    w(ws, 3, 8, "Intern / Developer")                        # Project Manager
    w(ws, 5, 2, "1.0")                                       # Doc version
    w(ws, 5, 5, str(date.today()))                           # Last updated

    # Column headers (row 8)
    raid_headers = [
        "Sr No", "RAID Category", "ID", "Description", "Impact",
        "Probability", "Priority", "Mitigation / Response", "Owner", "Status"
    ]
    for col, h in enumerate(raid_headers, start=2):
        hdr(ws, 8, col, h)

    raid_items = [
        # Risks
        (1, "Risk", "R1", "TensorFlow not available in target environment",
         "Deep learning model cannot be trained", "Medium", "Medium",
         "DL training wrapped in try/except ImportError; pipeline continues with classical models only",
         "Developer", "Closed"),
        (2, "Risk", "R2", "NumPy/scikit-learn version conflicts across environments",
         "AttributeError at runtime; system unusable", "High", "High",
         "Pin exact versions in requirements.txt; use virtual environment",
         "Developer", "Closed"),
        (3, "Risk", "R3", "Dataset column names don't match model input columns",
         "Shape mismatch error during prediction", "Medium", "High",
         "Preprocessor uses column names detected at fit time; app sends matching columns",
         "Developer", "Closed"),
        (4, "Risk", "R4", "scikit-learn version mismatch when loading serialized artifacts",
         "InconsistentVersionWarning; potential prediction errors", "Medium", "High",
         "Retrain pipeline on same environment; artifacts and runtime must match",
         "Developer", "Active"),
        (5, "Risk", "R5", "Streamlit breaking changes in newer versions",
         "UI components may stop working", "Low", "Medium",
         "Streamlit version pinned in requirements.txt",
         "Developer", "Closed"),
        (6, "Risk", "R6", "Insufficient training data leading to poor model accuracy",
         "Unreliable career predictions for end users", "Medium", "High",
         "Dataset should have ≥10 samples per class; UserWarning emitted if not",
         "Developer", "Monitoring"),
        (7, "Risk", "R7", "Model artifacts corrupted or missing at app startup",
         "Application cannot serve predictions", "Low", "High",
         "FileNotFoundError caught at startup; user-friendly message shown with instructions",
         "Developer", "Closed"),
        # Assumptions
        (8, "Assumption", "A1", "Python 3.9+ is installed on the target machine",
         "System cannot run if Python is absent or outdated", "-", "High",
         "Document minimum Python version in README and requirements",
         "Developer", "Closed"),
        (9, "Assumption", "A2", "All dependencies in requirements.txt can be installed via pip",
         "Setup fails if packages unavailable", "-", "Medium",
         "Use standard PyPI packages; test installation in clean venv",
         "Developer", "Closed"),
        (10, "Assumption", "A3", "Training dataset has a clearly identified target column (career label)",
         "Pipeline fails to split or train without target column", "-", "High",
         "--target CLI argument allows specifying target column; defaults to last column",
         "Developer", "Closed"),
        (11, "Assumption", "A4", "Training dataset has ≥50 total samples and ≥10 per class",
         "Model accuracy will be poor with very few samples", "-", "Medium",
         "UserWarning emitted for classes with <10 samples",
         "Developer", "Closed"),
        (12, "Assumption", "A5", "App is run from the project root directory",
         "Relative paths to artifacts/ and src/ will break", "-", "High",
         "Document this in README; use os.path for all file paths",
         "Developer", "Closed"),
        (13, "Assumption", "A6", "train_pipeline.py is run before app.py",
         "App raises FileNotFoundError at startup if artifacts missing", "-", "High",
         "FileNotFoundError caught gracefully; user shown instructions to run training first",
         "Developer", "Closed"),
        # Issues
        (14, "Issue", "I1", "numpy.long removed in NumPy 1.24+ — broke scikit-learn 1.9.0",
         "System entirely unusable; AttributeError on any ML operation", "-", "High",
         "Downgraded scikit-learn to 1.5.0 / upgraded Streamlit to 1.58.0",
         "Developer", "Closed"),
        (15, "Issue", "I2", "Streamlit 1.35.0 incompatible with NumPy 2.x",
         "Streamlit UI fails to start", "-", "High",
         "Upgraded to Streamlit 1.58.0 which supports NumPy 1.26.4",
         "Developer", "Closed"),
        (16, "Issue", "I3", "Artifacts trained with scikit-learn 1.9.0, loaded with 1.5.0",
         "Predictions unreliable; version warning at every inference call", "-", "Medium",
         "Retrained pipeline with scikit-learn 1.5.0; artifacts regenerated",
         "Developer", "Closed"),
        (17, "Issue", "I4", "Form raw_input dict contained extra columns not in trained preprocessor",
         "PredictorError on every prediction attempt; demo broken", "-", "High",
         "Filtered raw_input in app.py to only send gpa, technical_skills, interests",
         "Developer", "Closed"),
        (18, "Issue", "I5", "5 deep learning tests skipped when TensorFlow not installed",
         "Incomplete test coverage for neural network component", "-", "Low",
         "Tests use pytest.importorskip — will auto-run when TF is installed",
         "Developer", "Open"),
        # Dependencies
        (19, "Dependency", "D1", "Python 3.9+ must be installed",
         "System cannot run", "-", "Critical",
         "Document in README; verify in setup instructions",
         "Developer", "Closed"),
        (20, "Dependency", "D2", "scikit-learn 1.5.0 required",
         "Model training and preprocessing unavailable", "-", "High",
         "Pinned in requirements.txt",
         "Developer", "Closed"),
        (21, "Dependency", "D3", "Streamlit 1.35+ required for web UI",
         "Web interface unavailable", "-", "High",
         "Pinned in requirements.txt (using 1.58.0)",
         "Developer", "Closed"),
        (22, "Dependency", "D4", "TensorFlow 2.16.1 optional for deep learning",
         "DL model skipped; classical models still work", "-", "Medium",
         "Optional install; pipeline continues without TF",
         "Developer", "Open"),
        (23, "Dependency", "D5", "train_pipeline.py must run before app.py",
         "App cannot serve predictions without artifacts", "-", "High",
         "Documented clearly in README and Project Report",
         "Developer", "Closed"),
    ]

    for row_data in raid_items:
        sr, category, rid, desc, impact, prob, priority, mitigation, owner, status = row_data
        row = 8 + sr
        values = [sr, category, rid, desc, impact, prob, priority, mitigation, owner, status]
        for col, val in enumerate(values, start=2):
            w(ws, row, col, val, wrap=True)

    col_widths = {"B": 8, "C": 14, "D": 10, "E": 45, "F": 35, "G": 12, "H": 12, "I": 45, "J": 18, "K": 12}
    for col, width in col_widths.items():
        ws.column_dimensions[col].width = width

    wb.save(path)
    print(f"✅ Filled: RAID Log.xlsx")


# ── 4. REQUIREMENT ELICITATION QUESTIONNAIRE ────────────────────────────────

def fill_req_questionnaire():
    path = f"{FOLDER}/Requirement Elicitation Questionnaire.xlsx"
    wb = openpyxl.load_workbook(path)
    ws = wb["Sheet1"]

    w(ws, 3, 2, "Student / Developer")
    w(ws, 4, 2, "Machine Learning / AI")
    w(ws, 5, 2, "Career Path Prediction & Guidance System")

    qa_pairs = [
        ("Who are the primary users of this system?",
         "Students who want personalized career path recommendations based on their academic profile."),
        ("Who are the secondary users?",
         "Developers and interns who maintain, retrain, and update the ML models."),
        ("What problem does this system solve?",
         "Students often don't know which career path suits their skills. This system provides data-driven predictions with actionable course recommendations."),
        ("What data will the system use for training?",
         "A labeled CSV or Excel dataset with student records: academic scores, skills, interests, and a career label column."),
        ("What file formats must be supported?",
         "CSV (.csv) and Excel (.xlsx, .xls)."),
        ("What should happen if the file is missing or wrong format?",
         "Raise a descriptive error: FileNotFoundError for missing files, ValueError for unsupported formats."),
        ("How should missing values be handled?",
         "Numeric: impute with column median. Categorical: impute with column mode (most frequent value)."),
        ("How should the dataset be split?",
         "80% training, 20% test, using stratified sampling to preserve class distribution."),
        ("Which ML models should be trained?",
         "Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Logistic Regression."),
        ("Should a deep learning model be included?",
         "Yes — TensorFlow/Keras neural network with ≥2 hidden layers and EarlyStopping (patience=5)."),
        ("How should the best model be selected?",
         "By highest weighted F1-score across all trained models."),
        ("Should the model retrain on every prediction?",
         "No. Model trained offline, saved as artifact. Web app loads saved model at startup."),
        ("What metrics should be reported per model?",
         "Accuracy, weighted Precision, weighted Recall, weighted F1-score, Confusion Matrix."),
        ("What should the system output per student submission?",
         "Predicted career label + confidence score (percentage)."),
        ("How many career labels should be supported?",
         "10: Data Scientist, Software Engineer, Machine Learning Engineer, Data Analyst, Web Developer, Cybersecurity Analyst, Business Analyst, UX Designer, Network Engineer, Database Administrator."),
        ("Should the system recommend courses?",
         "Yes — at least 3 courses per career with title, provider name, and clickable URL."),
        ("What if no courses are mapped for a predicted career?",
         "Return a default set of at least 3 general career development courses."),
        ("Should predictions be deterministic?",
         "Yes — same input must always produce the same career label and confidence score."),
        ("What type of interface is required?",
         "Web-based form using Streamlit. Accessible from any browser at localhost:8501."),
        ("What form fields should be shown?",
         "GPA/Overall Score (required), Technical Skills (required), Areas of Interest (required), Soft Skills (optional), Extracurricular Activities (optional)."),
        ("What happens if a required field is left empty?",
         "Show field-level validation error message. Do not call the prediction model."),
        ("What happens if model artifacts are missing at startup?",
         "Display user-friendly error: 'Please run the training pipeline first (python train_pipeline.py)'"),
        ("What happens on unexpected prediction error?",
         "Display generic error without stack traces: 'An unexpected error occurred. Please try again.'"),
        ("How fast should predictions be?",
         "Career label returned within 3 seconds of form submission."),
        ("What Python version should be supported?",
         "Python 3.9 and above."),
        ("Should the system work offline?",
         "Yes — both training and inference run locally with no internet required."),
        ("How should the codebase be organized?",
         "Separate modules: data_loader, preprocessor, eda_module, trainer, selector, predictor, recommender, validator, app. All public functions must have docstrings."),
        ("Should there be automated tests?",
         "Yes — unit tests with pytest and property-based tests with hypothesis."),
        ("How should dependencies be managed?",
         "requirements.txt with all dependencies pinned to exact versions."),
    ]

    start_row = 8
    for i, (q, a) in enumerate(qa_pairs):
        row = start_row + i * 2
        w(ws, row,   1, f"Q{i+1}. {q}", bold=True, wrap=True)
        w(ws, row+1, 1, f"A: {a}", wrap=True)

    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 55

    wb.save(path)
    print(f"✅ Filled: Requirement Elicitation Questionnaire.xlsx")


# ── 5. SRS ────────────────────────────────────────────────────────────────────

def fill_srs():
    path = f"{FOLDER}/Software Requirement Specifications (SRS).xlsx"
    wb = openpyxl.load_workbook(path)
    ws = wb["GPI - SRS"]

    srs_data = [
        ("BACKGROUND\n(Available in the Problem Statement)",
         "Students often struggle to identify the most suitable career path aligned with their "
         "academic strengths, technical skills, and personal interests. Traditional career "
         "counselling is expensive, inconsistent, and not data-driven. This system addresses "
         "the gap by using machine learning to analyse student profiles and predict the most "
         "suitable career path automatically."),
        ("PROJECT OVERVIEW\n(Please refer the objective section of the Problem Statement)",
         "The Career Path Prediction & Guidance System is a standalone Python web application "
         "that accepts structured student data, trains multiple ML/DL models, selects the best "
         "performer, and delivers career predictions with curated course recommendations via a "
         "Streamlit UI."),
        ("PROBLEM STATEMENT",
         "Build a machine learning system that: (1) trains on historical student data, "
         "(2) predicts a career label for new students, (3) provides a confidence score, "
         "and (4) recommends relevant courses for the predicted career."),
        ("OBJECTIVE",
         "1. Load and preprocess student datasets (CSV/Excel)\n"
         "2. Generate EDA visualizations\n"
         "3. Train Random Forest, SVM, KNN, Logistic Regression, and Keras neural network\n"
         "4. Evaluate all models and auto-select the best by weighted F1-score\n"
         "5. Serve predictions via Streamlit web UI\n"
         "6. Return ≥3 course recommendations per predicted career"),
        ("FUNCTIONAL REQUIREMENTS",
         "FR-01: Load CSV/Excel datasets (FileNotFoundError & ValueError on invalid input)\n"
         "FR-02: Preprocess data (median imputation, OneHotEncoding, MinMaxScaling, 80/20 stratified split)\n"
         "FR-03: Generate EDA plots (distributions, heatmap, class balance) saved as PNG\n"
         "FR-04: Train 4 classical ML models with fault tolerance per model\n"
         "FR-05: Train Keras neural network with EarlyStopping (patience=5)\n"
         "FR-06: Evaluate all models: accuracy, precision, recall, F1, confusion matrix\n"
         "FR-07: Select best model by F1, serialize model + preprocessor + label encoder\n"
         "FR-08: Streamlit form with labeled fields, placeholders, help text\n"
         "FR-09: Return career label + confidence score; deterministic predictions\n"
         "FR-10: Return ≥3 courses per career; default courses for unknown labels\n"
         "FR-11: Validate required fields; show field-level errors; graceful error handling\n"
         "FR-12: Modular codebase; requirements.txt; docstrings on all public functions"),
        ("NON-FUNCTIONAL REQUIREMENTS",
         "NFR-01: Prediction response < 3 seconds\n"
         "NFR-02: Python 3.9+ compatibility\n"
         "NFR-03: Fully offline operation (no internet at inference time)\n"
         "NFR-04: ≥59 automated tests passing\n"
         "NFR-05: All public functions documented with docstrings\n"
         "NFR-06: Deterministic predictions (same input → same output always)"),
        ("MODULES / COMPONENTS",
         "src/data_loader.py — CSV/Excel loading\n"
         "src/preprocessor.py — Imputation, encoding, scaling, stratified split\n"
         "src/eda_module.py — Distribution plots, heatmap, class balance chart\n"
         "src/trainer.py — Train RF, SVM, KNN, LR, Keras NN; evaluate all models\n"
         "src/selector.py — Select best model by F1; save Joblib artifacts\n"
         "src/predictor.py — Load artifacts; serve predictions; wrap errors\n"
         "src/recommender.py — Static course map for 10 career labels\n"
         "src/validator.py — Validate student form input\n"
         "src/models.py — Shared dataclasses (StudentInput, EvaluationResult, Course, PredictionResult)\n"
         "src/exceptions.py — PredictorError custom exception\n"
         "app.py — Streamlit UI entry point\n"
         "train_pipeline.py — Offline CLI training script"),
        ("SUPPORTED CAREER LABELS",
         "1. Data Scientist\n2. Software Engineer\n3. Machine Learning Engineer\n"
         "4. Data Analyst\n5. Web Developer\n6. Cybersecurity Analyst\n"
         "7. Business Analyst\n8. UX Designer\n9. Network Engineer\n10. Database Administrator"),
        ("TECHNOLOGY STACK",
         "Language: Python 3.9+\nWeb UI: Streamlit 1.35+\nML: scikit-learn 1.5.0\n"
         "DL: TensorFlow 2.16.1 / Keras\nData: Pandas 2.2.2, NumPy 1.26.4\n"
         "Visualization: Matplotlib 3.9.0, Seaborn 0.13.2\n"
         "Serialization: Joblib 1.4.2\nTesting: pytest 8.2.2, hypothesis 6.103.1"),
        ("TESTING STRATEGY",
         "Unit Tests (pytest):\n"
         "  - test_data_loader.py: 20 tests — CSV/Excel loading, error handling\n"
         "  - test_trainer.py: 16 tests — classical models, evaluation metrics, DL architecture\n"
         "  - test_validator.py: 28 tests — required fields, GPA range/type, empty lists\n\n"
         "Property-Based Tests (hypothesis):\n"
         "  12 correctness properties defined — preprocessing idempotence, split count preservation,\n"
         "  stratification, no data leakage, model serialization, prediction determinism,\n"
         "  confidence score range, recommender coverage, input validation, EDA file creation\n\n"
         "Integration Test:\n"
         "  train_pipeline.py runs end-to-end producing all 3 artifacts without errors"),
        ("HOW TO RUN",
         "Step 1: python -m venv venv && source venv/bin/activate\n"
         "Step 2: pip install -r requirements.txt\n"
         "Step 3: python train_pipeline.py --data data/sample.csv --target career_label\n"
         "Step 4: streamlit run app.py\n"
         "Open: http://localhost:8501"),
    ]

    start_row = 4
    for i, (question, answer) in enumerate(srs_data):
        row = start_row + i
        w(ws, row, 1, question, bold=True, wrap=True)
        w(ws, row, 2, answer, wrap=True)

    ws.column_dimensions["A"].width = 35
    ws.column_dimensions["B"].width = 90
    for r in range(start_row, start_row + len(srs_data)):
        ws.row_dimensions[r].height = 120

    wb.save(path)
    print(f"✅ Filled: Software Requirement Specifications (SRS).xlsx")


# ── 6. PROJECT SCHEDULE ──────────────────────────────────────────────────────

def fill_project_schedule():
    path = f"{FOLDER}/Project Schedule.xlsx"
    wb = openpyxl.load_workbook(path)

    # Fill Guide sheet basics
    ws_guide = wb["Guide"]
    w(ws_guide, 5, 2, "Student / Developer")  # Document created By

    # Use the EXAMPLE sheet structure or Project Schedule sheet
    ws = wb["Project Schedule"]

    # Write project header info in known cells
    w(ws, 4, 3, "2025-07-01")   # Project start date

    # Build a clean task table starting from row 10
    task_headers = ["Sr No", "Phase", "Task ID", "Task Name", "Owner",
                    "Start Date", "End Date", "Duration (Days)", "Status", "Notes"]
    for col, h in enumerate(task_headers, start=2):
        hdr(ws, 9, col, h)

    tasks = [
        (1,  "Phase 1 – Scaffold",    "T-01", "Create directory structure (src/, tests/, artifacts/, eda_output/, data/)", "Developer", "2025-07-01", "2025-07-01", 1, "Completed", ""),
        (2,  "Phase 1 – Scaffold",    "T-02", "Define shared data models (StudentInput, EvaluationResult, Course, PredictionResult)", "Developer", "2025-07-01", "2025-07-01", 1, "Completed", "src/models.py"),
        (3,  "Phase 1 – Scaffold",    "T-03", "Define PredictorError custom exception", "Developer", "2025-07-01", "2025-07-01", 1, "Completed", "src/exceptions.py"),
        (4,  "Phase 1 – Scaffold",    "T-04", "Configure test suite (conftest.py + Hypothesis profile)", "Developer", "2025-07-01", "2025-07-01", 1, "Completed", "tests/conftest.py"),
        (5,  "Phase 1 – Scaffold",    "T-05", "Pin all dependencies in requirements.txt", "Developer", "2025-07-01", "2025-07-01", 1, "Completed", "requirements.txt"),
        (6,  "Phase 2 – Data Layer",  "T-06", "Implement data_loader.py (load_dataset, get_dataset_info)", "Developer", "2025-07-02", "2025-07-02", 1, "Completed", "src/data_loader.py"),
        (7,  "Phase 2 – Data Layer",  "T-07", "Write 20 unit tests for data_loader.py", "Developer", "2025-07-02", "2025-07-02", 1, "Completed", "tests/test_data_loader.py"),
        (8,  "Phase 2 – Data Layer",  "T-08", "Implement preprocessor.py (Preprocessor class + stratified_split)", "Developer", "2025-07-03", "2025-07-04", 2, "Completed", "src/preprocessor.py"),
        (9,  "Phase 2 – Data Layer",  "T-09", "Checkpoint: all data layer tests pass (20/20)", "Developer", "2025-07-04", "2025-07-04", 1, "Completed", ""),
        (10, "Phase 3 – EDA",         "T-10", "Implement eda_module.py (distributions, heatmap, class balance)", "Developer", "2025-07-05", "2025-07-05", 1, "Completed", "src/eda_module.py"),
        (11, "Phase 4 – Training",    "T-11", "Implement trainer.py (train_classical_models, train_deep_learning_model, evaluate_model, evaluate_all)", "Developer", "2025-07-06", "2025-07-08", 3, "Completed", "src/trainer.py"),
        (12, "Phase 4 – Training",    "T-12", "Write 16 unit tests for trainer.py (11 active, 5 skip pending TF)", "Developer", "2025-07-08", "2025-07-08", 1, "Completed", "tests/test_trainer.py"),
        (13, "Phase 4 – Training",    "T-13", "Implement selector.py (select_best_model, save_artifacts)", "Developer", "2025-07-09", "2025-07-09", 1, "Completed", "src/selector.py"),
        (14, "Phase 4 – Training",    "T-14", "Checkpoint: training pipeline tests pass (30/30)", "Developer", "2025-07-09", "2025-07-09", 1, "Completed", ""),
        (15, "Phase 5 – Prediction",  "T-15", "Implement predictor.py (Predictor class with artifact loading)", "Developer", "2025-07-10", "2025-07-10", 1, "Completed", "src/predictor.py"),
        (16, "Phase 5 – Prediction",  "T-16", "Implement recommender.py (COURSE_MAP for 10 careers + DEFAULT_COURSES)", "Developer", "2025-07-10", "2025-07-10", 1, "Completed", "src/recommender.py"),
        (17, "Phase 5 – Validation",  "T-17", "Implement validator.py (validate_student_input)", "Developer", "2025-07-11", "2025-07-11", 1, "Completed", "src/validator.py"),
        (18, "Phase 5 – Validation",  "T-18", "Write 28 unit tests for validator.py", "Developer", "2025-07-11", "2025-07-11", 1, "Completed", "tests/test_validator.py"),
        (19, "Phase 6 – UI",          "T-19", "Implement app.py (Streamlit form, validation, prediction display, course links)", "Developer", "2025-07-12", "2025-07-13", 2, "Completed", "app.py"),
        (20, "Phase 6 – UI",          "T-20", "Implement train_pipeline.py (end-to-end CLI training script)", "Developer", "2025-07-14", "2025-07-14", 1, "Completed", "train_pipeline.py"),
        (21, "Phase 6 – Integration", "T-21", "Final checkpoint: 59 tests passing; pipeline runs end-to-end", "Developer", "2025-07-15", "2025-07-15", 1, "Completed", ""),
        (22, "Phase 6 – Integration", "T-22", "Fix environment issues (numpy/scikit-learn version conflicts)", "Developer", "2025-07-15", "2025-07-15", 1, "Completed", ""),
        (23, "Phase 6 – Integration", "T-23", "Fix form column mismatch (app.py raw_input filtering)", "Developer", "2025-07-15", "2025-07-15", 1, "Completed", ""),
        (24, "Phase 7 – Docs",        "T-24", "Fill all project deliverable documents", "Developer", "2025-07-16", "2025-07-16", 1, "Completed", "src/Deliverables/"),
    ]

    for row_data in tasks:
        sr, phase, tid, task, owner, start, end, dur, status, notes = row_data
        row = 9 + sr
        values = [sr, phase, tid, task, owner, start, end, dur, status, notes]
        for col, val in enumerate(values, start=2):
            w(ws, row, col, val, wrap=(col == 5))  # wrap task name

    col_widths = {"B": 6, "C": 22, "D": 10, "E": 60, "F": 18, "G": 12, "H": 12, "I": 16, "J": 14, "K": 30}
    for col, width in col_widths.items():
        ws.column_dimensions[col].width = width

    wb.save(path)
    print(f"✅ Filled: Project Schedule.xlsx")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Filling Excel deliverables...\n")
    fill_project_charter()
    fill_lessons_learnt()
    fill_raid_log()
    fill_req_questionnaire()
    fill_srs()
    fill_project_schedule()
    print("\nAll Excel deliverables filled successfully!")
    print("Note: Project Report.docx and WBS.pptx are non-Excel formats.")
    print("Use the companion -Content.md files to copy content into those manually.")
