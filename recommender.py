"""
Recommendation module for the Career Path Prediction & Guidance System.

Maps predicted career labels to curated lists of online courses. Provides a
fallback default course list for any label not present in the static mapping.
"""

from __future__ import annotations

from src.models import Course

# ---------------------------------------------------------------------------
# Default courses — returned when a career label has no dedicated mapping.
# Must contain at least 3 generic career-development courses.
# ---------------------------------------------------------------------------

DEFAULT_COURSES: list[Course] = [
    Course(
        title="Learning How to Learn",
        provider="Coursera",
        url="https://www.coursera.org/learn/learning-how-to-learn",
    ),
    Course(
        title="Career Planning: A Pathway to Employment",
        provider="edX",
        url="https://www.edx.org/course/career-planning-a-pathway-to-employment",
    ),
    Course(
        title="Professional Skills for the Workplace",
        provider="Coursera",
        url="https://www.coursera.org/specializations/professional-skills-for-the-workplace",
    ),
]

# ---------------------------------------------------------------------------
# Static course mapping: career_label -> list[Course]
# Each label contains at least 3 courses with non-empty title, provider, url.
# ---------------------------------------------------------------------------

COURSE_MAP: dict[str, list[Course]] = {
    "Data Scientist": [
        Course(
            title="IBM Data Science Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/ibm-data-science",
        ),
        Course(
            title="Applied Data Science with Python Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/data-science-python",
        ),
        Course(
            title="Data Science MicroMasters",
            provider="edX",
            url="https://www.edx.org/micromasters/uc-san-diegox-data-science",
        ),
        Course(
            title="Python for Data Science and Machine Learning Bootcamp",
            provider="Udemy",
            url="https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/",
        ),
    ],
    "Software Engineer": [
        Course(
            title="CS50's Introduction to Computer Science",
            provider="edX",
            url="https://www.edx.org/course/introduction-computer-science-harvardx-cs50x",
        ),
        Course(
            title="Meta Back-End Developer Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/meta-back-end-developer",
        ),
        Course(
            title="Software Engineering Essentials",
            provider="edX",
            url="https://www.edx.org/course/software-engineering-essentials",
        ),
        Course(
            title="The Complete Web Developer in 2024",
            provider="Udemy",
            url="https://www.udemy.com/course/the-complete-web-developer-zero-to-mastery/",
        ),
    ],
    "Machine Learning Engineer": [
        Course(
            title="Machine Learning Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/machine-learning-introduction",
        ),
        Course(
            title="Deep Learning Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/deep-learning",
        ),
        Course(
            title="Machine Learning Engineering for Production (MLOps) Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops",
        ),
        Course(
            title="Practical Deep Learning for Coders",
            provider="fast.ai",
            url="https://course.fast.ai/",
        ),
    ],
    "Data Analyst": [
        Course(
            title="Google Data Analytics Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/google-data-analytics",
        ),
        Course(
            title="Data Analysis with Python",
            provider="Coursera",
            url="https://www.coursera.org/learn/data-analysis-with-python",
        ),
        Course(
            title="Excel to MySQL: Analytic Techniques for Business Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/excel-mysql",
        ),
        Course(
            title="Data Visualisation with Tableau Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/data-visualization",
        ),
    ],
    "Web Developer": [
        Course(
            title="Meta Front-End Developer Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/meta-front-end-developer",
        ),
        Course(
            title="The Web Developer Bootcamp",
            provider="Udemy",
            url="https://www.udemy.com/course/the-web-developer-bootcamp/",
        ),
        Course(
            title="Full-Stack Web Development with React Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/full-stack-react",
        ),
        Course(
            title="Responsive Web Design",
            provider="freeCodeCamp",
            url="https://www.freecodecamp.org/learn/2022/responsive-web-design/",
        ),
    ],
    "Cybersecurity Analyst": [
        Course(
            title="Google Cybersecurity Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/google-cybersecurity",
        ),
        Course(
            title="IBM Cybersecurity Analyst Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst",
        ),
        Course(
            title="Cybersecurity Fundamentals",
            provider="edX",
            url="https://www.edx.org/course/cybersecurity-fundamentals",
        ),
        Course(
            title="CompTIA Security+ (SY0-701) Complete Course",
            provider="Udemy",
            url="https://www.udemy.com/course/securityplus/",
        ),
    ],
    "Business Analyst": [
        Course(
            title="Business Analysis Fundamentals",
            provider="Udemy",
            url="https://www.udemy.com/course/business-analysis-ba/",
        ),
        Course(
            title="Business Analytics Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/business-analytics",
        ),
        Course(
            title="Professional Certificate in Business Analysis",
            provider="edX",
            url="https://www.edx.org/professional-certificate/business-analysis",
        ),
        Course(
            title="Agile Business Analysis",
            provider="Coursera",
            url="https://www.coursera.org/learn/agile-business-analysis",
        ),
    ],
    "UX Designer": [
        Course(
            title="Google UX Design Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/google-ux-design",
        ),
        Course(
            title="User Experience Research and Design Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/michiganux",
        ),
        Course(
            title="UX Design Fundamentals",
            provider="Coursera",
            url="https://www.coursera.org/learn/ux-design-fundamentals",
        ),
        Course(
            title="UI/UX Design Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/ui-ux-design",
        ),
    ],
    "Network Engineer": [
        Course(
            title="Cisco Networking Basics Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/networking-basics",
        ),
        Course(
            title="The Complete Networking Fundamentals Course",
            provider="Udemy",
            url="https://www.udemy.com/course/complete-networking-fundamentals-course-ccna-start/",
        ),
        Course(
            title="Computer Networking",
            provider="edX",
            url="https://www.edx.org/course/computer-networking",
        ),
        Course(
            title="Google IT Support Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/google-it-support",
        ),
    ],
    "Database Administrator": [
        Course(
            title="IBM Data Engineering Professional Certificate",
            provider="Coursera",
            url="https://www.coursera.org/professional-certificates/ibm-data-engineer",
        ),
        Course(
            title="Database Management Essentials",
            provider="Coursera",
            url="https://www.coursera.org/learn/database-management",
        ),
        Course(
            title="The Ultimate MySQL Bootcamp",
            provider="Udemy",
            url="https://www.udemy.com/course/the-ultimate-mysql-bootcamp-go-from-sql-beginner-to-expert/",
        ),
        Course(
            title="PostgreSQL for Everybody Specialization",
            provider="Coursera",
            url="https://www.coursera.org/specializations/postgresql-for-everybody",
        ),
    ],
}


def get_recommendations(career_label: str) -> list[Course]:
    """Return a curated list of courses for the given career label.

    Looks up ``career_label`` in the static :data:`COURSE_MAP`. If the label
    is not found — for example, when the model predicts an unexpected class —
    the function returns :data:`DEFAULT_COURSES` instead of raising an
    exception, guaranteeing that the caller always receives at least 3 courses.

    Parameters
    ----------
    career_label : str
        The predicted career path label (e.g., ``"Data Scientist"``).

    Returns
    -------
    list[Course]
        A list of :class:`~src.models.Course` objects. Always contains at
        least 3 items with non-empty ``title``, ``provider``, and ``url``.

    Examples
    --------
    >>> courses = get_recommendations("Data Scientist")
    >>> len(courses) >= 3
    True
    >>> courses = get_recommendations("Unknown Career")
    >>> len(courses) >= 3
    True
    """
    return COURSE_MAP.get(career_label, DEFAULT_COURSES)
