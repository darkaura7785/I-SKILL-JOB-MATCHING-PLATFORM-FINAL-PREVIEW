"""
Keyword Extractor Module — v2 (India-aware, timestamp-based experience)
=========================================================================
Extracts skills, work experience (professional + internship), and education
from resumes using lightweight NLP (regex + rule-based parsing).

Key improvements over v1:
• pdfplumber primary PDF reader (better layout) with PyPDF2 fallback
• Date-timestamp scanning to calculate experience from actual job history
• Internship experience tracked separately
• India-specific degree patterns (BTech, MTech, BCA, MCA, BCom, MCom, etc.)
• Skills taxonomy enriched from india_professional_skills_intelligence.csv
• All fields default to 0 / "Not Specified" — never None on resume upload
"""

import io
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# ── Optional heavy imports ──────────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ── Path helpers ────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INDIA_CSV = os.path.join(
    os.path.dirname(_BASE_DIR),          # project root
    "india_professional_skills_intelligence.csv"
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SKILLS TAXONOMY  (core + enriched from India dataset)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Core technology skills (unchanged from v1, extended)
TECH_SKILLS_TAXONOMY: set = {
    # Programming Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "c", "go",
    "golang", "rust", "ruby", "php", "swift", "kotlin", "r", "scala",
    "dart", "shell", "bash", "sql", "vba",
    # Frontend
    "react", "react.js", "next.js", "vue", "vue.js", "angular", "svelte",
    "html", "html5", "css", "css3", "tailwind", "tailwind css", "bootstrap",
    "sass", "scss", "redux", "graphql", "rest api", "restful apis", "webpack",
    "vite", "webgl", "ui design", "responsive design",
    # Backend
    "node.js", "express", "express.js", "django", "flask", "fastapi",
    "spring boot", "asp.net", "rails", "microservices", "grpc", "serverless",
    "system design", "rest apis",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "sqlite", "dynamodb", "cassandra", "oracle", "mariadb", "firebase",
    "supabase", "prisma",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
    "kubernetes", "terraform", "ci/cd", "github actions", "jenkins",
    "gitlab ci", "ansible", "prometheus", "grafana", "linux", "nginx",
    "helm", "devops", "cloud computing", "sre", "cloud security",
    # Data Science & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pytorch", "tensorflow", "keras", "scikit-learn",
    "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "data analysis",
    "data science", "tableau", "power bi", "spark", "hadoop", "airflow",
    "databricks", "llm", "llms", "transformers", "hugging face", "mlops",
    "prompt engineering", "statistics", "data visualization",
    "r", "excel", "statistical analysis",
    # Product & Agile
    "agile", "scrum", "kanban", "product management", "jira",
    "figma", "user research", "wireframing", "prototyping", "ux research",
    "product strategy", "a/b testing", "roadmapping", "stakeholder management",
    # Security
    "cybersecurity", "network security", "penetration testing", "siem",
    "incident response", "vulnerability assessment", "cryptography", "owasp",
    "soc 2", "compliance", "ethical hacking", "iso 27001", "risk assessment",
    # Finance & Accounting (India-relevant)
    "tally", "sap", "gst", "taxation", "auditing", "financial modeling",
    "financial analysis", "accounting", "ifrs", "gaap",
    # Digital Marketing (India-relevant)
    "seo", "sem", "google analytics", "social media marketing",
    "content marketing", "email marketing", "performance marketing",
    "content writing", "copywriting", "seo writing", "content strategy",
    # HR
    "talent acquisition", "hris", "performance management",
    "employee engagement", "labour law compliance", "payroll",
    # Sales & BD
    "salesforce", "crm", "b2b sales", "lead generation", "account management",
    "negotiation",
    # Core Engineering
    "autocad", "solidworks", "mechanical design", "quality control",
    "project management", "six sigma", "supply chain management",
    "logistics", "vendor management", "process improvement",
    # Soft skills / Tools
    "git", "github",
}

# Non-technical skills for expanded role coverage (administrative, healthcare, education,
# hospitality, trades, agriculture, domestic, transport, manufacturing, retail, government,
# legal, banking, security, beauty, media)
NON_TECHNICAL_SKILLS: set = {
    # Administrative
    "typing", "data entry", "filing", "office management", "ms office",
    "receptionist duties", "telephone handling", "scheduling", "record keeping",
    "documentation",
    # Healthcare
    "patient care", "nursing", "first aid", "medical records", "pharmacy",
    "physiotherapy", "lab technician skills", "infection control",
    "vital signs monitoring", "health assessment", "anm training", "asha worker training",
    # Education
    "teaching", "lesson planning", "classroom management", "curriculum development",
    "child development", "tutoring", "special education", "educational assessment",
    "student counseling", "e-learning",
    # Hospitality
    "cooking", "food preparation", "food safety", "housekeeping", "hotel management",
    "front desk operations", "banquet management", "bartending", "room service",
    "guest relations", "kitchen management",
    # Construction & Trades
    "plumbing", "electrical wiring", "carpentry", "masonry", "welding",
    "painting", "hvac", "blueprint reading", "scaffolding", "concrete work",
    "pipe fitting",
    # Agriculture
    "farming", "crop management", "irrigation", "soil testing", "organic farming",
    "livestock management", "agricultural machinery", "pest control", "harvesting",
    "seed treatment",
    # Domestic Services
    "childcare", "elderly care", "home cleaning", "laundry", "cooking meals",
    "pet care", "babysitting", "home organization",
    # Transport & Logistics
    "driving", "route planning", "fleet management", "warehouse management",
    "inventory management", "forklift operation", "delivery management",
    "dispatch operations",
    # Manufacturing
    "machine operation", "assembly line", "quality inspection", "packaging",
    "production planning", "cnc operation", "safety protocols", "material handling",
    # Retail
    "retail sales", "visual merchandising", "cash handling", "customer service",
    "point of sale", "stock management", "store management", "upselling",
    # Government / Public
    "public administration", "policy analysis", "government regulations",
    "tender management", "rti compliance", "public speaking",
    # Legal
    "legal research", "contract drafting", "litigation support",
    "intellectual property", "legal compliance", "court procedures",
    # Banking & Insurance
    "loan processing", "kyc compliance", "insurance underwriting", "credit analysis",
    "mutual funds", "banking operations", "neft rtgs", "financial advisory",
    # Security
    "physical security", "cctv monitoring", "access control", "fire safety",
    "crowd management", "surveillance",
    # Beauty & Wellness
    "hair styling", "makeup artistry", "skin care", "spa therapy", "nail art",
    "salon management",
    # Media
    "video editing", "photography", "graphic design", "social media management",
    "event management", "public relations", "presentation",
    # Healthcare additional
    "veterinary basics",
    # Networking (for IT support)
    "networking",
}

# India domain skills pulled from the CSV's top_skills column (pre-compiled set)
INDIA_DOMAIN_SKILLS: set = {
    # Software Engineering
    "system design", "rest apis", "spring boot", "microservices", "java",
    "node.js", "react", "sql", "git", "python", "javascript",
    # AI/ML
    "pytorch", "tensorflow", "nlp", "llms", "mlops", "deep learning",
    "computer vision", "prompt engineering",
    # Data
    "power bi", "tableau", "r", "statistics", "excel",
    "data visualization", "machine learning",
    # Cloud
    "aws", "gcp", "azure", "kubernetes", "terraform", "docker",
    "ci/cd", "linux",
    # Product
    "agile", "scrum", "jira", "product strategy", "user research",
    "roadmapping", "stakeholder management",
    # Design
    "figma", "wireframing", "prototyping", "design systems",
    # Marketing
    "seo", "sem", "google analytics", "social media marketing",
    "content marketing", "email marketing", "performance marketing",
    # Finance
    "tally", "sap", "gst", "auditing", "financial modeling", "taxation",
    # HR
    "talent acquisition", "hris", "performance management",
    "employee engagement", "labour law compliance", "payroll",
    # Sales
    "salesforce", "crm", "b2b sales", "lead generation",
    "account management", "negotiation",
    # Engineering
    "autocad", "solidworks", "mechanical design", "quality control",
    # Cybersecurity
    "siem", "ethical hacking", "cloud security", "iso 27001",
    "risk assessment", "penetration testing",
    # Content
    "content writing", "copywriting", "seo writing", "content strategy",
    "editing",
    # Operations
    "six sigma", "supply chain management", "logistics",
    "vendor management", "process improvement",
}

# Merge all sets
TECH_SKILLS_TAXONOMY = TECH_SKILLS_TAXONOMY | INDIA_DOMAIN_SKILLS | NON_TECHNICAL_SKILLS

# ── Canonical display name map ───────────────────────────────────────────────
CANONICAL_SKILL_MAP: Dict[str, str] = {
    "react.js": "React", "react": "React",
    "next.js": "Next.js",
    "vue.js": "Vue", "vue": "Vue",
    "express.js": "Express", "express": "Express",
    "golang": "Go", "go": "Go",
    "node.js": "Node.js",
    "tailwind css": "Tailwind CSS", "tailwind": "Tailwind CSS",
    "sklearn": "Scikit-Learn", "scikit-learn": "Scikit-Learn",
    "amazon web services": "AWS", "aws": "AWS",
    "google cloud": "GCP", "gcp": "GCP",
    "natural language processing": "NLP", "nlp": "NLP",
    "html5": "HTML", "html": "HTML",
    "css3": "CSS", "css": "CSS",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
    "rest apis": "REST APIs", "rest api": "REST APIs",
    "restful apis": "REST APIs",
    "llms": "LLMs", "llm": "LLMs",
    "b2b sales": "B2B Sales",
    "seo": "SEO", "sem": "SEM",
    "hris": "HRIS", "crm": "CRM",
    "gst": "GST", "sap": "SAP",
    "ci/cd": "CI/CD",
    "iso 27001": "ISO 27001",
    "siem": "SIEM",
    "six sigma": "Six Sigma",
    "ms office": "MS Office",
    "hvac": "HVAC",
    "cnc operation": "CNC Operation",
    "cctv monitoring": "CCTV Monitoring",
    "kyc compliance": "KYC Compliance",
    "neft rtgs": "NEFT/RTGS",
    "rti compliance": "RTI Compliance",
    "anm training": "ANM Training",
    "asha worker training": "ASHA Worker Training",
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EDUCATION PATTERNS  (India-aware)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Ordered from highest to lowest — first match wins.
#
# Each entry is (pattern, label, flags).  The flag lives on the entry rather
# than on the pattern because JavaScript has no scoped inline flags
# ("(?i:...)"), and the browser build has to apply exactly the same rules.
#
# Bare abbreviations are matched CASE-SENSITIVELY and must either be written
# with full stops (B.E.) or be a token that is not an English word (BTech,
# MSc, MBA).  Otherwise ordinary prose collides with degree abbreviations:
# "should be able to work" reads as "B.E.", "call me tomorrow" as "M.E." and
# "MS Office" as a Master's degree.  "MS" additionally carries a deny-list
# of Microsoft product names.
_MS_PRODUCTS = (
    "office", "word", "excel", "powerpoint", "publisher", "outlook",
    "teams", "sharepoint", "sql", "server", "azure", "dynamics", "visio",
    "project", "onenote", "onedrive", "one drive", "forms", "store",
    "translator", "whiteboard", "power bi", "powerbi", "page", "messenger",
)


def _ci(word: str) -> str:
    """Builds a case-insensitive character class for a literal word.

    Used where the pattern itself has to stay case-sensitive (``MS`` must
    not match a lower-case "ms") but the word it excludes may be written
    any way round — "MS Office", "Ms Office" and "MS office" all have to
    be recognised as a Microsoft product.
    """
    return "[" + "".join(
        c.upper() + c.lower() if c.isalpha() else c for c in word
    ) + "]"


_MS_PRODUCTS_RE = "(?:" + "|".join(_ci(p) for p in _MS_PRODUCTS) + ")"

EDUCATION_PATTERNS: List[Tuple[str, str, str]] = [
    # Doctorate
    (r'\b(?:ph\.?d|doctor\s+of\s+philosophy|doctorate|d\.?phil)\b',
     "Doctorate (Ph.D.)", "i"),
    # Masters — Indian + international
    (r'\bmaster\s+of\s+technology\b', "Master's (M.Tech)", "i"),
    (r'\bMTech\b', "Master's (M.Tech)", ""),
    (r'\bM\.Tech\b', "Master's (M.Tech)", ""),
    (r'\bmaster\s+of\s+engineering\b', "Master's (M.E.)", "i"),
    (r'\bM\.E\.?(?![A-Za-z])', "Master's (M.E.)", ""),
    (r'\bmaster\s+of\s+computer\s+applications\b', "Master's (MCA)", "i"),
    (r'\bMCA\b', "Master's (MCA)", ""),
    (r'\bmaster\s+of\s+science\b', "Master's (M.Sc)", "i"),
    (r'\bMSc\b', "Master's (M.Sc)", ""),
    (r'\bM\.Sc\.?(?![A-Za-z])', "Master's (M.Sc)", ""),
    (r'\bmaster\s+of\s+business\b', "Master's (MBA)", "i"),
    (r'\bMBA\b', "Master's (MBA)", ""),
    (r'\bmaster\s+of\s+commerce\b', "Master's (M.Com)", "i"),
    (r'\bM\.Com\b', "Master's (M.Com)", ""),
    (r'\bmaster\s+of\s+arts\b', "Master's (M.A.)", "i"),
    (r'\bM\.A\.?(?![A-Za-z])', "Master's (M.A.)", ""),
    (r'\bmaster\'?s\b(?!\s+of\b)', "Master's Degree", "i"),
    (r'\bmasters\b(?!\s+of\b)', "Master's Degree", "i"),
    (r'\bMS(?![A-Za-z])(?!\s+' + _MS_PRODUCTS_RE + r')', "Master's Degree", ""),
    (r'\bM\.S\.?(?![A-Za-z])', "Master's Degree", ""),
    # Bachelors — Indian
    (r'\bbachelor\s+of\s+technology\b', "Bachelor's (B.Tech)", "i"),
    (r'\bBTech\b', "Bachelor's (B.Tech)", ""),
    (r'\bB\.Tech\b', "Bachelor's (B.Tech)", ""),
    (r'\bbachelor\s+of\s+engineering\b', "Bachelor's (B.E.)", "i"),
    (r'\bB\.E\.?(?![A-Za-z])', "Bachelor's (B.E.)", ""),
    (r'\bbachelor\s+of\s+computer\s+applications\b', "Bachelor's (BCA)", "i"),
    (r'\bBCA\b', "Bachelor's (BCA)", ""),
    (r'\bbachelor\s+of\s+science\b', "Bachelor's (B.Sc)", "i"),
    (r'\bBSc\b', "Bachelor's (B.Sc)", ""),
    (r'\bB\.Sc\.?(?![A-Za-z])', "Bachelor's (B.Sc)", ""),
    (r'\bbachelor\s+of\s+business\b', "Bachelor's (BBA)", "i"),
    (r'\bBBA\b', "Bachelor's (BBA)", ""),
    (r'\bbachelor\s+of\s+commerce\b', "Bachelor's (B.Com)", "i"),
    (r'\bB\.Com\b', "Bachelor's (B.Com)", ""),
    (r'\bbachelor\s+of\s+arts\b', "Bachelor's (B.A.)", "i"),
    (r'\bB\.A\.?(?![A-Za-z])', "Bachelor's (B.A.)", ""),
    (r'\bbachelor\'?s\b(?!\s+of\b)', "Bachelor's Degree", "i"),
    (r'\bbachelors?\b(?!\s+of\b)', "Bachelor's Degree", "i"),
    (r'\bB\.S\.?(?![A-Za-z])', "Bachelor's Degree", ""),
    # Diploma / Associate
    (r'\b(?:diploma|polytechnic)\b', "Diploma", "i"),
    (r'\bassociate\'?s?\b|\bassociate\s+degree\b', "Associate's Degree", "i"),
    # ITI / Vocational
    (r'\b(?:iti|industrial\s+training\s+institute|vocational\s+training)\b',
     "ITI Certificate", "i"),
    # Higher Secondary / 12th
    (r'\b(?:12th(?:\s+pass)?|higher\s+secondary|hsc|intermediate|'
     r'plus\s+two|\+2)\b', "12th Pass", "i"),
    # Secondary / 10th
    (r'\b(?:10th(?:\s+pass)?|secondary\s+school|ssc|sslc|matriculation)\b',
     "10th Pass", "i"),
    # Bootcamp / Cert
    (r'\b(?:bootcamp|certificat(?:ion|e))\b',
     "Bootcamp / Professional Certification", "i"),
]

# Degree field mapping for India disciplines.
# Each entry is (pattern, field, flags); see EDUCATION_PATTERNS for why the
# flag lives on the entry.
DEGREE_FIELD_PATTERNS: List[Tuple[str, str, str]] = [
    (r'\b(?:computer\s*science|information\s*technology)\b',
     "Computer Science / IT", "i"),
    (r'\(IT\)', "Computer Science / IT", ""),
    (r'\bIT\s+degree\b', "Computer Science / IT", "i"),
    (r'\bdegree\s+in\s+IT\b', "Computer Science / IT", "i"),
    (r'(?<![A-Za-z])(?:CS|CSE|SE)(?![A-Za-z])',
     "Computer Science / IT", ""),
    (r'\b(?:electronics|electrical|vlsi|embedded)\b',
     "Electronics / Electrical", "i"),
    (r'(?<![A-Za-z])(?:ECE|EEE)(?![A-Za-z])',
     "Electronics / Electrical", ""),
    (r'\b(?:mechanical|mechatronics|production\s+engineering)\b',
     "Mechanical Engineering", "i"),
    (r'\b(?:civil\s+engineering|civil)\b', "Civil Engineering", "i"),
    (r'\b(?:commerce|finance|accounting)\b', "Commerce / Finance", "i"),
    (r'(?<![A-Za-z])(?:BCom|MCom|CA|CFA)(?![A-Za-z])',
     "Commerce / Finance", ""),
    (r'\b(?:business\s+(?:administration|management|studies)|'
     r'management\s+studies)\b', "Business Administration", "i"),
    (r'(?<![A-Za-z])(?:MBA|BBA)(?![A-Za-z])',
     "Business Administration", ""),
    (r'\b(?:arts|humanities|literature|history|political\s+science|sociology)\b',
     "Arts / Humanities", "i"),
    (r'\b(?:life\s+sciences?|biology|biotechnology|biochemistry|pharmacy)\b',
     "Life Sciences", "i"),
    (r'\b(?:data\s+science|data\s+analytics|statistics)\b',
     "Data Science / Analytics", "i"),
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATE / TIMESTAMP EXPERIENCE EXTRACTION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

# Patterns for date ranges like "Jun 2021 – Present", "2019 – 2022", "Jan'20 - Mar'22"
_DATE_RANGE_PATTERN = re.compile(
    r"""
    (?:
        # Month-Year (start)
        (?P<sm>[A-Za-z]{3,9})[\s'\-\.]*(?P<sy>\d{4})
        |
        # Year only (start)
        (?P<sy2>\d{4})
    )
    \s*[-–—/to]+\s*
    (?:
        # Present / Current / Ongoing
        (?P<end_present>present|current|ongoing|now|till\s+date|till\s+now)
        |
        # Month-Year (end)
        (?P<em>[A-Za-z]{3,9})[\s'\-\.]*(?P<ey>\d{4})
        |
        # Year only (end)
        (?P<ey2>\d{4})
    )
    """,
    re.VERBOSE | re.IGNORECASE
)

# Internship section markers
_INTERNSHIP_MARKERS = re.compile(
    r'\b(intern(?:ship)?|trainee|apprentice|industrial\s+training|summer\s+project|winter\s+project)\b',
    re.IGNORECASE
)

# ── Section headings ───────────────────────────────────────────────────────
# Almost every resume separates the timeline of paid work from the timeline
# of study.  Recognising the headings lets us tell the two apart, so a degree
# like "B.Sc Nursing  2013 - 2016" stops being counted as six years of work.
#
# The order matters: the first pattern that matches a heading line wins.
_SECTION_PATTERNS = [
    ("internship", re.compile(
        r"^\s*(?:\w+\s+){0,2}internships?(?:\s+experience)?\s*:?\s*$", re.IGNORECASE)),
    ("experience", re.compile(
        r"^\s*(?:(?:work|professional|relevant|employment|career|job|industry|"
        r"general)\s+)?experience(?:\s+history)?\s*:?\s*$", re.IGNORECASE)),
    ("education", re.compile(
        r"^\s*(?:education(?:al\s+background)?|academic\s+(?:background|"
        r"qualifications?|record)?|qualifications?|degrees?|"
        r"training(?:\s+and\s+education)?)\s*:?\s*$", re.IGNORECASE)),
    ("certifications", re.compile(
        r"^\s*certifications?\s*:?\s*$", re.IGNORECASE)),
    ("projects", re.compile(
        r"^\s*(?:academic\s+|key\s+|major\s+)?projects?\s*:?\s*$", re.IGNORECASE)),
]

# Sections whose date ranges describe study or credentials, not employment.
_NON_WORK_SECTIONS = ("education", "certifications")


def _parse_month_year(month_str: Optional[str], year_str: Optional[str]) -> Optional[datetime]:
    """Converts a matched month (name or None) and year string to a datetime."""
    if not year_str:
        return None
    try:
        year = int(year_str.strip())
        if year < 1980 or year > datetime.now().year + 1:
            return None
        month = 1
        if month_str:
            month = _MONTH_MAP.get(month_str.strip().lower()[:3], 1)
        return datetime(year, month, 1)
    except (ValueError, AttributeError):
        return None


def _extract_date_ranges(text: str) -> List[Dict]:
    """
    Returns list of dicts:  {start: datetime, end: datetime, is_present: bool}
    by scanning all date-range patterns in the text.
    """
    ranges = []
    for m in _DATE_RANGE_PATTERN.finditer(text):
        g = m.groupdict()
        # Parse start
        start = _parse_month_year(g.get("sm"), g.get("sy") or g.get("sy2"))
        # Parse end
        is_present = bool(g.get("end_present"))
        end = datetime.now() if is_present else _parse_month_year(
            g.get("em"), g.get("ey") or g.get("ey2")
        )
        if start and end and end >= start:
            ranges.append({"start": start, "end": end, "is_present": is_present})
    return ranges


def _split_into_sections(text: str) -> List[Dict]:
    """
    Splits a resume into sections using its headings.

    Returns a list of {"name", "start", "end"} where "name" is the section
    name (``None`` for the untitled preamble) and the offsets delimit the
    section's text.  A heading line belongs to the section it opens.
    """
    sections: List[Dict] = []
    current_name: Optional[str] = None
    current_start = 0

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        matched_name: Optional[str] = None
        if stripped and len(stripped) <= 60:
            for name, pattern in _SECTION_PATTERNS:
                if pattern.match(stripped):
                    matched_name = name
                    break

        if matched_name is not None:
            # Close the previous section just before the heading line.
            if current_start < offset:
                sections.append({
                    "name": current_name,
                    "start": current_start,
                    "end": offset,
                })
            current_name = matched_name
            current_start = offset

        offset += len(line)

    if current_start < len(text):
        sections.append({
            "name": current_name,
            "start": current_start,
            "end": len(text),
        })

    return sections


def _section_at(sections: List[Dict], position: int) -> Optional[str]:
    """Returns the name of the section containing ``position``."""
    for section in sections:
        if section["start"] <= position < section["end"]:
            return section["name"]
    return None


def _ranges_near_internship(text: str) -> List[Dict]:
    """
    Classifies every date range in the text as work, internship, or neither.

    The classification is section-aware: ranges under an EDUCATION or
    CERTIFICATIONS heading describe study, not employment, and are dropped
    entirely.  A range counts as an internship when it sits under an
    INTERNSHIP heading or when the line it appears on (or the line right
    above it) names an internship — which is how resumes normally read,
    e.g. "Nursing Intern - City Care Centre   Jan 2016 - May 2016".
    """
    sections = _split_into_sections(text)
    lines = text.splitlines(keepends=True)

    # Offsets at which each line starts, so a character position can be
    # mapped back to a line index.
    line_starts: List[int] = []
    running = 0
    for line in lines:
        line_starts.append(running)
        running += len(line)

    def line_index_of(position: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= position:
                lo = mid
            else:
                hi = mid - 1
        return lo

    all_ranges: List[Dict] = []
    for m in _DATE_RANGE_PATTERN.finditer(text):
        g = m.groupdict()
        start = _parse_month_year(g.get("sm"), g.get("sy") or g.get("sy2"))
        is_present = bool(g.get("end_present"))
        end = datetime.now() if is_present else _parse_month_year(
            g.get("em"), g.get("ey") or g.get("ey2")
        )
        if not (start and end and end >= start):
            continue

        section = _section_at(sections, m.start())
        if section in _NON_WORK_SECTIONS:
            continue  # Study dates, not employment.

        idx = line_index_of(m.start())
        context = lines[idx] if idx < len(lines) else ""
        if idx > 0:
            context = lines[idx - 1] + " " + context

        is_internship = (
            section == "internship"
            or bool(_INTERNSHIP_MARKERS.search(context))
        )

        all_ranges.append({
            "start": start, "end": end,
            "is_present": is_present,
            "is_internship": is_internship
        })
    return all_ranges


def _calculate_experience_from_ranges(
    ranges: List[Dict]
) -> Tuple[float, int]:
    """
    Merges overlapping date ranges and returns
    (professional_years, internship_months).

    Professional experience is reported in years (1 decimal place) because
    that is how the role profiles express their thresholds; internships are
    reported in whole months, because a placement is naturally measured in
    months and the UI asks for it in months.
    """
    professional = [r for r in ranges if not r.get("is_internship")]
    internships  = [r for r in ranges if r.get("is_internship")]

    def merged_months(range_list: List[Dict]) -> float:
        """Merge overlapping intervals and sum months."""
        if not range_list:
            return 0.0
        sorted_r = sorted(range_list, key=lambda x: x["start"])
        merged = [sorted_r[0].copy()]
        for r in sorted_r[1:]:
            if r["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], r["end"])
            else:
                merged.append(r.copy())
        return float(sum(
            (r["end"].year - r["start"].year) * 12
            + (r["end"].month - r["start"].month)
            for r in merged
        ))

    professional_years = round(max(0.0, merged_months(professional) / 12.0), 1)
    internship_months = int(round(max(0.0, merged_months(internships))))
    return professional_years, internship_months


# ── Fallback: text-regex experience extractor (if no date ranges found) ─────
def _extract_experience_years_from_text(text: str) -> Optional[float]:
    """
    Extracts a stated experience claim like '5+ years of experience' as a float.
    Used when no date-stamp ranges are found.
    """
    patterns = [
        r'(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s*'
        r'(?:experience|work\s+experience|professional\s+experience|industry\s+experience)',
        r'(?:experience|worked\s+for)\s*(?:of)?\s*:?\s*(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)',
        r'(\d{1,2}(?:\.\d)?)\s*\+\s*(?:years?|yrs?)',
        r'(?:total|overall)\s+experience\s*:?\s*(\d{1,2}(?:\.\d)?)'
    ]
    found = []
    for pattern in patterns:
        for match in re.findall(pattern, text.lower()):
            try:
                yr = float(match)
                if 0 <= yr <= 40:
                    found.append(yr)
            except ValueError:
                pass
    return max(found) if found else None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EDUCATION EXTRACTION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_education_level(text: str) -> str:
    """
    Identifies the highest level of education using India-aware patterns.
    Returns the most specific matched label (ordered highest → lowest).

    Each pattern carries its own case-sensitivity flag, so abbreviated
    degrees ("B.Sc", "MBA", "MCA") are matched case-sensitively while
    spelled-out degrees ("Bachelor of Science") are matched case-insensitively.
    """
    for pattern, label, flags in EDUCATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE if flags == "i" else 0):
            return label
    return "Not Specified"


def extract_degree_field(text: str) -> str:
    """Best-effort extraction of the candidate's degree specialisation."""
    for pattern, field, flags in DEGREE_FIELD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE if flags == "i" else 0):
            return field
    return ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SKILLS EXTRACTION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_skills_from_text(text: str) -> List[str]:
    """
    Scans input text for known skills using boundary-aware regex.
    Returns deduplicated, sorted canonical skill names.
    """
    found: set = set()
    # Pad text to ease boundary matching at start/end
    text_lower = f" {text.lower()} "

    # Sort by length desc so longer phrases are matched before substrings
    sorted_skills = sorted(TECH_SKILLS_TAXONOMY, key=len, reverse=True)
    multi_word = [s for s in sorted_skills if " " in s or "." in s or "/" in s or "-" in s]
    single_word = [s for s in sorted_skills if s not in multi_word]

    for skill in multi_word:
        escaped = re.escape(skill)
        if re.search(rf'(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])', text_lower):
            found.add(CANONICAL_SKILL_MAP.get(skill, skill.title()))

    for skill in single_word:
        escaped = re.escape(skill)
        if re.search(rf'\b{escaped}\b', text_lower):
            found.add(CANONICAL_SKILL_MAP.get(skill, skill.title()))

    return sorted(found)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FILE TEXT EXTRACTION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_text_from_file(file_obj, filename: str) -> str:
    """
    Extracts raw text from PDF, DOCX, TXT, or CSV upload.
    PDF: tries pdfplumber first (better layout), falls back to PyPDF2.
    All errors return a non-empty string describing the problem.
    """
    ext = filename.lower()
    text = ""

    try:
        if ext.endswith(".pdf"):
            text = _extract_pdf_text(file_obj)

        elif ext.endswith(".docx"):
            if docx is None:
                return "python-docx not installed — cannot parse DOCX file."
            doc = docx.Document(file_obj)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext.endswith(".csv"):
            if pd is not None:
                df = pd.read_csv(file_obj)
                text = df.to_string()
            else:
                raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
                text = raw.decode("utf-8", errors="ignore")

        else:  # .txt and fallback
            raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
            text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")

    except Exception as exc:
        return f"Error parsing {filename}: {exc}"

    return text


def _extract_pdf_text(file_obj) -> str:
    """
    Attempts pdfplumber (preserves columns/layout better than PyPDF2).
    Falls back to PyPDF2 if pdfplumber is absent or fails.
    """
    # Make the stream seekable for two-pass attempts
    raw_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw_bytes, str):
        return raw_bytes  # Already plain text, unusual edge case

    pages_text: List[str] = []

    # ── Pass 1: pdfplumber ──────────────────────────────────────────────────
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text(x_tolerance=2, y_tolerance=3)
                    if extracted:
                        pages_text.append(extracted)
            if pages_text:
                return "\n".join(pages_text)
        except Exception:
            pages_text = []  # Fall through to PyPDF2

    # ── Pass 2: PyPDF2 ─────────────────────────────────────────────────────
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            if pages_text:
                return "\n".join(pages_text)
        except Exception as exc:
            return f"PDF extraction failed: {exc}"

    return "No PDF library available (install pdfplumber or PyPDF2)."


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PARSE ENTRY POINT                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_resume(file_or_text, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a resume and returns a complete profile dict.  ALL numeric fields
    default to 0 (never None); all string fields default to "Not Specified".

    Returns
    -------
    {
        "skills"             : List[str]   — canonical skill names
        "experience_years"   : float       — professional years (0 if unknown)
        "internship_months"  : int         — internship duration in months (0 if none)
        "education"          : str         — highest degree label
        "degree_field"       : str         — specialisation / discipline
        "raw_text_preview"   : str
        "word_count"         : int
        "raw_text"           : str
    }
    """
    # ── 1. Obtain raw text ──────────────────────────────────────────────────
    if filename and hasattr(file_or_text, "read"):
        raw_text = extract_text_from_file(file_or_text, filename)
    else:
        raw_text = str(file_or_text)

    # ── 2. Skills ───────────────────────────────────────────────────────────
    skills = extract_skills_from_text(raw_text)

    # ── 3. Experience (timestamp-first, text-regex fallback) ─────────────────
    date_ranges = _ranges_near_internship(raw_text)
    professional_yrs: float = 0.0
    internship_months: int = 0

    if date_ranges:
        professional_yrs, internship_months = _calculate_experience_from_ranges(date_ranges)

    # If timestamp parsing gave 0 professional years, try text-stated claim
    if professional_yrs == 0.0:
        stated = _extract_experience_years_from_text(raw_text)
        if stated is not None:
            professional_yrs = float(stated)

    # ── 4. Education ─────────────────────────────────────────────────────────
    education = extract_education_level(raw_text)
    degree_field = extract_degree_field(raw_text)

    # ── 5. Return complete profile ────────────────────────────────────────────
    words = raw_text.split()
    return {
        "skills": skills,
        "experience_years": professional_yrs,        # 0.0 default, never None
        "internship_months": internship_months,      # separate internship count
        "education": education,
        "degree_field": degree_field,
        "raw_text_preview": raw_text[:300].strip() + ("..." if len(raw_text) > 300 else ""),
        "word_count": len(words),
        "raw_text": raw_text
    }
