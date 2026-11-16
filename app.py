"""
Skill-to-Job Matching Platform — v2
====================================
AI-Powered Career Recommendation Engine built with Streamlit.

Workflow:
1. Resume Upload  →  auto-extraction of skills, experience & education
2. Lightweight NLP (regex + India-aware taxonomy)
3. Role Compatibility Scoring (Top 4 Roles)
4. Job Openings Recommendations
"""

import os
import sys
from typing import Any, Dict

import streamlit as st

# ── Ensure package-relative imports work ────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.keyword_extractor import (
    parse_resume,
    TECH_SKILLS_TAXONOMY,
    CANONICAL_SKILL_MAP,
)
from utils.matcher import rank_top_roles
from utils.job_fetcher import fetch_jobs_for_role, SUPPORTED_COUNTRIES

# ── Shared UI vocabulary ─────────────────────────────────────────────────────
EDU_OPTIONS = [
    "Not Specified",
    "10th Pass",
    "12th Pass",
    "ITI Certificate",
    "Bootcamp / Professional Certification",
    "Diploma",
    "Associate's Degree",
    "Bachelor's (B.A.)", "Bachelor's (B.Com)", "Bachelor's (BCA)",
    "Bachelor's (B.Sc)", "Bachelor's (BBA)", "Bachelor's (B.E.)",
    "Bachelor's (B.Tech)", "Bachelor's Degree",
    "Master's (M.A.)", "Master's (M.Com)", "Master's (MCA)",
    "Master's (M.Sc)", "Master's (MBA)", "Master's (M.E.)",
    "Master's (M.Tech)", "Master's Degree",
    "Doctorate (Ph.D.)",
]

GENDER_OPTIONS = ["Prefer not to say", "Female", "Male", "Other"]

# Maps the UI label to the token the scoring engine understands.
GENDER_MAP = {
    "Prefer not to say": "neutral",
    "Female": "female",
    "Male": "male",
    "Other": "neutral",
}

SCOPE_DOMESTIC = "Domestic (India)"
SCOPE_INTERNATIONAL = "International"

AGE_DISCLAIMER = (
    "**How age & gender are used:** age and gender only modulate *suitability* "
    "(physical demand of a role, and any statutory preference such as female staff "
    "in nursing or male staff for certain driving posts). They are a small part of the "
    "score (4 % and 3 %) and never remove a role from your list."
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skill-to-Job Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = os.path.join(CURRENT_DIR, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.caption(
        "Match your skills to market roles with lightweight, India-aware NLP "
        "and live job alignment."
    )
    st.markdown("---")
    st.caption(
        "Add your age, gender and preferred job market below — they only fine-tune "
        "suitability, they never hide a role from you."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Skill-to-Job Matching Platform</div>
    <div class="app-subtitle">Map candidate competencies to market roles with lightweight NLP and real-time job alignment.</div>
</div>
""", unsafe_allow_html=True)

# ── Initialise session defaults ───────────────────────────────────────────────
_DEFAULTS = {
    "extracted_skills": [],
    "user_experience": 0.0,
    "internship_months": 0,
    "user_education": "Not Specified",
    "degree_field": "",
    "user_age": 0.0,
    "user_gender": "Prefer not to say",
    "job_scope": SCOPE_DOMESTIC,
    "target_country": "India",
    "source_mode": "manual",
    "analyzed": False,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ── Input Tabs ────────────────────────────────────────────────────────────────
tab_upload, tab_manual = st.tabs([
    "Upload Resume / CV",
    "Manual Skill Entry",
])

def _stamp_demographics(prefix: str) -> None:
    """Marks a tab as the one whose demographics were edited most recently.

    Streamlit renders every tab on every rerun, so both tab blocks exist at
    once. Each keeps its own widget keys and the most recently edited one
    wins — that way a candidate can set age on the resume tab, then switch to
    the manual tab to change the target country, and both survive.
    """
    st.session_state["demo_source"] = prefix


def _render_demographics(prefix: str) -> None:
    """Renders the age / gender / job-scope inputs shared by both tabs."""
    st.markdown("##### Personal Details & Job Search Scope")

    current = _resolve_demographics()

    c_demo1, c_demo2, c_demo3 = st.columns(3)
    with c_demo1:
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=70,
            value=int(current["age"]),
            step=1,
            key=f"demo_{prefix}_age",
            on_change=_stamp_demographics,
            args=(prefix,),
            help="Leave at 0 if you prefer not to share. Age only affects physical-demand suitability.",
        )
    with c_demo2:
        gender = st.selectbox(
            "Gender",
            options=GENDER_OPTIONS,
            index=(
                GENDER_OPTIONS.index(current["gender"])
                if current["gender"] in GENDER_OPTIONS
                else 0
            ),
            key=f"demo_{prefix}_gender",
            on_change=_stamp_demographics,
            args=(prefix,),
            help="Used only where a role carries a statutory preference. It never removes a role.",
        )
    with c_demo3:
        scope = st.radio(
            "Job search scope",
            options=[SCOPE_DOMESTIC, SCOPE_INTERNATIONAL],
            index=(
                [SCOPE_DOMESTIC, SCOPE_INTERNATIONAL].index(current["scope"])
                if current["scope"] in [SCOPE_DOMESTIC, SCOPE_INTERNATIONAL]
                else 0
            ),
            key=f"demo_{prefix}_scope",
            on_change=_stamp_demographics,
            args=(prefix,),
        )

    if scope == SCOPE_INTERNATIONAL:
        country = st.text_input(
            "Target country",
            value=current["country"],
            key=f"demo_{prefix}_country",
            on_change=_stamp_demographics,
            args=(prefix,),
            help="Type any country, e.g. 'UAE', 'UK', 'Canada'. Listings are filtered to that market.",
        )
        st.caption(
            f"Searching listings for **{country.strip() or 'all international markets'}** — "
            f"live search links will point at that country's job boards."
        )
    else:
        country = "India"
        st.caption("Searching listings for **India**.")

    # NOTE: the widget keys (demo_<prefix>_age/gender/scope/country) are owned
    # by Streamlit once the widgets above exist, so they must not be written
    # here — assigning to them raises StreamlitWidgetAlreadyInstantiatedError.
    # The on_change callback (_stamp_demographics) already records which tab
    # was edited last, and _resolve_demographics() reads the widget keys.

    st.markdown(AGE_DISCLAIMER)


def _resolve_demographics() -> Dict[str, Any]:
    """Returns the demographics currently in force and syncs session state.

    The most recently edited tab wins; before any edit, the resume tab leads
    if a resume is loaded and the manual tab leads otherwise.
    """
    source = st.session_state.get("demo_source")
    if not source:
        source = "upload" if st.session_state.get("source_mode") == "resume" else "manual"

    def _from(prefix: str) -> Dict[str, Any]:
        return {
            "age": st.session_state.get(f"demo_{prefix}_age", 0),
            "gender": st.session_state.get(f"demo_{prefix}_gender", "Prefer not to say"),
            "scope": st.session_state.get(f"demo_{prefix}_scope", SCOPE_DOMESTIC),
            "country": st.session_state.get(f"demo_{prefix}_country", "United Arab Emirates"),
        }

    values = _from(source)

    st.session_state["user_age"] = float(values["age"] or 0)
    st.session_state["user_gender"] = values["gender"]
    st.session_state["job_scope"] = values["scope"]
    st.session_state["target_country"] = (
        "India" if values["scope"] == SCOPE_DOMESTIC else (values["country"] or "India")
    )
    return values


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Resume Upload  (primary, preferred)
# ════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("##### Upload Candidate File")
    st.info(
        "**Skills, experience years, internship duration, and education are extracted automatically.**  "
        "No manual entry needed after upload."
    )

    uploaded_file = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT, CSV",
        type=["pdf", "docx", "txt", "csv"],
        help="The parser extracts skills, work-history timestamps, internship durations, and education level.",
    )

    st.markdown("---")
    _render_demographics("upload")

    if uploaded_file is not None:
        # Only parse when the file itself changes.  Re-parsing on every rerun
        # would overwrite the values the candidate edited under
        # "Fine-tune extracted values" (the expander below).
        if st.session_state.get("parsed_file_name") != uploaded_file.name:
            with st.spinner("Analysing resume — extracting skills, timeline, and education…"):
                parsed = parse_resume(uploaded_file, uploaded_file.name)

            # ── Store in session state (auto-fill everything) ─────────────────
            st.session_state["extracted_skills"] = parsed.get("skills", [])
            st.session_state["user_experience"]  = parsed.get("experience_years", 0.0)
            st.session_state["internship_months"] = parsed.get("internship_months", 0)
            st.session_state["user_education"]   = parsed.get("education", "Not Specified")
            st.session_state["degree_field"]     = parsed.get("degree_field", "")
            st.session_state["source_mode"]      = "resume"
            st.session_state["parsed_file_name"] = uploaded_file.name
            st.session_state["parsed_word_count"] = parsed.get("word_count", 0)
    else:
        # The upload was cleared — hand control back to the manual tab.
        st.session_state["parsed_file_name"] = None
        if st.session_state.get("source_mode") == "resume":
            st.session_state["source_mode"] = "manual"

    if uploaded_file is not None:
        exp_val     = st.session_state["user_experience"]
        intern_val  = st.session_state["internship_months"]
        edu_val     = st.session_state["user_education"]
        field_val   = st.session_state["degree_field"]
        skills_list = st.session_state["extracted_skills"]

        # ── Success banner ────────────────────────────────────────────────────
        st.success(
            f"Parsed **{uploaded_file.name}** — "
            f"{st.session_state.get('parsed_word_count', 0)} words"
        )

        # ── Summary metrics ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Detected Skills", len(skills_list))
        with c2:
            st.metric("Professional Exp.", f"{exp_val} yrs")
        with c3:
            st.metric("Internship Exp.", f"{intern_val} months")
        with c4:
            edu_display = edu_val if edu_val != "Not Specified" else "Not detected"
            st.metric("Education", edu_display)

        if field_val:
            st.caption(f"Degree field detected: **{field_val}**")

        # ── Skill tags ────────────────────────────────────────────────────────
        if skills_list:
            st.markdown("**Identified Skills:**")
            tags_html = " ".join(
                f"<span class='skill-tag skill-tag-matched'>{s}</span>"
                for s in skills_list
            )
            st.markdown(f"<div class='tag-container'>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.warning(
                "No technical skills matched the built-in taxonomy.  "
                "Switch to **Manual Skill Entry** or ensure your resume uses standard skill names."
            )

        # ── Inline adjustments (optional, not required) ───────────────────────
        with st.expander("Fine-tune extracted values (optional)"):
            st.caption("The fields below were auto-detected from your resume. Edit only if an extracted value is wrong.")
            adj_exp    = st.slider(
                "Professional experience (years):",
                0.0, 30.0, float(exp_val), 0.5,
                key="adj_exp_input",
            )
            adj_intern = st.number_input(
                "Internship experience (months):",
                min_value=0,
                value=int(intern_val),
                step=1,
                key="adj_intern_input",
            )
            adj_edu = st.selectbox(
                "Highest education level:",
                options=EDU_OPTIONS,
                index=EDU_OPTIONS.index(edu_val) if edu_val in EDU_OPTIONS else 0,
                key="adj_edu_input",
            )
            if st.button("Apply adjustments", key="apply_adjustments_btn"):
                st.session_state["user_experience"]  = adj_exp
                st.session_state["internship_months"] = int(adj_intern)
                st.session_state["user_education"]   = adj_edu
                st.success("Adjustments saved.")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Manual Entry  (fallback)
# ════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown("##### Manual Competency Specification")

    st.markdown("---")
    _render_demographics("manual")
    st.markdown("---")

    # Only accept manual input if no resume has been uploaded this session
    if st.session_state["source_mode"] == "resume":
        st.info("A resume was uploaded — values are auto-filled. Switch to **Upload Resume** tab to view them, or clear the upload to enter manually.")
    else:
        # Build canonical skill options
        skill_options = sorted({CANONICAL_SKILL_MAP.get(s, s.title()) for s in TECH_SKILLS_TAXONOMY})

        # Safe defaults that exist in the canonicalized set
        safe_defaults = [s for s in ["Python", "Sql", "Machine Learning", "Data Analysis", "Git"] if s in skill_options]

        manual_skills = st.multiselect(
            "Select your core skills (technical and non-technical):",
            options=skill_options,
            default=safe_defaults,
            key="manual_skills_input",
        )

        col_exp, col_edu = st.columns(2)
        with col_exp:
            manual_exp    = st.slider(
                "Professional experience (years):",
                0.0, 20.0, 2.0, 0.5,
                key="manual_exp_input",
            )
            manual_intern = st.number_input(
                "Internship experience (months):",
                min_value=0,
                value=0,
                step=1,
                key="manual_intern_input",
            )
        with col_edu:
            manual_edu = st.selectbox(
                "Highest education level:",
                options=EDU_OPTIONS,
                index=EDU_OPTIONS.index("Bachelor's (B.Tech)"),
                key="manual_edu_input",
            )

        st.session_state["extracted_skills"] = manual_skills
        st.session_state["user_experience"]  = manual_exp
        st.session_state["internship_months"] = int(manual_intern)
        st.session_state["user_education"]   = manual_edu
        st.session_state["source_mode"]      = "manual"

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  ANALYSE BUTTON
# ════════════════════════════════════════════════════════════════════════════
if st.button("Analyse & Match Top Roles", type="primary", use_container_width=True):
    st.session_state["analyzed"] = True

# ════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.get("analyzed"):
    active_skills  = st.session_state.get("extracted_skills", [])
    active_exp     = st.session_state.get("user_experience", 0.0)
    active_intern  = st.session_state.get("internship_months", 0)
    active_edu     = st.session_state.get("user_education", "Not Specified")
    active_age     = float(st.session_state.get("user_age", 0.0) or 0.0)
    active_gender  = GENDER_MAP.get(
        st.session_state.get("user_gender", "Prefer not to say"),
        "neutral",
    )
    active_scope   = st.session_state.get("job_scope", SCOPE_DOMESTIC)
    active_country = st.session_state.get("target_country", "India")

    if not active_skills:
        st.error("Please provide at least one skill to calculate job compatibility.")
    else:
        st.markdown("### Top 4 Role Compatibility Matches")
        scope_bits = []
        if active_intern > 0:
            scope_bits.append(
                f"**{active_exp} yrs** professional + **{active_intern} months** internship (counted at 40 %)"
            )
        else:
            scope_bits.append(f"**{active_exp} yrs** experience")
        scope_bits.append(f"Education: {active_edu}")
        if active_age > 0:
            scope_bits.append(f"Age: {int(active_age)}")
        if active_gender != "neutral":
            scope_bits.append(f"Gender: {active_gender}")
        scope_bits.append(
            f"Market: {active_country if active_scope == SCOPE_INTERNATIONAL else 'India (domestic)'}"
        )
        st.caption("Scoring basis: " + " | ".join(scope_bits))

        ranked_roles = rank_top_roles(
            user_skills=active_skills,
            user_experience=active_exp,
            user_education=active_edu,
            internship_months=active_intern,
            user_age=active_age,
            user_gender=active_gender,
            top_n=4,
        )

        for idx, role in enumerate(ranked_roles):
            score = role["compatibility_score"]
            badge_cls = "score-high" if score >= 75 else ("score-mid" if score >= 50 else "score-low")

            st.markdown(f"""
            <div class="match-card">
                <div class="match-card-header">
                    <div>
                        <span style="font-size:0.8rem;color:#64748b;font-weight:600;">RANK #{idx+1} • {role['category'].upper()}</span>
                        <div class="match-role-title">{role['title']}</div>
                    </div>
                    <span class="score-badge {badge_cls}">{score}% Match</span>
                </div>
                <div style="font-size:0.9rem;color:#475569;margin-bottom:0.75rem;">
                    {role['description']}
                </div>
                <div style="font-size:0.8rem;color:#64748b;">
                    Typical salary (India): <strong>₹{role.get('typical_salary_lpa','—')} LPA</strong>
                    &nbsp;|&nbsp; Min. experience: <strong>{role['min_experience']} yr(s)</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(score / 100.0)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.caption("Matched Skills:")
                if role["matched_skills"]:
                    tags = " ".join(
                        f"<span class='skill-tag skill-tag-matched'>{s}</span>"
                        for s in role["matched_skills"]
                    )
                else:
                    tags = "<span style='font-size:0.8rem;color:#94a3b8;'>None</span>"
                st.markdown(f"<div class='tag-container'>{tags}</div>", unsafe_allow_html=True)
            with col_m2:
                st.caption("Recommended Skills to Learn:")
                if role["missing_skills"]:
                    tags = " ".join(
                        f"<span class='skill-tag skill-tag-missing'>{s}</span>"
                        for s in role["missing_skills"]
                    )
                else:
                    tags = "<span style='font-size:0.8rem;color:#15803d;'>All core skills covered!</span>"
                st.markdown(f"<div class='tag-container'>{tags}</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

        # ── Job Openings ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Curated Live Job Openings")
        st.caption("Aggregated opportunities aligned with your top matched role profile.")

        c_filter1, c_filter2, c_filter3 = st.columns([2, 1, 1])
        with c_filter1:
            selected_role = st.selectbox(
                "Filter openings by ranked role:",
                options=[r["title"] for r in ranked_roles],
                index=0,
            )
        with c_filter2:
            work_type = st.selectbox("Work Arrangement:", ["All", "Remote", "Hybrid", "On-site"])
        with c_filter3:
            market_options = ["All"] + [c for c in SUPPORTED_COUNTRIES]
            default_market = "India" if active_scope == SCOPE_DOMESTIC else active_country
            if default_market not in market_options:
                default_market = "All"
            listing_market = st.selectbox(
                "Listing market:",
                options=market_options,
                index=market_options.index(default_market),
            )

        jobs = fetch_jobs_for_role(
            selected_role,
            work_type_filter=work_type,
            limit=5,
            country_filter=None if listing_market == "All" else listing_market,
            country=active_country,
        )

        if jobs:
            for job in jobs:
                st.markdown(f"""
                <div class="job-listing-card">
                    <div>
                        <div style="font-size:1.05rem;font-weight:600;color:#0f172a;">{job.get('job_title')}</div>
                        <div class="job-meta-row">
                            <span>{job.get('company')}</span>
                            <span>{job.get('location')}</span>
                            <span>{job.get('work_type')}</span>
                            <span>{job.get('salary_range')}</span>
                            <span>{job.get('country', 'India')}</span>
                        </div>
                        <div style="font-size:0.85rem;color:#475569;margin-top:0.5rem;">
                            <strong>Skills:</strong> {job.get('skills_required')}
                        </div>
                    </div>
                    <div>
                        <a href="{job.get('apply_url','#')}" target="_blank" class="apply-button-link">Apply Now</a>
                        <a href="{job.get('live_search_linkedin','#')}" target="_blank" class="apply-button-link" style="margin-top:0.4rem;background:#0f172a;">Search LinkedIn</a>
                        <a href="{job.get('live_search_indeed','#')}" target="_blank" class="apply-button-link" style="margin-top:0.4rem;background:#2557a7;">Search Indeed</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(
                f"No specific openings found for '{selected_role}'"
                f"{f' in {listing_market}' if listing_market != 'All' else ''}"
                f" with '{work_type}' arrangement."
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Skill-to-Job Matching Platform • India-aware NLP • Deployable on Streamlit Cloud")
