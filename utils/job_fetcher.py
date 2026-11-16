"""
Job Fetcher Utility
Fetches active job postings for recommended roles from the curated local dataset.

Matching is score-based rather than first-match-wins: a role title is tokenised
and every opening is ranked by IDF-weighted token overlap, with the job title
weighted far above the skills list, plus a phrase-containment bonus and a light
seniority nudge.

The IDF weighting is what stops generic title words from dominating. Tokens like
"engineer", "analyst", and "executive" occur in many openings, so they earn
little; distinctive tokens like "finance", "design", or "operations" earn a lot.
Without it, "Finance Analyst" ranked a data-analyst opening first purely because
both titles contain "analyst".
"""

import math
import os
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd

# Resolved path for sample_jobs.csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_jobs.csv")

# Columns the scoring logic depends on.
REQUIRED_COLUMNS = ("job_title", "skills_required")

_TOKEN_RE = re.compile(r"[^a-z0-9+#.]+")

# Words that carry no role identity on their own.
_STOPWORDS = {"and", "the", "for", "with", "our", "core", "team"}

# Seniority vocabulary in a role title -> experience_level label in the dataset.
_SENIORITY_HINTS = {
    "junior": "Entry-Level",
    "associate": "Entry-Level",
    "trainee": "Entry-Level",
    "graduate": "Entry-Level",
    "senior": "Senior",
    "lead": "Senior",
    "principal": "Senior",
    "head": "Senior",
}

# Relative weight of a token found in skills_required vs. in the job title.
_SKILLS_WEIGHT = 0.30
# Bonus (in units of the corpus max IDF) when the whole role name is in the title.
_PHRASE_BONUS = 2.5
# Bonus (same units) when the opening's seniority matches the role's.
_SENIORITY_BONUS = 0.25

# Cached IDF table, invalidated when the dataset size changes.
_idf_cache: Optional[Tuple[int, Dict[str, float]]] = None

# Country -> Indeed domain. Indeed runs a per-country site, and the generic
# "www.indeed.com" mostly surfaces US postings, which is useless for a
# candidate searching in India or the UAE.
INDEED_DOMAINS: Dict[str, str] = {
    "India": "in.indeed.com",
    "United States": "www.indeed.com",
    "United Kingdom": "uk.indeed.com",
    "Canada": "ca.indeed.com",
    "Australia": "au.indeed.com",
    "New Zealand": "nz.indeed.com",
    "United Arab Emirates": "ae.indeed.com",
    "Saudi Arabia": "sa.indeed.com",
    "Qatar": "qa.indeed.com",
    "Kuwait": "kw.indeed.com",
    "Bahrain": "bh.indeed.com",
    "Oman": "om.indeed.com",
    "Singapore": "sg.indeed.com",
    "Malaysia": "my.indeed.com",
    "Indonesia": "id.indeed.com",
    "Thailand": "th.indeed.com",
    "Vietnam": "vn.indeed.com",
    "Philippines": "ph.indeed.com",
    "China": "cn.indeed.com",
    "Hong Kong": "hk.indeed.com",
    "Japan": "jp.indeed.com",
    "South Korea": "kr.indeed.com",
    "Germany": "de.indeed.com",
    "France": "fr.indeed.com",
    "Netherlands": "nl.indeed.com",
    "Spain": "es.indeed.com",
    "Italy": "it.indeed.com",
    "Switzerland": "ch.indeed.com",
    "Ireland": "ie.indeed.com",
    "Poland": "pl.indeed.com",
    "Sweden": "se.indeed.com",
    "Norway": "no.indeed.com",
    "Denmark": "dk.indeed.com",
    "Finland": "fi.indeed.com",
    "Brazil": "br.indeed.com",
    "Mexico": "mx.indeed.com",
    "South Africa": "za.indeed.com",
    "Nigeria": "ng.indeed.com",
    "Kenya": "ke.indeed.com",
    "Sri Lanka": "lk.indeed.com",
    "Bangladesh": "bd.indeed.com",
    "Nepal": "np.indeed.com",
    "Pakistan": "pk.indeed.com",
}

# Country -> LinkedIn geographic ID. LinkedIn takes these in the `geoId`
# query parameter; without one the search is silently restricted to the
# visitor's own region.
LINKEDIN_GEO_IDS: Dict[str, str] = {
    "India": "102713980",
    "United States": "103644278",
    "United Kingdom": "101165590",
    "Canada": "101174742",
    "Australia": "101452733",
    "New Zealand": "101452733",
    "United Arab Emirates": "104116990",
    "Saudi Arabia": "101274290",
    "Qatar": "104022890",
    "Kuwait": "103063795",
    "Bahrain": "101464670",
    "Oman": "101063539",
    "Singapore": "102454443",
    "Malaysia": "102499462",
    "Indonesia": "102311430",
    "Thailand": "105076800",
    "Vietnam": "102056428",
    "Philippines": "102311430",
    "China": "102291662",
    "Hong Kong": "102291662",
    "Japan": "101355337",
    "South Korea": "101748736",
    "Germany": "101282230",
    "France": "105015130",
    "Netherlands": "102890719",
    "Spain": "105015130",
    "Italy": "105015130",
    "Switzerland": "101453733",
    "Ireland": "101165590",
    "Poland": "105072130",
    "Sweden": "101549249",
    "Norway": "101722218",
    "Denmark": "105115390",
    "Finland": "102454443",
    "Brazil": "106057199",
    "Mexico": "101807712",
    "South Africa": "104035573",
    "Nigeria": "101314322",
    "Kenya": "101009733",
    "Sri Lanka": "101065882",
    "Bangladesh": "101056248",
    "Nepal": "101012061",
    "Pakistan": "101049155",
}

# Fallbacks used when a country is not in the tables above.
DEFAULT_INDEED_DOMAIN = "www.indeed.com"
DEFAULT_LINKEDIN_GEO_ID = ""

# Free-text country names typed by the user -> canonical dataset country.
COUNTRY_ALIASES: Dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "dubai": "United Arab Emirates",
    "abu dhabi": "United Arab Emirates",
    "saudi": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "rsa": "South Africa",
    "nz": "New Zealand",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "republic of korea": "South Korea",
    "korea": "South Korea",
    "prc": "China",
    "mainland china": "China",
    "people's republic of china": "China",
    "uae dubai": "United Arab Emirates",
}

# The set of countries the UI offers in the job-scope picker.
SUPPORTED_COUNTRIES: List[str] = sorted(INDEED_DOMAINS.keys())


def _country_key(country: Optional[str]) -> str:
    """Normalises a country name for table lookups (case/space tolerant).

    Handles the abbreviations candidates actually type ("UAE", "UK", "USA")
    before falling back to a straight name match.
    """
    if not country:
        return ""
    cleaned = re.sub(r"\s+", " ", str(country)).strip()
    for known in INDEED_DOMAINS:
        if known.lower() == cleaned.lower():
            return known
    alias = COUNTRY_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    return cleaned


def _tokens(text: Any) -> List[str]:
    """Lowercases and splits text into comparison tokens, dropping stopwords.

    Two-character tokens are kept deliberately — "hr", "ui", "ux" and "ai" are
    the most distinctive part of the roles that use them.
    """
    return [
        t for t in _TOKEN_RE.split(str(text).lower())
        if len(t) >= 2 and t not in _STOPWORDS
    ]


def load_dataset_jobs() -> List[Dict[str, Any]]:
    """Loads default sample jobs from local CSV dataset."""
    if not os.path.exists(DATA_PATH):
        print(f"sample_jobs.csv not found at {DATA_PATH}")
        return []
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        # A malformed row (e.g. an unquoted comma inside a salary figure) lands
        # here. Name the file and the reason so it is not mistaken for an empty
        # dataset.
        print(f"Could not parse {DATA_PATH}: {type(e).__name__}: {e}")
        return []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"{DATA_PATH} is missing required column(s): {', '.join(missing)}")
        return []
    return df.to_dict(orient="records")


def _title_idf(jobs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Builds an IDF table over job titles so common title words score lower."""
    global _idf_cache
    n = len(jobs)
    if _idf_cache is not None and _idf_cache[0] == n:
        return _idf_cache[1]

    doc_freq: Counter = Counter()
    for job in jobs:
        doc_freq.update(set(_tokens(job.get("job_title", ""))))

    idf = {tok: math.log(1.0 + n / df) for tok, df in doc_freq.items()}
    _idf_cache = (n, idf)
    return idf


def _score_job(
    job: Dict[str, Any],
    role_tokens: List[str],
    role_phrase: str,
    desired_level: Optional[str],
    idf: Dict[str, float],
    max_idf: float,
) -> float:
    """Scores one opening against a role title. Returns 0.0 for no overlap."""
    title_tokens = set(_tokens(job.get("job_title", "")))
    skill_tokens = set(_tokens(job.get("skills_required", "")))

    score = 0.0
    for tok in role_tokens:
        weight = idf.get(tok, max_idf)
        if tok in title_tokens:
            score += weight
        elif tok in skill_tokens:
            score += _SKILLS_WEIGHT * weight

    if score <= 0.0:
        return 0.0

    # Whole role name present in the title, e.g. "Data Analyst" in
    # "Business Data Analyst".
    if role_phrase and role_phrase in " ".join(_tokens(job.get("job_title", ""))):
        score += _PHRASE_BONUS * max_idf

    if desired_level and desired_level.lower() in str(job.get("experience_level", "")).lower():
        score += _SENIORITY_BONUS * max_idf

    return score


def build_search_links(
    job_title: str,
    location: str = "",
    country: str = "India",
) -> Dict[str, str]:
    """Generates direct live search links for LinkedIn and Indeed.

    The query is percent-encoded: titles such as "Finance Analyst - GST &
    Compliance" contain characters that would otherwise terminate the query
    string and silently drop the rest of the search terms.

    The target country picks the Indeed country domain and the LinkedIn
    geographic ID, so a candidate looking for jobs in the UAE does not end up
    on an India-only Indeed site.
    """
    query = quote_plus(f"{job_title} {location}".strip())
    key = _country_key(country)
    indeed_domain = INDEED_DOMAINS.get(key, DEFAULT_INDEED_DOMAIN)
    geo_id = LINKEDIN_GEO_IDS.get(key, DEFAULT_LINKEDIN_GEO_ID)

    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={query}"
    if geo_id:
        linkedin_url += f"&geoId={geo_id}"
    return {
        "linkedin": linkedin_url,
        "indeed": f"https://{indeed_domain}/jobs?q={query}",
    }


def _with_search_links(
    job: Dict[str, Any],
    fallback_title: str,
    country: str = "India",
) -> Dict[str, Any]:
    """Copies a job record and attaches live LinkedIn/Indeed search URLs."""
    job_copy = dict(job)
    links = build_search_links(
        job.get("job_title", fallback_title),
        job.get("location", ""),
        country=job.get("country") or country,
    )
    job_copy["live_search_linkedin"] = links["linkedin"]
    job_copy["live_search_indeed"] = links["indeed"]
    return job_copy


def fetch_jobs_for_role(
    role_title: str,
    location_filter: Optional[str] = None,
    work_type_filter: Optional[str] = None,
    limit: int = 5,
    country_filter: Optional[str] = None,
    country: str = "India",
) -> List[Dict[str, Any]]:
    """
    Fetches job listings matching the specified role title, best match first.

    Openings are scored on IDF-weighted token overlap against the role title
    (see module docstring), then filtered by country, location and work
    arrangement. If nothing scores above zero, falls back to the leading
    openings in the dataset so the UI still has something to show.

    `country_filter` restricts results to openings tagged with that country
    ("all" or None means no restriction). `country` is the market the live
    search links should point at.
    """
    all_jobs = load_dataset_jobs()
    if not all_jobs:
        return []

    idf = _title_idf(all_jobs)
    max_idf = math.log(1.0 + len(all_jobs))

    role_tokens = _tokens(role_title)
    role_phrase = " ".join(role_tokens)
    desired_level = next(
        (lvl for tok, lvl in _SENIORITY_HINTS.items() if tok in role_tokens),
        None,
    )

    def _country_matches(job: Dict[str, Any]) -> bool:
        if not country_filter or country_filter.lower() == "all":
            return True
        target = _country_key(country_filter).lower()
        return _country_key(job.get("country", "")).lower() == target

    def _location_matches(job: Dict[str, Any]) -> bool:
        if not location_filter:
            return True
        return location_filter.lower() in str(job.get("location", "")).lower()

    def _work_type_matches(job: Dict[str, Any]) -> bool:
        if not work_type_filter or work_type_filter.lower() == "all":
            return True
        return work_type_filter.lower() in str(job.get("work_type", "")).lower()

    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for position, job in enumerate(all_jobs):
        # Country filter
        if not _country_matches(job):
            continue
        # Location filter
        if not _location_matches(job):
            continue
        # Work type filter
        if not _work_type_matches(job):
            continue

        score = _score_job(job, role_tokens, role_phrase, desired_level, idf, max_idf)
        if score > 0.0:
            # `position` keeps the sort stable and deterministic on ties.
            scored.append((score, position, job))

    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [_with_search_links(job, role_title, country) for _, _, job in scored[:limit]]

    # No scored matches — return leading general openings, honouring filters.
    fallback = []
    for job in all_jobs:
        if not _work_type_matches(job):
            continue
        if not _location_matches(job):
            continue
        if not _country_matches(job):
            continue
        fallback.append(_with_search_links(job, role_title, country))
        if len(fallback) >= limit:
            break
    return fallback


def fetch_all_ranked_jobs(
    top_roles: List[Dict[str, Any]],
    limit_per_role: int = 3,
    country_filter: Optional[str] = None,
    country: str = "India",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches job openings grouped by each ranked role in top_roles.
    Returns dictionary mapping role title to list of job records.
    """
    results = {}
    for role in top_roles:
        role_title = role.get("title", "")
        jobs = fetch_jobs_for_role(
            role_title,
            limit=limit_per_role,
            country_filter=country_filter,
            country=country,
        )
        results[role_title] = jobs
    return results
