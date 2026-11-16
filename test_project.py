import streamlit as st
import pandas as pd
print('Streamlit version:', st.__version__)
print('Pandas version:', pd.__version__)

# Test the core modules
from models.keyword_extractor import parse_resume, TECH_SKILLS_TAXONOMY, CANONICAL_SKILL_MAP
from utils.matcher import rank_top_roles
from utils.job_fetcher import load_dataset_jobs, fetch_jobs_for_role

# Test loading jobs
jobs = load_dataset_jobs()
print('Jobs loaded:', len(jobs))

# Test ranking with sample data
roles = rank_top_roles(['Python', 'JavaScript', 'Git'], 2.0, "Bachelor's (B.Tech)", 6, 25.0, 'male')
print('Roles ranked:', len(roles))

# Test job fetching
jobs2 = fetch_jobs_for_role('Software Engineer', limit=3)
print('Jobs fetched:', len(jobs2))

print('SUCCESS')