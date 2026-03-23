import os
import streamlit as st
from typing import List, Dict, Any

# -----------------------------
# LLM PROVIDER CONFIGURATION
# -----------------------------
# Supported providers: "google", "openai", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", _get_secret("OPENAI_API_KEY", ""))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", _get_secret("ANTHROPIC_API_KEY", ""))

# -----------------------------
# GOOGLE GEMINI CONFIG
# -----------------------------
def get_google_ai_token() -> str:
    """Get Google Gemini API key."""
    token = _get_secret("GOOGLE_AI_API_KEY", "")
    if token and len(token) > 10:
        return token
    token = os.getenv("GOOGLE_AI_API_KEY", "")
    if token and len(token) > 10:
        return token
    return ""

GOOGLE_AI_API_KEY = get_google_ai_token()

GOOGLE_AI_MODELS = {
    "gemini_pro": "gemini-1.5-pro-latest",
    "gemini_flash": "gemini-1.5-flash-latest",
    "gemini_pro_002": "gemini-1.5-pro-002",
    "flash_20_lite": "gemini-2.0-flash-lite",
    "flash_20": "gemini-2.0-flash",
    "flash_25_lite": "gemini-2.5-flash-lite",
}

# OpenAI and Anthropic defaults
OPENAI_MODELS = {
    "main": os.getenv("OPENAI_MODEL_MAIN", "gpt-4o-mini"),
    "chat": os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"),
}

ANTHROPIC_MODELS = {
    "main": os.getenv("ANTHROPIC_MODEL_MAIN", "claude-3-5-sonnet-20241022"),
    "chat": os.getenv("ANTHROPIC_MODEL_CHAT", "claude-3-5-sonnet-20241022"),
}

# Unified model names
if LLM_PROVIDER == "google":
    MAIN_MODEL = GOOGLE_AI_MODELS["flash_20"]
    SCORING_MODEL = GOOGLE_AI_MODELS["flash_20"]
    CHATBOT_MODEL = GOOGLE_AI_MODELS["flash_20"]
    DATA_EXTRACTION_MODEL = GOOGLE_AI_MODELS["flash_20"]
elif LLM_PROVIDER == "openai":
    MAIN_MODEL = OPENAI_MODELS["main"]
    SCORING_MODEL = OPENAI_MODELS["main"]
    CHATBOT_MODEL = OPENAI_MODELS["chat"]
    DATA_EXTRACTION_MODEL = OPENAI_MODELS["main"]
elif LLM_PROVIDER == "anthropic":
    MAIN_MODEL = ANTHROPIC_MODELS["main"]
    SCORING_MODEL = ANTHROPIC_MODELS["main"]
    CHATBOT_MODEL = ANTHROPIC_MODELS["chat"]
    DATA_EXTRACTION_MODEL = ANTHROPIC_MODELS["main"]
else:
    MAIN_MODEL = GOOGLE_AI_MODELS["flash_25_lite"]
    SCORING_MODEL = GOOGLE_AI_MODELS["flash_25_lite"]
    CHATBOT_MODEL = GOOGLE_AI_MODELS["flash_25_lite"]
    DATA_EXTRACTION_MODEL = GOOGLE_AI_MODELS["flash_25_lite"]

# -----------------------------
# FILE / SCORING CONFIG
# -----------------------------
UPLOAD_FOLDER = "./data/uploads"
CACHE_PATH = "./data/cache"
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB

SCORING_WEIGHTS = {
    "relevance_score": 0.70,
    "experience_match": 0.20,
    "skills_match": 0.10,
}

GEMINI_SETTINGS = {
    "data_extraction": {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2000,
        "timeout": 45,
    },
    "relevance_scoring": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 800,
        "timeout": 30,
    },
    "chatbot": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 60,
        "max_output_tokens": 400,
        "timeout": 25,
        "system_instruction": """You are an expert HR consultant and resume screening specialist.
You are hiring for M Group Services, which delivers essential infrastructure services across
water, energy, transport (highways, rail and aviation) and telecoms in the UK and Ireland,
with a strong safety-first and client/customer-centric culture.

Provide specific, actionable, professional advice about candidates and hiring. When you compare
candidates, be explicit about sector fit, HSE mindset, and experience on regulated UK/IE
infrastructure projects.""",
    },
}

MIN_SCORE_THRESHOLD = 0.3
TOP_CANDIDATES = 15

# -----------------------------
# JOB TEMPLATES (M GROUP)
# -----------------------------
JOB_TEMPLATES = {
    "MGroup Site Engineer - Water & Highways": """
Site Engineer role for M Group Services on water / highways civils projects in the UK and Ireland.
- Experience on water, utilities or major civils pipeline / highways schemes
- Strong setting-out, surveying and QA skills on trunk mains, drainage and structures
- Ability to coordinate plant, materials, workforce and subcontractors on live sites
- Working knowledge of CDM, SHEQ and environmental requirements on regulated infrastructure
- Competent with drawings, ITPs, as-builts and AutoCAD-style records
- Comfortable liaising with clients, local authorities, utilities and third parties
- HNC/HND or degree in Civil Engineering or similar; site-based mindset
""",
    "MGroup Project Manager - Substations & Energy": """
Project Manager role delivering electrical transmission / substation projects for M Group Services.
- Proven experience leading multi-disciplinary M&E or civils projects in the energy sector
- Background in National Grid / DNO frameworks and UK utilities environments
- Strong NEC contract, commercial and risk management capability
- Full project lifecycle responsibility for time, cost, quality and safety performance
- Able to manage design, construction, commissioning and stakeholder engagement
- Deep understanding of UK H&S regulations (HSWA, CDM, EAWR) and permit systems
- Professional qualification in engineering / construction and recognised PM credential (e.g. APM)
""",
    "MGroup Construction / Site Manager - Infrastructure": """
Construction / Site Manager role delivering water, energy, highways or rail infrastructure.
- Track record managing civils delivery teams on complex linear or site-based works
- Planning and sequencing works, coordinating multiple crews and subcontractors
- Enforcing safety-first culture and high standards of quality and environmental performance
- Experience with temporary works, traffic management and working near live services
- Comfortable producing and reviewing RAMS, permits, progress reports and forecasts
- Strong client and stakeholder interface skills on regulated infrastructure programmes
""",
    "MGroup Delivery Manager - Utilities Programmes": """
Delivery Manager role overseeing programmes of utility infrastructure works.
- Experience delivering multi-site works across water, energy or telecoms frameworks
- Programme-level planning, resource allocation and performance management
- Ability to drive productivity, efficiency and right-first-time delivery
- Strong commercial awareness and experience with target-cost / NEC-style contracts
- Skilled at client reporting, risk management and continuous improvement initiatives
""",
    "MGroup Foreman / Supervisor - Civils & Utilities": """
Working Foreman / Supervisor role on civils and utilities projects.
- Hands-on supervision of gangs delivering earthworks, drainage, pipelines or highways works
- Ensuring safe systems of work, tool-box talks and daily briefings
- Checking quality, setting out information and adherence to design
- Supporting site engineers and managers with records, progress and problem solving
""",
    "MGroup Service Support / Coordination": """
Service Support / Coordination role within M Group Services.
- Coordinating work orders, schedules, street works or permits for field delivery teams
- Supporting customer and client communication for planned and reactive works
- Updating systems, trackers and reports for programme delivery
- Familiarity with UK utilities, highways or telecoms environments beneficial
""",
}

# -----------------------------
# PROMPTS
# -----------------------------
GEMINI_PROMPTS = {
    "data_extraction": """You are an expert resume parser. Extract structured information from this resume text.

Resume Text:
{resume_text}

Extract JSON only with the following structure:
{{
  "candidate_name": "...",
  "contact_info": {{
     "email": "...",
     "phone": "...",
     "location": "...",
     "linkedin": "..."
  }},
  "professional_summary": "...",
  "work_experience": [
    {{
      "position": "...",
      "company": "...",
      "duration": "...",
      "key_responsibilities": ["..."],
      "achievements": ["..."]
    }}
  ],
  "education": [
    {{
      "degree": "...",
      "institution": "...",
      "graduation_year": "...",
      "gpa": "..."
    }}
  ],
  "technical_skills": ["..."],
  "certifications": ["..."],
  "total_years_experience": 0,
  "key_achievements": ["..."],
  "languages": ["..."],
  "soft_skills": ["..."]
}}
""",
    "relevance_scoring": """You are an expert HR consultant evaluating candidates for M Group Services.

M Group Services delivers essential infrastructure services across water, energy,
transport (highways, rail and aviation) and telecoms in the UK and Ireland, with a
safety-first, client- and customer-centric culture.

JOB REQUIREMENTS:
{job_description}

CANDIDATE DATA (JSON):
{candidate_data}

Return ONLY JSON with:
{{
  "overall_score": 0-100,
  "category_scores": {{
      "technical_skills": 0-100,
      "experience_relevance": 0-100,
      "education": 0-100,
      "career_progression": 0-100,
      "soft_skills": 0-100,
      "cultural_fit": 0-100
  }},
  "strengths": ["..."],
  "concerns": ["..."],
  "recommendation": "Strong Match / Good Match / Moderate Match / Poor Match",
  "key_reasons": "...",
  "interview_focus": ["..."],
  "role_fit_analysis": "...",
  "growth_potential": "..."
}}
""",
    "detailed_analysis": """You are a senior HR consultant assessing a candidate for an M Group Services role.

M Group works on regulated infrastructure across water, energy, highways, rail & aviation
and telecoms in the UK and Ireland. Explicitly discuss:
- Sector fit for these markets
- Health, safety and environmental mindset
- Understanding of UK regulations (CDM, NEC, etc.)
- Ability to work with clients, utilities, local authorities

JOB POSITION:
{job_title}

JOB REQUIREMENTS:
{job_description}

CANDIDATE DATA:
{candidate_data}

Provide a detailed plain-text analysis following this structure:
1. Candidate Overview
2. Key Strengths
3. Areas for Consideration / Gaps
4. Competency Analysis (technical, HSE, leadership)
5. Interview Recommendations
6. Overall Recommendation & Fit
7. Growth Potential
""",
    "comprehensive_chat": """You are an expert HR consultant and resume screening specialist for M Group Services.

CONTEXT:
{context}

USER QUESTION:
{question}

Answer professionally and concretely, focusing on:
- Which candidates best fit which M Group roles
- Sector experience (water, energy, highways, rail, aviation, telecom)
- HSE/CDM mindset and experience on regulated UK/IE infrastructure
- Trade-offs between candidates and recommended interview shortlists.
""",
}

# -----------------------------
# SKILL CATEGORIES
# -----------------------------
SKILL_CATEGORIES = {
    "civil_engineering": [
        "civil engineering", "site engineer", "setting out", "surveying", "earthworks",
        "drainage", "structures", "trunk main", "pipeline", "temporary works", "autocad"
    ],
    "utilities_infrastructure": [
        "water", "wastewater", "clean water", "potable water", "sewerage",
        "electricity transmission", "distribution", "substation", "overhead line",
        "gas distribution", "utilities", "power networks",
        "highways", "roads", "motorway",
        "rail", "railway", "airport", "aviation",
        "telecom", "telecommunications", "fibre", "fiber", "fibre optic"
    ],
    "hse_compliance": [
        "health and safety", "hse", "sheq", "cdm", "cdm regulations",
        "risk assessment", "method statement", "rams",
        "nebosh", "iosh", "smsts", "sssts",
        "permit to work", "ptw", "safe system of work", "environmental management"
    ],
    "project_controls": [
        "nec", "nec3", "nec4", "contract management",
        "commercial management", "programme management", "planning",
        "ms project", "primavera", "p6",
        "cost control", "budgeting", "forecasting", "risk management"
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "project management", "client management", "stakeholder management"
    ],
}

M_GROUP_CONTEXT: Dict[str, Any] = {
    "employer_name": "M Group Services",
    "markets": ["water", "energy", "highways", "rail & aviation", "telecom"],
    "regions": ["United Kingdom", "Republic of Ireland"],
    "culture": [
        "safety-first",
        "client and customer-centric",
        "regulated utilities and critical infrastructure",
    ],
}

# -----------------------------
# DIRECTORY & INFO HELPERS
# -----------------------------
def create_directories():
    for d in [UPLOAD_FOLDER, CACHE_PATH, "./data/temp_uploads", "./logs"]:
        os.makedirs(d, exist_ok=True)

def get_model_info() -> Dict[str, Any]:
    if LLM_PROVIDER == "google":
        provider_name = "Google Gemini"
        api_ok = bool(GOOGLE_AI_API_KEY)
    elif LLM_PROVIDER == "openai":
        provider_name = "OpenAI"
        api_ok = bool(OPENAI_API_KEY)
    else:
        provider_name = "Anthropic Claude"
        api_ok = bool(ANTHROPIC_API_KEY)

    return {
        "deployment_type": "🤖 Multi-Provider LLM (Google/OpenAI/Anthropic)",
        "main_model": {
            "name": MAIN_MODEL,
            "full_name": MAIN_MODEL,
            "provider": provider_name,
            "type": "Large Language Model",
            "status": "✅ Ready" if api_ok else "❌ No API Key",
        },
    }

def validate_setup() -> List[str]:
    issues: List[str] = []
    if LLM_PROVIDER == "google":
        if not GOOGLE_AI_API_KEY:
            issues.append("❌ Missing GOOGLE_AI_API_KEY for Gemini")
        else:
            issues.append("✅ Google Gemini configured")
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            issues.append("❌ Missing OPENAI_API_KEY for OpenAI")
        else:
            issues.append("✅ OpenAI configured")
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            issues.append("❌ Missing ANTHROPIC_API_KEY for Anthropic Claude")
        else:
            issues.append("✅ Anthropic Claude configured")
    else:
        issues.append(f"❌ Unknown LLM_PROVIDER={LLM_PROVIDER}")
    return issues

def get_gemini_prompt(template_key: str, **kwargs) -> str:
    template = GEMINI_PROMPTS.get(template_key, "")
    return template.format(**kwargs)

def check_api_health() -> Dict[str, Any]:
    if LLM_PROVIDER == "google":
        api_ok = bool(GOOGLE_AI_API_KEY)
    elif LLM_PROVIDER == "openai":
        api_ok = bool(OPENAI_API_KEY)
    else:
        api_ok = bool(ANTHROPIC_API_KEY)

    return {
        "provider": LLM_PROVIDER,
        "available": api_ok,
        "model": MAIN_MODEL,
    }

MULTI_ANALYSIS_SETTINGS = {
    "max_resumes_per_batch": 20,
    "max_jds_per_batch": 10,
    "max_concurrent_analyses": 50,
    "progress_update_interval": 1,
    "result_caching_enabled": True,
    "comprehensive_chat_context_limit": 5000,
    "score_matrix_display_limit": 15,
}

EXPORT_SETTINGS = {
    "csv_filename_template": "multi_analysis_{timestamp}.csv",
    "include_detailed_analysis": True,
    "include_score_breakdown": True,
    "include_metadata": True,
}
