import json
from typing import Dict, Any

from config import GEMINI_SETTINGS, get_gemini_prompt
from models import llm_provider

from parser import EnhancedResumeParser
from config import (
    get_gemini_prompt,
    GEMINI_SETTINGS,
    check_api_health,
    SCORING_WEIGHTS,
)
from models import llm_provider


class GoogleAIManager:
    """LLM-backed manager for data extraction, scoring, and chat."""

    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        """Call LLM to extract structured data from resume text."""
        prompt = get_gemini_prompt("data_extraction", resume_text=resume_text)
        system_prompt = "You are an expert resume parser. Return ONLY valid JSON."

        raw = llm_provider.generate_text(
            system_prompt=system_prompt,
            user_content=prompt,
            max_tokens=GEMINI_SETTINGS["data_extraction"]["max_output_tokens"],
        )

        # Guard: LLM returned nothing or only whitespace
        if not raw or not raw.strip():
            return {
                "success": False,
                "error": "LLM returned empty response in extract_resume_data",
                "raw": raw,
            }

        try:
            data = json.loads(raw)
            return {
                "success": True,
                "data": data,
                "raw": raw,  # keep for debugging
            }
        except Exception as e:
            # IMPORTANT: propagate both error and raw text
            return {
                "success": False,
                "error": f"JSON parse error in extract_resume_data: {e}",
                "raw": raw,
            }
            
    def score_candidate(self, job_description: str, candidate_json: Dict[str, Any]) -> Dict[str, Any]:
        """Score a candidate vs a JD using the LLM, returning structured scores."""
        prompt = get_gemini_prompt(
            "relevance_scoring",
            job_description=job_description,
            candidate_data=json.dumps(candidate_json),
        )
        system_prompt = (
            "You are an expert HR consultant scoring candidates for M Group Services. "
            "Return ONLY valid JSON following the requested schema."
        )

        raw = llm_provider.generate_text(
            system_prompt=system_prompt,
            user_content=prompt,
            max_tokens=GEMINI_SETTINGS["relevance_scoring"]["max_output_tokens"],
        )

        # Guard: empty / whitespace response
        if not raw or not raw.strip():
            return {
                "success": False,
                "error": "LLM returned empty response in score_candidate",
                "raw": raw,
            }

        try:
            data = json.loads(raw)
        except Exception as e:
            return {
                "success": False,
                "error": f"JSON parse error in score_candidate: {e}",
                "raw": raw,
            }

        # Compute composite score safely
        overall = float(data.get("overall_score", 0)) / 100.0
        cat = data.get("category_scores", {}) or {}
        relevance = float(cat.get("experience_relevance", 0)) / 100.0
        skills = float(cat.get("technical_skills", 0)) / 100.0

        composite = (
            0.7 * overall +
            0.2 * relevance +
            0.1 * skills
        )

        return {
            "success": True,
            "scores": {
                "composite_score": composite,
                "relevance_score": relevance,
                "skills_score": skills,
                "experience_score": relevance,  # or a separate field if you have it
            },
            "analysis": data,
            "raw": raw,
        }

    def detailed_analysis(self, job_title: str, job_description: str, candidate_json: Dict[str, Any]) -> str:
        prompt = get_gemini_prompt(
            "detailed_analysis",
            job_title=job_title,
            job_description=job_description,
            candidate_data=json.dumps(candidate_json),
        )
        sys_prompt = "You are a senior HR consultant for M Group Services."
        return llm_provider.generate_text(
            system_prompt=sys_prompt,
            user_content=prompt,
            max_tokens=1200,
        )

    def comprehensive_chat_response(self, question: str, context: str) -> Dict[str, Any]:
        sys_prompt = GEMINI_SETTINGS["chatbot"]["system_instruction"]
        messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        answer = llm_provider.chat(
            system_prompt=sys_prompt,
            messages=messages,
            max_tokens=GEMINI_SETTINGS["chatbot"]["max_output_tokens"],
        )
        return {"success": True, "answer": answer}


class GoogleAIScreeningEngine:
    """Engine used by the Streamlit app, now provider-agnostic via llm_provider."""

    def __init__(self):
        self.parser = EnhancedResumeParser()
        self.google_ai_manager = GoogleAIManager()
        self.processing_stats = {
            "google_ai_calls": 0,
            "avg_processing_time": 0.0,
        }

    def get_system_status(self) -> Dict[str, Any]:
        health = check_api_health()
        return {
            "ready_for_screening": bool(health.get("available", False)),
            "google_ai_status": {
                "connection_status": {
                    "connected": bool(health.get("available", False)),
                    "error": None if health.get("available", False) else "API key missing or invalid",
                },
                "models": {
                    "main_model": health.get("model", "unknown"),
                },
            },
            "processing_stats": self.processing_stats,
        }

    def process_single_resume(self, file_path: str, filename: str, job_description: str) -> Dict[str, Any]:
        """End-to-end processing for one resume vs one JD."""
        try:
            # 1) parse file → text (already done by your parser)
            parsed = self.parser.parse_resume(file_path, filename)
            if not parsed.get("success"):
                return {
                    "status": "failed",
                    "error": parsed.get("error", "Parser failed"),
                }

            resume_text = parsed.get("cleaned_text") or parsed.get("raw_text", "")

            # 2) Extract structured data via LLM
            extraction = self.extract_resume_data(resume_text)
            if not extraction.get("success"):
                return {
                    "status": "failed",
                    "error": extraction.get("error", "Extraction failed"),
                    "raw": extraction.get("raw"),
                }

            candidate_json = extraction.get("data", {})

            # 3) Score candidate via LLM
            scoring = self.score_candidate(job_description, candidate_json)
            if not scoring.get("success"):
                return {
                    "status": "failed",
                    "error": scoring.get("error", "Scoring failed"),
                    "raw": scoring.get("raw"),
                    "analysis": {
                        "extracted_data": candidate_json,
                    },
                }

            # 4) Successful path
            return {
                "status": "completed",
                "scores": scoring["scores"],
                "analysis": {
                    "extracted_data": candidate_json,
                    "scoring_details": scoring["analysis"],
                },
                "raw": scoring.get("raw"),
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": f"Unhandled exception in process_single_resume: {e}",
            }
