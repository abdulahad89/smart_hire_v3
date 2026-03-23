import json
from typing import Dict, Any

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
        prompt = get_gemini_prompt("data_extraction", resume_text=resume_text)
        sys_prompt = "You are an expert resume parser. Return strictly valid JSON."
        raw = llm_provider.generate_text(
            system_prompt=sys_prompt,
            user_content=prompt,
            max_tokens=GEMINI_SETTINGS["data_extraction"]["max_output_tokens"],
        )

        # NEW: guard against empty response
        if not raw or not raw.strip():
            return {
                "success": False,
                "error": "LLM returned empty response for data_extraction",
                "raw": raw,
            }
        #End Here
        
        try:
            data = json.loads(raw)
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": f"JSON parse error: {e}", "raw": raw}

    def score_candidate(self, job_description: str, candidate_json: Dict[str, Any]) -> Dict[str, Any]:
        prompt = get_gemini_prompt(
            "relevance_scoring",
            job_description=job_description,
            candidate_data=json.dumps(candidate_json),
        )
        sys_prompt = "You are an expert HR consultant scoring candidates for M Group Services."
        raw = llm_provider.generate_text(
            system_prompt=sys_prompt,
            user_content=prompt,
            max_tokens=GEMINI_SETTINGS["relevance_scoring"]["max_output_tokens"],
        )

        # NEW: guard against empty response
        if not raw or not raw.strip():
            return {
                "success": False,
                "error": "LLM returned empty response for relevance_scoring",
                "raw": raw,
            }
        #End here

        try:
            data = json.loads(raw)
        except Exception as e:
            return {
                "success": False,
                "error": f"JSON parse error: {e}",
                "raw": raw,
            }

        overall = data.get("overall_score", 0) / 100.0
        cats = data.get("category_scores", {})
        rel = cats.get("technical_skills", 0) / 100.0
        exp = cats.get("experience_relevance", 0) / 100.0

        comp = (
            SCORING_WEIGHTS["relevance_score"] * overall
            + SCORING_WEIGHTS["experience_match"] * exp
            + SCORING_WEIGHTS["skills_match"] * rel
        )

        scores = {
            "composite_score": comp,
            "relevance_score": overall,
            "experience_score": exp,
            "skills_score": rel,
        }
        return {"success": True, "scores": scores, "raw": data}

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
        """Parse resume, extract structured data, score against JD."""
        parse_result = self.parser.parse_resume(file_path, filename)
        if not parse_result.get("success"):
            return {
                "status": "failed",
                "error": parse_result.get("error", "Unknown parsing error"),
            }

        resume_text = parse_result.get("cleaned_text", "")

        # Extract structured data via LLM
        extraction = self.google_ai_manager.extract_resume_data(resume_text)
        extracted_data = extraction.get("data", {}) if extraction.get("success") else {}

        # Score candidate
        scoring = self.google_ai_manager.score_candidate(job_description, extracted_data or {})
        if not scoring.get("success"):
            return {
                "status": "failed",
                "error": scoring.get("error", "Scoring failed"),
            }

        scores = scoring["scores"]

        return {
            "status": "completed",
            "scores": scores,
            "analysis": {
                "parsed_data": parse_result,
                "extracted_data": extracted_data,
                "scoring_raw": scoring.get("raw", {}),
            },
        }
