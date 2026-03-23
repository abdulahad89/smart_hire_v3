# models/google_screening_engine.py

import time
import json
from typing import Dict, Any, List

from parser import EnhancedResumeParser
from config import (
    GEMINI_SETTINGS,
    get_gemini_prompt,
    check_api_health,
    get_model_info,
    M_GROUP_CONTEXT,
)
from models import llm_provider


class GoogleAIManager:
    """
    Thin manager that app.py can call via:
      screening_engine.google_ai_manager.extract_resume_data(...)
      screening_engine.google_ai_manager.comprehensive_chat_response(...)
    It delegates to methods on the parent engine.
    """

    def __init__(self, engine: "GoogleAIScreeningEngine"):
        self.engine = engine

    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        return self.engine.extract_resume_data(resume_text)

    def comprehensive_chat_response(self, question: str, context: str) -> Dict[str, Any]:
        """
        High‑level chat over the entire portfolio, used by AI Chatbot tab.
        """
        try:
            # Build prompt from config
            prompt = get_gemini_prompt(
                "comprehensive_chat",
                question=question,
                context=context,
            )

            system_prompt = GEMINI_SETTINGS["chatbot"].get(
                "system_instruction",
                "You are an expert HR consultant.",
            )

            # Call unified chat helper
            start = time.time()
            answer = llm_provider.chat(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GEMINI_SETTINGS["chatbot"]["max_output_tokens"],
            )
            duration = time.time() - start
            self.engine._record_llm_call(duration)

            return {
                "success": True,
                "answer": answer,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class GoogleAIScreeningEngine:
    """
    Screening engine used by the Streamlit app.

    Responsibilities:
    - Parse resumes with EnhancedResumeParser
    - Call LLM for resume data extraction and scoring
    - Provide system status & basic metrics
    """

    def __init__(self) -> None:
        # Parser for PDFs / DOCXs
        self.parser = EnhancedResumeParser()

        # Manager object expected by app.py
        self.google_ai_manager = GoogleAIManager(self)

        # Simple stats for sidebar
        self.google_ai_calls = 0
        self.total_processing_time = 0.0

    # ------------------------------------------------------------------
    # LLM helpers & metrics
    # ------------------------------------------------------------------

    def _record_llm_call(self, duration_sec: float) -> None:
        self.google_ai_calls += 1
        self.total_processing_time += duration_sec

    # ------------------------------------------------------------------
    # LLM‑BASED RESUME EXTRACTION
    # ------------------------------------------------------------------

    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        """
        Call LLM to extract structured data from resume text.
        Returns dict with keys: success(bool), data(dict), raw(str) or error(str).
        """
        prompt = get_gemini_prompt("data_extraction", resume_text=resume_text)
        system_prompt = "You are an expert resume parser. Return ONLY valid JSON."

        start = time.time()
        raw = llm_provider.generate_text(
            system_prompt=system_prompt,
            user_content=prompt,
            max_tokens=GEMINI_SETTINGS["data_extraction"]["max_output_tokens"],
        )
        duration = time.time() - start
        self._record_llm_call(duration)

        if not raw or not raw.strip():
            return {
                "success": False,
                "error": "LLM returned empty response in extract_resume_data",
                "raw": raw,
            }

        try:
            data = json.loads(raw)
            return {"success": True, "data": data, "raw": raw}
        except Exception as e:
            return {
                "success": False,
                "error": f"JSON parse error in extract_resume_data: {e}",
                "raw": raw,
            }

    # ------------------------------------------------------------------
    # LLM‑BASED SCORING
    # ------------------------------------------------------------------

    def score_candidate(self, job_description: str, candidate_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a candidate vs a JD using the LLM.
        Returns dict with keys: success(bool), scores(dict), analysis(dict), raw(str) or error(str).
        """
        prompt = get_gemini_prompt(
            "relevance_scoring",
            job_description=job_description,
            candidate_data=json.dumps(candidate_json),
        )

        system_prompt = (
            "You are an expert HR consultant scoring candidates for M Group Services. "
            "Return ONLY valid JSON following the requested schema."
        )

        start = time.time()
        raw = llm_provider.generate_text(
            system_prompt=system_prompt,
            user_content=prompt,
            max_tokens=GEMINI_SETTINGS["relevance_scoring"]["max_output_tokens"],
        )
        duration = time.time() - start
        self._record_llm_call(duration)

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

        composite = 0.7 * overall + 0.2 * relevance + 0.1 * skills

        return {
            "success": True,
            "scores": {
                "composite_score": composite,
                "relevance_score": relevance,
                "skills_score": skills,
                "experience_score": relevance,
            },
            "analysis": data,
            "raw": raw,
        }

    # ------------------------------------------------------------------
    # END‑TO‑END: ONE RESUME + ONE JD
    # ------------------------------------------------------------------

    def process_single_resume(self, file_path: str, filename: str, job_description: str) -> Dict[str, Any]:
        """
        Full pipeline for one resume against one JD:
        parse -> LLM extract -> LLM score.
        """
        try:
            # 1) Parse resume file
            parsed = self.parser.parse_resume(file_path, filename)
            if not parsed.get("success"):
                return {
                    "status": "failed",
                    "error": parsed.get("error", "Parser failed"),
                }

            resume_text = parsed.get("cleaned_text") or parsed.get("raw_text", "")

            # 2) Extract structured info via LLM
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

            # 4) Success path
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
            # Make sure failures are visible in RAW LLM OUTPUT
            return {
                "status": "failed",
                "error": f"Unhandled exception in process_single_resume: {e}",
                "raw": {
                    "status": "failed",
                    "error": f"Unhandled exception in process_single_resume: {e}",
                },
            }

    # ------------------------------------------------------------------
    # SYSTEM STATUS FOR SIDEBAR
    # ------------------------------------------------------------------

    def get_system_status(self) -> Dict[str, Any]:
        """
        Return status dict used in the System Status sidebar/tab.

        Expected keys in app.py:
          - ready_for_screening (bool)
          - google_ai_status: {
                "connection_status": {"connected": bool, "error": str|None},
                "models": {"main_model": str}
            }
          - processing_stats: {
                "google_ai_calls": int,
                "avg_processing_time": float
            }
        """
        health = check_api_health()
        model_info = get_model_info()

        connected = health.get("available", False)
        main_model = health.get("model") or model_info["main_model"]["full_name"]

        avg_time = (
            self.total_processing_time / self.google_ai_calls
            if self.google_ai_calls > 0
            else 0.0
        )

        return {
            "ready_for_screening": connected,
            "google_ai_status": {
                "connection_status": {
                    "connected": connected,
                    "error": None if connected else "API key missing or invalid",
                },
                "models": {
                    "main_model": main_model,
                },
            },
            "processing_stats": {
                "google_ai_calls": self.google_ai_calls,
                "avg_processing_time": avg_time,
            },
            "employer_context": M_GROUP_CONTEXT,
        }
