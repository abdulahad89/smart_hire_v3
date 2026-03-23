import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import List, Dict, Any

from parser import EnhancedResumeParser
from config import create_directories, JOB_TEMPLATES
from models.google_screening_engine import GoogleAIScreeningEngine

# ---------------------------------
# PAGE CONFIG & SESSION STATE
# ---------------------------------
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "screening_engine" not in st.session_state:
    st.session_state.screening_engine = None
if "resumes_data" not in st.session_state:
    st.session_state.resumes_data = []
if "job_descriptions" not in st.session_state:
    st.session_state.job_descriptions = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "chatbot_history" not in st.session_state:
    st.session_state.chatbot_history = []
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Resume Upload"


# ---------------------------------
# MAIN
# ---------------------------------
def main():
    st.title("SmartHire.AI")
    st.markdown("### Multi-Resume vs Multi-JD Analysis")

    create_directories()

    if st.session_state.screening_engine is None:
        with st.spinner(" Initializing AI system..."):
            try:
                engine = GoogleAIScreeningEngine()
                st.session_state.screening_engine = engine
                st.success(" AI system initialized!")
            except Exception as e:
                st.error(f" System initialization failed: {e}")
                st.info(" Check your API keys and LLM_PROVIDER")
                return

    render_sidebar()
    render_main_content()


# ---------------------------------
# SIDEBAR
# ---------------------------------
def render_sidebar():
    st.sidebar.markdown("# Navigation")
    for tab in [
        "Resume Upload",
        "Job Descriptions",
        "Analysis & Results",
        "AI Chatbot",
        "System Status",
    ]:
        if st.sidebar.button(tab, key=f"nav_{tab}", use_container_width=True):
            st.session_state.selected_tab = tab

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Actions**")
    if st.sidebar.button("Refresh System"):
        st.session_state.screening_engine = None
        st.rerun()
    if st.sidebar.button("Clear All Data"):
        st.session_state.resumes_data = []
        st.session_state.job_descriptions = []
        st.session_state.analysis_results = []
        st.session_state.chatbot_history = []
        st.rerun()
    if st.sidebar.button("Export Results") and st.session_state.analysis_results:
        export_comprehensive_results()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Current Session")
    st.sidebar.metric("Resumes Loaded", len(st.session_state.resumes_data))
    st.sidebar.metric("Job Descriptions", len(st.session_state.job_descriptions))
    st.sidebar.metric("Analyses Complete", len(st.session_state.analysis_results))


# ---------------------------------
# MAIN CONTENT ROUTING
# ---------------------------------
def render_main_content():
    tab = st.session_state.selected_tab
    if tab == "Resume Upload":
        render_resume_upload()
    elif tab == "Job Descriptions":
        render_job_descriptions()
    elif tab == "Analysis & Results":
        render_analysis_results()
    elif tab == "AI Chatbot":
        render_enhanced_chatbot()
    elif tab == "System Status":
        render_system_status()


# ---------------------------------
# RESUME UPLOAD
# ---------------------------------
def render_resume_upload():
    st.header("Resume Upload & Management")

    if not st.session_state.screening_engine:
        st.error("System not initialized. Please refresh.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Upload New Resumes")
        uploaded_files = st.file_uploader(
            "Select resume files",
            accept_multiple_files=True,
            type=["pdf", "docx", "doc", "txt"],
            help="Upload PDF, Word documents, or text files",
            key="resume_uploader",
        )
        if uploaded_files:
            if st.button("Process & Add Resumes", type="primary"):
                process_and_store_resumes(uploaded_files)

    with col2:
        st.subheader("Current Resume Bank")
        if st.session_state.resumes_data:
            st.success(f"{len(st.session_state.resumes_data)} resumes loaded")
            with st.expander("Resume Details"):
                for i, r in enumerate(st.session_state.resumes_data):
                    st.write(f"**{i+1}.** {r['filename']}")
                    name = r.get("extracted_data", {}).get("candidate_name")
                    if name:
                        st.write(f"   {name}")
        else:
            st.info("No resumes loaded yet")

    if st.session_state.resumes_data:
        st.markdown("---")
        st.subheader("Resume Bank Management")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Preview All Resumes"):
                st.info("Preview not implemented in this minimal build.")
        with c2:
            if st.button("Resume Statistics"):
                st.info("Statistics not implemented in this minimal build.")
        with c3:
            if st.button("Clear Resume Bank"):
                st.session_state.resumes_data = []
                st.rerun()


def save_temp_file(uploaded_file) -> str:
    temp_dir = "./data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def process_and_store_resumes(uploaded_files):
    engine = st.session_state.screening_engine
    progress = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded_files):
        status.text(f"Processing {f.name}...")
        temp_path = save_temp_file(f)

        try:
            if not hasattr(engine, "parser") or engine.parser is None:
                engine.parser = EnhancedResumeParser()
            parse_res = engine.parser.parse_resume(temp_path, f.name)
            if not parse_res.get("success"):
                st.error(f"Failed to parse {f.name}: {parse_res.get('error')}")
                continue

            raw_text = parse_res.get("cleaned_text", "")
            # Optional extra extraction via LLM
            extraction = engine.google_ai_manager.extract_resume_data(raw_text)
            extracted_data = extraction.get("data", {}) if extraction.get("success") else {}

            st.session_state.resumes_data.append(
                {
                    "filename": f.name,
                    "file_size": f.size,
                    "upload_time": datetime.now().isoformat(),
                    "raw_text": raw_text,
                    "parsed_data": parse_res,
                    "extracted_data": extracted_data,
                    "processing_status": "completed",
                }
            )
            st.success(f"{f.name} processed successfully")

        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        progress.progress((i + 1) / len(uploaded_files))

    status.empty()
    progress.empty()


# ---------------------------------
# JOB DESCRIPTIONS
# ---------------------------------
def render_job_descriptions():
    st.header("Job Descriptions Management")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Add New Job Description")
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Predefined Templates"],
            horizontal=True,
            key="jd_input_method",
        )
        title = st.text_input("Job Title", placeholder="e.g.,Site Engineer")

        # M Group metadata
        mgroup_sector = st.selectbox(
            "Sector (optional)",
            ["", "Water", "Energy", "Highways", "Rail & Aviation", "Telecom"],
        )
        mgroup_region = st.selectbox(
            "Primary region (optional)",
            ["", "England", "Scotland", "Wales", "Northern Ireland", "Republic of Ireland"],
        )
        mgroup_role_family = st.selectbox(
            "Role family (optional)",
            ["", "Site Engineer", "Construction Manager", "Foreman / Supervisor",
             "Project Manager", "Delivery Manager", "Service Support / Office"],
        )

        jd_text = ""
        if input_method == "Text Input":
            jd_text = st.text_area(
                "Job Description",
                height=180,
                placeholder="Paste or type the JD here ...",
            )
        else:
            template = st.selectbox("Select template:", list(JOB_TEMPLATES.keys()))
            jd_text = JOB_TEMPLATES[template]
            st.text_area("Template JD", value=jd_text, height=180, disabled=True)

        st.markdown("---")
        if st.button("Add Job Description", type="primary"):
            if not title.strip():
                st.error("Please enter a job title")
            elif not jd_text.strip():
                st.error("Please provide a job description")
            else:
                jd = {
                    "title": title.strip(),
                    "description": jd_text.strip(),
                    "input_method": input_method,
                    "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mgroup_sector": mgroup_sector or "",
                    "mgroup_region": mgroup_region or "",
                    "mgroup_role_family": mgroup_role_family or "",
                }
                st.session_state.job_descriptions.append(jd)
                st.success(f"Added job description: {title}")
                st.rerun()

    with c2:
        st.subheader("Current Job Descriptions")
        if not st.session_state.job_descriptions:
            st.info("No job descriptions added yet.")
        else:
            st.success(f"{len(st.session_state.job_descriptions)} JDs loaded")
            for i, jd in enumerate(st.session_state.job_descriptions):
                with st.expander(f"{i+1}. {jd['title']}"):
                    st.write(f"**Method:** {jd['input_method']}")
                    st.write(f"**Added:** {jd.get('created_time')}")
                    st.write(f"**Preview:** {jd['description'][:160]}...")
                    ctx = []
                    if jd.get("mgroup_sector"):
                        ctx.append(f"Sector: {jd['mgroup_sector']}")
                    if jd.get("mgroup_region"):
                        ctx.append(f"Region: {jd['mgroup_region']}")
                    if jd.get("mgroup_role_family"):
                        ctx.append(f"Role family: {jd['mgroup_role_family']}")
                    if ctx:
                        st.write("**Context:** " + " | ".join(ctx))
                    if st.button("Delete", key=f"delete_jd_{i}"):
                        st.session_state.job_descriptions.pop(i)
                        st.rerun()


def build_mgroup_job_context(jd: Dict[str, Any]) -> str:
    base = jd.get("description", "")
    bits = [
        "Employer: Delivering essential infrastructure across "
        "water, energy, highways, rail & aviation, and telecom in the UK & Ireland "
        "with a safety-first, client-centric culture."
    ]
    if jd.get("mgroup_sector"):
        bits.append(f"Sector: {jd['mgroup_sector']}.")
    if jd.get("mgroup_region"):
        bits.append(f"Region: {jd['mgroup_region']}.")
    if jd.get("mgroup_role_family"):
        bits.append(f"Role family: {jd['mgroup_role_family']}.")
    return base + "\n\n" + " ".join(bits)


# ---------------------------------
# ANALYSIS
# ---------------------------------
def render_analysis_results():
    st.header("Multi-Resume vs Multi-JD Analysis")
    if not st.session_state.resumes_data:
        st.warning("Please upload resumes first.")
        return
    if not st.session_state.job_descriptions:
        st.warning("Please add job descriptions first.")
        return

    #st.info(
    #    "Scoring is tuned for Utilities Services infrastructure roles "
    #    "across water, energy, highways, rail & aviation, and telecom."
    #)

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_resumes = st.multiselect(
            "Select Resumes",
            options=list(range(len(st.session_state.resumes_data))),
            default=list(range(len(st.session_state.resumes_data))),
            format_func=lambda i: st.session_state.resumes_data[i]["filename"],
        )
    with c2:
        selected_jds = st.multiselect(
            "Select Job Descriptions",
            options=list(range(len(st.session_state.job_descriptions))),
            default=list(range(len(st.session_state.job_descriptions))),
            format_func=lambda i: st.session_state.job_descriptions[i]["title"],
        )
    with c3:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Scoring", "Detailed Analysis", "Comprehensive Report"],
        )

    if st.button("Start Multi-Analysis", type="primary"):
        if not selected_resumes or not selected_jds:
            st.warning("Please select both resumes and job descriptions.")
        else:
            run_comprehensive_analysis(selected_resumes, selected_jds, analysis_depth)

    if st.session_state.analysis_results:
        st.markdown("---")
        render_comprehensive_results()


def run_comprehensive_analysis(selected_resumes, selected_jds, analysis_depth):
    engine = st.session_state.screening_engine
    results = []
    total = len(selected_resumes) * len(selected_jds)
    progress = st.progress(0)
    status = st.empty()
    count = 0

    for ri in selected_resumes:
        r = st.session_state.resumes_data[ri]
        for ji in selected_jds:
            jd = st.session_state.job_descriptions[ji]
            count += 1
            progress.progress(count / total)
            status.text(f"Analyzing {r['filename']} vs {jd['title']} ({count}/{total})")
            try:
                temp_path = create_temp_file_from_data(r["raw_text"], r["filename"])
                jd_aug = build_mgroup_job_context(jd)
                res = engine.process_single_resume(temp_path, r["filename"], jd_aug)
                res["resume_index"] = ri
                res["jd_index"] = ji
                res["jd_title"] = jd["title"]
                res["analysis_depth"] = analysis_depth
                res["analysis_timestamp"] = datetime.now().isoformat()
                results.append(res)
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Analysis failed for {r['filename']} vs {jd['title']}: {e}")

    st.session_state.analysis_results = results
    progress.empty()
    status.empty()
    st.success(f"Completed {len(results)} analyses!")


def create_temp_file_from_data(text_data: str, filename: str) -> str:
    temp_dir = "./data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"temp_{filename}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text_data)
    return path


# ---------------------------------
# RESULTS RENDERING
# ---------------------------------
def create_kpi_card(title, value, delta=None):
    html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.9rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    ">
      <h3 style="margin: 0; font-size: 1.6rem;">{value}</h3>
      <p style="margin: 0; font-size: 0.9rem;">{title}</p>
      {f'<p style="margin: 0.3rem 0 0; font-size: 0.8rem;">{delta}</p>' if delta else ""}
    </div>
    """
    return html


def compute_mgroup_insights(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen: Dict[int, Dict[str, Any]] = {}
    for r in successful_results:
        idx = r.get("resume_index")
        if idx is None or idx in seen:
            continue
        text = (st.session_state.resumes_data[idx].get("raw_text") or "").lower()
        stats = {
            "infra": any(
                k in text
                for k in [
                    "water", "wastewater", "substation", "pipeline", "highways",
                    "rail", "aviation", "telecom", "fibre", "fiber"
                ]
            ),
            "hse": any(
                k in text
                for k in [
                    "health and safety", "hse", "sheq", "cdm", "nebosh",
                    "iosh", "rams", "permit to work"
                ]
            ),
            "uk_ie": any(
                k in text
                for k in [
                    "united kingdom", "uk", "england", "scotland", "wales",
                    "northern ireland", "republic of ireland", "dublin", "london"
                ]
            ),
            "projects": text.count("project"),
        }
        seen[idx] = stats

    total = max(len(seen), 1)
    return {
        "infra_experienced_candidates": sum(1 for v in seen.values() if v["infra"]),
        "hse_focused_candidates": sum(1 for v in seen.values() if v["hse"]),
        "uk_ie_candidates": sum(1 for v in seen.values() if v["uk_ie"]),
        "avg_projects_per_candidate": sum(v["projects"] for v in seen.values()) / total,
    }


def render_comprehensive_results():
    st.subheader("Analysis Results Dashboard")
    results = st.session_state.analysis_results
    success = [r for r in results if r.get("status") == "completed"]
    if not success:
        st.info("No successful analyses yet.")
        return

    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(create_kpi_card("Total Analyses", str(len(results))), unsafe_allow_html=True)
    with c2:
        st.markdown(
            create_kpi_card("Successful", str(len(success)),
                            delta=f"{len(success)/len(results)*100:.1f}% success"),
            unsafe_allow_html=True,
        )
    with c3:
        avg = np.mean([r["scores"]["composite_score"] for r in success])
        st.markdown(create_kpi_card("Average Score", f"{avg:.3f}"), unsafe_allow_html=True)
    with c4:
        top = max([r["scores"]["composite_score"] for r in success])
        st.markdown(create_kpi_card("Top Score", f"{top:.3f}"), unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        strong = len([r for r in success if r["scores"]["composite_score"] >= 0.8])
        st.markdown(
            create_kpi_card("Excellent Matches", str(strong), delta="Score ≥ 0.8"),
            unsafe_allow_html=True,
        )
    with c6:
        good = len([r for r in success if 0.6 <= r["scores"]["composite_score"] < 0.8])
        st.markdown(
            create_kpi_card("Good Matches", str(good), delta="Score 0.6 - 0.8"),
            unsafe_allow_html=True,
        )
    with c7:
        avg_rel = np.mean([r["scores"].get("relevance_score", 0) for r in success])
        st.markdown(
            create_kpi_card("Avg Relevance", f"{avg_rel:.3f}"),
            unsafe_allow_html=True,
        )
    with c8:
        avg_exp = np.mean([r["scores"].get("experience_score", 0) for r in success])
        st.markdown(
            create_kpi_card("Avg Experience", f"{avg_exp:.3f}"),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Insights")
    stats = compute_mgroup_insights(success)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            create_kpi_card(
                "Civils/Utilities Experience",
                str(stats["infra_experienced_candidates"]),
                delta="Water/energy/highways/rail/telecom keywords",
            ),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            create_kpi_card(
                "HSE/CDM Indicators",
                str(stats["hse_focused_candidates"]),
                delta="NEBOSH/IOSH/CDM/permits/RAMS",
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            create_kpi_card(
                "UK & Ireland Based",
                str(stats["uk_ie_candidates"]),
                delta="Location in UK/IE",
            ),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            create_kpi_card(
                "Avg 'Project' Mentions",
                f"{stats['avg_projects_per_candidate']:.1f}",
                delta="Proxy for project intensity",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    render_detailed_table(success)


def render_detailed_table(success: List[Dict[str, Any]]):
    st.subheader("Detailed Results Table")
    rows = []
    for r in success:
        res = st.session_state.resumes_data[r["resume_index"]]
        jd = st.session_state.job_descriptions[r["jd_index"]]
        scores = r["scores"]
        extracted = r.get("analysis", {}).get("extracted_data", {})
        name = extracted.get("candidate_name", res["filename"])
        rows.append(
            {
                "Candidate": name,
                "Resume": res["filename"],
                "Job": jd["title"],
                "Overall": round(scores["composite_score"], 3),
                "Relevance": round(scores.get("relevance_score", 0), 3),
                "Experience": round(scores.get("experience_score", 0), 3),
                "Skills": round(scores.get("skills_score", 0), 3),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=420)


# ---------------------------------
# CHATBOT
# ---------------------------------
def render_enhanced_chatbot():
    st.header("AI Chatbot")
    st.markdown("Ask portfolio-level questions about all resumes and analyses.")

    engine = st.session_state.screening_engine
    if not engine:
        st.error("System not initialized.")
        return

    if st.session_state.chatbot_history:
        st.subheader("Conversation History")
        for ex in st.session_state.chatbot_history:
            st.markdown(f"**You:** {ex['question']}")
            st.markdown(f"**AI:** {ex['answer']}")
            st.caption(ex.get("timestamp", ""))
            st.markdown("---")

    st.subheader("Ask About Your Talent Pool")
    with st.form(key="chat_form", clear_on_submit=True):
        q = st.text_input(
            "Question",
            placeholder="e.g., Which candidates are best for Site Engineer vs Project Manager roles?",
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            submit = st.form_submit_button("⏎", type="primary", use_container_width=True)
        with c2:
            clear = st.form_submit_button("🧹 Clear")
    if clear:
        st.session_state.chatbot_history = []
        st.rerun()
    if submit and q:
        process_enhanced_chat(q)


def process_enhanced_chat(question: str):
    engine = st.session_state.screening_engine
    context_parts = [
        "EMPLOYER: Utilites Services (water, energy, highways, rail & aviation, telecom; UK & Ireland)."
    ]
    if st.session_state.resumes_data:
        context_parts.append(f"RESUMES ({len(st.session_state.resumes_data)}):")
        for i, r in enumerate(st.session_state.resumes_data[:10]):
            ed = r.get("extracted_data", {})
            name = ed.get("candidate_name", r["filename"])
            yrs = ed.get("total_years_experience", "N/A")
            context_parts.append(f"- {name}: {yrs} years experience.")
    if st.session_state.job_descriptions:
        context_parts.append(f"JOB DESCRIPTIONS ({len(st.session_state.job_descriptions)}):")
        for jd in st.session_state.job_descriptions:
            context_parts.append(f"- {jd['title']}: {jd['description'][:80]}...")
    context = "\n".join(context_parts)

    with st.spinner("AI analyzing context..."):
        resp = engine.google_ai_manager.comprehensive_chat_response(question, context)
    if resp.get("success"):
        st.session_state.chatbot_history.append(
            {
                "question": question,
                "answer": resp["answer"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.rerun()
    else:
        st.error(resp.get("error", "Chat failed"))


# ---------------------------------
# SYSTEM STATUS & EXPORT
# ---------------------------------
def render_system_status():
    st.header("System Status")
    engine = st.session_state.screening_engine
    if not engine:
        st.error("System not initialized.")
        return
    status = engine.get_system_status()
    if status.get("ready_for_screening"):
        st.success("System Ready for Multi-Analysis")
    else:
        st.error("System not ready. Check API keys and provider.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Provider & Model")
        st.json(status.get("google_ai_status", {}))
    with c2:
        st.subheader("Session")
        st.metric("Resumes", len(st.session_state.resumes_data))
        st.metric("JDs", len(st.session_state.job_descriptions))
        st.metric("Analyses", len(st.session_state.analysis_results))
    with c3:
        st.subheader("Processing Stats")
        st.json(status.get("processing_stats", {}))


def export_comprehensive_results():
    if not st.session_state.analysis_results:
        st.warning("No results to export.")
        return
    rows = []
    for r in st.session_state.analysis_results:
        if r.get("status") != "completed":
            continue
        res = st.session_state.resumes_data[r["resume_index"]]
        jd = st.session_state.job_descriptions[r["jd_index"]]
        scores = r["scores"]
        rows.append(
            {
                "Resume": res["filename"],
                "Job": jd["title"],
                "Composite_Score": scores.get("composite_score", 0),
                "Relevance_Score": scores.get("relevance_score", 0),
                "Experience_Score": scores.get("experience_score", 0),
                "Skills_Score": scores.get("skills_score", 0),
                "Timestamp": r.get("analysis_timestamp", ""),
            }
        )
    if not rows:
        st.warning("No completed results to export.")
        return
    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        data=csv,
        file_name=f"mgroup_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
