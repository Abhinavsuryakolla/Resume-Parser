# import re
# import json
# import spacy
# import fitz  # PyMuPDF for better PDF parsing
# import streamlit as st
# from spacy.matcher import PhraseMatcher
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Load skills from JSON file safely
# try:
#     with open("skills.json", "r", encoding="utf-8") as f:
#         skill_list = json.load(f)
# except (FileNotFoundError, json.JSONDecodeError):
#     skill_list = []

# # Create a PhraseMatcher object for skills
# phrase_matcher = PhraseMatcher(nlp.vocab)
# patterns = [nlp(skill) for skill in skill_list]
# phrase_matcher.add("SKILLS", None, *patterns)

# def extract_name(text):
#     nlp_text = nlp(text)
#     for ent in nlp_text.ents:
#         if ent.label_ == "PERSON":
#             return ent.text
#     return "Unknown"

# def get_email_addresses(text):
#     return re.findall(r'[\w\.-]+@[\w\.-]+', text)

# def get_mobile_numbers(text):
#     return re.findall(r'\b\d{10}\b', text)

# def extract_skills(text):
#     doc = nlp(text)
#     matches = phrase_matcher(doc)
#     return list({doc[start:end].text for match_id, start, end in matches})

# def extract_experience(text):
#     return "Experience details extracted (Placeholder)"

# def extract_education(text):
#     return "Education details extracted (Placeholder)"

# def extract_location(text):
#     return "Location details extracted (Placeholder)"

# def extract_text(file):
#     try:
#         with fitz.open(stream=file.read(), filetype="pdf") as doc:
#             text = "\n".join([page.get_text("text") for page in doc])
#         return text if text.strip() else "No text extracted."
#     except Exception as e:
#         return f"Error extracting text: {str(e)}"

# def calculate_similarity(resume_text, job_desc_text):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
#     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
#     return similarity[0][0]

# # Streamlit UI
# st.set_page_config(page_title="Resume Parser", layout="wide")
# st.title("📄 Resume Parser & Job Description Matcher")

# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f5f7fa;
#     }
#     div.stButton > button:first-child {
#         background-color: #4CAF50;
#         color: white;
#         font-size: 16px;
#     }
#     div.stButton > button:first-child:hover {
#         background-color: #45a049;
#     }
# </style>
# """, unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("📂 Upload Resume(s)")
#     uploaded_resumes = st.file_uploader("Upload one or multiple resumes in PDF format.", type=["pdf"], accept_multiple_files=True, key="resume")

# with col2:
#     st.subheader("📜 Upload Job Description")
#     job_description = st.file_uploader("Upload a job description.", type=["pdf"], key="job")

# st.markdown("---")

# if "parsed_data" not in st.session_state:
#     st.session_state.parsed_data = []

# data = st.session_state.parsed_data

# if st.button("🔍 Parse Resumes"):
#     st.session_state.parsed_data = []  # Clear old data
#     for resume in uploaded_resumes:
#         resume_text = extract_text(resume)
#         if not resume_text:
#             st.warning(f"Could not extract text from {resume.name}.")
#             continue
#         st.session_state.parsed_data.append({
#             "name": extract_name(resume_text),
#             "email": get_email_addresses(resume_text),
#             "phone": get_mobile_numbers(resume_text),
#             "skills": extract_skills(resume_text),
#             "experience": extract_experience(resume_text),
#             "education": extract_education(resume_text),
#             "location": extract_location(resume_text),
#             "text": resume_text,  # Store text for ranking
#             "score": None  # Placeholder for ranking
#         })
#     data = st.session_state.parsed_data

# if data:
#     st.subheader("📜 Parsed Resumes")
#     for idx, res in enumerate(data):
#         with st.container():
#             st.markdown(f"### 📝 Resume {idx+1}: {res['name']}")
#             st.write(f"📧 Email: {res['email']}")
#             st.write(f"📞 Phone: {res['phone']}")
#             st.write(f"📍 Location: {res['location']}")
#             st.write(f"🎓 Education: {res['education']}")
#             st.write(f"💼 Experience: {res['experience']}")
#             st.write(f"🛠️ Skills: {', '.join(res['skills'])}")
#             st.markdown("---")

# if job_description and st.button("📊 Match & Rank"):
#     job_desc_text = extract_text(job_description)
#     if not job_desc_text:
#         st.error("Could not extract text from the job description file.")
#     else:
#         ranked_results = []
#         for res in data:
#             if not res["text"]:
#                 continue
#             res["score"] = calculate_similarity(res["text"], job_desc_text)
#             ranked_results.append(res)
        
#         sorted_results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)
#         st.subheader("🏆 Ranked Resumes")
#         for idx, res in enumerate(sorted_results):
#             with st.container():
#                 st.markdown(f"### 🥇 Rank {idx+1}: {res['name']} - Score: {res['score']:.2f}")
#                 st.write(f"📧 Email: {res['email']}")
#                 st.write(f"📞 Phone: {res['phone']}")
#                 st.write(f"📍 Location: {res['location']}")
#                 st.write(f"🎓 Education: {res['education']}")
#                 st.write(f"💼 Experience: {res['experience']}")
#                 st.write(f"🛠️ Skills: {', '.join(res['skills'])}")
#                 st.markdown("---")



import re
import json
import spacy
import fitz
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from spacy.matcher import Matcher, PhraseMatcher
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from fpdf import FPDF  # Ensure you have installed fpdf via pip install fpdf

# Load spaCy model for NER, similarity and embeddings
# Replace the existing spaCy load line



import os
import sys

# Force install dependencies before anything else
if "streamlit" in sys.modules:
    os.system("pip install --upgrade pip")
    os.system("pip install -r requirements.txt --force-reinstall")

# Verify spaCy installation
try:
    import spacy
except ImportError:
    os.system("pip install spacy==3.7.4")
    import spacy

# Verify model installation
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# import spacy

# try:
#     nlp = spacy.load("en_core_web_md")
# except OSError:
#     from spacy.cli import download
#     download("en_core_web_md")
#     nlp = spacy.load("en_core_web_md")
geolocator = Nominatim(user_agent="resume_parser", timeout=10)

# ------------------------- SKILLS LOADING CODE -------------------------
def load_skills():
    try:
        with open("skills.json", "r", encoding="utf-8") as f:
            skill_data = json.load(f)
            
            # Flatten the nested structure
            flat_skills = []
            for category, subcategories in skill_data.items():
                if isinstance(subcategories, dict):  # For nested categories like Non-Technical
                    for subcat, skills in subcategories.items():
                        flat_skills.extend(skills)
                else:
                    flat_skills.extend(subcategories)
            
            # Add common variations
            variations = {
                "NLP": ["Natural Language Processing"],
                "CI/CD": ["Continuous Integration/Continuous Deployment"],
                "SEO": ["Search Engine Optimization"],
                "AI": ["Artificial Intelligence"],
                "ML": ["Machine Learning"]
            }
            
            expanded_skills = []
            for skill in flat_skills:
                expanded_skills.append(skill)
                if skill in variations:
                    expanded_skills.extend(variations[skill])
            
            return list(set(expanded_skills))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading skills: {str(e)}")
        return []

skill_list = load_skills()

# Initialize PhraseMatcher for skills extraction (case-insensitive)
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill.lower()) for skill in skill_list]
phrase_matcher.add("SKILLS", None, *patterns)
# --------------------------------------------------------------------

# --------------------- Improved Candidate Name Extraction ---------------------
def extract_candidate_name(text):
    """
    Try to extract candidate name using heuristics:
    - Check the first 5 non-empty lines for a line containing at least two words.
    - Prefer lines that are mostly alphabetic (ignoring phone numbers, emails, etc.).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:5]:
        words = line.split()
        if len(words) >= 2:
            # Check if most characters are alphabetic
            alpha_count = sum(c.isalpha() for c in line)
            if alpha_count / len(line) > 0.5:
                return line
    return lines[0] if lines else "Unknown"
# --------------------------------------------------------------------

# --------------------- Section Extraction ---------------------
def extract_section(text, header):
    """
    Extract text from a section that starts with a header (e.g., "TECHNICAL SKILLS")
    until the next all-caps header or end of document.
    """
    pattern = re.compile(header + r"(.*?)(?:\n[A-Z][A-Z\s]+(?:\n|$)|$)", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

# --------------------- Updated Candidate Summary Extraction ---------------------
def extract_candidate_summary(resume_text):
    """
    Build a summary for matching using candidate name, technical skills, education details,
    work experience, certifications, and coding profiles (e.g., LeetCode, CodeChef, CodeForces, GitHub).
    """
    # Candidate name extraction
    name = extract_candidate_name(resume_text)
    
    # Extract technical skills section
    tech_skills = extract_section(resume_text, "TECHNICAL SKILLS")
    
    # Extract education details using EnhancedResumeParser
    parser = EnhancedResumeParser()
    education = " ".join(parser.extract_education(resume_text))
    
    # Extract work experience and certifications sections
    work_experience = extract_section(resume_text, "WORK EXPERIENCE")
    certifications = extract_section(resume_text, "CERTIFICATIONS")
    
    # Detect coding profiles by checking for common keywords
    coding_profiles = ""
    profile_keywords = ["leetcode", "codechef", "codeforces", "github"]
    lower_text = resume_text.lower()
    for keyword in profile_keywords:
        if keyword in lower_text:
            coding_profiles += f" {keyword.capitalize()} "
    
    # Build and return the complete candidate summary
    summary = f"{name}. {tech_skills}. {education}. {work_experience}. {certifications}. {coding_profiles}"
    return summary
# --------------------------------------------------------------------

class EnhancedResumeParser:
    def __init__(self):
        self.matcher = Matcher(nlp.vocab)
        self._add_patterns()
        
    def _add_patterns(self):
        education_pattern = [
            {"ENT_TYPE": "PERSON", "OP": "+"},
            {"LOWER": {"IN": ["bachelor", "master", "phd", "bs", "ms"]}},
            {"LOWER": {"IN": ["of", "in"]}, "OP": "?"},
            {"ENT_TYPE": "ORG"}
        ]
        self.matcher.add("EDUCATION", [education_pattern])
        
        experience_pattern = [
            {"ENT_TYPE": "DATE", "OP": "+"},
            {"LOWER": {"IN": ["years", "yrs"]}},
            {"LOWER": {"IN": ["experience", "exp"]}}
        ]
        self.matcher.add("EXPERIENCE", [experience_pattern])
    
    def extract_experience(self, text):
        years = set(re.findall(r'\b(\d+)\+? years?\b', text, re.I))
        return f"{max(years, key=lambda x: int(x))} years experience" if years else "Experience not specified"
    
    def extract_education(self, text):
        doc = nlp(text)
        degrees = []
        for match_id, start, end in self.matcher(doc):
            if nlp.vocab.strings[match_id] == "EDUCATION":
                degrees.append(doc[start:end].text)
        return degrees if degrees else ["Education details not found"]
    
    def extract_location(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                try:
                    location = geolocator.geocode(ent.text)
                    return f"{ent.text} ({location.latitude}, {location.longitude})" if location else ent.text
                except:
                    return ent.text
        return "Location not specified"

# --------------------- UPDATED SKILL EXTRACTION ---------------------
def extract_skills(text):
    doc = nlp(text.lower())
    matches = phrase_matcher(doc)
    
    # Load skills from the JSON file (preserves the original categorization)
    with open("skills.json", "r") as f:
        skill_db = json.load(f)
    
    detected_skills = set()
    skill_categories = set()
    
    for match_id, start, end in matches:
        skill = doc[start:end].text.title()  # Convert to title case
        # Check for matches in each category (handle nested categories)
        for category, subcategories in skill_db.items():
            if isinstance(subcategories, dict):
                for subcat, skills in subcategories.items():
                    if skill in skills:
                        skill_categories.add(f"{category}/{subcat}")
                        detected_skills.add(skill)
            elif skill in subcategories:
                skill_categories.add(category)
                detected_skills.add(skill)
    
    return {
        "skills_list": list(detected_skills),
        "categories": list(skill_categories)
    }
# --------------------------------------------------------------------

# ------------------ IMPROVED MATCHER USING MULTIPLE APPROACHES ------------------
class AdvancedMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        
    def calculate_similarity(self, resume_text, job_desc_text):
        # Use the candidate summary instead of the full resume text
        candidate_summary = extract_candidate_summary(resume_text)
        
        # 1. TF-IDF similarity (word frequency overlap)
        tfidf_matrix = self.vectorizer.fit_transform([candidate_summary, job_desc_text])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # 2. Semantic similarity using spaCy embeddings
        resume_doc = nlp(candidate_summary)
        job_doc = nlp(job_desc_text)
        spacy_score = resume_doc.similarity(job_doc)
        
        # 3. Skills overlap score
        job_skills = set(extract_skills(job_desc_text)["skills_list"])
        resume_skills = set(extract_skills(resume_text)["skills_list"])
        skills_overlap = len(job_skills & resume_skills) / (len(job_skills) if job_skills else 1)
        
        # Adjusted weights: 60% semantic, 20% TF-IDF, 20% skills overlap
        combined_score = (spacy_score * 0.6) + (tfidf_score * 0.2) + (skills_overlap * 0.2)
        return combined_score
# ----------------------------------------------------------------------------

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(title, fontsize=20)
    plt.axis("off")
    st.pyplot(plt)

def extract_text(file):
    try:
        # For PyMuPDF (fitz) v1.18.0+
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text("text") for page in doc])
        return text if text.strip() else "No text extracted."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def generate_pdf_report(results):
    # Generate a PDF report using FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resume Analysis Report", ln=1, align="C")
    for candidate in results:
        pdf.cell(200, 10, txt=f"Candidate: {candidate['name']}", ln=1)
        pdf.cell(200, 10, txt=f"Match Score: {candidate['score']}", ln=1)
        pdf.cell(200, 10, txt=f"Experience: {candidate['experience']}", ln=1)
        pdf.cell(200, 10, txt=f"Education: {candidate['education']}", ln=1)
        pdf.cell(200, 10, txt=f"Skills: {candidate['skills']}", ln=1)
        pdf.cell(200, 10, txt=f"Categories: {candidate['categories']}", ln=1)
        pdf.cell(200, 10, txt=f"Location: {candidate['location']}", ln=1)
        pdf.ln(5)
    return pdf.output(dest="S").encode("latin1")
    
def main():
    st.set_page_config(page_title="AI Resume Analyst", layout="wide")
    st.title("🚀 AI Resume Analyst 2.0")
    
    st.sidebar.header("⚙️ Analysis Settings")
    st.sidebar.write("Using advanced matching with semantic, TF-IDF, and skills overlap.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_resumes = st.file_uploader("📁 Upload Resumes (PDF/DOCX)", 
                                            type=["pdf", "docx"],
                                            accept_multiple_files=True)
        
    with col2:
        job_description = st.file_uploader("📑 Upload Job Description (for matching)", 
                                           type=["pdf", "docx"])
        min_score = st.slider("🔍 Minimum Match Score", 0.0, 1.0, 0.5)
    
    parser = EnhancedResumeParser()
    matcher = AdvancedMatcher()
    
    # --- Section 1: Parse Resumes ---
    if uploaded_resumes:
        if st.button("📄 Parse Resumes", key="parse_btn"):
            st.header("📝 Parsed Resume Details")
            parsed_results = []
            for resume in uploaded_resumes:
                resume_text = extract_text(resume)
                candidate_name = extract_candidate_name(resume_text)
                skills_data = extract_skills(resume_text)
                details = {
                    "Name": candidate_name,
                    "Experience": parser.extract_experience(resume_text),
                    "Education": ", ".join(parser.extract_education(resume_text)),
                    "Skills": ", ".join(skills_data["skills_list"]),
                    "Categories": ", ".join(skills_data["categories"]),
                    "Location": parser.extract_location(resume_text)
                }
                parsed_results.append(details)
                st.subheader(candidate_name)
                st.write(details)
                # Display the extracted candidate summary as well:
                st.write("**Candidate Summary:**")
                st.write(extract_candidate_summary(resume_text))
            st.success("Resume parsing complete.")
    
    # --- Section 2: Match Resumes with Job Description ---
    if uploaded_resumes and job_description:
        if st.button("🔍 Match Resumes", key="match_btn"):
            job_desc_text = extract_text(job_description)
            results = []
            for resume in uploaded_resumes:
                resume_text = extract_text(resume)
                score = matcher.calculate_similarity(resume_text, job_desc_text)
                if score >= min_score:
                    skills_data = extract_skills(resume_text)
                    results.append({
                        "name": extract_candidate_name(resume_text),
                        "score": round(score, 2),
                        "experience": parser.extract_experience(resume_text),
                        "education": ", ".join(parser.extract_education(resume_text)),
                        "skills": ", ".join(skills_data["skills_list"]),
                        "categories": ", ".join(skills_data["categories"]),
                        "location": parser.extract_location(resume_text),
                        "text": resume_text
                    })
            
            # Sort results by descending score
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            if results:
                st.header("📊 Analysis Dashboard")
                
                # Global Score Distribution Chart
                st.subheader("📈 Global Score Distribution")
                df = pd.DataFrame(results)
                st.bar_chart(df.set_index("name")["score"])
                
                # Display each candidate with separate analysis
                st.subheader("🏆 Candidate Analyses")
                for candidate in results:
                    with st.expander(f"{candidate['name']} - Match Score: {candidate['score']:.0%}"):
                        st.write(f"**Experience:** {candidate['experience']}")
                        st.write(f"**Education:** {candidate['education']}")
                        st.write(f"**Skills:** {candidate['skills']}")
                        st.write(f"**Categories:** {candidate['categories']}")
                        st.write(f"**Location:** {candidate['location']}")
                        
                        # Candidate-specific Skill Gap Analysis
                        candidate_skills = set(extract_skills(candidate['text'])["skills_list"])
                        job_skills = set(extract_skills(job_desc_text)["skills_list"])
                        missing_skills = job_skills - candidate_skills
                        st.write("**Skill Gap Analysis:**")
                        if missing_skills:
                            st.write(f"Missing Skills: {', '.join(missing_skills)}")
                        else:
                            st.write("No missing skills!")
                        
                        # Candidate-specific Keyword Comparison
                        st.write("**Keyword Comparison:**")
                        col_kw1, col_kw2 = st.columns(2)
                        with col_kw1:
                            plot_wordcloud(candidate['text'], f"{candidate['name']} Resume Keywords")
                        with col_kw2:
                            plot_wordcloud(job_desc_text, "Job Description Keywords")
                
                # Global Analysis Sections (if desired, these remain separate)
                st.subheader("🔍 Global Skill Gap Analysis")
                global_job_skills = set(extract_skills(job_desc_text)["skills_list"])
                all_resume_skills = set().union(*[set(r['skills'].split(", ")) for r in results])
                global_missing = global_job_skills - all_resume_skills
                if global_missing:
                    st.warning(f"Missing skills across all candidates: {', '.join(global_missing)}")
                else:
                    st.success("All required skills covered in candidate pool!")
                
                st.subheader("📚 Global Keyword Comparison")
                col_global1, col_global2 = st.columns(2)
                with col_global1:
                    plot_wordcloud(job_desc_text, "Job Description Keywords")
                with col_global2:
                    plot_wordcloud(" ".join([r["text"] for r in results]), "Combined Resume Keywords")
                
                # Data Export Section: Download PDF Report
                st.subheader("💾 Export Results")
                if st.button("📥 Download Analysis Report"):
                    pdf = generate_pdf_report(results)
                    st.download_button("Download PDF", pdf, file_name="resume_analysis.pdf")
            else:
                st.warning("No candidates meet the minimum score criteria.")

if __name__ == "__main__":
    main()
