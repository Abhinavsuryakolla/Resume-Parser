# -*- coding: utf-8 -*-
# ---------------------- DEPENDENCY INSTALLATION ----------------------
import os
import sys

if "streamlit" in sys.modules:
    os.system("pip install --upgrade pip")
    os.system("pip install -r requirements.txt --force-reinstall --no-cache-dir")

# ---------------------- IMPORTS AFTER DEPENDENCIES -------------------
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
from fpdf import FPDF
from docx import Document  # Added DOCX support


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


# ---------------------- SPA CY MODEL INITIALIZATION ------------------

@st.cache_resource
def load_nlp_model():
    import spacy, subprocess
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        return spacy.load("en_core_web_md")

nlp = load_nlp_model()

# ---------------------- GEOLOCATION CONFIGURATION --------------------
geolocator = Nominatim(
    user_agent="resume_parser_app_v2",
    timeout=20,
    domain="nominatim.openstreetmap.org"
)

# ------------------------- SKILLS LOADING CODE -----------------------
@st.cache_data
def load_skills():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        skills_path = os.path.join(current_dir, "skills.json")
        
        with open(skills_path, "r", encoding="utf-8") as f:
            skill_data = json.load(f)
            
            flat_skills = []
            for category, subcategories in skill_data.items():
                if isinstance(subcategories, dict):
                    for subcat, skills in subcategories.items():
                        flat_skills.extend(skills)
                else:
                    flat_skills.extend(subcategories)
            
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
    except Exception as e:
        st.error(f"Critical skills error: {str(e)}")
        return []

skill_list = load_skills()
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill.lower()) for skill in skill_list]
phrase_matcher.add("SKILLS", None, *patterns)

# --------------------- IMPROVED NAME EXTRACTION ---------------------
def extract_candidate_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Enhanced regex for international names
    name_pattern = r"^[A-Za-zÀ-ÿ\-\.']+(?: [A-Za-zÀ-ÿ\-\.']+){1,3}$"
    for line in lines[:5]:
        if re.match(name_pattern, line):
            return line.title()
    
    # NER fallback with prioritization
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return persons[0] if persons else lines[0] if lines else "Unknown"

# # --------------------- SECTION EXTRACTION ---------------------------
# def extract_section(text, header):
#     pattern = re.compile(
#         rf'(?i){re.escape(header)}[\s:]*(.*?)(?=\n\s*[A-Z]{{3,}}|\Z)',
#         re.DOTALL
#     )
#     match = pattern.search(text)
#     return match.group(1).strip() if match else ""

# # --------------------- RESUME PARSER CLASS --------------------------
# class EnhancedResumeParser:
#     def __init__(self):
#         self.matcher = Matcher(nlp.vocab)
#         self._add_patterns()
        
#     def _add_patterns(self):
#         self.matcher.add("EDUCATION", [
#             [{"LOWER": {"IN": ["bachelor", "master", "phd", "bs", "ms"]}}],
#             [{"LOWER": "degree"}, {"LOWER": "in"}]
#         ])
        
#         self.matcher.add("EXPERIENCE", [
#             [{"ENT_TYPE": "DATE"}, {"LOWER": {"IN": ["years", "yrs"]}}]
#         ])
    
#     def extract_experience(self, text):
#         doc = nlp(text)
#         experiences = re.findall(r'(\d+)\+? (?:years?|yrs?)', text, re.I)
#         return f"{max(experiences, default=0)} years" if experiences else "Not specified"
    
#     def extract_education(self, text):
#         doc = nlp(text)
#         degrees = []
#         for match_id, start, end in self.matcher(doc):
#             if self.matcher.vocab.strings[match_id] == "EDUCATION":
#                 degrees.append(doc[start:end].text)
#         return degrees if degrees else ["Education not found"]
    
#     def extract_location(self, text):
#         doc = nlp(text)
#         locations = []
#         for ent in doc.ents:
#             if ent.label_ in ["GPE", "LOC"]:
#                 try:
#                     loc = geolocator.geocode(ent.text, exactly_one=True)
#                     if loc:
#                         locations.append(f"{ent.text} ({loc.latitude:.4f}, {loc.longitude:.4f})")
#                     else:
#                         locations.append(ent.text)
#                 except Exception as e:
#                     locations.append(ent.text)
#         return ", ".join(list(set(locations))[:3]) if locations else "Not found"

# # ---------------------- FILE EXTRACTION -----------------------------
# def extract_text(file):
#     try:
#         if file.type == "application/pdf":
#             with fitz.open(stream=file.read(), filetype="pdf") as doc:
#                 return "\n".join([page.get_text("text") for page in doc])
#         elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#             doc = Document(file)
#             return "\n".join([para.text for para in doc.paragraphs])
#         else:
#             return "Unsupported file format"
#     except Exception as e:
#         st.error(f"File error: {str(e)}")
#         return ""

# # ---------------------- REST OF YOUR ORIGINAL CODE ------------------
# # [Keep all your existing code for summary extraction, matcher class, 
# #  visualization functions, and main application logic exactly as is]

# if __name__ == "__main__":
#     main()









# import re
# import json
# import spacy
# import fitz
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from spacy.matcher import Matcher, PhraseMatcher
# from wordcloud import WordCloud
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from geopy.geocoders import Nominatim
# from fpdf import FPDF  # Ensure you have installed fpdf via pip install fpdf

# # Load spaCy model for NER, similarity and embeddings
# # Replace the existing spaCy load line



# import os
# import sys

# # Force install dependencies before anything else
# if "streamlit" in sys.modules:
#     os.system("pip install --upgrade pip")
#     os.system("pip install -r requirements.txt --force-reinstall")

# # Verify spaCy installation
# try:
#     import spacy
# except ImportError:
#     os.system("pip install spacy==3.7.4")
#     import spacy

# # Verify model installation
# try:
#     nlp = spacy.load("en_core_web_md")
# except OSError:
#     os.system("python -m spacy download en_core_web_md")
#     nlp = spacy.load("en_core_web_md")

# # import spacy

# # try:
# #     nlp = spacy.load("en_core_web_md")
# # except OSError:
# #     from spacy.cli import download
# #     download("en_core_web_md")
# #     nlp = spacy.load("en_core_web_md")
# geolocator = Nominatim(user_agent="resume_parser", timeout=10)

# # ------------------------- SKILLS LOADING CODE -------------------------
# def load_skills():
#     try:
#         with open("skills.json", "r", encoding="utf-8") as f:
#             skill_data = json.load(f)
            
#             # Flatten the nested structure
#             flat_skills = []
#             for category, subcategories in skill_data.items():
#                 if isinstance(subcategories, dict):  # For nested categories like Non-Technical
#                     for subcat, skills in subcategories.items():
#                         flat_skills.extend(skills)
#                 else:
#                     flat_skills.extend(subcategories)
            
#             # Add common variations
#             variations = {
#                 "NLP": ["Natural Language Processing"],
#                 "CI/CD": ["Continuous Integration/Continuous Deployment"],
#                 "SEO": ["Search Engine Optimization"],
#                 "AI": ["Artificial Intelligence"],
#                 "ML": ["Machine Learning"]
#             }
            
#             expanded_skills = []
#             for skill in flat_skills:
#                 expanded_skills.append(skill)
#                 if skill in variations:
#                     expanded_skills.extend(variations[skill])
            
#             return list(set(expanded_skills))
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         st.error(f"Error loading skills: {str(e)}")
#         return []

# skill_list = load_skills()

# # Initialize PhraseMatcher for skills extraction (case-insensitive)
# phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
# patterns = [nlp.make_doc(skill.lower()) for skill in skill_list]
# phrase_matcher.add("SKILLS", None, *patterns)
# # --------------------------------------------------------------------

# # --------------------- Improved Candidate Name Extraction ---------------------
# def extract_candidate_name(text):
#     """
#     Try to extract candidate name using heuristics:
#     - Check the first 5 non-empty lines for a line containing at least two words.
#     - Prefer lines that are mostly alphabetic (ignoring phone numbers, emails, etc.).
#     """
#     lines = [line.strip() for line in text.splitlines() if line.strip()]
#     for line in lines[:5]:
#         words = line.split()
#         if len(words) >= 2:
#             # Check if most characters are alphabetic
#             alpha_count = sum(c.isalpha() for c in line)
#             if alpha_count / len(line) > 0.5:
#                 return line
#     return lines[0] if lines else "Unknown"
# # --------------------------------------------------------------------

# # --------------------- Section Extraction ---------------------
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
    
