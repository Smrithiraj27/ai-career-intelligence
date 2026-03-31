from flask import Flask, render_template, request
import os
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CAREER ROLES
career_roles = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "statistics", "deep learning"],
    "Web Developer": ["html", "css", "javascript", "react", "node.js", "git"],
    "Backend Developer": ["python", "java", "sql", "mysql", "api", "django"],
    "AI Engineer": ["python", "deep learning", "machine learning", "tensorflow", "nlp"]
}

# SKILLS DATABASE
skills_list = [
    "python", "java", "c++", "c", "javascript",
    "html", "css", "sql", "mysql",
    "machine learning", "deep learning", "ai",
    "data science", "pandas", "numpy",
    "flask", "django", "react", "node.js",
    "git", "github"
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['resume']
    job_desc = request.form['job_desc']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # PDF TEXT EXTRACTION
    text = ""
    pdf = fitz.open(filepath)
    for page in pdf:
        text += page.get_text()

    text_lower = text.lower()
    job_desc_lower = job_desc.lower()

    # SKILL DETECTION
    detected_skills = []
    for skill in skills_list:
        if re.search(rf"\b{skill}\b", text_lower):
            detected_skills.append(skill)

    detected_skills = list(set(detected_skills))

    # ATS MATCHING
    matched_skills = []
    missing_skills = []

    for skill in skills_list:
        if re.search(rf"\b{skill}\b", job_desc_lower):
            if re.search(rf"\b{skill}\b", text_lower):
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

    # WEIGHTED SCORING SYSTEM
    skill_weight = 0.6
    similarity_weight = 0.4

    skill_score = 0
    if len(matched_skills) + len(missing_skills) > 0:
        skill_score = len(matched_skills) / (len(matched_skills) + len(missing_skills))

    # NLP SIMILARITY (ML PART)
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([text, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # FINAL SCORE
    final_score = int((skill_score * skill_weight + similarity * similarity_weight) * 100)

    # ROLE PREDICTION
    role_scores = {}

    for role, skills in career_roles.items():
        score = 0
        for skill in skills:
            if re.search(rf"\b{skill}\b", text_lower):
                score += 1
        role_scores[role] = score

    if all(score == 0 for score in role_scores.values()):
        predicted_role = "Not enough data"
    else:
        predicted_role = max(role_scores, key=role_scores.get)

    #ROLE GAP
    role_missing = []
    if predicted_role in career_roles:
        for skill in career_roles[predicted_role]:
            if not re.search(rf"\b{skill}\b", text_lower):
                role_missing.append(skill)

    # ROADMAP
    roadmap = []

    if predicted_role == "Web Developer":
        roadmap = [
            "Learn advanced JavaScript",
            "Master React.js",
            "Build portfolio projects",
            "Learn APIs & backend basics",
            "Deploy projects online"
        ]

    elif predicted_role == "Data Scientist":
        roadmap = [
            "Master Python & libraries",
            "Learn Machine Learning",
            "Work on datasets",
            "Build ML projects",
            "Learn deployment"
        ]

    elif predicted_role == "Backend Developer":
        roadmap = [
            "Master backend language",
            "Learn databases deeply",
            "Build APIs",
            "Learn system design basics",
            "Deploy backend apps"
        ]

    elif predicted_role == "AI Engineer":
        roadmap = [
            "Learn ML & DL concepts",
            "Work with TensorFlow/PyTorch",
            "Build AI models",
            "Work on NLP/CV projects",
            "Deploy models"
        ]

    # AI-LIKE FEEDBACK (SMART LOGIC)
    feedback = []

    if similarity < 0.3:
        feedback.append("Your resume is not aligned with the job description.")

    if similarity >= 0.3 and similarity < 0.6:
        feedback.append("Your resume is moderately aligned but needs improvement.")

    if similarity >= 0.6:
        feedback.append("Your resume is well aligned with the job role.")

    if missing_skills:
        feedback.append("You are missing key skills required for this role.")

    if len(detected_skills) < 5:
        feedback.append("Your resume lacks sufficient technical depth.")

    if "project" not in text_lower:
        feedback.append("Add strong projects to increase impact.")

    if "internship" not in text_lower:
        feedback.append("Include real-world experience.")

    return render_template(
        "result.html",
        score=final_score,
        matched=matched_skills,
        missing=missing_skills,
        skills=detected_skills,
        text=text,
        role=predicted_role,
        role_missing=role_missing,
        roadmap=roadmap,
        feedback=feedback,
        similarity=round(similarity * 100, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)