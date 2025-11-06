"""
SafeRx Backend - FastAPI Application
AI Medical Prescription Verification System
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
import io
import json
import hashlib
import jwt
import requests
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from fpdf import FPDF
import base64

# Database imports
try:
    from pymongo import MongoClient
    USE_MONGODB = True
except ImportError:
    import sqlite3
    USE_MONGODB = False

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "saferx_db"

# Initialize FastAPI
app = FastAPI(title="SafeRx API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Initialize NLP Model (Hugging Face)
print("Loading NLP model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
except Exception as e:
    print(f"Warning: Could not load NER model: {e}")
    ner_pipeline = None

# Initialize QA Model for Chatbot
try:
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
except Exception as e:
    print(f"Warning: Could not load QA model: {e}")
    qa_pipeline = None

# Database Setup
if USE_MONGODB:
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        users_collection = db["users"]
        prescriptions_collection = db["prescriptions"]
        print("Connected to MongoDB")
    except Exception as e:
        print(f"MongoDB connection failed: {e}, falling back to SQLite")
        USE_MONGODB = False

if not USE_MONGODB:
    # SQLite Setup
    conn = sqlite3.connect("saferx.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            allergies TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            prescription_text TEXT,
            extracted_drugs TEXT,
            interactions TEXT,
            dosage_check TEXT,
            alternatives TEXT,
            safety_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print("Connected to SQLite")

# Pydantic Models
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    allergies: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PrescriptionText(BaseModel):
    text: str
    user_email: str

class DrugQuery(BaseModel):
    drugs: List[str]

class ChatQuery(BaseModel):
    question: str
    context: Optional[str] = None

# Helper Functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {str(e)}")

def extract_drugs_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract drug names and medical entities using Hugging Face NER"""
    if not ner_pipeline:
        # Fallback: Simple extraction based on common patterns
        return fallback_drug_extraction(text)
    
    try:
        entities = ner_pipeline(text)
        drugs = []
        
        for entity in entities:
            if entity['entity_group'] in ['DRUG', 'MEDICATION', 'CHEMICAL', 'B-DRUG', 'I-DRUG']:
                drugs.append({
                    "name": entity['word'],
                    "confidence": entity['score']
                })
        
        # Enhanced extraction with pattern matching
        import re
        drug_patterns = [
            r'\b([A-Z][a-z]+(?:cillin|mycin|prazole|statin|ine|ol|ide|ate))\b',
            r'\b(Aspirin|Ibuprofen|Paracetamol|Amoxicillin|Metformin|Lisinopril)\b'
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if not any(d['name'].lower() == match.lower() for d in drugs):
                    drugs.append({"name": match, "confidence": 0.8})
        
        return drugs
    except Exception as e:
        print(f"NLP extraction error: {e}")
        return fallback_drug_extraction(text)

def fallback_drug_extraction(text: str) -> List[Dict[str, Any]]:
    """Fallback drug extraction using pattern matching"""
    import re
    common_drugs = [
        "Amoxicillin", "Ibuprofen", "Paracetamol", "Aspirin", "Metformin",
        "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole", "Azithromycin"
    ]
    
    found_drugs = []
    for drug in common_drugs:
        if drug.lower() in text.lower():
            found_drugs.append({"name": drug, "confidence": 0.7})
    
    # Extract dosage patterns
    dosage_pattern = r'(\d+)\s*(mg|ml|g|mcg)'
    dosages = re.findall(dosage_pattern, text, re.IGNORECASE)
    
    return found_drugs

def check_drug_interactions(drugs: List[str]) -> List[Dict[str, Any]]:
    """Check drug interactions using fallback database (case-insensitive)"""
    interactions = []

    # Normalize drug names for consistency
    normalized_drugs = [d.strip().capitalize() for d in drugs]

    # Fallback database (case-insensitive)
    fallback_db = {
        frozenset(["Ibuprofen", "Aspirin"]): {
            "description": "Both are NSAIDs and can cause stomach bleeding or ulcers when taken together.",
            "severity": "High",
        },
        frozenset(["Amoxicillin", "Methotrexate"]): {
            "description": "Amoxicillin can increase Methotrexate toxicity and cause side effects.",
            "severity": "High",
        },
        frozenset(["Ibuprofen", "Lisinopril"]): {
            "description": "Ibuprofen may reduce the effectiveness of Lisinopril and affect kidney function.",
            "severity": "Moderate",
        },
        frozenset(["Aspirin", "Warfarin"]): {
            "description": "Both thin blood and can cause severe bleeding when combined.",
            "severity": "High",
        },
        frozenset(["Paracetamol", "Alcohol"]): {
            "description": "May increase risk of liver damage if used together regularly.",
            "severity": "Moderate",
        },
    }

    # Check all pairs for matches (case-insensitive using frozenset)
    for i in range(len(normalized_drugs)):
        for j in range(i + 1, len(normalized_drugs)):
            d1, d2 = normalized_drugs[i], normalized_drugs[j]
            pair_key = frozenset([d1, d2])

            if pair_key in fallback_db:
                data = fallback_db[pair_key]
                interactions.append({
                    "drug1": d1,
                    "drug2": d2,
                    "description": data["description"],
                    "severity": data["severity"]
                })

    # Debug output
    if not interactions:
        print(f"[DEBUG] No interactions found for {normalized_drugs}")
    else:
        print(f"[DEBUG] Found interactions: {interactions}")

    return interactions


def get_fallback_interactions(drugs: List[str]) -> List[Dict[str, Any]]:
    """Fallback interaction database"""
    known_interactions = {
        ("Ibuprofen", "Aspirin"): {"description": "May increase risk of bleeding and gastric irritation", "severity": "Moderate"},
        ("Amoxicillin", "Methotrexate"): {"description": "May increase methotrexate toxicity", "severity": "High"},
        ("Ibuprofen", "Lisinopril"): {"description": "May reduce effectiveness of blood pressure medication", "severity": "Moderate"},
    }
    
    interactions = []
    for i, drug1 in enumerate(drugs):
        for drug2 in drugs[i+1:]:
            key = tuple(sorted([drug1, drug2]))
            if key in known_interactions:
                interactions.append({
                    "drug1": drug1,
                    "drug2": drug2,
                    **known_interactions[key]
                })
    
    return interactions

def check_dosage_by_age(drugs: List[str], age: int) -> List[Dict[str, Any]]:
    """Validate dosage based on age"""
    dosage_recommendations = []
    
    # Age categories
    if age < 2:
        category = "Infant"
        factor = 0.1
    elif age < 12:
        category = "Child"
        factor = 0.5
    elif age < 18:
        category = "Adolescent"
        factor = 0.75
    elif age < 65:
        category = "Adult"
        factor = 1.0
    else:
        category = "Senior"
        factor = 0.75
    
    # Standard adult dosages (reference)
    standard_dosages = {
        "Amoxicillin": {"adult": 500, "unit": "mg", "frequency": "3x/day"},
        "Ibuprofen": {"adult": 400, "unit": "mg", "frequency": "3x/day"},
        "Paracetamol": {"adult": 500, "unit": "mg", "frequency": "4x/day"},
        "Aspirin": {"adult": 300, "unit": "mg", "frequency": "3x/day"},
    }
    
    for drug in drugs:
        if drug in standard_dosages:
            adult_dose = standard_dosages[drug]["adult"]
            recommended = int(adult_dose * factor)
            
            dosage_recommendations.append({
                "drug": drug,
                "age_category": category,
                "recommended_dose": recommended,
                "unit": standard_dosages[drug]["unit"],
                "frequency": standard_dosages[drug]["frequency"],
                "notes": f"Adjusted for {category.lower()} (factor: {factor})"
            })
    
    return dosage_recommendations

def suggest_alternatives(drugs: List[str]) -> List[Dict[str, Any]]:
    """Suggest safer alternative drugs"""
    alternatives_db = {
        "Ibuprofen": {
            "alternative": "Paracetamol",
            "reason": "Safer for children and those with stomach issues",
            "category": "Pain reliever"
        },
        "Aspirin": {
            "alternative": "Paracetamol",
            "reason": "Lower risk of bleeding and Reye's syndrome in children",
            "category": "Pain reliever"
        },
        "Amoxicillin": {
            "alternative": "Azithromycin",
            "reason": "Alternative for penicillin-allergic patients",
            "category": "Antibiotic"
        },
    }
    
    suggestions = []
    for drug in drugs:
        if drug in alternatives_db:
            suggestions.append({
                "original_drug": drug,
                **alternatives_db[drug]
            })
    
    return suggestions

def generate_pdf_report(data: Dict[str, Any], filename: str = "report.pdf") -> str:
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "SafeRx - Prescription Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    # Date
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Extracted Drugs
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Extracted Drugs:", ln=True)
    pdf.set_font("Arial", "", 10)
    for drug in data.get("drugs", []):
        pdf.cell(0, 8, f"  - {drug['name']}", ln=True)
    pdf.ln(5)
    
    # Interactions
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Drug Interactions:", ln=True)
    pdf.set_font("Arial", "", 10)
    interactions = data.get("interactions", [])
    if interactions:
        for inter in interactions:
            pdf.multi_cell(0, 8, f"  - {inter['drug1']} + {inter['drug2']}: {inter['description']} (Severity: {inter['severity']})")
    else:
        pdf.cell(0, 8, "  No significant interactions detected", ln=True)
    pdf.ln(5)
    
    # Dosage Recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Dosage Recommendations:", ln=True)
    pdf.set_font("Arial", "", 10)
    for dose in data.get("dosage_recommendations", []):
        pdf.cell(0, 8, f"  - {dose['drug']}: {dose['recommended_dose']}{dose['unit']} - {dose['frequency']}", ln=True)
    pdf.ln(5)
    
    # Alternatives
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Alternative Suggestions:", ln=True)
    pdf.set_font("Arial", "", 10)
    for alt in data.get("alternatives", []):
        pdf.multi_cell(0, 8, f"  - Replace {alt['original_drug']} with {alt['alternative']}: {alt['reason']}")
    pdf.ln(5)
    
    # Safety Score
    pdf.set_font("Arial", "B", 14)
    safety_score = data.get("safety_score", 50)
    color = (0, 255, 0) if safety_score > 70 else (255, 165, 0) if safety_score > 40 else (255, 0, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Overall Safety Score: {safety_score}/100", ln=True)
    
    # Save
    output_path = f"reports/{filename}"
    os.makedirs("reports", exist_ok=True)
    pdf.output(output_path)
    return output_path

# API Endpoints

@app.get("/")
def root():
    return {
        "message": "SafeRx API v1.0",
        "status": "active",
        "endpoints": ["/auth/register", "/auth/login", "/analyze", "/interactions", "/dosage", "/alternatives", "/report", "/history"]
    }

@app.post("/auth/register")
def register(user: UserRegister):
    """Register a new user"""
    hashed_password = hash_password(user.password)
    
    if USE_MONGODB:
        # Check if user exists
        if users_collection.find_one({"email": user.email}):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user_data = {
            "name": user.name,
            "email": user.email,
            "password": hashed_password,
            "age": user.age,
            "gender": user.gender,
            "allergies": user.allergies,
            "created_at": datetime.utcnow()
        }
        users_collection.insert_one(user_data)
    else:
        # SQLite
        try:
            cursor.execute("""
                INSERT INTO users (name, email, password, age, gender, allergies)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user.name, user.email, hashed_password, user.age, user.gender, user.allergies))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    return {"message": "User registered successfully", "email": user.email}

@app.post("/auth/login")
def login(credentials: UserLogin):
    """Authenticate user and return JWT token"""
    hashed_password = hash_password(credentials.password)
    
    if USE_MONGODB:
        user = users_collection.find_one({"email": credentials.email, "password": hashed_password})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user_data = {"name": user["name"], "email": user["email"], "age": user.get("age")}
    else:
        cursor.execute("SELECT name, email, age FROM users WHERE email = ? AND password = ?", 
                      (credentials.email, hashed_password))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user_data = {"name": user[0], "email": user[1], "age": user[2]}
    
    token = create_access_token({"email": credentials.email})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_data
    }

@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(None)):
    """Analyze uploaded prescription image"""
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")
    
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        extracted_text = extract_text_from_image(image_bytes)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from image")

        return {
            "extracted_text": extracted_text,
            "message": "Text extracted successfully. Use /analyze/text to process."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@app.post("/analyze/text")
def analyze_text(prescription: PrescriptionText, token_data: dict = Depends(verify_token)):
    """Analyze prescription text and extract drugs, interactions, and compute safety score"""

    print("=== [DEBUG] /analyze/text called ===")
    print("Incoming text (first 200 chars):", prescription.text[:200])

    # --- Step 1: Extract drugs ---
    drugs = extract_drugs_nlp(prescription.text)
    if not drugs:
        print("[DEBUG] No drugs detected.")
        return {
            "drugs": [],
            "message": "No drugs detected in the prescription text",
            "safety_score": 20,
            "risk_level": "High"
        }

    drug_names = [d["name"] for d in drugs]
    print(f"[DEBUG] Extracted drugs: {drug_names}")

    # --- Step 2: Get user age ---
    user_email = prescription.user_email
    age = 30  # Default
    if USE_MONGODB:
        user = users_collection.find_one({"email": user_email})
        if user and user.get("age"):
            age = user["age"]
    else:
        cursor.execute("SELECT age FROM users WHERE email = ?", (user_email,))
        result = cursor.fetchone()
        if result and result[0]:
            age = result[0]
    print(f"[DEBUG] User age: {age}")

    # --- Step 3: Check interactions, dosage, and alternatives ---
    interactions = check_drug_interactions(drug_names)
    dosage_recommendations = check_dosage_by_age(drug_names, age)
    alternatives = suggest_alternatives(drug_names)

    # --- Step 4: Compute improved safety score ---
    try:
        confidences = []
        for d in drugs:
            c = d.get("confidence", 0.6)
            if c > 1:  # normalize if score is in 0–100
                c = c / 100.0
            confidences.append(max(0.0, min(1.0, float(c))))
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        safety_score = int(100 * avg_conf)  # start based on confidence
        debug_reason = [f"Base from confidence: {safety_score}"]

        # Interaction penalty (based on severity)
        severity_penalty = {"High": 30, "Moderate": 20, "Low": 10, "Unknown": 10}
        total_penalty = 0
        for inter in interactions:
            sev = inter.get("severity", "Unknown")
            total_penalty += severity_penalty.get(sev, 10)
        if total_penalty:
            safety_score -= total_penalty
            debug_reason.append(f"-{total_penalty} for interactions")

        # Penalize very short or meaningless text
        if len(prescription.text.split()) < 5:
            safety_score -= 20
            debug_reason.append("-20 for too short/unclear text")

        # If no drugs, limit to 40
        if len(drugs) == 0:
            safety_score = min(safety_score, 40)
            debug_reason.append("Capped at 40 (no valid drugs)")

        # Clamp value between 0 and 100
        safety_score = max(0, min(100, safety_score))
        print(f"[DEBUG] Safety score computed: {safety_score} | Reasons: {debug_reason}")

    except Exception as e:
        print("⚠️ Error computing safety score:", e)
        safety_score = 50

    # --- Step 5: Compile final result ---
    result = {
        "drugs": drugs,
        "interactions": interactions,
        "dosage_recommendations": dosage_recommendations,
        "alternatives": alternatives,
        "safety_score": safety_score,
        "risk_level": "High" if safety_score < 40 else "Medium" if safety_score < 70 else "Low"
    }

    print(f"[DEBUG] Final result summary -> Drugs: {drug_names} | Score: {safety_score} | Risk: {result['risk_level']}")

    # --- Step 6: Save analysis to database ---
    if USE_MONGODB:
        prescriptions_collection.insert_one({
            "user_email": user_email,
            "prescription_text": prescription.text,
            "analysis_result": result,
            "created_at": datetime.utcnow()
        })
    else:
        cursor.execute("""
            INSERT INTO prescriptions 
            (user_email, prescription_text, extracted_drugs, interactions, dosage_check, alternatives, safety_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_email,
            prescription.text,
            json.dumps(drugs),
            json.dumps(interactions),
            json.dumps(dosage_recommendations),
            json.dumps(alternatives),
            safety_score
        ))
        conn.commit()

    return result

@app.post("/interactions")
def get_interactions(query: DrugQuery, token_data: dict = Depends(verify_token)):
    """Check drug interactions"""
    interactions = check_drug_interactions(query.drugs)
    return {"interactions": interactions}

@app.post("/dosage")
def get_dosage(query: DrugQuery, age: int, token_data: dict = Depends(verify_token)):
    """Get dosage recommendations by age"""
    recommendations = check_dosage_by_age(query.drugs, age)
    return {"dosage_recommendations": recommendations}

@app.post("/alternatives")
def get_alternatives(query: DrugQuery, token_data: dict = Depends(verify_token)):
    """Get alternative drug suggestions"""
    alternatives = suggest_alternatives(query.drugs)
    return {"alternatives": alternatives}

@app.post("/report/generate")
def generate_report(data: Dict[str, Any], token_data: dict = Depends(verify_token)):
    """Generate PDF report"""
    filename = f"prescription_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = generate_pdf_report(data, filename)
    
    return {
        "message": "Report generated successfully",
        "filename": filename,
        "download_url": f"/report/download/{filename}"
    }

@app.get("/report/download/{filename}")
def download_report(filename: str, token_data: dict = Depends(verify_token)):
    """Download generated PDF report"""
    file_path = f"reports/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.get("/history")
def get_history(email: str, token_data: dict = Depends(verify_token)):
    """Get user's prescription history"""
    if USE_MONGODB:
        prescriptions = list(prescriptions_collection.find(
            {"user_email": email},
            {"_id": 0}
        ).sort("created_at", -1).limit(10))
    else:
        cursor.execute("""
            SELECT prescription_text, extracted_drugs, interactions, safety_score, created_at
            FROM prescriptions
            WHERE user_email = ?
            ORDER BY created_at DESC
            LIMIT 10
        """, (email,))
        rows = cursor.fetchall()
        prescriptions = []
        for row in rows:
            prescriptions.append({
                "prescription_text": row[0],
                "extracted_drugs": json.loads(row[1]) if row[1] else [],
                "interactions": json.loads(row[2]) if row[2] else [],
                "safety_score": row[3],
                "created_at": row[4]
            })
    
    return {"history": prescriptions}

@app.post("/chat")
def chat(query: ChatQuery, token_data: dict = Depends(verify_token)):
    """AI Medical Assistant using Hugging Face QA model with enhanced context"""
    if not qa_pipeline:
        return {"answer": "Chatbot service is currently unavailable"}

    # Expanded and meaningful context for better answers
    medical_context = """
    Amoxicillin is an antibiotic used to treat bacterial infections. 
    Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain and inflammation.
    Generally, Amoxicillin and Ibuprofen can be taken together safely, but they should be used
    under a doctor's supervision if the patient has kidney or stomach issues.

    Paracetamol is a safe pain reliever and fever reducer for most people, including children. 
    The usual dose for children is based on body weight, around 10–15 mg per kg every 4 to 6 hours.
    Avoid exceeding the recommended dose as it may cause liver damage.

    Aspirin should not be given to children with viral infections due to risk of Reye's syndrome. 
    It may also irritate the stomach, so it should be taken with food and avoided in people with ulcers.

    Lisinopril is a medication for blood pressure control. It should not be mixed with Ibuprofen
    for long periods, as this can reduce kidney function.

    Always consult a healthcare provider before combining medications or adjusting dosages.
    """

    try:
        result = qa_pipeline(question=query.question, context=medical_context)
        return {
            "question": query.question,
            "answer": result["answer"].capitalize(),
            "confidence": f"{result['score']*100:.2f}%",
        }
    except Exception as e:
        return {
            "question": query.question,
            "answer": "I'm unable to answer that right now. Please consult a healthcare provider.",
            "error": str(e)
        }
