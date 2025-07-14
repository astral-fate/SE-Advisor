# se_advisor_v11_final_with_csv_export.py - FINAL VERSION ALIGNED WITH PAPER + CSV EXPORT + CONFUSION MATRIX

import os
import json
import fitz  # PyMuPDF
import pandas as pd # Added for CSV export
from openai import OpenAI
from pdf2image import convert_from_path
from sklearn.metrics import f1_score, precision_score, recall_score
import re

# ==============================================================================
# --- 1. CONFIGURATION & CONSTANTS (ALIGNED WITH PAPER) ---
# ==============================================================================

# --- API and File Paths ---
try:
    from google.colab import userdata
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
except (ImportError, AttributeError):
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# Define base path and output path
DRIVE_BASE_PATH = '/content/drive/My Drive/'
TRANSCRIPT_FOLDER_PATH = os.path.join(DRIVE_BASE_PATH, 'transcripts/')
CSV_OUTPUT_PATH = os.path.join(TRANSCRIPT_FOLDER_PATH, 'output_llama3_with_matrix.csv')


# --- Course Definitions (FROM TABLE 1 IN THE PAPER) ---
PREREQUISITE_COURSES = {
    "SE101": "Computer Systems Principles and Programming",
    "SE102": "Relational Database Systems",
    "SE103": "The Software Development Process",
    "SE105": "Object-Oriented Software Development using UML",
    "SE109": "Programming in the Large",
}

# --- Ground Truth (INFERRED FROM PAPER'S DATASET AND FINDINGS) ---
# This is the authoritative "University Board Decision" for each applicant.
# Mapped from App-ID in Table 2 to our filenames.
GROUND_TRUTH_RAW = {
    'AOU IInformation Technology & Computing Track.pdf': [
        "Relational Database Systems",
         "Object-Oriented Software Development using UML",
          "The Software Development Process",
           "Programming in the Large" 
           ],
    'arab open university computer Science.pdf': [
        "Relational Database Systems"
        ],
    'Administrative science KAU.pdf': [
        'Relational Database Systems', 
        'Programming in the Large'
        ],

    'Business information system.pdf': [
        "Programming in the Large",
        "Relational Database Systems", 
        "Object-Oriented Software Development using UML",
        "The Software Development Process"
          ],

    'Computer engineering munfia university.pdf': [
        'Object-Oriented Software Development using UML',
        "Relational Database Systems",
        "Programming in the Large"
           ],

    'Computer science future university.pdf': [
        'Object-Oriented Software Development using UML',
        'Relational Database Systems'
         ],
    'Geology lypis university.pdf': [
        "Programming in the Large",
        "Relational Database Systems",
        "Object-Oriented Software Development using UML",
        "The Software Development Process"
           ],

    'Higher Institute for Computer Sciences and Information Systems cs.pdf': [ 
        "Relational Database Systems",
        "Object-Oriented Software Development using UML",
        "The Software Development Process"
          ],

    'Higher Technological Institute BIS.pdf':  [
        "Programming in the Large",
        "Relational Database Systems",
        "Object-Oriented Software Development using UML",
        "The Software Development Process"
           ],

    'Physics with computer science azhar.pdf': [
        "Programming in the Large",
        "Computer Systems Principles and Programming"
         ],

    'alex university BIS.pdf': [
        "Relational Database Systems",
        "Object-Oriented Software Development using UML",
        "The Software Development Process",
        "Computer Systems Principles and Programming"
           ],

    'clicical pharma cairo university.pdf':  [
        "Programming in the Large",
        "Relational Database Systems",
        "Object-Oriented Software Development using UML",
           "The Software Development Process"
           ],

    'communication engineering Helwan university.pdf': [
        "Relational Database Systems",
        "Object-Oriented Software Development using UML",
        "Programming in the Large",
        "Object-Oriented Software Development using UML"
           ],
           
    'future academy BIS.pdf': [
        "Relational Database Systems", 
        "The Software Development Process", 
        "Programming in the Large", 
        "Object-Oriented Software Development using UML"
        ],

    'master commernce cairo university.pdf':  [
        "Relational Database Systems", 
        'Object-Oriented Software Development using UML', 
        "Programming in the Large"
        ],
    'physics semi bio.pdf':   [
        "Relational Database Systems", 
        'Object-Oriented Software Development using UML', 
        "Programming in the Large", 
        'Object-Oriented Software Development using UML'
        ],
    'taif university math.pdf':  [
        "Relational Database Systems",
        'Object-Oriented Software Development using UML',
        "Programming in the Large",
        'Object-Oriented Software Development using UML'
           ],
}

# ==============================================================================
# --- 2. DOCUMENT & TEXT PROCESSING UTILITIES ---
# ==============================================================================

def get_document_text(file_path: str) -> str:
    """Extracts text from a PDF, falling back to OCR if needed."""
    try:
        with fitz.open(file_path) as doc:
            text = "".join(page.get_text() for page in doc)
        if len(text.strip()) < 200:
            print(f"⚠️ Low text count. Initiating OCR for {os.path.basename(file_path)}...")
            images = convert_from_path(file_path)
            text = "".join(pytesseract.image_to_string(img, lang='eng+ara') for img in images)
        return text
    except Exception as e:
        print(f"❌ Text extraction failed for {os.path.basename(file_path)}: {e}")
        return ""

# ==============================================================================
# --- 3. NUANCED REASONING ENGINE (LLM) ---
# ==============================================================================

class NuancedLLMReasoner:
    """LLM Reasoner designed to replicate the paper's methodology."""

    def __init__(self, api_key: str):
        if not api_key: raise ValueError("NVIDIA API key is missing.")
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        self.model = "meta/llama3-70b-instruct"
        self.prereq_details = """
        - **SE101: Computer Systems Principles and Programming**: Key Topics are C Programming, Pointers, Memory Management, Basic Data Structures. Look for low-level systems programming.
        - **SE102: Relational Database Systems**: Key Topics are Data Modeling (ER), Relational Model, SQL. Look for database design and query language courses.
        - **SE103: The Software Development Process**: Key Topics are SDLC, Development Methodologies (Agile, Waterfall), Requirements Engineering. Look for "Software Engineering" or "Systems Development" courses.
        - **SE105: Object-Oriented Software Development using UML**: Key Topics are OOP Concepts, UML Diagrams, Design Patterns. Look for "Object-Oriented Analysis and Design" or similar.
        - **SE109: Programming in the Large**: Key Topics are Java Fundamentals, Class Definition, Advanced OOP. Look for a second-level or advanced programming course, often with Java or C++.
        """

    def analyze(self, transcript_text: str) -> dict:
        """Analyzes curriculum based on key topics for each prerequisite."""
        prompt = f"""
        You are an expert member of the University Board of Professional Studies for the Master's in Software Engineering. Your task is to perform a zero-shot semantic assessment of an applicant's academic plan to determine prerequisite equivalence.

        **METHODOLOGY:**
        Your decision must be based on evidence of the **Key Topics** for each of the five prerequisites listed below. You must reason about the entire curriculum provided. Do not just match titles; infer coverage from course collections.

        **PREREQUISITE KNOWLEDGE BASE:**
        {self.prereq_details}

        **INSTRUCTIONS:**
        1. Analyze the complete 'APPLICANT'S ACADEMIC PLAN' provided below.
        2. For each of the five prerequisites, determine if the applicant's plan provides sufficient evidence that they have covered the key topics.
        3. A single course on the plan might satisfy a prerequisite (e.g., "Software Engineering" for SE103).
        4. Sometimes, a collection of courses implies coverage (e.g., "Programming 1" + "Advanced Programming" for SE109).
        5. Be realistic. A "Clinical Pharmacy" or "Geology" degree is highly unlikely to cover any of these specific computer science topics. A "Computer Science" or "Computer Engineering" degree is very likely to cover most or all.
        6. Return ONLY a single, valid JSON object with your final decision. The keys in the 'met_prerequisites' list must be the official codes (e.g., "SE101", "SE102").

        **APPLICANT'S ACADEMIC PLAN:**
        ---
        {transcript_text[:50000]}
        ---

        **RETURN ONLY THE FOLLOWING JSON STRUCTURE:**
        {{
          "met_prerequisites": ["A list of course codes (e.g., 'SE101', 'SE105') that are met"],
          "degree_type": "A brief, inferred degree title (e.g., computer_science, geology, business)"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            return json.loads(json_match.group(0)) if json_match else json.loads(result_text)
        except Exception as e:
            print(f"❌ LLM Analysis Error: {e}")
            return None

# ==============================================================================
# --- 4. SE-ADVISOR SYSTEM (PAPER-ALIGNED) ---
# ==============================================================================

class SEAdvisor:
    """The main SE-Advisor system, designed to replicate the paper's architecture."""
    def __init__(self, llm_reasoner: NuancedLLMReasoner):
        self.llm_reasoner = llm_reasoner

    def assess_applicant(self, transcript_text: str) -> tuple:
        print("🧠 Invoking LLM Reasoning Engine...")
        llm_analysis = self.llm_reasoner.analyze(transcript_text)
        if not llm_analysis:
            return [], [], "error", "unknown"
        met_codes = llm_analysis.get("met_prerequisites", [])
        degree_type = llm_analysis.get("degree_type", "unknown")
        met_prerequisites = [PREREQUISITE_COURSES[code] for code in met_codes if code in PREREQUISITE_COURSES]
        all_prereq_names = list(PREREQUISITE_COURSES.values())
        missing_prerequisites = [name for name in all_prereq_names if name not in met_prerequisites]
        print(f"✅ LLM analysis complete. Inferred degree: {degree_type}. Met: {len(met_prerequisites)}/5.")
        return missing_prerequisites, met_prerequisites, "high", degree_type

# ==============================================================================
# --- 5. REPORTING & EVALUATION ---
# ==============================================================================

def print_final_analysis(filename, results):
    """Prints a user-friendly summary of the final analysis."""
    missing, met, _, degree_type = results
    total_prereqs = len(PREREQUISITE_COURSES)
    print("\n" + "-" * 70 + f"\n📄 FINAL ASSESSMENT FOR: {filename}\n" + "-" * 70)
    print(f"🎓 Inferred Degree Program: {degree_type.replace('_', ' ').title()}")
    print(f"✅ Prerequisites Fulfilled ({len(met)}/{total_prereqs}):")
    if met:
        for prereq in sorted(met): print(f"  - {prereq}")
    else:
        print("  - None")
    print(f"\n📚 Recommended Complementary Courses ({len(missing)}/{total_prereqs}):")
    if missing:
        for prereq in sorted(missing): print(f"  - {prereq}")
    else:
        print("  - None. Applicant is fully prepared.")
    print("-" * 70)


# ==============================================================================
# --- 6. MAIN EXECUTION ---
# ==============================================================================

def main():
    """Main execution block to run the SE-Advisor and evaluate its performance."""
    if not NVIDIA_API_KEY:
        print("❌ FATAL ERROR: NVIDIA_API_KEY is not set.")
        return

    try:
        from google.colab import drive
        print("🚀 Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        print("✅ Drive mounted successfully.")
    except ImportError:
        print("ℹ️ Not running in Google Colab, skipping Drive mount.")

    llm_reasoner = NuancedLLMReasoner(api_key=NVIDIA_API_KEY)
    advisor = SEAdvisor(llm_reasoner=llm_reasoner)

    all_true_labels, all_predictions = [], []
    csv_results = []

    print("\n" + "=" * 80)
    print("🚀 Starting SE-Advisor v11 (Llama3, Paper-Aligned with CSV Export & Matrix)")
    print(f"🎯 Evaluating against {len(PREREQUISITE_COURSES)} prerequisites and an authoritative ground truth.")
    print("=" * 80 + "\n")

    if not os.path.exists(TRANSCRIPT_FOLDER_PATH):
        print(f"❌ ERROR: Transcript folder not found at '{TRANSCRIPT_FOLDER_PATH}'")
        return

    for filename, true_codes in sorted(GROUND_TRUTH.items()):
        file_path = os.path.join(TRANSCRIPT_FOLDER_PATH, filename)
        if not os.path.exists(file_path):
            print(f"❓ SKIPPING: File '{filename}' not found in Drive.")
            continue
            
        print(f"🔄 Processing Applicant File: {filename}")
        raw_text = get_document_text(file_path)
        if not raw_text: continue

        missing, met, _, _ = advisor.assess_applicant(raw_text)
        print_final_analysis(filename, (missing, met, "high", ""))
        
        all_prereq_names = list(PREREQUISITE_COURSES.values())
        met_true_names = [PREREQUISITE_COURSES[code] for code in true_codes]
        actual_complementary = [name for name in all_prereq_names if name not in met_true_names]
        
        csv_results.append({
            'filename': filename,
            'predicted_complementary_courses': ' | '.join(sorted(missing)),
            'actual_complementary_courses': ' | '.join(sorted(actual_complementary))
        })

        y_true = [1 if name in met_true_names else 0 for name in all_prereq_names]
        y_pred = [1 if name in met else 0 for name in all_prereq_names]
        all_true_labels.extend(y_true)
        all_predictions.extend(y_pred)

    # --- Final Performance Evaluation ---
    if all_true_labels:
        precision = precision_score(all_true_labels, all_predictions, zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
        
        print("\n" + "="*80)
        print("🎯 FINAL PERFORMANCE (vs. University Board Decisions)")
        print("="*80)
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1-Score:  {f1:.2f}")
        
        # ** NEW: Added detailed confusion matrix breakdown **
        print(f"\n📊 AGGREGATE CONFUSION MATRIX:")
        tp = sum(1 for t, p in zip(all_true_labels, all_predictions) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(all_true_labels, all_predictions) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(all_true_labels, all_predictions) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(all_true_labels, all_predictions) if t == 0 and p == 0)
        print(f"• True Positives:  {tp}")
        print(f"• False Positives: {fp}")
        print(f"• False Negatives: {fn}")
        print(f"• True Negatives:  {tn}")
     
    # --- Save results to CSV ---
    if csv_results:
        print("\n" + "="*80)
        print(f"💾 Saving results to CSV file...")
        try:
            os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
            results_df = pd.DataFrame(csv_results)
            results_df.to_csv(CSV_OUTPUT_PATH, index=False)
            print(f"✅ Successfully saved results to: {CSV_OUTPUT_PATH}")
        except Exception as e:
            print(f"❌ FAILED to save CSV file. Error: {e}")
        print("="*80)


if __name__ == '__main__':
    main()
