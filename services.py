import google.generativeai as genai
import PyPDF2
import docx
import os
import json
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"}
)



def read_pdf(file_path, max_pages=30):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = min(len(reader.pages), max_pages)
            for page in reader.pages[:num_pages]:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text


def read_docx(file_path, max_pages=30):
    text = ""
    try:
        doc = docx.Document(file_path)
        limit_words = max_pages * 500
        words_count = 0
        for para in doc.paragraphs:
            para_text = para.text + "\n"
            text += para_text
            words_count += len(para_text.split())
            if words_count >= limit_words: break
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""
    return text


def split_text(text, chunk_size=2000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]



def generate_questions(text_chunk):
    prompt = """
    Act as a university professor. Analyze the provided text.
    Generate 5 diverse questions (mix of factual, explanatory, and analytical).
    For each question, provide a clear Model Answer.

    IMPORTANT OUTPUT FORMAT:
    Output strictly a JSON list of objects. Each object must have keys: "question" and "answer".
    """
    try:
        full_prompt = f"{prompt}\n\nProject Content:\n---\n{text_chunk}\n---"
        response = model.generate_content(full_prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def evaluate_answers(combined_data):
    data_str = json.dumps(combined_data, ensure_ascii=False)
    prompt = f"""
    You are a university professor grading an exam.
    Input Data (JSON): Keys are Questions. Values are Lists: [Model Answer, Student Answer].
    ---
    {data_str}
    ---
    Task: Compare 'Student Answer' against 'Model Answer'.
    Output: JSON list of objects: {{"question": "...", "score": X, "feedback": "..."}} (Score out of 10).
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Error evaluating: {e}")
        return []