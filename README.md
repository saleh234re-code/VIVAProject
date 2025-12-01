# VIVA API - AI-Powered Project Discussion System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi&logoColor=white)
![Gemini AI](https://img.shields.io/badge/AI-Google%20Gemini-orange?logo=google&logoColor=white)
![Deployed](https://img.shields.io/badge/Deployed%20on-Railway-0B0D0E?logo=railway&logoColor=white)

## Project Description
This is the Backend API for the VIVA Graduation Project. It acts as an intelligent engine that simulates a university project discussion (Viva). The system allows students to upload their project documentation, generates relevant discussion questions using **Google's Gemini AI**, and evaluates student answers with grading and feedback.

## Key Features
* **Document Parsing:** Extracts text from project files (supports `.pdf` and `.docx`).
* **AI Question Generation:** Uses NLP (Gemini Pro) to analyze project content and generate diverse questions (factual, analytical, explanatory).
* **Automated Grading:** Compares student answers against model answers and provides a score (0-10) with constructive feedback.
* **High Performance:** Built with **FastAPI** for asynchronous processing and high speed.
* **Swagger Documentation:** Interactive API documentation for easy testing and integration.

## Tech Stack
* **Framework:** FastAPI
* **Language:** Python 3.10
* **AI Model:** Google Generative AI (Gemini 2.5 Flash)
* **File Handling:** PyPDF2, python-docx
* **Deployment:** Railway

---

## API Documentation & Usage

**Base URL (Live Server):** `https://web-production-d763e.up.railway.app`

**Interactive Docs (Swagger UI):** [Click here to view API Docs](https://web-production-d763e.up.railway.app/docs)

### 1️⃣ Upload Project & Generate Questions
* **Endpoint:** `POST /upload-project`
* **Description:** Uploads the project file and returns a list of generated questions.
* **Input:** `multipart/form-data` (Key: `file`)
* **Response Example:**
    ```json
    {
      "status": "success",
      "data": [
        {
          "question": "What is the main objective of the platform?",
          "answer": "The main objective is to provide..."
        }
      ]
    }
    ```

### 2️⃣ Evaluate Student Answers
* **Endpoint:** `POST /evaluate`
* **Description:** Grades the student's answers based on the AI model's knowledge.
* **Input (JSON):**
    ```json
    {
      "qa_pairs": {
        "Question Text": ["Model Answer", "Student Answer"]
      }
    }
    ```
* **Response Example:**
    ```json
    {
      "status": "success",
      "evaluation": [
        {
          "question": "Question Text",
          "score": 8,
          "feedback": "Good answer, but missed one key point..."
        }
      ]
    }
    ```

---

## Local Installation (For Developers)

If you want to run this project locally on your machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mustafaEssam10/VIVA_API.git](https://github.com/mustafaEssam10/VIVA_API.git)
    cd VIVA_API
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

5.  **Run the Server:**
    ```bash
    uvicorn main:app --reload
    ```

---

## 👤 Author
**Mustafa Essam** Computer Science Student | Software engineer 
[GitHub Profile](https://github.com/mustafaEssam10)
