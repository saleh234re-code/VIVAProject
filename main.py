from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import services

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Memory Storage
# =========================
latest_project_data = []
latest_audio_path = None


# =========================
# 1️⃣ Upload PDF / DOCX
# =========================
@app.post("/upload-project")
async def upload_project(file: UploadFile = File(...)):
    global latest_project_data

    temp_filename = f"temp_{file.filename}"

    try:
        with open(temp_filename, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if temp_filename.endswith(".pdf"):
            text = services.read_pdf(temp_filename)

        elif temp_filename.endswith(".docx"):
            text = services.read_docx(temp_filename)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text")

        chunks = services.split_text(text)
        full_qa_list = services.generate_questions(chunks[0])

        if not full_qa_list:
            raise HTTPException(status_code=500, detail="AI failed to generate questions")

        latest_project_data = full_qa_list
        services.save_project_result(full_qa_list)

        questions_only = [item["question"] for item in full_qa_list]

        return {
            "status": "success",
            "questions": questions_only
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# =========================
# 2️⃣ Upload Video
# =========================
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    global latest_audio_path

    temp_filename = f"temp_{file.filename}"

    try:
        with open(temp_filename, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = services.process_video(temp_filename)
        latest_audio_path = result["audio_path"]

        return {
            "status": "success",
            "message": "Audio extracted successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# =========================
# 3️⃣ Get Extracted Audio
# =========================
@app.get("/get-video-analysis")
async def get_video_analysis():
    result = services.get_video_result()

    if not result:
        raise HTTPException(status_code=404, detail="No video analysis found")

    return result

# =========================
# 4️⃣ Evaluation
# =========================
class EvaluationRequest(BaseModel):
    question_index: int
    user_answer: str


@app.post("/evaluation")
async def evaluate_answer(data: EvaluationRequest):
    project_data = services.get_project_result()

    if not project_data:
        raise HTTPException(status_code=400, detail="No project data found")

    if data.question_index >= len(project_data):
        raise HTTPException(status_code=400, detail="Invalid question index")

    correct_answer = project_data[data.question_index]["answer"]

    score = 1 if data.user_answer.strip().lower() == correct_answer.strip().lower() else 0

    return {
        "question": project_data[data.question_index]["question"],
        "your_answer": data.user_answer,
        "correct_answer": correct_answer,
        "score": score
    }


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)