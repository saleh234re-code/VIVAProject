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

# =========================
# 5. Body Language — Start Session
# =========================
import threading
import numpy as np
from vivasum_pro_analyzer import VivasumProAnalyzer

# Global body language session state
bl_session = {
    "analyzer":   None,
    "running":    False,
    "thread":     None,
    "results":    None,
}

def run_bl_thread():
    """Runs body language analyzer in background thread."""
    bl_session["analyzer"].run_session()
    bl_session["running"] = False

    analyzer = bl_session["analyzer"]
    if not analyzer.history['total']:
        bl_session["results"] = {"error": "No data captured"}
        return

    avg_eye     = float(np.mean(analyzer.history['eye']))
    avg_hand    = float(np.mean(analyzer.history['hand']))
    avg_posture = float(np.mean(analyzer.history['posture']))
    final_score = float(np.mean(analyzer.history['total']))
    face_vis    = ((analyzer.total_frames - analyzer.no_face_frames)
                   / max(1, analyzer.total_frames)) * 100
    analysis_rate = (analyzer.analyzed_frames
                     / max(1, analyzer.total_frames)) * 100

    grade, emoji, tips = VivasumProAnalyzer._evaluate_grade(
        final_score, avg_eye, avg_hand, avg_posture
    )

    bl_session["results"] = {
        "eye_contact":     round(avg_eye, 1),
        "hand_gestures":   round(avg_hand, 1),
        "posture":         round(avg_posture, 1),
        "total_score":     round(final_score, 1),
        "grade":           grade,
        "grade_emoji":     emoji,
        "tips":            tips,
        "analyzed_frames": analyzer.analyzed_frames,
        "total_frames":    analyzer.total_frames,
        "analysis_rate":   round(analysis_rate, 1),
        "face_visibility": round(face_vis, 1),
    }

    # Save results to database
    try:
        from database import SessionLocal, BodyLanguageScore
        db = SessionLocal()
        bl_record = BodyLanguageScore(
            ImagePath="webcam_session",
            Score=round(final_score, 2),
            AnalysisText=f"Eye: {avg_eye:.1f}% | Hand: {avg_hand:.1f}% | Posture: {avg_posture:.1f}% | Grade: {grade}"
        )
        db.add(bl_record)
        db.commit()
        print(f"Body Language saved to DB with ID: {bl_record.id}")
        db.close()
    except Exception as e:
        print(f"Database error: {e}")


@app.post("/body-language/start")
async def start_body_language():
    """Starts the body language analysis session."""
    if bl_session["running"]:
        return {"status": "error", "message": "Session already running"}

    bl_session["results"]  = None
    bl_session["analyzer"] = VivasumProAnalyzer(window_size=20)
    bl_session["running"]  = True

    bl_session["thread"] = threading.Thread(target=run_bl_thread, daemon=True)
    bl_session["thread"].start()

    return {"status": "started", "message": "Body language session started"}


@app.get("/body-language/status")
async def body_language_status():
    """Returns current session status."""
    return {
        "running":     bl_session["running"],
        "has_results": bl_session["results"] is not None,
    }


@app.get("/body-language/report")
async def get_body_language_report():
    """Returns the full body language report after session ends."""
    if bl_session["running"]:
        return {"status": "in_progress", "message": "Session still running"}

    if bl_session["results"] is None:
        raise HTTPException(status_code=404, detail="No results yet")

    return {"status": "ready", "data": bl_session["results"]}
