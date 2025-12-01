from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import services

app = FastAPI()

latest_project_data = []


@app.post("/upload-project")
async def upload_project(file: UploadFile = File(...)):
    global latest_project_data

    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)


        text = ""
        if temp_filename.endswith('.pdf'):
            text = services.read_pdf(temp_filename)
        elif temp_filename.endswith('.docx'):
            text = services.read_docx(temp_filename)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text")


        chunks = services.split_text(text)
        full_qa_list = services.generate_questions(chunks[0])


        latest_project_data = full_qa_list


        questions_only = [item['question'] for item in full_qa_list]

        return {
            "status": "success",
            "questions": questions_only
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


class StudentSubmission(BaseModel):
    student_answers: list[str]


@app.post("/evaluate")
async def evaluate_student(data: StudentSubmission):
    global latest_project_data

    if not latest_project_data:
        raise HTTPException(status_code=400, detail="No project uploaded yet. Please upload a file first.")

    if len(data.student_answers) != len(latest_project_data):
        raise HTTPException(status_code=400, detail="Answer count mismatch. Make sure you answered all questions.")

    combined_data = {}
    for i, item in enumerate(latest_project_data):
        question_text = item['question']
        model_answer = item['answer']
        student_answer = data.student_answers[i]

        combined_data[question_text] = [model_answer, student_answer]

    results = services.evaluate_answers(combined_data)

    return {"status": "success", "evaluation": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)