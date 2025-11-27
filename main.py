from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import services  # بنستورد ملف الخدمات اللي عملناه

app = FastAPI()


# 1. Endpoint: رفع المشروع وتوليد الأسئلة
@app.post("/upload-project")
async def upload_project(file: UploadFile = File(...)):
    # حفظ الملف مؤقتاً
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # قراءة الملف باستخدام دوال services
        text = ""
        if temp_filename.endswith('.pdf'):
            text = services.read_pdf(temp_filename)
        elif temp_filename.endswith('.docx'):
            text = services.read_docx(temp_filename)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text")

        # تقسيم النص وتوليد الأسئلة (بناخد أول جزء فقط كمثال لسرعة الرد)
        chunks = services.split_text(text)
        questions = services.generate_questions(chunks[0])

        return {"status": "success", "data": questions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # تنظيف: مسح الملف المؤقت
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# نموذج البيانات للتصحيح
class EvaluateRequest(BaseModel):
    qa_pairs: dict  # {question: [model_ans, student_ans]}
    # ملحوظة: الـ Front-end هيبعت الداتا جاهزة بالشكل ده أو نعمل دالة تجميع هنا


# 2. Endpoint: تصحيح الإجابات
@app.post("/evaluate")
async def evaluate_student(request: EvaluateRequest):
    results = services.evaluate_answers(request.qa_pairs)
    return {"status": "success", "evaluation": results}


# لتشغيل السيرفر لو عملت run للملف ده مباشرة
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)