from fastapi import FastAPI, File, UploadFile
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from airflow.api.client.local_client import Client


ALLOWED_EXTENSIONS = set(["csv"])
UPLOAD_FOLDER = "./upload"
DOWNLOAD_FOLDER = "./download/"

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictParam(BaseModel):
    target_date: str


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/api/v1/upload-trial-input")
async def upload(file: UploadFile = File(...)):
    if file and allowed_file(file.filename):
        filename = file.filename
        fileobj = file.file
        upload_dir = open(os.path.join(UPLOAD_FOLDER, filename), "wb+")
        shutil.copyfileobj(fileobj, upload_dir)
        upload_dir.close()
        return {"result": "Success", "message": f"successfully uploaded: {filename}"}
    if file and not allowed_file(file.filename):
        return {
            "result": "Failed",
            "message": "file is not uploaded or extension is not allowed",
        }


@app.post("/api/v1/execute-trial-operation")
async def execute(param: PredictParam):
    try:
        c = Client(None, None)
        c.trigger_dag(
            dag_id="trial_operation",
            conf={
                "target_date": param.target_date,
                "work_dir": "~/Code/between_poc_and_production/notebooks/trial/",
            },
        )
        return {"result": "Success", "message": "successfully triggered"}
    except Exception as e:
        return {"result": "Failed", "message": f"error occured: {e}"}


@app.get("/api/v1/download-trial-output")
async def download():
    return FileResponse(
        DOWNLOAD_FOLDER + "pred_20210101.csv",
        filename="pred_20210101.csv",
        media_type="text/csv",
    )
