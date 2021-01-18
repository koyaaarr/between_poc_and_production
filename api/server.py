from fastapi import FastAPI
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # リクエストbodyを定義するために必要

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


class LearnParam(BaseModel):
    test: str


# curl -X POST http://localhost:8000/hello -H "Content-Type: application/json" -d "{'test':'hello'}"
@app.post("/hello")  # methodとendpointの指定
async def hello(param: LearnParam):
    print(param)
    # subprocess.call("airflow dags trigger test_dag".split(" "))
    return {"text": param}
