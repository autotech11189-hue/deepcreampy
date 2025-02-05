import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.server.task import DecensorItem

app = FastAPI()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/upload_files", response_model=list[str])
async def upload_files(files: list[UploadFile]):
    ids = []
    for file in files:
        img_id = uuid.uuid4()
        ids.append(img_id)
        ext = Path(file.filename).suffix
        file_path = DATA_DIR / f"{img_id}{ext}"
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())
    return ids



class DecensorRequest(BaseModel):
    imgs: list[DecensorItem]


@app.post("/decensor", response_model=StreamingResponse)
async def decensor(files: DecensorRequest):
    pass
