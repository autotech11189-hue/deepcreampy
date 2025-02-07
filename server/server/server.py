import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from .myqueue import task_queue
from .task import DecensorItem

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

DATA_DIR = Path("temp")
if DATA_DIR.exists():
    import shutil

    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/upload_files", response_model=list[str])
async def upload_files(files: list[UploadFile]):
    ids = []
    for file in files:
        img_id = str(uuid.uuid4())
        ids.append(img_id)
        ext = Path(file.filename).suffix
        file_path = DATA_DIR / f"{img_id}{ext}"
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())
    return ids


class DecensorRequest(BaseModel):
    imgs: list[DecensorItem]


@app.put("/decensor", response_class=StreamingResponse)
async def decensor(files: DecensorRequest):
    pass


@app.get("/tasks", response_model=list[(str, int)])
async def tasks() -> list[(str, int)]:
    return [(x.item_id, len(x.items)) for x in task_queue.queue]


@app.delete("/task/{task_id}", response_class=StreamingResponse)
async def cancel_task(task_id: str):
    task = next((x for x in task_queue.queue if x.item_id == task_id))
    if task is None:
        raise HTTPException(status_code=422, detail="Task not found")
    task_queue.remove(task)
