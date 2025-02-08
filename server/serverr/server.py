import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .instance import executor_instances, get_file_path
from .myqueue import task_queue, QueueElement, wait_in_queue
from .task import DecensorItem

app = FastAPI()
check_disconnect = False

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

DATA_DIR = Path("temp")
if DATA_DIR.exists():
    import shutil

    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/images", response_model=list[str])
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


@app.delete("/image/{image_id}")
async def delete_file(image_id: str):
    os.remove(get_file_path(image_id))


class DecensorRequest(BaseModel):
    imgs: list[DecensorItem]


async def stream(messages):
    while True:
        message = await messages.get()
        yield message
        # 200 == success, 201 == cancel, 202 == error
        if message[0] >= 200:
            break


def notify(code: int, data: bytes, messages: asyncio.Queue):
    encoded_result = code.to_bytes(1, 'big') + len(data).to_bytes(4, 'big') + data
    messages.put_nowait(encoded_result)


async def while_streaming(req: Request, items: list[DecensorItem]):
    task = QueueElement(req, items)
    task_queue.add_task(task)

    messages = asyncio.Queue()

    def notify_internal(code: int, data: bytes) -> None:
        notify(code, data, messages)

    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait_in_queue(task, notify_internal, check_disconnect=check_disconnect))
    return streaming_response


@app.post("/decensor", response_class=StreamingResponse)
async def decensor(req: Request, files: DecensorRequest):
    return await while_streaming(req, files.imgs)


@app.get("/tasks", response_model=list[(str, int)])
async def tasks() -> list[(str, int)]:
    return [(x.item_id, len(x.items)) for x in task_queue.queue] + [(x.busy, -1) for x in
                                                                    executor_instances.list if x.busy is not None]


@app.delete("/task/{task_id}", response_class=StreamingResponse)
async def cancel_task(task_id: str):
    task = next((x for x in task_queue.queue if x.item_id == task_id))
    if task is None:
        executor = next((x for x in executor_instances.list if x.busy == task_id))
        if executor is None:
            raise HTTPException(status_code=422, detail="Task not found")
        executor.stop = True
    task_queue.remove(task)
