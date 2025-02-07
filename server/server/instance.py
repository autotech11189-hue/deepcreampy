import asyncio
import os
from asyncio import Event, Lock
from typing import List, Optional, Callable

from PIL import Image
from fastapi import HTTPException
from pydantic import BaseModel

from lib_cream_py import ColorMask, RawMask
from lib_cream_py import InpaintNN, decensor_image_variations
from lib_cream_py.util import apply_variant
from server.server.task import DecensorItem
from ..local import generate_out_path, MaskInfo

NotifyType = Optional[Callable[[int, Optional[bytes]], None]]


def get_img(id: str) -> Image.Image:
    return Image.open(get_file_path(id))


def get_file_path(id: str) -> str:
    for file in os.listdir("./temp"):
        filename_without_ext, _ = os.path.splitext(file)

        if filename_without_ext == id:
            return os.path.join("./temp", file)

    raise HTTPException(status_code=422, detail="File " + id + " not found")


class ExecutorInstance(BaseModel):
    _model_mosaic: Optional[InpaintNN] = None
    _model_bar: Optional[InpaintNN] = None

    busy: bool = False

    @property
    def model_mosaic(self) -> InpaintNN:
        if self._model_mosaic is None:
            self._model_mosaic = InpaintNN("./models/mosaic.keras")
        return self._model_mosaic

    @property
    def model_bar(self) -> InpaintNN:
        if self._model_bar is None:
            self._model_bar = InpaintNN("./models/bar.keras")
        return self._model_bar

    def free_executor(self):
        self.busy = False

    async def sent(self, items: list[DecensorItem]):
        await self.sent_stream(items, None)

    async def sent_stream(self, items: list[DecensorItem], sender: NotifyType):
        # todo: sent progres
        # todo: cancel inbetween images
        for index, item in enumerate(items):
            sender(6, bytes(index))
            img = get_img(item.img_id)
            save_image = lambda i, out_img: out_img.save(generate_out_path(item.output, get_file_path(item.img_id), i))

            mask = MaskInfo(item.mask)
            if mask.file:
                mask_gen = lambda i, ori, colored: RawMask(apply_variant(get_img(mask.file), i))
            else:
                mask_gen = lambda i, ori, colored: ColorMask(colored if item.is_mosaic else ori, rgb=mask.rgb)
            await asyncio.to_thread(
                decensor_image_variations(self.model_mosaic if item.is_moasic else self.model_bar, img, img, mask_gen,
                                          item.variations, item.is_moasic, save_image))
            sender(7, bytes(index))


class Executors:
    def __init__(self):
        self.list: List[ExecutorInstance] = []
        self.lock: Lock = Lock()
        self.event = Event()

    def register(self, instance: ExecutorInstance):
        self.list.append(instance)

    def free_executors(self) -> int:
        return len([item for item in self.list if not item.busy])

    async def _find_instance(self):
        while True:
            instance = next((x for x in self.list if x.busy == False), None)
            if instance is not None:
                return instance
            # todo: cricial error: warn should never happen
            await self.event.wait()

    async def find_executor(self) -> ExecutorInstance:
        async with self.lock:  # Using async with for lock management
            instance = await self._find_instance()
            instance.busy = True
            return instance

    async def free_executor(self, instance: ExecutorInstance):
        from myqueue import task_queue
        instance.free_executor()
        self.event.set()
        self.event.clear()
        await task_queue.update_event()


executor_instances: Executors = Executors()
