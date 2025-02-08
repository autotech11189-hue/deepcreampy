import asyncio
import os
from asyncio import Event, Lock
from typing import List, Optional, Callable

from PIL import Image
from fastapi import HTTPException
from pydantic import BaseModel

from lib_cream_py import InpaintNN, decensor_image_variations, Logger, apply_variant, ColorMask, RawMask
from .task import DecensorItem
from ..local import generate_out_path, MaskInfo

NotifyType = Optional[Callable[[int, Optional[bytes]], None]]


# todo: is this thread save
class WebLogger(Logger):
    def __init__(self, sender):
        self.sender = sender

    def warn(self, id: str, info=None):
        match id:
            case _:
                self.sender(17, id.encode('utf-8'))

    def info(self, id: str, info=None):
        match id:
            case "apply-variant":
                self.sender(3, bytes(info))
            case "generate-mask":
                self.sender(4, None)
            case "finished":
                self.sender(5, None)
            case "remove-alpha":
                self.sender(6, None)
            case "find-regions":
                self.sender(7, None)
            case "decensor-segment":
                region_counter, region_count = info
                self.sender(8, (bytes(region_counter) + bytes(region_count)))
            case "restore-alpha":
                self.sender(9, None)
            case _:
                self.sender(10, id.encode('utf-8'))

    def debug(self, id: str, info=None):
        match id:
            case "found-regions":
                region_count = info
                self.sender(11, region_count.encode('utf-8'))
            case _:
                self.sender(12, id.encode('utf-8'))

    def error(self, id: str, info=None):
        match id:
            case "no-regions":
                self.sender(13, None)
            case "missing-model":
                # "Missing Model, download model\nRead: https://github.com/deeppomf/DeepCreamPy/blob/master/docs/INSTALLATION.md#run-code-yourself"
                self.sender(14, None)
            case "bounding-box-out-of-bounds":
                x1_square, y1_square, x2_square, y2_square = info
                self.sender(15, None)
            case _:
                self.sender(16, id.encode('utf-8'))


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

    busy: Optional[str] = False
    stop = False

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
        self.busy = None

    async def sent(self, items: list[DecensorItem]):
        await self.sent_stream(items, None)

    async def sent_stream(self, items: list[DecensorItem], sender: NotifyType):
        logger = WebLogger(sender) if sender else Logger()
        for index, item in enumerate(items):
            (self.model_mosaic if item.is_moasic else self.model_bar).logger = logger
            if self.stop:
                sender(201, None)
                self.stop = False
                break
            sender(1, bytes(index) + bytes(len(items)))
            img = get_img(item.img_id)
            save_image = lambda i, out_img: out_img.save(generate_out_path(item.output, get_file_path(item.img_id), i))
            mask = MaskInfo(item.mask)
            if mask.file:
                mask_img = get_img(mask.file)
                mask_gen = lambda i, ori, colored: RawMask(apply_variant(mask_img, i))
            else:
                mask_gen = lambda i, ori, colored: ColorMask(colored if item.is_mosaic else ori, rgb=mask.rgb)
            try:
                await asyncio.to_thread(
                    decensor_image_variations(self.model_mosaic if item.is_moasic else self.model_bar, img, img,
                                              mask_gen,
                                              item.variations, item.is_moasic, save_image, logger=logger))
            except Exception as e:
                sender(202, None)
                return
            sender(2, bytes(index) + bytes(len(items)))
        sender(200, None)


class Executors:
    def __init__(self):
        self.list: List[ExecutorInstance] = []
        self.lock: Lock = Lock()
        self.event = Event()

    def register(self, instance: ExecutorInstance):
        self.list.append(instance)

    def free_executors(self) -> int:
        return len([item for item in self.list if item.busy is None])

    async def _find_instance(self):
        while True:
            instance = next((x for x in self.list if x.busy is None), None)
            if instance is not None:
                return instance
            # todo: cricial error: warn should never happen
            await self.event.wait()

    async def find_executor(self, task_id: str) -> ExecutorInstance:
        async with self.lock:  # Using async with for lock management
            instance = await self._find_instance()
            instance.busy = task_id
            return instance

    async def free_executor(self, instance: ExecutorInstance):
        from myqueue import task_queue
        instance.free_executor()
        self.event.set()
        self.event.clear()
        await task_queue.update_event()


executor_instances: Executors = Executors()
