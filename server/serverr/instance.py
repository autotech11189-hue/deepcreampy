import asyncio
import os
from asyncio import Event, Lock
from typing import List, Optional, Callable

from PIL import Image
from fastapi import HTTPException

from lib_cream_py import InpaintNN, decensor_image_variations, Logger, apply_variant, ColorMask, RawMask
from .task import DecensorItem


class MaskInfo:
    def __init__(self, v: str):
        if v.startswith("rgb-"):
            self.file = False
            try:
                rgb_values = v[4:].split(",")
                self.rgb = tuple(int(value) for value in rgb_values)
                if not all(0 <= value <= 255 for value in self.rgb):
                    raise ValueError("RGB values must be between 0 and 255")
                if len(self.rgb) != 3:
                    raise ValueError("RGB values must be 3 numbers")
                self.rgb = (self.rgb[0], self.rgb[1], self.rgb[2])
            except ValueError:
                raise Exception("Invalid RGB format. Expected format: rgb-R,G,B")
        elif v.startswith("file-"):
            self.file = True
            self.mask_name = v[5:]
            if len(self.mask_name) == 0:
                raise ValueError("Mask name must not be empty")
        else:
            raise ValueError(f"Unknown mask type: {v}")


def generate_out_path(out_folder: str, file_path: str, index: int) -> str:
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))

    new_file_name = f"{file_name}_{index}{file_extension}"

    return os.path.join(out_folder, new_file_name)


NotifyType = Optional[Callable[[int, bytes], None]]


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
                self.sender(4, b"")
            case "finished":
                self.sender(5, b"")
            case "remove-alpha":
                self.sender(6, b"")
            case "find-regions":
                self.sender(7, b"")
            case "decensor-segment":
                region_counter, region_count = info
                self.sender(8, (bytes(region_counter) + bytes(region_count)))
            case "restore-alpha":
                self.sender(9, b"")
            case _:
                self.sender(10, id.encode('utf-8'))

    def debug(self, id: str, info=None):
        match id:
            case "found-regions":
                region_count = info
                self.sender(11, bytes(region_count))
            case _:
                self.sender(12, id.encode('utf-8'))

    def error(self, id: str, info=None):
        match id:
            case "no-regions":
                self.sender(13, b"")
            case "missing-model":
                # "Missing Model, download model\nRead: https://github.com/deeppomf/DeepCreamPy/blob/master/docs/INSTALLATION.md#run-code-yourself"
                self.sender(14, b"")
            case "bounding-box-out-of-bounds":
                x1_square, y1_square, x2_square, y2_square = info
                self.sender(15, b"")
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


class ExecutorInstance:
    _model_mosaic: Optional[InpaintNN] = None
    _model_bar: Optional[InpaintNN] = None

    busy: Optional[str] = None
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
            (self.model_mosaic if item.is_mosaic else self.model_bar).logger = logger
            if self.stop:
                sender(201, b"")
                self.stop = False
                break
            sender(1, bytes(index) + bytes(len(items)))
            img = get_img(item.img_id)
            os.makedirs(item.output, exist_ok=True)
            save_image = lambda i, out_img: out_img.save(generate_out_path(item.output, item.output_name, i))
            mask = MaskInfo(item.mask)
            if mask.file:
                mask_img = get_img(mask.mask_name)
                mask_gen = lambda i, ori, colored: RawMask(apply_variant(mask_img, i))
            else:
                mask_gen = lambda i, ori, colored: ColorMask(colored if item.is_mosaic else ori, rgb=mask.rgb)
            try:
                await asyncio.to_thread(
                    lambda: decensor_image_variations(
                        self.model_mosaic if item.is_mosaic else self.model_bar,
                        img, img, mask_gen, item.variations, item.is_mosaic, save_image, logger=logger
                    )
                )
            except Exception as e:
                sender(202, b"")
                return
            sender(2, bytes(index) + bytes(len(items)))
        sender(200, b"")


class Executors:
    def __init__(self):
        self.list: List[ExecutorInstance] = [ExecutorInstance()]
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
        from .myqueue import task_queue
        instance.free_executor()
        self.event.set()
        self.event.clear()
        await task_queue.update_event()


executor_instances: Executors = Executors()
