import os
from typing import List, Tuple

from PIL import Image

from lib_cream_py import RawMask, ColorMask, decensor_image_variations, apply_variant, Logger


class CliLogger(Logger):
    def warn(self, id: str, info=None):
        match id:
            case _:
                print("[WARN]", "UNKNOWN", id)

    def info(self, id: str, info=None):
        match id:
            case "apply-variant":
                print("[INFO]", "Apply variant", info)
            case "generate-mask":
                print("[INFO]", "Generating mask")
            case "finished":
                print("[INFO]", "Finished processing")
            case "remove-alpha":
                print("[INFO]", "Removing alpha channel")
            case "find-regions":
                print("[INFO]", "Trying to find regions")
            case "decensor-segment":
                region_counter, region_count = info
                print("[INFO]", "Processing", region_counter, "segment of", region_count)
            case "restore-alpha":
                print("[INFO]", "Restoring alpha channel")
            case _:
                print("[INFO]", "UNKNOWN", id)

    def debug(self, id: str, info=None):
        match id:
            case "found-regions":
                region_count = info
                print("[DEBUG]", "Found", region_count, "regions")
            case _:
                print("[DEBUG]", "UNKNOWN", id)

    def error(self, id: str, info=None):
        match id:
            case "no-regions":
                print("[ERROR]", "Couldn't find any regions")
            case "missing-model":
                print(
                    "[ERROR] Missing Model, download model\nRead: https://github.com/deeppomf/DeepCreamPy/blob/master/docs/INSTALLATION.md#run-code-yourself")
            case "bounding-box-out-of-bounds":
                x1_square, y1_square, x2_square, y2_square = info
                print("[ERROR]", "Bounding box out of bounds", x1_square, y1_square, x2_square, y2_square)
            case _:
                print("[ERROR]", "UNKNOWN", id)


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


def process(path: str, out_path: str, mask: MaskInfo, model, variations: int, is_mosaic: bool, cleanup: bool, logger: Logger):
    imgs = get_imgs(path)
    if mask.file:
        imgs = join_mask(imgs, mask.mask_name)
        for (img, mask) in imgs:
            save_image = lambda i, out_img: out_img.save(generate_out_path(out_path, img.path, i))
            mask_gen = lambda i, x, y: RawMask(apply_variant(mask.img, i))
            decensor_image_variations(model, img.img, img.img, mask_gen, variations, is_mosaic, save_image, logger=logger)
            if cleanup:
                try:
                    img.delete()
                    mask.delete()
                except Exception as e:
                    print("[ERROR] Failed to delete img: ", img.path)
    else:
        for img in imgs:
            try:
                save_image = lambda i, out_img: out_img.save(generate_out_path(out_path, img.path, i))
                mask_gen = lambda i, ori, colored: ColorMask(colored if is_mosaic else ori, rgb=mask.rgb)
                decensor_image_variations(model, img.img, img.img, mask_gen, variations, is_mosaic, save_image, logger=logger)
            except Exception as e:
                print("[ERROR]", e)
            if cleanup:
                try:
                    img.delete()
                except Exception as e:
                    print("[ERROR] Failed to delete img: ", img.path)


class MyImage:
    def __init__(self, path: str):
        self.img = Image.open(path)
        self.path = path

    def delete(self):
        os.remove(self.path)


def get_imgs(path: str) -> list[MyImage]:
    img_list = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            img_list.append(MyImage(file_path))
        except Exception as e:
            print("[WARN] Failed to load image:", e)
    return img_list


def join_mask(imgs: List[MyImage], suffix: str) -> List[Tuple[MyImage, MyImage]]:
    no_suffix_imgs = {}
    mask_imgs = {}
    result = []

    for img in imgs:
        base_name = os.path.splitext(os.path.basename(img.path))[0]
        if base_name.endswith(suffix):
            mask_imgs[base_name[:-len(suffix)]] = img
        else:
            no_suffix_imgs[base_name] = img

    for base_name, img in no_suffix_imgs.items():
        if base_name in mask_imgs:
            mask_img = mask_imgs[base_name]
            result.append((img, mask_img))

    return result
