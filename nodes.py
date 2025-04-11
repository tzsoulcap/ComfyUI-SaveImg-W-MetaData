import os
import hashlib
from datetime import datetime
import json
import piexif
import piexif.helper
from PIL import Image, ExifTags, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
import node_helpers
import torch


def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading the entire file into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp


def make_pathname(filename, counter, time_format):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%counter", str(counter))
    return filename


def make_filename(filename, counter, time_format):
    filename = make_pathname(filename, counter, time_format)

    return get_timestamp(time_format) if filename == "" else filename

class TagImageNode:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_tags"
    CATEGORY = "CAPImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tags": ("STRING", {"default": "", "multiline": True})}}

    def get_tags(self, tags):
        return (tags,)

class SeedGenerator:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_seed"
    CATEGORY = "CAPImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    def get_seed(self, seed):
        return (seed,)


class StringLiteral:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_string"
    CATEGORY = "CAPImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"default": "", "multiline": True})}}

    def get_string(self, string):
        return (string,)


class SizeLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "CAPImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8})}}

    def get_int(self, int):
        return (int,)


class IntLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "CAPImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 0, "min": 0, "max": 1000000})}}

    def get_int(self, int):
        return (int,)


class CfgLiteral:
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"
    CATEGORY = "CAPImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})}}

    def get_float(self, float):
        return (float,)


class CheckpointSelector:
    CATEGORY = 'CAPImageSaverTools/utils'
    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"),)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),}}

    def get_names(self, ckpt_name):
        return (ckpt_name,)


class SamplerSelector:
    CATEGORY = 'CAPImageSaverTools/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (comfy.samplers.KSampler.SAMPLERS,)}}

    def get_names(self, sampler_name):
        return (sampler_name,)


class SchedulerSelector:
    CATEGORY = 'CAPImageSaverTools/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (comfy.samplers.KSampler.SCHEDULERS,)}}

    def get_names(self, scheduler):
        return (scheduler,)

class LoadImageWithMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "image": (sorted(files),  {"image_upload": True}),   
                    },
                }
    CATEGORY = "CAPLoadImageWithMetadata"
    ''' Return order:
        positive prompt(string), negative prompt(string), seed(int), steps(int), cfg(float), 
        width(int), height(int)
    '''
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "title", "tags")
    FUNCTION = "get_image_data"

    def get_image_data(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)

        exif_dict = piexif.load(img.info["exif"])
        title = exif_dict["0th"][piexif.ImageIFD.ImageDescription].decode('utf-8')
        # tags = exif_dict["0th"][piexif.ImageIFD.XPKeywords]
        tags = ""
        print("--------------------------------")
        print(f"title: {title}")
        print(f"tags: {tags}")

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        
        return (output_image, output_mask, title, tags)


class ImageSaveWithMetadata:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": f'%time', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'],),
            },
            "optional": {
                # "positive": ("STRING", {"default": 'unknown', "multiline": True}),
                # "negative": ("STRING", {"default": 'unknown', "multiline": True}),
                "title": ("STRING", {"default": 'unknown', "multiline": True}),
                "tags": ("STRING", {"default": '', "multiline": True}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "counter": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "CAPImageSaverTools"

    def save_files(self, images, title, tags, quality_jpeg_or_webp,
                   lossless_webp, counter, filename, path, extension, time_format, prompt=None, extra_pnginfo=None):
        
        print(f"title: {title}")
        print(f"tags: {tags}")

        filename = make_filename(filename, counter, time_format)

        path = make_pathname(path, counter, time_format)

        comment = f"{handle_whitespace(title)}"
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        filenames = self.save_images(images, output_path, filename, comment, tags, extension, quality_jpeg_or_webp, lossless_webp, prompt, extra_pnginfo)

        subfolder = os.path.normpath(path)
        return {"ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)}}

    def save_images(self, images, output_path, filename_prefix, comment, tags, extension, quality_jpeg_or_webp, lossless_webp, prompt=None, extra_pnginfo=None) -> list[str]:
        img_count = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            if extension == 'png':
                metadata = PngInfo()
                metadata.add_text("parameters", comment)
                metadata.add_text("tags", handle_whitespace(tags))

                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename = f"{filename_prefix}.png"
                img.save(os.path.join(output_path, filename), pnginfo=metadata, compress_level=1)
            else:
                filename = f"{filename_prefix}.{extension}"
                file = os.path.join(output_path, filename)
                img.save(file, quality=quality_jpeg_or_webp, lossless=lossless_webp)
                # exif_dict = piexif.load(file)
                image = Image.open(file)
                # Load existing EXIF data or create a new EXIF dictionary if none exists
                try:
                    exif_dict = piexif.load(image.info["exif"])
                except KeyError:
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

                description = f"{comment}"
                list_tags = f"{tags}"
                print(f"list_tags: {list_tags}")
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
                exif_dict["0th"][piexif.ImageIFD.XPKeywords] = list_tags.encode('utf-16le')
                exif_bytes = piexif.dump(exif_dict)
                # piexif.insert(exif_bytes, file)
                image.save(file, exif=exif_bytes, quality=quality_jpeg_or_webp)

            paths.append(filename)
            img_count += 1
        return paths


NODE_CLASS_MAPPINGS = {
    "CAP Checkpoint Selector": CheckpointSelector,
    "CAP Save Image w/Metadata": ImageSaveWithMetadata,
    "CAP Load Image with Metadata": LoadImageWithMetadata,
    "CAP Tag Image": TagImageNode,
    "CAP Sampler Selector": SamplerSelector,
    "CAP Scheduler Selector": SchedulerSelector,
    "CAP Seed Generator": SeedGenerator,
    "CAP String Literal": StringLiteral,
    "CAP Width/Height Literal": SizeLiteral,
    "CAP Cfg Literal": CfgLiteral,
    "CAP Int Literal": IntLiteral,
}
