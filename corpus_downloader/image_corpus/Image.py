import logging
import os
from io import BytesIO
from typing import List, Optional

import PIL.Image
import numpy as np
from PIL import ImageDraw, ImageOps
from corpus_downloader.requests import requests
from corpus_downloader.settings import Settings
from h5py import Dataset
from shapely.geometry import shape, Polygon, MultiPolygon
from skimage.transform import resize
from skimage.util import pad

LOGGER = logging.getLogger(__name__)


class Image(object):
    def __init__(self, image_id: str, labels: Optional[str] = None, annotations: Optional[List[dict]] = None, _auth=None):
        self.image_id = image_id
        self.label = labels
        self.annotations = annotations
        self._auth = _auth

    def __repr__(self):
        return self.image_id

    @property
    def bytes(self):
        url = Settings().GET_IMAGE_BYTES.format(image_id=self.image_id)
        return requests.get(url, auth=self._auth).content

    @staticmethod
    def get_patch(img: np.ndarray, tile_x: int, tile_y: int):
        size = img.shape[0] // tile_y, img.shape[1] // tile_x
        for y0, y1 in zip(range(0, img.shape[0] + 1, size[0]), range(0 + size[0], img.shape[0] + 1, size[0])):
            for x0, x1 in zip(range(0, img.shape[1] + 1, size[1]), range(0 + size[1], img.shape[1] + 1, size[1])):
                yield img[y0:y1, x0:x1]

    def save_to_jpeg(self, path: str, width: int, height: int, drop_smaller: bool, tile_x: int, tile_y: int):
        try:
            image = PIL.Image.open(BytesIO(self.bytes)).convert('RGB')
            if drop_smaller and (image.size[0] < width or image.size[1] < height):
                return
            elif tile_x * tile_y == 1:
                image_path = os.path.join(path, self.image_id + '.jpg')
                image = self.pad_and_resize_image(np.array(image), width=width, height=height)
                PIL.Image.fromarray(image).save(image_path)
            else:
                image_path = os.path.join(path, self.image_id + '-{patch_nb}' + '.jpg')
                image = self.pad_and_resize_image(np.array(image), width=width, height=height)
                for i, patch in enumerate(self.get_patch(image, tile_y=tile_x, tile_x=tile_x)):
                    image_path = image_path.format(patch_nb=i)
                    PIL.Image.fromarray(patch).save(image_path)
        except Exception as e:
            LOGGER.exception(e)

    def _get_mask_array(self, annotations, height, width, label_map):
        masks = {label: PIL.Image.new('L', (height, width)) for label in label_map}
        for annotation in annotations:
            if "label" in annotation["properties"].keys():
                # get a drawer for this layer
                draw = ImageDraw.Draw(masks[annotation["properties"]["label"]])
                # parse polygon
                polygon = shape(annotation['geometry'])

                # draw real label
                if type(polygon) == Polygon:
                    draw.polygon(list(zip(*polygon.exterior.xy)), fill=255)
                    for interior in polygon.interiors:
                        draw.polygon(list(zip(*interior.xy)), fill=0)
                elif type(polygon) == MultiPolygon:
                    for poly in polygon:
                        draw.polygon(list(zip(*poly.exterior.xy)), fill=255)
                        for interior in poly.interiors:
                            draw.polygon(list(zip(*interior.xy)), fill=0)
                else:
                    raise NotImplementedError

        # flip data because of PIL coordinate system
        for label in masks:
            masks[label] = ImageOps.flip(masks[label])
        # create a list of mask in the right order in order to create a stacked numpy
        masks_list = []
        for label in label_map:
            masks_list.append(masks[label])

        # stack all layer
        numpy_mask = np.stack(masks_list, axis=-1)

        # append the remaining layer
        remaining = np.invert(np.any(numpy_mask, axis=-1)).astype(np.uint8) * 255

        # add remaining layer
        numpy_mask = np.append(numpy_mask, np.expand_dims(remaining, -1), axis=-1)
        return numpy_mask

    @staticmethod
    def resize_with_ratio(img: np.ndarray, width: int, height: int) -> np.ndarray:
        g_dim = np.argmax(img.shape)
        resize_ratio = (np.array(img.shape[:2]) / np.array([height, width]))[g_dim]
        new_size = (np.array(img.shape[:2]) / resize_ratio).astype(np.int)
        return resize(img, new_size, order=3, preserve_range=True, mode='constant').astype(np.uint8)

    @staticmethod
    def pad_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
        pad_height = (height - img.shape[0]) // 2
        pad_width = (width - img.shape[1]) // 2
        # the addition is here to make sure the resulting size is not affected by  '//2'
        pad_dim = ((pad_height, pad_height + (height - 2 * pad_height - img.shape[0])),
                   (pad_width, pad_width + (width - 2 * pad_width - img.shape[1])),
                   (0, 0))

        return pad(img, pad_width=pad_dim, mode="constant")

    @staticmethod
    def pad_and_resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
        img = img.copy()
        img = Image.resize_with_ratio(img, width=width, height=height)
        img = Image.pad_image(img, width=width, height=height)
        return img

    def save_to_h5(self, data: Dataset, label: Dataset, start_index: int, width: int, height: int, drop_smaller: bool, label_map: dict, tile_x: int, tile_y: int):
        image = PIL.Image.open(BytesIO(self.bytes)).convert('RGB')
        if drop_smaller and (image.size[0] < width or image.size[1] < height):
            return
        else:
            np_image = np.array(image)
            mask = self._get_mask_array(annotations=self.annotations, height=np_image.shape[0], width=np_image.shape[1], label_map=label_map)
            np_image = self.pad_and_resize_image(img=np_image, height=height, width=width)
            mask = self.pad_and_resize_image(img=mask, height=height, width=width)
            for i, patch in enumerate(self.get_patch(np_image, tile_x=tile_x, tile_y=tile_y)):
                data[start_index + i, ...] = patch
            for i, patch in enumerate(self.get_patch(mask, tile_x=tile_x, tile_y=tile_y)):
                label[start_index + i, ...] = patch
