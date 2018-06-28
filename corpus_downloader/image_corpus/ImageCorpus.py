import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import makedirs
from random import random
from typing import List, Optional

import h5py
import numpy as np
from requests.compat import quote

from corpus_downloader.requests import requests
from corpus_downloader.utils.log_iter import log_iter
from . import Image
from ..auth import Auth
from ..settings import Settings

LOGGER = logging.getLogger(__name__)


class ImageCorpus(object):
    def __init__(self, corpus_id: str, label_type: str, name: str, labels: List[str], labelled_count: int, _auth=None):
        self.corpus_id = quote(corpus_id)
        self.label_type = label_type
        self.name = name
        self.labels = labels
        self.labelled_count = labelled_count
        self._auth = _auth

    @classmethod
    def from_id(cls, corpus_id: str, auth: Auth, output: Optional[str]):
        url = Settings().GET_CORPUS_URL.format(corpus_id=corpus_id)
        body = requests.get(url, auth=auth).json()
        return cls(corpus_id=corpus_id, label_type=body['label_type'], name=output or body['name'], labels=body['labels'], labelled_count=body['images_labelled_count'], _auth=auth)

    def download(self, test_size: float, width: int, height: int, drop_smaller: bool, tile_x: int, tile_y: int):
        LOGGER.info("Downloading {}".format(self.name))
        if drop_smaller:
            LOGGER.info("Ignore file lower than {}.{}".format(width, height))
        if self.label_type == 'CLASSIFICATION':
            self._download_classification(test_size=test_size, width=width, height=height, drop_smaller=drop_smaller, tile_x=tile_x, tile_y=tile_y)
        if self.label_type == 'SEMANTIC_SEGMENTATION':
            self._download_semantic_seg(test_size=test_size, width=width, height=height, drop_smaller=drop_smaller, tile_x=tile_x, tile_y=tile_y)
        else:
            raise NotImplementedError('Don\'t know how to download {} corpus'.format(self.label_type))

    def _download_classification(self, test_size: float, width: int, height: int, drop_smaller: bool, tile_x: int, tile_y: int):
        futures = []
        with ThreadPoolExecutor(8) as executor:
            makedirs(self.name, exist_ok=True)
            for image in self.images:
                if random() > test_size:
                    image_path = os.path.join(self.name, "train", image.label)
                    makedirs(image_path, exist_ok=True)
                    futures.append(executor.submit(image.save_to_jpeg, path=image_path, width=width, height=height, drop_smaller=drop_smaller, tile_x=tile_x, tile_y=tile_y))
                else:
                    image_path = os.path.join(self.name, "test", image.label)
                    makedirs(image_path, exist_ok=True)
                    futures.append(executor.submit(image.save_to_jpeg, path=image_path, width=width, height=height, drop_smaller=drop_smaller, tile_x=tile_x, tile_y=tile_y))
            for future in log_iter(as_completed(futures, timeout=60 * 60), total=len(futures)):
                future.result()  # raise exception

    def _download_semantic_seg(self, test_size: float, width: int, height: int, drop_smaller: bool, tile_x: int, tile_y: int):
        os.makedirs(self.name, exist_ok=True)
        label_map = {name: id for name, id in zip(self.labels, range(len(self.labels)))}
        with h5py.File(os.path.join(self.name, self.name + '.h5'), "w") as file:
            patch_height, patch_width = height // tile_y, width // tile_x
            train_data = file.create_dataset("train/data", shape=(100, patch_height, patch_width, 3), maxshape=(None, patch_height, patch_width, None), dtype=np.uint8, compression="gzip")
            eval_data = file.create_dataset("eval/data", shape=(100, patch_height, patch_width, 3), maxshape=(None, patch_height, patch_width, None), dtype=np.uint8, compression="gzip")
            # Shape is None, None, None, len(labels) + 1 since we need to keep 1 channel for the remaining layer
            train_label = file.create_dataset("train/label", shape=(100, patch_height, patch_width, len(self.labels) + 1), maxshape=(None, patch_height, patch_width, len(self.labels) + 1),
                                              dtype=np.uint8, compression="gzip")
            eval_label = file.create_dataset("eval/label", shape=(100, patch_height, patch_width, len(self.labels) + 1), maxshape=(None, patch_height, patch_width, len(self.labels) + 1),
                                             dtype=np.uint8, compression="gzip")
            file.attrs['label-map'] = json.dumps(label_map)
            tr_index, ev_index = 0, 0
            with ThreadPoolExecutor(8) as executor:
                futures = []
                for image in log_iter(self.images, total=self.labelled_count):
                    nb_img_to_append = tile_x * tile_y
                    if random() > test_size:
                        if train_data.shape[0] - tr_index - nb_img_to_append < 5:
                            train_data.resize(tr_index + 100, axis=0)
                            train_label.resize(tr_index + 100, axis=0)
                        futures.append(
                                executor.submit(image.save_to_h5, data=train_data, label=train_label, start_index=tr_index, label_map=label_map, height=height, width=width, drop_smaller=drop_smaller,
                                                tile_x=tile_x, tile_y=tile_y))
                        tr_index += nb_img_to_append
                    else:
                        if eval_data.shape[0] - ev_index - nb_img_to_append < 5:
                            eval_data.resize(ev_index + 100, axis=0)
                            eval_label.resize(ev_index + 100, axis=0)
                        futures.append(
                                executor.submit(image.save_to_h5, data=eval_data, label=eval_label, start_index=ev_index, label_map=label_map, height=height, width=width, drop_smaller=drop_smaller,
                                                tile_x=tile_x, tile_y=tile_y))
                        ev_index += nb_img_to_append
            for future in as_completed(futures, timeout=60 * 60):
                future.result()  # raise exception
            train_data.resize(tr_index, axis=0)
            train_label.resize(tr_index, axis=0)
            eval_data.resize(ev_index, axis=0)
            eval_label.resize(ev_index, axis=0)

    @property
    def images(self):
        offset = 0
        url = Settings().GET_ALL_IMAGES_URL.format(corpus_id=self.corpus_id)
        limit = 5000
        while True:
            LOGGER.info("fetching new page of images")
            images = requests.get(url, auth=self._auth, params={'limit': limit, 'offset': offset, 'labelled': True}).json()
            annotated_images = []
            if len(images) > 0:
                for image in images:
                    if 'labels' in image and image['labels'] != '':
                        yield Image(image_id=image['id'], labels=image['labels'], _auth=self._auth)
                    elif 'annotated' in image and image['annotated'] is True:
                        annotated_images.append(image)
                with ThreadPoolExecutor(32) as executor:
                    futures = {}
                    for image in annotated_images:
                        annotation_url = Settings().GET_IMAGE_ANNOTATION.format(image_id=image['id'])
                        futures[executor.submit(requests.get, annotation_url, auth=self._auth)] = image['id']
                    for annotation in as_completed(futures.keys(), 20):
                        image_id = futures[annotation]
                        annotations = annotation.result().json()
                        if len(annotations) > 0:
                            yield Image(image_id=image_id, annotations=annotations, _auth=self._auth)
                offset += limit
            else:
                return
