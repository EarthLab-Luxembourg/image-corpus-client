import logging
from typing import Iterator, Optional

from io import StringIO
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def log_iter(iterable: Iterator, total: Optional[int] = None):
"""Create a log visiable iteration using TQDM

:param iterable: iterable elements
:type iterable: Iterator
:param total: total number of elements in the iterable, defaults to None
:param total: Optional[int], optional
:return: tqdm object as log compatible object
:rtype: tqdm class
"""
    class LogFileObject(StringIO):
        def write(self, log):
            log = log.replace('\r', '').replace('\n', '')
            if len(log) > 0:
                logging.info(log)

        def flush(self, *args, **kwargs):
            pass

    return tqdm(iterable, file=LogFileObject(), mininterval=3, ascii=True, ncols=80, total=total)
