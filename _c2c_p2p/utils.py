# -*- coding: utf-8 -*-
"""
Created on Tuesday 10 25 19:05:29 2022

@author: Jeff
"""
import logging
import threading as mt
from threading import Lock
import time
import traceback

import pandas as pd

from .common import OUTPUT_LOC

BATCH_SIZE = 100000


class BufferList:

    def __init__(self):
        self._buffer = []
        self._write_stack = []
        self._lock = Lock()
        self._count = 0
        self._daemon = mt.Thread(target=self._run).start()
        self._writer = mt.Thread(target=self._write).start()

    def append(self, item):
        self._lock.acquire()
        self._buffer.append(item)
        self._lock.release()

    def _run(self):
        try:
            while True:
                if len(self._buffer) > BATCH_SIZE:
                    self._lock.acquire()
                    self._write_stack.append(self._buffer[:BATCH_SIZE])
                    self._buffer = self._buffer[BATCH_SIZE:]
                    self._lock.release()
                time.sleep(10)

        except Exception as esp:
            logging.error(traceback.format_exc())

    def _write(self):
        try:
            while True:
                if self._write_stack:
                    logging.debug(f'writing data {self._count}~{self._count+BATCH_SIZE-1}')
                    lines = pd.DataFrame(self._write_stack.pop(0))
                    lines.to_csv(
                        f'{OUTPUT_LOC}/_temp_file/{self._count}~{self._count+BATCH_SIZE-1}.csv', encoding='utf-8-sig')
                    self._count += BATCH_SIZE
                    del lines
                time.sleep(10)

        except Exception as esp:
            logging.error(traceback.format_exc())
