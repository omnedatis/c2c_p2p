# -*- coding: utf-8 -*-
"""
Created on Tuesday 10 25 19:05:29 2022

@author: Jeff
"""
import json
import logging
import os
import pickle
import threading as mt
from threading import Lock
import time
import traceback
from typing import NamedTuple, Optional, Dict, Tuple

import pandas as pd

from _c2c_p2p import OUTPUT_LOC, SPLITER, PK, PK2, Dtypes, dataRangeType

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


class SetCode(NamedTuple):
    code:str
    table:str
    column:str
    setmap:Dict[str, int]

    @property
    def name(self):
        return SPLITER.join([self.table, self.column, self.code])
    
    @property
    def keys(self):
        return list(self.setmap.keys())
    
    @property
    def values(self):
        return list(self.setmap.values())
    
    def decode(self, key):
        return {v:k for k, v in self.setmap.items()}[key]


class SetCodeManager:
     
    CS_MAP = {}

    @classmethod
    def register(cls, set_code:str, set_info:SetCode):
        if set_code in cls.CS_MAP:
            invalid = False if len(set_info) == len(cls.CS_MAP[set_code]) else True
            registered = cls.CS_MAP[set_code]
            for old, new in zip(registered, set_info):
                invalid &= old != new
            if invalid:
                raise RuntimeError(f'inconsistent definition of {set_code} encountered')
        if not isinstance(set_info, SetCode):
            raise TypeError(f'column info can only be of type {SetCode}')
        cls.CS_MAP[set_code] = set_info

    @classmethod
    def get(cls, code:str) -> SetCode:
        if code not in cls.CS_MAP:
            raise KeyError(f'code {code} not found')
        return cls.CS_MAP[code]

    @classmethod
    def dump(cls):
        pickle.dump(cls.CS_MAP, open(OUTPUT_LOC+'/columnsets.pkl', 'wb'))
        json.dump(cls.CS_MAP, open(
            OUTPUT_LOC+'/columnsets.json', 'w', encoding='utf-8'), ensure_ascii=False)

    @classmethod
    def load(cls):
        if os.path.isfile(OUTPUT_LOC+'/columnsets.pkl'):
            registries:dict = pickle.load(open(OUTPUT_LOC+'/columnsets.pkl', 'rb'))
            for key, reg in registries.items():
                cls.register(key, reg)   


class ExtendedColumn(NamedTuple):
    code:str
    table:str
    column:str
    label:Optional[str]=None
    dtype:Optional[Dtypes]=None
    methods:Optional[Tuple[str]]=None
    nullable:Optional[bool]=None
    range:Optional[dataRangeType]=None # comparison may cause issues
    transforms:Optional[Tuple[str]]=None

    @property
    def name(self):
        return SPLITER.join([self.table, self.column, self.code])


class ColumnManager:
    
    COL_MAP = {}

    @classmethod
    def register(cls, col_code:str, col_info:ExtendedColumn):
        if col_code in cls.COL_MAP and (not col_code in [PK, PK2]):
            invalid = False if len(col_info) == len(cls.COL_MAP[col_code]) else True
            registered = cls.COL_MAP[col_code]
            for old, new in zip(registered, col_info):
                invalid &= old != new
            if invalid:
                raise RuntimeError(f'inconsistent definition of {col_code} encountered')
        if not isinstance(col_info, ExtendedColumn):
            raise TypeError(f'column info can only be of type {ExtendedColumn}')
        cls.COL_MAP[col_code] = col_info
    
    @classmethod
    def get(cls, code:str) -> ExtendedColumn:
        if code not in cls.COL_MAP:
            raise KeyError(f'code {code} not found')
        return cls.COL_MAP[code]

    @classmethod
    def dump(cls):
        pickle.dump(cls.COL_MAP, open(OUTPUT_LOC+'/columns.pkl', 'wb'))
        json.dump(cls.COL_MAP, open(
            OUTPUT_LOC+'/columns.json', 'w', encoding='utf-8'), ensure_ascii=False)

    @classmethod
    def load(cls):
        if os.path.isfile(OUTPUT_LOC+'/columns.pkl'):
            registries:dict = pickle.load(open(OUTPUT_LOC+'/columns.pkl', 'rb'))
            for key, reg in registries.items():
                cls.register(key, reg)

