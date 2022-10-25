# -*- coding: utf-8 -*-
from collections import namedtuple
import json
import os
import pickle
from enum import Enum
from typing import Callable, Dict, Generator, List, NamedTuple, Union, Optional, Tuple, Any

import pandas as pd

SCHEMA_CONFIG_LOC = './_c2c_p2p/_local_db_config.json'
OUTPUT_LOC = './_c2c_p2p/_c2c_p2p_out'
LOG_LOC = OUTPUT_LOC + '/log'
PK = 'customerid'
PK2 = 'prod_code'
SPLITER = '`'

Task = namedtuple('Task', ['x', 'y', 'name', 'task'])

dataRangeType = Dict[str, Optional[Union[int, float]]]
columnAttrType = Dict[str, Union[str, bool, dataRangeType]]
tableFieldType = Dict[str, columnAttrType]
configType = Dict[str, Dict[str, Union[str, tableFieldType]]]

taskArgType = Union[str, List[str]]
dataGeneratorType = Callable[[Tuple[taskArgType, taskArgType]], Generator[Tuple[pd.DataFrame, pd.Series], Any, None]]
dataFuncType = Callable[[Tuple[taskArgType, taskArgType]], Tuple[pd.DataFrame, pd.DataFrame]]


class AlgorithmCodes(str, Enum):
    REG = 'reg'
    DTC = 'dtc'

class Dtypes(str, Enum):
    FLOAT = 'float'
    INT = 'integer'
    SET = 'set'
    DATE = 'date'
    STRING = 'string'

class TableNames(str, Enum):
    PROD_INFO = '产品类'
    CLIENT_TRANS = '客户交易'
    CLIENT_HODING = '客户持仓'
    CLIENT_TPYE = '客户类别'

class RefNames(NamedTuple):
    file_name: str
    main_def: str
    set_def: str
    value_def: str
    target_file: str


class SchemaTableRefs(RefNames, Enum):
    PROD_INFO = RefNames('./产品类.xls', '产品类_分析', '产品类_组别编号', '产品类_原值编号', '产品类')
    CLIENT_TRANS = RefNames('./客户交易.xls', '客户交易_分析', '客户交易_组别编号', '客户交易_原值编号', '客户交易')
    CLIENT_HODING = RefNames('./客户持仓.xls', '客户持仓_分析', '客户持仓_组别编号', '客户持仓_原值编号', '客户持仓')
    CLIENT_TPYE = RefNames('./客户类别.xls', '客户类别_分析', '客户类别_组别编号', '客户类别_原值编号', '客户类别')


class ValueColumns(str, Enum):
    LO = '左开区间'
    RO = '右开区间'
    LC = '左闭区间'
    RC = '右闭区间'
    NAME = '名称'
    VALUEDTYPE = '资料型态'

class DataCateNames(str, Enum):
    VALUE = '原值'
    SET = '组别'

    @classmethod
    def get(cls, value: str) -> 'DataCateNames':
        for each in cls:
            if value == each:
                return each
        return None


class FieldInfoNames(str, Enum):
    CODE = '变数代码'
    LABEL = '自变量分析标签'
    SUBGROUP = '自变数集'
    NAME = '变数名称'
    NULLABLE = '可否空值'
    CATE = '组别/原值'
    CATE_CODE = '组别/原值编号'


class SetCode(NamedTuple):
    table:str
    column:str
    s_dict:dict

    @property
    def name(self):
        return SPLITER.join([self.table, self.column])
    
    @property
    def keys(self):
        return list(self.s_dict.keys())

    @property
    def values(self):
        return list(self.s_dict.values())

    def __getitem__(self, key):
        return self.s_dict[key]


class SetCodeManager:
     
    CS_MAP = {}

    @classmethod
    def register(cls, set_code:str, set_info:SetCode):
        if set_code in cls.CS_MAP:
            invalid = False if len(set_info) == len(cls.COL_MAP[set_code]) else True
            registered = cls.CS_MAP[set_code]
            for old, new in zip(registered, set_info):
                invalid &= old != new
            if invalid:
                raise RuntimeError(f'inconsistent definition of {set_code} encountered')
        if not isinstance(set_info, ExtendedColumn):
            raise TypeError(f'column info can only be of type {type(ExtendedColumn)}')
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
            raise TypeError(f'column info can only be of type {type(ExtendedColumn)}')
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
