# -*- coding: utf-8 -*-
from collections import namedtuple
from enum import Enum
from typing import Dict, List, NamedTuple, Union, Optional

SCHEMA_CONFIG_LOC = './_c2c_p2p/_local_db_config.json'
OUTPUT_LOC = './_c2c_p2p/_c2c_p2p_out'
LOG_LOC = OUTPUT_LOC + '/log'


class RefNames(NamedTuple):
    file_name: str
    main_def: str
    set_def: str
    value_def: str
    ds_sheet: str
    target_file: str


class Intervals(str, Enum):
    LO = '左开区间'
    RO = '右开区间'
    LC = '左闭区间'
    RC = '右闭区间'


class typeNames(str, Enum):
    VALUE = '原值'
    SET = '组别'

    @classmethod
    def get(cls, value: str) -> 'typeNames':
        for each in cls:
            if value == each:
                return each
        return None


class SchemaTableRefs(RefNames, Enum):
    PROD_INFO = RefNames('./产品类.xls', '产品类_分析', '产品类_组别编号',
                         '产品类_原值编号', '产品类_分析', '产品类')
    CLIENT_TRANS = RefNames('./客户交易.xls', '客户交易_分析',
                            '客户交易_组别编号', '客户交易_原值编号', '客户交易订单类', '客户交易')
    CLIENT_HODING = RefNames('./客户持仓.xls', '客户持仓_分析',
                             '客户持仓_组别编号', '客户持仓_原值编号', '客户持仓类', '客户持仓')
    CLIENT_TPYE = RefNames('./客户类别.xls', '客户类别_分析',
                           '客户类别_组别编号', '客户类别_原值编号', '客户类别', '客户类别')


class FieldInfoNames(str, Enum):
    CODE = '变数代码'
    LABEL = '自变量分析标签'
    CATEGORY = '自变数集'
    NAME = '变数名称'
    NULLABLE = '可否空值'
    TYPE = '组别/原值'
    SET_CODE = '组别/原值编号'


dataRangeType = Union[Dict[str, Optional[Union[int, float]]], List[int]]
columnAttrType = Dict[str, Union[str, bool, dataRangeType]]
tableType = Dict[str, columnAttrType]
configType = Dict[str, Dict[str, Union[str, tableType]]]

ExtendedColumn = namedtuple(
    'ExtendedColumn', ['code', 't_name', 'label', 'name', 'method'])
SPLITER = '`'
Task = namedtuple('Task', ['x', 'y', 'name', 'task'])
